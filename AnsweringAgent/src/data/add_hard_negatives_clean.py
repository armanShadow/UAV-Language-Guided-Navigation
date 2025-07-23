#!/usr/bin/env python3
"""
Clean Hard Negative Mining Script for AVDN Dataset

Simplified version focusing on:
- Visual K-NN for hard negatives
- Visual clustering for diverse negatives  
- Simple phrase reuse limits for diversity
- Fast performance with minimal complexity

Usage:
    python add_hard_negatives_clean.py --image-dir /path/to/images --split train
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import datetime
import time
from typing import Dict, List, Tuple, Any, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import argparse
from tqdm import tqdm
import random
import sys

# Add the parent directory to the path to access modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.Normalizer import AnsweringAgentNormalizer
    from config import Config
except ImportError:
    from Normalizer import AnsweringAgentNormalizer
    from config import Config

from transformers import T5Tokenizer


class CleanHardNegativeMiner:
    """Clean implementation of hard negative mining with essential features only."""
    
    def __init__(self, config: Config, tokenizer, k_nn: int = 100, cosine_threshold: float = 0.2,
                 diverse_ratio: float = 0.0, min_answer_length: int = 20,
                 min_visual_similarity: float = 0.15, phrase_reuse_limit: int = 3,
                 sliding_window_size: int = 1000):
        """
        Initialize the clean hard negative miner.
        
        Args:
            config: Configuration object
            tokenizer: Tokenizer for text processing
            k_nn: Number of nearest neighbors for visual similarity
            cosine_threshold: Text similarity threshold for hard negatives
            diverse_ratio: Ratio of samples to try diverse first (0.0 = always hard first)
            min_answer_length: Minimum answer length to consider
            min_visual_similarity: Minimum visual similarity for hard negatives
            phrase_reuse_limit: Base reuse limit for phrase diversity
            sliding_window_size: Number of recent phrases to track for diversity
        """
        self.config = config
        self.tokenizer = tokenizer
        self.k_nn = k_nn
        self.cosine_threshold = cosine_threshold
        self.diverse_ratio = diverse_ratio
        self.min_answer_length = min_answer_length
        self.min_visual_similarity = min_visual_similarity
        self.phrase_reuse_limit = phrase_reuse_limit
        self.sliding_window_size = sliding_window_size
        
        # GPU settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 64
        
        # Initialize normalizer for answer quality checking
        self.normalizer = AnsweringAgentNormalizer(tokenizer, config, generate_mpnet_embeddings=True)
        
        # Visual features and models
        self.visual_features = {}
        self.visual_knn = None
        self.visual_indices = []
        self.visual_clusters = None
        self.cluster_labels = None
        
        # Text features cache
        self.text_features = {}
        
        # Phrase diversity tracking (sliding window)
        self.used_phrases = {}
        
        # Answer quality components
        self.blacklist_embeddings = {}
        self.answer_embedding_cache = {}
        self.answer_quality_cache = {}
        self.semantic_similarity_threshold = 0.75
        
        # Answer quality blacklist
        self.answer_blacklist = {
            'short_affirmative': [
                'yes', 'exactly', 'correct', 'right', 'true', 'sure', 'okay', 'ok',
                "that's correct", "that's right", "that's true", "you are correct", "absolutely"
            ],
            'generic_responses': [
                'destiny is exactly that', 'that is correct', 'you are right', 'that is it',
                'yes that is correct', 'yes exactly', 'exactly that'
            ],
            'minimal_answers': [
                'go', 'turn', 'move', 'head', 'fly', 'navigate',
                'proceed', 'continue', 'advance', 'straight ahead'
            ]
        }
    
    def _normalize_answer(self, answer: str) -> str:
        """Consistent normalization for all caches."""
        return answer.lower().strip() if answer else ""
    
    def _initialize_answer_quality_filter(self):
        """Initialize semantic filtering for answer quality."""
        print("🔍 Initializing answer quality filter...")
        
        # Generate embeddings for blacklisted phrases
        for category, phrases in self.answer_blacklist.items():
            for phrase in phrases:
                try:
                    embedding = self.normalizer.generate_mpnet_embedding(phrase)
                    self.blacklist_embeddings[phrase] = embedding
                except Exception as e:
                    print(f"⚠️ Error generating embedding for '{phrase}': {e}")
        
        print(f"✅ Loaded {len(self.blacklist_embeddings)} blacklist embeddings")
    
    def _precompute_embeddings_and_quality(self, dataset: Dict[int, Dict[str, Any]]):
        """Precompute embeddings and answer quality for performance."""
        print("🔄 Precomputing embeddings and answer quality...")
        
        # Collect unique answers
        unique_answers = set()
        for item in dataset.values():
            answer = item.get('answer', '')
            if answer:
                normalized = self._normalize_answer(answer)
                unique_answers.add(normalized)
        
        print(f"  Processing {len(unique_answers)} unique answers...")
        
        # Precompute embeddings and quality
        good_count = 0
        bad_count = 0
        
        for normalized_answer in tqdm(unique_answers, desc="Computing embeddings"):
            # Compute embedding
            try:
                embedding = self.normalizer.generate_mpnet_embedding(normalized_answer)
                self.answer_embedding_cache[normalized_answer] = embedding
            except Exception as e:
                print(f"⚠️ Failed to generate embedding for '{normalized_answer[:50]}...': {e}")
                # Create zero embedding as fallback to maintain consistency
                self.answer_embedding_cache[normalized_answer] = np.zeros(768, dtype=np.float32)
            
            # Compute quality
            is_good = self._evaluate_answer_quality(normalized_answer)
            self.answer_quality_cache[normalized_answer] = is_good
            
            if is_good:
                good_count += 1
            else:
                bad_count += 1
        
        # Add precomputed quality to dataset items
        for item in dataset.values():
            answer = item.get('answer', '')
            if answer:
                normalized = self._normalize_answer(answer)
                item['_precomputed_quality'] = self.answer_quality_cache.get(normalized, False)
        
        print(f"✅ Precomputed {len(self.answer_embedding_cache)} embeddings")
        print(f"✅ Quality: {good_count} good, {bad_count} bad answers")
    
    def _evaluate_answer_quality(self, normalized_answer: str) -> bool:
        """Internal method to evaluate answer quality."""
        if not normalized_answer or len(normalized_answer) < self.min_answer_length:
            return False
        
        # Check semantic similarity to blacklisted phrases
        if self.blacklist_embeddings and normalized_answer in self.answer_embedding_cache:
            answer_embedding = self.answer_embedding_cache[normalized_answer]
            
            for blacklist_embedding in self.blacklist_embeddings.values():
                similarity = np.dot(answer_embedding, blacklist_embedding)
                if similarity > self.semantic_similarity_threshold:
                    return False
        
        return True
    
    def _is_good_answer(self, answer: str, dataset_item: Dict = None) -> bool:
        """Check if answer meets quality requirements."""
        # Use precomputed quality if available
        if dataset_item and '_precomputed_quality' in dataset_item:
            return dataset_item['_precomputed_quality']
        
        normalized = self._normalize_answer(answer)
        return self.answer_quality_cache.get(normalized, False)
    
    def _is_phrase_diverse(self, answer: str) -> bool:
        """Check if phrase satisfies diversity requirements."""
        if not answer:
            return False
        
        normalized = self._normalize_answer(answer)
        
        # Check reuse limits
        current_uses = self.used_phrases.get(normalized, 0)
        max_uses = self._get_reuse_limit(len(normalized))
        
        return current_uses < max_uses
    
    def _get_reuse_limit(self, phrase_length: int) -> int:
        """Get reuse limit based on phrase length."""
        if phrase_length < 60:
            return 1
        elif phrase_length < 100:
            return 2
        else:
            return self.phrase_reuse_limit
    
    def _track_phrase_usage(self, answer: str):
        """Track phrase usage with sliding window."""
        if not answer:
            return
        
        normalized = self._normalize_answer(answer)
        self.used_phrases[normalized] = self.used_phrases.get(normalized, 0) + 1
        
        # Maintain sliding window
        if len(self.used_phrases) > self.sliding_window_size:
            # Remove oldest entry
            oldest_key = next(iter(self.used_phrases))
            del self.used_phrases[oldest_key]
    
    def _extract_visual_features(self, current_view: torch.Tensor) -> np.ndarray:
        """Extract visual features from current view."""
        if current_view.device != self.device:
            current_view = current_view.to(self.device)
        
        # Simple adaptive pooling
        features = F.adaptive_avg_pool2d(current_view.unsqueeze(0), (8, 8))
        features = features.view(-1)
        features = features / (torch.norm(features) + 1e-8)
        
        return features.cpu().numpy()
    
    def _extract_text_features(self, text: str) -> np.ndarray:
        """Extract text features using cached MPNet embeddings."""
        normalized = self._normalize_answer(text)
        
        # All embeddings should be precomputed and cached
        if normalized in self.answer_embedding_cache:
            return self.answer_embedding_cache[normalized]
        
        # This should never happen if precomputation worked correctly
        print(f"⚠️ Missing cached embedding for: '{normalized[:50]}...'")
        print(f"   This indicates a bug in precomputation!")
        
        # Emergency fallback - but this should never be reached
        try:
            embedding = self.normalizer.generate_mpnet_embedding(normalized)
            self.answer_embedding_cache[normalized] = embedding  # Cache for future
            return embedding
        except Exception:
            # Last resort zero embedding
            zero_embedding = np.zeros(768, dtype=np.float32)
            self.answer_embedding_cache[normalized] = zero_embedding
            return zero_embedding
    
    def _build_visual_knn(self, dataset: Dict[int, Dict[str, Any]]):
        """Build K-NN model for visual similarity."""
        print("🔍 Building visual K-NN model...")
        
        visual_features_list = []
        self.visual_indices = []
        
        # Process in batches for better performance
        items_list = list(dataset.items())
        batch_size = self.batch_size
        
        for batch_start in tqdm(range(0, len(items_list), batch_size), desc="Extracting visual features"):
            batch_end = min(batch_start + batch_size, len(items_list))
            batch_items = items_list[batch_start:batch_end]
            
            batch_tensors = []
            batch_indices = []
            
            for idx, item in batch_items:
                try:
                    current_view = item['current_view_image']
                    batch_tensors.append(current_view)
                    batch_indices.append(idx)
                except Exception:
                    continue
            
            if not batch_tensors:
                continue
            
            # Process batch
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            try:
                # Batch process visual features
                features = F.adaptive_avg_pool2d(batch_tensor, (8, 8))
                features = features.view(len(batch_tensors), -1)
                features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
                 
                for i, idx in enumerate(batch_indices):
                    feature_np = features[i].cpu().numpy()
                    visual_features_list.append(feature_np)
                    self.visual_indices.append(idx)
                    self.visual_features[idx] = feature_np
            except Exception:
                # Fallback to individual processing for this batch
                for idx, current_view in zip(batch_indices, batch_tensors):
                    try:
                        features = self._extract_visual_features(current_view)
                        visual_features_list.append(features)
                        self.visual_indices.append(idx)
                        self.visual_features[idx] = features
                    except Exception:
                        continue
        
        if visual_features_list:
            visual_features_array = np.array(visual_features_list)
            k_neighbors = min(self.k_nn + 1, len(visual_features_array))
            self.visual_knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
            self.visual_knn.fit(visual_features_array)
            print(f"✅ Built K-NN model with {len(visual_features_list)} samples (K={k_neighbors})")
        else:
            print("❌ No visual features extracted!")
    
    def _build_visual_clusters(self, dataset: Dict[int, Dict[str, Any]], n_clusters: int = 30):
        """Build visual clusters for diverse negative sampling."""
        print("🔍 Building visual clusters...")
        
        if not self.visual_features:
            print("❌ No visual features available for clustering!")
            return
        
        visual_features_list = []
        for idx in self.visual_indices:
            visual_features_list.append(self.visual_features[idx])
        
        if len(visual_features_list) >= n_clusters:
            visual_features_array = np.array(visual_features_list)
            self.visual_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = self.visual_clusters.fit_predict(visual_features_array)
            print(f"✅ Built {n_clusters} visual clusters")
        else:
            # Fallback for small datasets
            self.cluster_labels = np.random.randint(0, 2, size=len(visual_features_list))
            print(f"✅ Using random cluster assignment for small dataset")
    
    def _find_hard_negative(self, anchor_idx: int, dataset: Dict[int, Dict[str, Any]]) -> Optional[tuple]:
        """Find hard negative using visual K-NN and text dissimilarity."""
        if anchor_idx not in self.visual_features or self.visual_knn is None:
            return None
        
        anchor_item = dataset[anchor_idx]
        anchor_features = self.visual_features[anchor_idx]
        anchor_answer = anchor_item.get('answer', '')
        anchor_instruction = anchor_item.get('first_instruction', '')
        
        # Get visual neighbors
        distances, indices = self.visual_knn.kneighbors([anchor_features])
        neighbor_indices = indices[0][1:]  # Skip self
        neighbor_distances = distances[0][1:]
        
        # Pre-compute anchor text features
        anchor_text_features = self._extract_text_features(anchor_answer)
        
        best_candidate = None
        lowest_text_similarity = float('inf')
        best_visual_similarity = None
        
        for i, pos in enumerate(neighbor_indices):
            sample_idx = self.visual_indices[pos]
            if sample_idx not in dataset:
                continue
            
            visual_similarity = 1.0 - neighbor_distances[i]
            
            # Skip if visual similarity too low
            if visual_similarity < self.min_visual_similarity:
                break
            
            neighbor_item = dataset[sample_idx]
            neighbor_answer = neighbor_item.get('answer', '')
            neighbor_instruction = neighbor_item.get('first_instruction', '')
            
            # Skip if same goal
            if anchor_instruction == neighbor_instruction:
                continue
            
            # Check answer quality
            if not self._is_good_answer(neighbor_answer, neighbor_item):
                continue
            
            # Check phrase diversity
            if not self._is_phrase_diverse(neighbor_answer):
                continue
            
            # Calculate text similarity
            neighbor_text_features = self._extract_text_features(neighbor_answer)
            text_similarity = np.dot(anchor_text_features, neighbor_text_features)
            
            # Check if text similarity is below threshold
            if text_similarity >= self.cosine_threshold:
                continue
            
            # Found a good candidate
            if text_similarity < lowest_text_similarity:
                best_candidate = sample_idx
                lowest_text_similarity = text_similarity
                best_visual_similarity = visual_similarity
        
        if best_candidate is not None:
            return (best_candidate, lowest_text_similarity, best_visual_similarity)
        
        return None
    
    def _find_diverse_negative(self, anchor_idx: int, dataset: Dict[int, Dict[str, Any]]) -> Optional[tuple]:
        """Find diverse negative from different visual cluster."""
        if anchor_idx not in self.visual_features or self.cluster_labels is None:
            return None
        
        anchor_item = dataset[anchor_idx]
        anchor_instruction = anchor_item.get('first_instruction', '')
        anchor_features = self.visual_features[anchor_idx]
        
        # Find anchor's cluster
        anchor_idx_in_array = self.visual_indices.index(anchor_idx)
        anchor_cluster = self.cluster_labels[anchor_idx_in_array]
        
        # Find candidates from different clusters
        candidates = []
        
        for i, cluster_label in enumerate(self.cluster_labels):
            sample_idx = self.visual_indices[i]
            if sample_idx in dataset and sample_idx != anchor_idx:
                neighbor_item = dataset[sample_idx]
                neighbor_instruction = neighbor_item.get('first_instruction', '')
                neighbor_answer = neighbor_item.get('answer', '')
                
                # Skip if same goal
                if anchor_instruction == neighbor_instruction:
                    continue
                
                # Check answer quality
                if not self._is_good_answer(neighbor_answer, neighbor_item):
                    continue
                
                # Check phrase diversity
                if not self._is_phrase_diverse(neighbor_answer):
                    continue
                
                neighbor_features = self.visual_features[sample_idx]
                visual_similarity = np.dot(anchor_features, neighbor_features)
                
                # Prioritize different clusters
                if cluster_label != anchor_cluster:
                    candidates.append((sample_idx, cluster_label, visual_similarity, True))
                else:
                    candidates.append((sample_idx, cluster_label, visual_similarity, False))
        
        if not candidates:
            return None
        
        # Sort by cluster diversity first, then by visual similarity (ascending for diversity)
        candidates.sort(key=lambda x: (not x[3], x[2]))
        
        selected_idx, selected_cluster, visual_similarity, is_different_cluster = candidates[0]
        return (selected_idx, anchor_cluster, selected_cluster, visual_similarity)
    
    def mine_hard_negatives(self, dataset: Dict[int, Dict[str, Any]], 
                           max_samples: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
        """Mine hard negatives for the dataset."""
        print("⛏️ Mining hard negatives...")
        
        # Initialize answer quality filter
        self._initialize_answer_quality_filter()
        
        # Precompute embeddings and answer quality
        self._precompute_embeddings_and_quality(dataset)
        
        # Build visual models
        self._build_visual_knn(dataset)
        self._build_visual_clusters(dataset)
        
        # Mine negatives
        negatives = {}
        samples_to_process = list(dataset.keys())[:max_samples] if max_samples else list(dataset.keys())
        
        stats = {
            'total_attempts': 0,
            'hard_attempts': 0,
            'diverse_attempts': 0,
            'hard_success': 0,
            'diverse_success': 0,
            'fallback_used': 0
        }
        
        start_time = time.time()
        
        for anchor_idx in tqdm(samples_to_process, desc="Mining negatives"):
            stats['total_attempts'] += 1
            
            # Decide strategy
            if random.random() < self.diverse_ratio:
                strategy_order = ["diverse", "hard"]
            else:
                strategy_order = ["hard", "diverse"]
            
            negative_result = None
            negative_type = None
            
            for strategy in strategy_order:
                if strategy == "hard":
                    stats['hard_attempts'] += 1
                    negative_result = self._find_hard_negative(anchor_idx, dataset)
                else:
                    stats['diverse_attempts'] += 1
                    negative_result = self._find_diverse_negative(anchor_idx, dataset)
                
                if negative_result is not None:
                    negative_type = strategy
                    if strategy != strategy_order[0]:
                        stats['fallback_used'] += 1
                    break
            
            if negative_result is None:
                continue
            
            # Track success
            if negative_type == "hard":
                stats['hard_success'] += 1
            else:
                stats['diverse_success'] += 1
            
            # Create negative data
            if negative_type == "hard":
                negative_idx, text_similarity, visual_similarity = negative_result
                validation_info = {
                    'negative_type_2': negative_type,
                    'text_similarity': float(text_similarity),
                    'visual_similarity': float(visual_similarity),
                    'mining_method': 'hard_negative_knn'
                }
            else:
                negative_idx, anchor_cluster, negative_cluster, visual_similarity = negative_result
                validation_info = {
                    'negative_type_2': negative_type,
                    'anchor_cluster': int(anchor_cluster),
                    'negative_cluster': int(negative_cluster),
                    'visual_similarity': float(visual_similarity),
                    'mining_method': 'diverse_negative_clustering'
                }
            
            negative_item = dataset[negative_idx]
            self._track_phrase_usage(negative_item.get('answer', ''))
            
            negative_data = {
                'negative_text_2': negative_item.get('answer', ''),
                'negative_context_2': negative_item.get('dialog_context', ''),
                'negative_question_2': negative_item.get('question', ''),
                'negative_first_instruction_2': negative_item.get('first_instruction', ''),
                'negative_visual_features_2': negative_item.get('current_view_image', None),
                'negative_type_2': negative_type,
                'map_name_2': negative_item.get('map_name', 'unknown'),
                'validation_metadata_2': validation_info
            }
            
            if self.tokenizer:
                negative_data['tokenized_negative_2'] = self.tokenizer(
                    negative_data['negative_text_2'],
                    max_length=self.config.model.max_answer_length if self.config else 128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
            
            negatives[anchor_idx] = negative_data
        
        # Print results
        total_time = time.time() - start_time
        success_rate = len(negatives) / stats['total_attempts'] * 100 if stats['total_attempts'] > 0 else 0
        
        print(f"\n📊 Mining Results:")
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Hard attempts: {stats['hard_attempts']} (success: {stats['hard_success']})")
        print(f"  Diverse attempts: {stats['diverse_attempts']} (success: {stats['diverse_success']})")
        print(f"  Fallback used: {stats['fallback_used']}")
        print(f"  Success rate: {len(negatives)}/{stats['total_attempts']} ({success_rate:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        
        hard_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'hard')
        diverse_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'diverse')
        
        print(f"  Hard negatives: {hard_count}, Diverse negatives: {diverse_count}")
        print(f"  Phrase diversity: {len(self.used_phrases)} unique phrases")
        
        if negatives:
            unique_phrases = set(data['negative_text_2'].lower().strip() for data in negatives.values())
            diversity_ratio = len(unique_phrases) / len(negatives)
            print(f"  Unique phrase ratio: {diversity_ratio:.3f}")
        
        return negatives
    
    def add_negatives_to_dataset(self, dataset: Dict[int, Dict[str, Any]], 
                                negatives: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """Add negatives to the existing dataset."""
        print("➕ Adding negatives to dataset...")
        
        updated_dataset = dataset.copy()
        
        for anchor_idx, negative_data in tqdm(negatives.items(), desc="Adding negatives"):
            if anchor_idx in updated_dataset:
                anchor_item = updated_dataset[anchor_idx]
                
                if 'contrastive_data' not in anchor_item:
                    anchor_item['contrastive_data'] = {}
                
                anchor_item['contrastive_data']['negative_text_2'] = negative_data['negative_text_2']
                anchor_item['contrastive_data']['tokenized_negative_2'] = negative_data['tokenized_negative_2']
                anchor_item['contrastive_data']['validation_metadata_negative_2'] = {
                    'negative_type_2': negative_data['negative_type_2'],
                    'map_name_2': negative_data['map_name_2'],
                    'mining_timestamp': datetime.datetime.now().isoformat(),
                    **negative_data['validation_metadata_2']
                }
                
                updated_dataset[anchor_idx] = anchor_item
        
        print(f"✅ Added negatives to {len(negatives)} samples")
        return updated_dataset


def load_dataset(config: Config, split: str) -> Dict[int, Dict[str, Any]]:
    """Load the processed dataset."""
    print(f"📊 Loading {split} dataset...")
    
    if split == 'train':
        try:
            from dataset import AnsweringDataset
        except ImportError:
            from data.dataset import AnsweringDataset
        dataset = AnsweringDataset.load_train_chunks(config.data.train_processed_path_dir)
    else:
        if split == 'val_seen':
            data_path = config.data.val_seen_processed_path
        else:
            data_path = config.data.val_unseen_processed_path
        
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
    
    print(f"✅ Loaded {len(dataset)} samples")
    return dataset


def save_dataset(dataset: Dict[int, Dict[str, Any]], config: Config, split: str):
    """Save the updated dataset."""
    print(f"💾 Saving updated {split} dataset...")
    
    if split == 'train':
        try:
            from dataset import AnsweringDataset
        except ImportError:
            from data.dataset import AnsweringDataset
        output_dir = config.data.train_processed_path_dir
        os.makedirs(output_dir, exist_ok=True)
        
        for file in os.listdir(output_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(output_dir, file))
        
        chunk_size = 1000
        AnsweringDataset.save_in_chunks(dataset, chunk_size, output_dir)
        print(f"✅ Saved train data in chunks to {output_dir}")
    else:
        if split == 'val_seen':
            output_path = config.data.val_seen_processed_path
        else:
            output_path = config.data.val_unseen_processed_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"✅ Saved {split} data to {output_path}")


def main():
    """Main function to add hard negatives to dataset."""
    parser = argparse.ArgumentParser(description='Clean hard negative mining for AVDN dataset')
    parser.add_argument('--config', type=str, default='config.py', help='Path to config file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val_seen', 'val_unseen'], 
                       help='Dataset split to process')
    parser.add_argument('--k-nn', type=int, default=100, help='Number of K-NN neighbors to consider')
    parser.add_argument('--cosine-threshold', type=float, default=0.2, 
                       help='Text similarity threshold for hard negatives')
    parser.add_argument('--max-samples', type=int, default=None, 
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--diverse-ratio', type=float, default=0.0,
                       help='Ratio of samples to try diverse first (0.0 = always hard first)')
    parser.add_argument('--min-answer-length', type=int, default=20,
                       help='Minimum answer length to consider')
    parser.add_argument('--min-visual-similarity', type=float, default=0.15,
                       help='Minimum visual similarity for hard negatives')
    parser.add_argument('--phrase-reuse-limit', type=int, default=3,
                       help='Base phrase reuse limit for diversity')
    parser.add_argument('--sliding-window-size', type=int, default=1000,
                       help='Number of recent phrases to track for diversity')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for GPU processing')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set up GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"🚀 Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Load configuration and dataset
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    dataset = load_dataset(config, args.split)
    
    print(f"🚀 Starting clean mining on GPU {args.gpu_id}")
    print(f"📊 Processing {len(dataset)} samples")
    
    # Create miner
    miner = CleanHardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        k_nn=args.k_nn,
        cosine_threshold=args.cosine_threshold,
        diverse_ratio=args.diverse_ratio,
        min_answer_length=args.min_answer_length,
        min_visual_similarity=args.min_visual_similarity,
        phrase_reuse_limit=args.phrase_reuse_limit,
        sliding_window_size=args.sliding_window_size
    )
    
    miner.batch_size = args.batch_size
    miner.device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    
    # Mine negatives
    hard_negatives = miner.mine_hard_negatives(dataset, max_samples=args.max_samples)
    updated_dataset = miner.add_negatives_to_dataset(dataset, hard_negatives)
    
    # Save results
    save_dataset(updated_dataset, config, args.split)


if __name__ == '__main__':
    main() 