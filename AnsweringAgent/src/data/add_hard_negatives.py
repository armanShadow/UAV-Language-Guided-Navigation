#!/usr/bin/env python3
"""
Hard Negative Mining Script for AVDN Dataset

This script adds hard negative samples to the existing dataset by:
1. Mining hard negatives using visual K-NN + least-similar instruction
2. Adding diverse negatives from outside nearest visual clusters
3. Adding one hard negative per anchor to the existing dataset
4. Preserving the current LM negative if it costs little

Usage:
    python add_hard_negatives.py --config config.py --split train --k-nn 50 --cosine-threshold 0.3
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import datetime
from typing import Dict, List, Tuple, Any, Optional
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import argparse
from tqdm import tqdm
import random
from collections import defaultdict

# Import local modules
from data.Normalizer import AnsweringAgentNormalizer
from config import Config
from transformers import T5Tokenizer

class HardNegativeMiner:
    """Mines hard negative samples using visual K-NN and least-similar instruction selection."""
    
    def __init__(self, config: Config, tokenizer, image_dir: str, k_nn: int = 30, cosine_threshold: float = 0.3,
                 use_diverse_negatives: bool = True, diverse_ratio: float = 0.3, min_answer_length: int = 20):
        """
        Initialize the hard negative miner.
        
        Args:
            config: Configuration object
            tokenizer: Tokenizer for text processing
            image_dir: Directory containing satellite images
            k_nn: Number of nearest neighbors to consider for visual similarity
            cosine_threshold: Threshold for considering instructions as dissimilar
            use_diverse_negatives: Whether to add diverse negatives from outside clusters
            diverse_ratio: Ratio of samples to use for diverse negative mining (default: 0.3 for 30/70 split)
            min_answer_length: Minimum answer length to consider (default: 20 characters)
        """
        self.config = config
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.k_nn = k_nn
        self.cosine_threshold = cosine_threshold
        self.use_diverse_negatives = use_diverse_negatives
        self.diverse_ratio = diverse_ratio
        self.min_answer_length = min_answer_length
        
        # GPU optimization settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 64  # Process images in batches for GPU efficiency
        self.num_workers = 4  # For parallel data loading
        
        # Initialize normalizer for image processing with MPNet embeddings
        self.normalizer = AnsweringAgentNormalizer(tokenizer, config, generate_mpnet_embeddings=True)
        
        # Storage for mined data
        self.visual_features = {}  # episode_id_turn_id -> visual_features
        self.text_features = {}    # episode_id_turn_id -> text_features
        self.episode_data = {}     # episode_id_turn_id -> episode_data
        
        # K-NN model for visual similarity
        self.visual_knn = None
        self.visual_indices = []
        
        # Clustering for diverse negatives
        self.visual_clusters = None
        self.cluster_labels = None
        
        # Enhanced answer quality filtering
        # Expanded blacklist (token + sentence variants)
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
        
        # Embedding-based blacklist for semantic similarity filtering
        self.blacklist_embeddings = {}  # Will be populated with embeddings of blacklisted phrases
        self.semantic_similarity_threshold = 0.88  # Slightly stricter (captures more semantically similar phrases)
        
        # Duplicate phrase tracking for better diversity
        self.used_phrases = {}  # phrase -> count
        self.max_phrase_reuse = 3  # Maximum times a phrase can be reused
    
    def extract_visual_features(self, current_view: torch.Tensor) -> np.ndarray:
        """
        Extract visual features from current view using a simple CNN.
        This is a lightweight feature extractor for K-NN mining.
        
        Args:
            current_view: Current view image tensor [3, H, W]
            
        Returns:
            Visual features as numpy array
        """
        # Move to GPU if available
        if current_view.device != self.device:
            current_view = current_view.to(self.device)
        
        # Simple feature extraction using average pooling
        # This is lightweight and sufficient for K-NN mining
        features = F.adaptive_avg_pool2d(current_view.unsqueeze(0), (8, 8))
        features = features.view(-1)
        
        # Normalize features on GPU
        features = features / (torch.norm(features) + 1e-8)
        
        # Move back to CPU for numpy operations
        return features.cpu().numpy()
    
    def extract_text_features(self, dialog_context: str) -> np.ndarray:
        """
        Extract text features from dialog context using MPNet embeddings or fallback method.
        
        Args:
            dialog_context: Dialog context string
            
        Returns:
            Text features as numpy array
        """
        # Use MPNet embeddings for better text similarity
        if hasattr(self, 'normalizer') and self.normalizer.generate_mpnet_embeddings:
            try:
                # Generate MPNet embedding
                embedding = self.normalizer.generate_mpnet_embedding(dialog_context)
                return embedding
            except Exception as e:
                print(f"⚠️ Error generating MPNet embedding: {e}")
                # Fallback to simple approach
                pass
        
        # Fallback: Improved TF-IDF like features for text similarity
        import re
        from collections import Counter
        
        # Clean and tokenize text
        text = dialog_context.lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        # Count word frequencies
        word_freq = Counter()
        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_freq[word] += 1
        
        # Create feature vector with fixed size for consistency
        max_features = 1000  # Limit feature size
        unique_words = list(word_freq.keys())[:max_features]
        
        if not unique_words:
            # If no meaningful words, return zero vector
            return np.zeros(100, dtype=np.float32)
        
        # Create feature vector
        features = np.zeros(len(unique_words))
        for i, word in enumerate(unique_words):
            features[i] = word_freq[word]
        
        # Normalize features
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _initialize_blacklist_embeddings(self):
        """Initialize embeddings for blacklisted phrases for semantic similarity checking."""
        if not hasattr(self, 'normalizer') or not self.normalizer.generate_mpnet_embeddings:
            print("⚠️ MPNet embeddings not available, using string-based filtering only")
            return
        
        print("🔍 Initializing blacklist embeddings for semantic filtering...")
        
        cache_path = os.path.join(os.path.dirname(__file__), 'blacklist_embeds.pkl')

        # Try loading cache first
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.blacklist_embeddings = pickle.load(f)
                print(f"✅ Loaded cached blacklist embeddings from {cache_path}")
                return
            except Exception as e:
                print(f"⚠️ Failed to load cached blacklist embeddings: {e}. Recomputing...")

        for category, phrases in self.answer_blacklist.items():
            for phrase in phrases:
                try:
                    # Generate embedding for blacklisted phrase
                    embedding = self.normalizer.generate_mpnet_embedding(phrase)
                    self.blacklist_embeddings[phrase] = embedding
                except Exception as e:
                    print(f"⚠️ Error generating embedding for '{phrase}': {e}")

        # Save cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.blacklist_embeddings, f)
            print(f"💾 Cached blacklist embeddings to {cache_path}")
        except Exception as e:
            print(f"⚠️ Could not cache blacklist embeddings: {e}")
    
    def _check_semantic_similarity_to_blacklist(self, answer: str) -> bool:
        """Check if answer is semantically similar to any blacklisted phrase."""
        if not self.blacklist_embeddings or not hasattr(self, 'normalizer'):
            return False  # Fall back to string-based filtering
        
        try:
            # Generate embedding for the answer
            answer_embedding = self.normalizer.generate_mpnet_embedding(answer)
            
            # Check similarity against all blacklisted embeddings
            for blacklisted_phrase, blacklist_embedding in self.blacklist_embeddings.items():
                similarity = np.dot(answer_embedding, blacklist_embedding)
                if similarity > 0.7 and getattr(self, 'debug_mode', False):
                    print(f"    ↪ sim({similarity:.2f}) to blacklist phrase '{blacklisted_phrase}'")
                if similarity > self.semantic_similarity_threshold:
                    return True  # Answer is semantically similar to a blacklisted phrase
            
            return False  # Answer is not semantically similar to any blacklisted phrase
            
        except Exception as e:
            print(f"⚠️ Error in semantic similarity check: {e}")
            return False  # Fall back to string-based filtering
    
    def is_good_answer(self, answer: str) -> bool:
        """
        Check if an answer is good enough for negative mining.
        
        Args:
            answer: Answer text to evaluate
            
        Returns:
            True if answer is good enough, False otherwise
        """
        if not answer or not isinstance(answer, str):
            return False
        
        # Check minimum length
        if len(answer.strip()) < self.min_answer_length:
            if hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
                print(f"    ❌ Filtered: too short ({len(answer.strip())} chars)")
            return False
        
        # Check for blacklisted phrases
        for category, phrases in self.answer_blacklist.items():
            for phrase in phrases:
                if phrase in answer.lower():
                    if hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
                        print(f"    ❌ Filtered: contains blacklisted phrase '{phrase}'")
                    return False
        
        # Check for semantic similarity to blacklisted phrases
        if self._check_semantic_similarity_to_blacklist(answer):
            if hasattr(self, 'debug_mode') and getattr(self, 'debug_mode', False):
                print(f"    ❌ Filtered: semantically similar to blacklisted phrase -> '{answer[:60]}{'...' if len(answer) > 60 else ''}'")
            return False
        
        return True
    
    def build_visual_clusters(self, dataset: Dict[int, Dict[str, Any]], n_clusters: int = 30):
        """
        Build visual clusters for diverse negative sampling.
        
        Args:
            dataset: Processed dataset dictionary
            n_clusters: Number of clusters to create
        """
        print("🔍 Building visual clusters for diverse negative sampling...")
        
        # Use existing features if already extracted, otherwise extract in batches
        if not self.visual_features:
            # Extract all visual features in batches
            items_list = list(dataset.items())
            batch_size = self.batch_size
            
            for batch_start in tqdm(range(0, len(items_list), batch_size), desc="Extracting visual features for clustering"):
                batch_end = min(batch_start + batch_size, len(items_list))
                batch_items = items_list[batch_start:batch_end]
                
                # Prepare batch tensors
                batch_tensors = []
                batch_indices = []
                
                for idx, item in batch_items:
                    try:
                        current_view = item['current_view_image']
                        batch_tensors.append(current_view)
                        batch_indices.append(idx)
                    except Exception as e:
                        print(f"⚠️ Error preparing item {idx}: {e}")
                        continue
                
                if not batch_tensors:
                    continue
                    
                # Stack tensors and move to GPU
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Extract features in batch
                try:
                    # Batch feature extraction
                    features = F.adaptive_avg_pool2d(batch_tensor, (8, 8))
                    features = features.view(len(batch_tensors), -1)
                    
                    # Normalize features on GPU
                    features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
                    
                    # Store results
                    for i, idx in enumerate(batch_indices):
                        feature_np = features[i].cpu().numpy()
                        self.visual_features[idx] = feature_np
                        
                except Exception as e:
                    print(f"⚠️ Error processing batch {batch_start}-{batch_end}: {e}")
                    continue
        
        # Prepare features list for clustering
        visual_features_list = []
        self.visual_indices = []
        for idx, features in self.visual_features.items():
            visual_features_list.append(features)
            self.visual_indices.append(idx)
        
        if visual_features_list:
            visual_features_array = np.array(visual_features_list)
            
            # Perform K-means clustering
            n_clusters = min(n_clusters, len(visual_features_array) // 10)  # Ensure reasonable cluster size
            self.visual_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = self.visual_clusters.fit_predict(visual_features_array)
            
            print(f"✅ Built {n_clusters} visual clusters with {len(visual_features_list)} samples")
            
            # Print cluster distribution
            unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
            print(f"📊 Cluster distribution: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        else:
            print("❌ No visual features extracted for clustering!")
    
    def find_diverse_negative(self, anchor_idx: int, dataset: Dict[int, Dict[str, Any]], debug_mode: bool = False) -> Optional[tuple]:
        """
        Find a diverse negative from outside the anchor's visual cluster.
        
        Args:
            anchor_idx: Index of the anchor sample
            dataset: Processed dataset dictionary
            debug_mode: Whether to enable debug logging
            
        Returns:
            Tuple of (negative_idx, anchor_cluster, negative_cluster, visual_similarity), or None if not found
        """
        if anchor_idx not in self.visual_features or self.visual_clusters is None:
            return None
        
        anchor_item = dataset[anchor_idx]
        anchor_first_instruction = anchor_item.get('first_instruction', '')
        anchor_features = self.visual_features[anchor_idx]
        
        # Find anchor's cluster
        anchor_idx_in_array = self.visual_indices.index(anchor_idx)
        anchor_cluster = self.cluster_labels[anchor_idx_in_array]
        
        # Find samples from different clusters
        different_cluster_candidates = []
        for i, cluster_label in enumerate(self.cluster_labels):
            # Enforce different clusters for diverse negatives
            if cluster_label != anchor_cluster:
                sample_idx = self.visual_indices[i]
                if sample_idx in dataset:
                    neighbor_item = dataset[sample_idx]
                    neighbor_first_instruction = neighbor_item.get('first_instruction', '')
                    neighbor_answer = neighbor_item.get('answer', '')
                    
                    # Ensure different goal
                    if anchor_first_instruction != neighbor_first_instruction:
                        # Skip if answer is not good enough
                        if not self.is_good_answer(neighbor_answer):
                            # is_good_answer already provides detailed debug output when debug_mode is enabled
                            continue
                        
                        # Check phrase diversity
                        if not self._is_phrase_diverse(neighbor_answer):
                            if debug_mode:
                                print(f"    ❌ Phrase not diverse: '{neighbor_answer[:50]}{'...' if len(neighbor_answer) > 50 else ''}'")
                            continue
                        
                        # Calculate visual similarity
                        neighbor_features = self.visual_features[sample_idx]
                        visual_similarity = np.dot(anchor_features, neighbor_features)
                        different_cluster_candidates.append((sample_idx, cluster_label, visual_similarity))
        
        # If no candidates from different clusters, try any cluster
        if not different_cluster_candidates:
            for i, cluster_label in enumerate(self.cluster_labels):
                sample_idx = self.visual_indices[i]
                if sample_idx in dataset and sample_idx != anchor_idx:
                    neighbor_item = dataset[sample_idx]
                    neighbor_first_instruction = neighbor_item.get('first_instruction', '')
                    neighbor_answer = neighbor_item.get('answer', '')
                    
                    # Ensure different goal
                    if anchor_first_instruction != neighbor_first_instruction:
                        # Skip if answer is not good enough
                        if not self.is_good_answer(neighbor_answer):
                            continue
                        
                        # Calculate visual similarity
                        neighbor_features = self.visual_features[sample_idx]
                        visual_similarity = np.dot(anchor_features, neighbor_features)
                        different_cluster_candidates.append((sample_idx, cluster_label, visual_similarity))
        
        # Randomly select from candidates
        if different_cluster_candidates:
            selected_idx, selected_cluster, visual_similarity = random.choice(different_cluster_candidates)
            return (selected_idx, anchor_cluster, selected_cluster, visual_similarity)
        
        return None
    
    def build_visual_knn(self, dataset: Dict[int, Dict[str, Any]]):
        """
        Build K-NN model for visual similarity.
        
        Args:
            dataset: Processed dataset dictionary
        """
        print("🔍 Building visual K-NN model...")
        
        visual_features_list = []
        self.visual_indices = []
        
        # Batch process visual features for GPU efficiency
        items_list = list(dataset.items())
        batch_size = self.batch_size
        
        for batch_start in tqdm(range(0, len(items_list), batch_size), desc="Extracting visual features"):
            batch_end = min(batch_start + batch_size, len(items_list))
            batch_items = items_list[batch_start:batch_end]
            
            # Prepare batch tensors
            batch_tensors = []
            batch_indices = []
            
            for idx, item in batch_items:
                try:
                    current_view = item['current_view_image']
                    batch_tensors.append(current_view)
                    batch_indices.append(idx)
                except Exception as e:
                    print(f"⚠️ Error preparing item {idx}: {e}")
                    continue
            
            if not batch_tensors:
                continue
                
            # Stack tensors and move to GPU
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract features in batch
            try:
                # Batch feature extraction
                features = F.adaptive_avg_pool2d(batch_tensor, (8, 8))
                features = features.view(len(batch_tensors), -1)
                
                # Normalize features on GPU
                features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
                
                # Store results
                for i, idx in enumerate(batch_indices):
                    feature_np = features[i].cpu().numpy()
                    visual_features_list.append(feature_np)
                    self.visual_indices.append(idx)
                    self.visual_features[idx] = feature_np
                    
            except Exception as e:
                print(f"⚠️ Error processing batch {batch_start}-{batch_end}: {e}")
                continue
        
        # Build K-NN model
        if visual_features_list:
            visual_features_array = np.array(visual_features_list)
            # Increase K for better coverage, especially for small datasets
            k_neighbors = min(max(self.k_nn + 1, 20), len(visual_features_array))
            self.visual_knn = NearestNeighbors(n_neighbors=k_neighbors, 
                                             metric='cosine')
            self.visual_knn.fit(visual_features_array)
            print(f"✅ Built K-NN model with {len(visual_features_list)} samples (K={k_neighbors})")
        else:
            print("❌ No visual features extracted!")
    
    def find_hard_negative(self, anchor_idx: int, dataset: Dict[int, Dict[str, Any]], debug_mode: bool = False) -> Optional[tuple]:
        """
        Find a hard negative for the given anchor.
        
        Strategy:
        1. Find K nearest visual neighbors
        2. Among neighbors, find instruction with lowest cosine similarity
        3. Ensure different goal (first instruction)
        
        Args:
            anchor_idx: Index of the anchor sample
            dataset: Processed dataset dictionary
            debug_mode: Whether to enable debug logging
            
        Returns:
            Tuple of (negative_idx, text_similarity, visual_similarity), or None if not found
        """
        if anchor_idx not in self.visual_features or self.visual_knn is None:
            return None
        
        anchor_item = dataset[anchor_idx]
        anchor_features = self.visual_features[anchor_idx]
        anchor_context = anchor_item.get('dialog_context', '')
        anchor_first_instruction = anchor_item.get('first_instruction', '')
        
        # Find K nearest visual neighbors
        distances, indices = self.visual_knn.kneighbors([anchor_features])
        
        # Skip the first neighbor (it's the anchor itself)
        neighbor_indices = indices[0][1:]
        neighbor_distances = distances[0][1:]  # Visual distances
        
        # Find neighbor with least similar instruction
        best_negative_idx = None
        lowest_text_similarity = float('inf')
        best_visual_similarity = None
        
        # Validation counters for this anchor
        total_neighbors = len(neighbor_indices)
        same_goal_count = 0
        bad_answer_count = 0
        threshold_fail_count = 0
        candidates_found = 0
        
        # Try with strict threshold first, then relax if needed
        thresholds_to_try = [0.25, 0.4, 0.6, 0.75]  # Tighter thresholds for better hard negatives
        
        for threshold in thresholds_to_try:
            for i, pos in enumerate(neighbor_indices):
                # Map from KNN position to actual dataset key
                sample_idx = self.visual_indices[pos]
                if sample_idx not in dataset:
                    continue
                neighbor_item = dataset[sample_idx]
                neighbor_context = neighbor_item.get('dialog_context', '')
                neighbor_first_instruction = neighbor_item.get('first_instruction', '')
                neighbor_answer = neighbor_item.get('answer', '')
                
                # Skip if same goal (first instruction)
                if anchor_first_instruction == neighbor_first_instruction:
                    same_goal_count += 1
                    continue
                
                # Skip if answer is not good enough
                if not self.is_good_answer(neighbor_answer):
                    bad_answer_count += 1  # is_good_answer already logs detailed reason when debug_mode is enabled
                    continue
                
                # Calculate text similarity
                anchor_text_features = self.extract_text_features(anchor_context)
                neighbor_text_features = self.extract_text_features(neighbor_context)
                
                # Cosine similarity
                text_similarity = np.dot(anchor_text_features, neighbor_text_features)
                
                # Get visual similarity (distance to similarity)
                visual_distance = neighbor_distances[i]
                visual_similarity = 1.0 - visual_distance  # Convert distance to similarity
                
                candidates_found += 1
                
                # We want the least similar text (lowest cosine similarity)
                if text_similarity < lowest_text_similarity and text_similarity < threshold:
                    # Check phrase diversity
                    neighbor_answer = neighbor_item.get('answer', '')
                    if self._is_phrase_diverse(neighbor_answer):
                        lowest_text_similarity = text_similarity
                        best_negative_idx = sample_idx
                        best_visual_similarity = visual_similarity
                elif text_similarity >= threshold:
                    threshold_fail_count += 1
            
            # If we found a negative, break
            if best_negative_idx is not None:
                break
        
        if best_negative_idx is not None:
            return (best_negative_idx, lowest_text_similarity, best_visual_similarity)

        # FINAL FALLBACK: pick the neighbour with the lowest text similarity regardless of threshold
        global_best_idx = None
        global_best_sim = float('inf')
        global_best_vis = None

        for i, pos in enumerate(neighbor_indices):
            sample_idx = self.visual_indices[pos]
            if sample_idx not in dataset:
                continue
            neighbor_item = dataset[sample_idx]
            neighbor_context = neighbor_item.get('dialog_context', '')
            neighbor_first_instruction = neighbor_item.get('first_instruction', '')
            neighbor_answer = neighbor_item.get('answer', '')

            # Same goal or bad answer still disqualify
            if anchor_first_instruction == neighbor_first_instruction:
                continue
            if not self.is_good_answer(neighbor_answer):
                continue

            anchor_text_features = self.extract_text_features(anchor_context)
            neighbor_text_features = self.extract_text_features(neighbor_context)
            text_sim = np.dot(anchor_text_features, neighbor_text_features)

            if text_sim < global_best_sim and self._is_phrase_diverse(neighbor_answer):
                global_best_sim = text_sim
                global_best_idx = sample_idx
                global_best_vis = 1.0 - neighbor_distances[i]

        if global_best_idx is not None:
            return (global_best_idx, global_best_sim, global_best_vis)
        
        return None
    
    def _is_phrase_diverse(self, answer: str) -> bool:
        """Check if answer phrase is diverse enough (not overused)."""
        if not answer:
            return False
        
        # Normalize answer for comparison
        normalized_answer = answer.lower().strip()
        
        # Check if this phrase has been used too many times
        if normalized_answer in self.used_phrases:
            if self.used_phrases[normalized_answer] >= self.max_phrase_reuse:
                return False
        
        return True
    
    def _track_phrase_usage(self, answer: str):
        """Track phrase usage for diversity."""
        if not answer:
            return
        
        normalized_answer = answer.lower().strip()
        self.used_phrases[normalized_answer] = self.used_phrases.get(normalized_answer, 0) + 1
    
    def mine_hard_negatives(self, dataset: Dict[int, Dict[str, Any]], 
                           max_samples: Optional[int] = None, debug_mode: bool = False) -> Dict[int, Dict[str, Any]]:
        """
        Mine hard negatives for the entire dataset.
        
        Args:
            dataset: Processed dataset dictionary
            max_samples: Maximum number of samples to process (for testing)
            
        Returns:
            Dictionary mapping anchor_idx -> negative_data (either hard or diverse)
        """
        print("⛏️ Mining hard negatives...")
        
        # Set debug mode for filtering
        self.debug_mode = debug_mode
        
        # Adjust filtering for small datasets
        if len(dataset) < 100:
            print("📊 Small dataset detected, using lenient filtering...")
            self.min_answer_length = max(15, self.min_answer_length - 5)  # Reduce minimum length
            # Only apply strict blacklist for small datasets
            self.answer_blacklist = {
                'short_affirmative': ['yes', 'exactly', 'correct', 'right'],
                'generic_responses': ['destiny is exactly that', 'that is correct'],
            }
            print(f"  Adjusted min_answer_length to {self.min_answer_length}")
            print(f"  Using lenient blacklist with {sum(len(phrases) for phrases in self.answer_blacklist.values())} phrases")
        
        # Initialize blacklist embeddings
        self._initialize_blacklist_embeddings()

        # Build visual K-NN model
        self.build_visual_knn(dataset)
        
        # Build visual clusters for diverse negatives
        if self.use_diverse_negatives:
            self.build_visual_clusters(dataset)
        
        # Mine negatives (only one per sample: 50% hard, 50% diverse)
        negatives = {}
        samples_to_process = list(dataset.keys())[:max_samples] if max_samples else list(dataset.keys())
        
        # Validation statistics
        validation_stats = {
            'total_attempts': 0,
            'hard_attempts': 0,
            'diverse_attempts': 0,
            'hard_success': 0,
            'diverse_success': 0,
            'fallback_used': 0,
            'no_candidates_found': 0
        }
        
        for anchor_idx in tqdm(samples_to_process, desc="Mining negatives"):
            validation_stats['total_attempts'] += 1

            # Decide mining order based on diverse_ratio probability
            if self.use_diverse_negatives and random.random() < self.diverse_ratio:
                strategy_order = ["diverse", "hard"]
            else:
                # Default to hard-first; if diverse mining disabled, list will only contain 'hard'
                strategy_order = ["hard"] if not self.use_diverse_negatives else ["hard", "diverse"]

            if debug_mode and validation_stats['total_attempts'] <= 3:
                print(f"\n🔍 Debug sample {validation_stats['total_attempts']}: anchor_idx={anchor_idx}")
                print(f"  Strategy order: {strategy_order}")
                print(f"  First instruction: {dataset[anchor_idx].get('first_instruction', 'N/A')}")
                print(f"  Current question: {dataset[anchor_idx].get('question', 'N/A')}")
                print(f"  Original answer: {dataset[anchor_idx].get('answer', 'N/A')}")

            negative_result = None
            negative_type = None

            for strategy in strategy_order:
                if strategy == "hard":
                    validation_stats['hard_attempts'] += 1
                    negative_result = self.find_hard_negative(anchor_idx, dataset, debug_mode)
                else:  # diverse
                    validation_stats['diverse_attempts'] += 1
                    negative_result = self.find_diverse_negative(anchor_idx, dataset, debug_mode)

                if negative_result is not None:
                    negative_type = strategy
                    # Record fallback usage if the first strategy failed and we switched
                    if strategy != strategy_order[0]:
                        validation_stats['fallback_used'] += 1
                    break

            # If no result after trying available strategies
            if negative_result is None:
                if debug_mode and validation_stats['total_attempts'] <= 3:
                    print(f"  ❌ No negative found after trying {strategy_order}")
                validation_stats['no_candidates_found'] += 1
                continue
            
            # At this point we must have a valid negative_result
            # Track success
            if negative_type == "hard":
                validation_stats['hard_success'] += 1
            else:
                validation_stats['diverse_success'] += 1

            # -----------------------------------------------------------------
            # Shared post-processing for both HARD and DIVERSE negatives
            # -----------------------------------------------------------------
            if debug_mode and validation_stats['total_attempts'] <= 3:
                print(f"  ✅ Found {negative_type} negative")
                # Get the negative item for detailed logging
                if negative_type == "hard":
                    negative_idx, text_similarity, visual_similarity = negative_result
                else:
                    negative_idx, anchor_cluster, negative_cluster, visual_similarity = negative_result
                
                negative_item = dataset[negative_idx]
                print(f"  Negative sample {negative_idx}:")
                print(f"    First instruction: {negative_item.get('first_instruction', 'N/A')}")
                print(f"    Current question: {negative_item.get('question', 'N/A')}")
                print(f"    Negative answer: {negative_item.get('answer', 'N/A')}")
                if negative_type == "hard":
                    print(f"    Text similarity: {text_similarity:.3f}")
                    print(f"    Visual similarity: {visual_similarity:.3f}")
                else:
                    print(f"    Visual similarity: {visual_similarity:.3f}")
                    print(f"    Clusters: {anchor_cluster} -> {negative_cluster}")

            # Build validation metadata specific to the negative type
            if negative_type == "hard":
                negative_idx, text_similarity, visual_similarity = negative_result
                validation_info = {
                    'negative_type_2': negative_type,
                    'text_similarity': float(text_similarity),
                    'visual_similarity': float(visual_similarity),
                    'mining_method': 'hard_negative_knn'
                }
            else:  # diverse
                negative_idx, anchor_cluster, negative_cluster, visual_similarity = negative_result
                validation_info = {
                    'negative_type_2': negative_type,
                    'anchor_cluster': int(anchor_cluster),
                    'negative_cluster': int(negative_cluster),
                    'visual_similarity': float(visual_similarity),
                    'mining_method': 'diverse_negative_clustering'
                }
            
            negative_item = dataset[negative_idx]
            
            # Track phrase usage for diversity
            self._track_phrase_usage(negative_item.get('answer', ''))
            
            # Create negative contrastive data (saved as negative_2 to avoid conflict with existing LM negative)
            negative_data = {
                'negative_text_2': negative_item.get('answer', ''),
                'negative_context_2': negative_item.get('dialog_context', ''),
                'negative_question_2': negative_item.get('question', ''),
                'negative_first_instruction_2': negative_item.get('first_instruction', ''),
                'negative_visual_features_2': negative_item.get('current_view_image', None),
                'negative_type_2': negative_type,  # Track which type this is
                'map_name_2': negative_item.get('map_name', 'unknown'),  # Save map name
                'validation_metadata_2': validation_info
            }

            # Tokenize negative text
            if self.tokenizer:
                negative_data['tokenized_negative_2'] = self.tokenizer(
                    negative_data['negative_text_2'],
                    max_length=self.config.model.max_answer_length if self.config else 128,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

            negatives[anchor_idx] = negative_data
        
        # Print detailed validation statistics
        print(f"📊 Validation Statistics:")
        print(f"  Total attempts: {validation_stats['total_attempts']}")
        print(f"  Hard attempts: {validation_stats['hard_attempts']} (success: {validation_stats['hard_success']})")
        print(f"  Diverse attempts: {validation_stats['diverse_attempts']} (success: {validation_stats['diverse_success']})")
        print(f"  Fallback used: {validation_stats['fallback_used']}")
        print(f"  No candidates found: {validation_stats['no_candidates_found']}")
        print(f"  Success rate: {len(negatives)}/{validation_stats['total_attempts']} ({len(negatives)/validation_stats['total_attempts']*100:.1f}%)")
        
        # Count types
        hard_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'hard')
        diverse_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'diverse')
        
        print(f"✅ Mined {len(negatives)} negatives total")
        print(f"📊 Hard negatives: {hard_count}, Diverse negatives: {diverse_count}")
        
        # Enhanced statistics
        if negatives:
            # Phrase diversity statistics
            unique_phrases = len(self.used_phrases)
            total_negatives = len(negatives)
            avg_reuse = sum(self.used_phrases.values()) / len(self.used_phrases) if self.used_phrases else 0
            print(f"📈 Phrase diversity: {unique_phrases} unique phrases, avg reuse: {avg_reuse:.2f}")
            
            # Answer quality statistics
            answer_lengths = [len(data['negative_text_2']) for data in negatives.values()]
            avg_length = sum(answer_lengths) / len(answer_lengths)
            print(f"📏 Answer quality: avg length {avg_length:.1f} chars")
            
            # Similarity statistics
            hard_sims = [data['validation_metadata_2']['text_similarity'] 
                        for data in negatives.values() 
                        if data.get('negative_type_2') == 'hard' and 'text_similarity' in data['validation_metadata_2']]
            if hard_sims:
                avg_hard_sim = sum(hard_sims) / len(hard_sims)
                print(f"🎯 Hard negative quality: avg text similarity {avg_hard_sim:.3f}")
            
            # Semantic filtering statistics
            if self.blacklist_embeddings:
                print(f"🔍 Semantic filtering: {len(self.blacklist_embeddings)} blacklist embeddings initialized")
                print(f"   Similarity threshold: {self.semantic_similarity_threshold}")
        
        return negatives
    
    def add_hard_negatives_to_dataset(self, dataset: Dict[int, Dict[str, Any]], 
                                     negatives: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Add negatives to the existing dataset.
        
        Args:
            dataset: Original dataset
            negatives: Mined negative data (hard or diverse)
            
        Returns:
            Updated dataset with negatives added
        """
        print("➕ Adding negatives to dataset...")
        
        updated_dataset = dataset.copy()
        
        for anchor_idx, negative_data in tqdm(negatives.items(), desc="Adding negatives"):
            if anchor_idx in updated_dataset:
                # Get the anchor item
                anchor_item = updated_dataset[anchor_idx]
                
                # Add negative to contrastive data
                if 'contrastive_data' not in anchor_item:
                    anchor_item['contrastive_data'] = {}
                
                # Add negative_2 data (second negative, avoiding conflict with existing LM negative)
                anchor_item['contrastive_data']['negative_text_2'] = negative_data['negative_text_2']
                anchor_item['contrastive_data']['tokenized_negative_2'] = negative_data['tokenized_negative_2']
                
                # Add validation metadata specifically for negative_2
                anchor_item['contrastive_data']['validation_metadata_negative_2'] = {
                    'negative_type_2': negative_data['negative_type_2'],
                    'map_name_2': negative_data['map_name_2'],
                    'mining_timestamp': datetime.datetime.now().isoformat(),
                    **negative_data['validation_metadata_2']  # Include detailed validation info
                }
                
                updated_dataset[anchor_idx] = anchor_item
        
        print(f"✅ Added negatives to {len(negatives)} samples")
        return updated_dataset

def load_dataset(config: Config, split: str) -> Dict[int, Dict[str, Any]]:
    """
    Load the processed dataset.
    
    Args:
        config: Configuration object
        split: Dataset split ('train', 'val_seen', 'val_unseen')
        
    Returns:
        Loaded dataset dictionary
    """
    print(f"📊 Loading {split} dataset...")
    
    if split == 'train':
        # Load train data from chunks
        from dataset import AnsweringDataset
        dataset = AnsweringDataset.load_train_chunks(config.data.train_processed_path_dir)
    else:
        # Load validation data
        if split == 'val_seen':
            data_path = config.data.val_seen_processed_path
        else:  # val_unseen
            data_path = config.data.val_unseen_processed_path
        
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
    
    print(f"✅ Loaded {len(dataset)} samples")
    return dataset

def save_dataset(dataset: Dict[int, Dict[str, Any]], config: Config, split: str):
    """
    Save the updated dataset.
    
    Args:
        dataset: Updated dataset
        config: Configuration object
        split: Dataset split
    """
    print(f"💾 Saving updated {split} dataset...")
    
    if split == 'train':
        # Save train data in chunks
        from dataset import AnsweringDataset
        output_dir = config.data.train_processed_path_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear existing chunks
        for file in os.listdir(output_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(output_dir, file))
        
        # Save in chunks
        chunk_size = 1000
        AnsweringDataset.save_in_chunks(dataset, chunk_size, output_dir)
        print(f"✅ Saved train data in chunks to {output_dir}")
    else:
        # Save validation data
        if split == 'val_seen':
            output_path = config.data.val_seen_processed_path
        else:  # val_unseen
            output_path = config.data.val_unseen_processed_path
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"✅ Saved {split} data to {output_path}")

def main():
    """Main function to add hard negatives to dataset."""
    parser = argparse.ArgumentParser(description='Add hard negative samples to AVDN dataset')
    parser.add_argument('--config', type=str, default='config.py', help='Path to config file')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val_seen', 'val_unseen'], 
                       help='Dataset split to process')
    parser.add_argument('--k-nn', type=int, default=30, help='Number of K-NN neighbors to consider')
    parser.add_argument('--cosine-threshold', type=float, default=0.3, 
                       help='Cosine similarity threshold for hard negatives')
    parser.add_argument('--max-samples', type=int, default=None, 
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--image-dir', type=str, required=True, 
                       help='Directory containing satellite images')
    parser.add_argument('--use-diverse-negatives', action='store_true', default=True,
                       help='Whether to add diverse negatives from outside clusters')
    parser.add_argument('--diverse-ratio', type=float, default=0.3,
                       help='Ratio of samples to use for diverse negative mining')
    parser.add_argument('--min-answer-length', type=int, default=20,
                       help='Minimum answer length to consider for negative mining')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for GPU processing')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU ID to use (0-9 for your 10 RTX 2080s)')
    
    # Dataset sharding arguments (useful for multi-GPU setups to avoid duplicated work)
    parser.add_argument('--num-shards', type=int, default=1,
                       help='Total number of dataset shards (typically = number of GPUs)')
    parser.add_argument('--shard-id', type=int, default=0,
                       help='Shard index for this process (0 ≤ shard_id < num_shards)')
    
    args = parser.parse_args()
    
    # Set GPU device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"🚀 Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Load configuration
    config = Config()
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    # Load dataset
    dataset = load_dataset(config, args.split)

    # Optional: shard dataset so each GPU/process works on a unique subset
    if args.num_shards > 1:
        original_size = len(dataset)
        dataset = {k: v for k, v in dataset.items() if (k % args.num_shards) == args.shard_id}
        print(f"🔀 Sharded dataset: keeping {len(dataset)} / {original_size} samples for shard {args.shard_id} of {args.num_shards}")
    
    # Initialize hard negative miner
    miner = HardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        image_dir=args.image_dir,
        k_nn=args.k_nn,
        cosine_threshold=args.cosine_threshold,
        use_diverse_negatives=args.use_diverse_negatives,
        diverse_ratio=args.diverse_ratio,
        min_answer_length=args.min_answer_length
    )
    
    # Update GPU settings
    miner.batch_size = args.batch_size
    miner.num_workers = args.num_workers
    if torch.cuda.is_available():
        miner.device = torch.device(f'cuda:{args.gpu_id}')
    
    # Mine hard negatives
    hard_negatives = miner.mine_hard_negatives(dataset, max_samples=args.max_samples)
    
    # Add hard negatives to dataset
    updated_dataset = miner.add_hard_negatives_to_dataset(dataset, hard_negatives)
    
    # Save updated dataset
    save_dataset(updated_dataset, config, args.split)
    
    print(f"🎉 Successfully added {len(hard_negatives)} negatives to {args.split} dataset!")

if __name__ == '__main__':
    main() 