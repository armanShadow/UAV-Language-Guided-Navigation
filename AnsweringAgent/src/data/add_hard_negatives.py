#!/usr/bin/env python3
"""
Hard Negative Mining Script for AVDN Dataset

This script adds hard negative samples to the existing dataset by:
1. Mining hard negatives using visual K-NN + least-similar instruction
2. Adding diverse negatives from outside nearest visual clusters
3. Adding one hard negative per anchor to the existing dataset
4. Enhanced semantic filtering with caching for better quality

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
import re
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
    
    def __init__(self, config: Config, tokenizer, image_dir: str, k_nn: int = 100, cosine_threshold: float = 0.2,
                 use_diverse_negatives: bool = True, diverse_ratio: float = 0.3, min_answer_length: int = 20):
        """
        Initialize the hard negative miner.
        
        Args:
            config: Configuration object
            tokenizer: Tokenizer for text processing
            image_dir: Directory containing satellite images
            k_nn: Number of nearest neighbors to consider for visual similarity (increased to 100)
            cosine_threshold: Threshold for considering instructions as dissimilar (lowered to 0.2)
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
        self.batch_size = 64
        self.num_workers = 4
        
        # Initialize normalizer for image processing with MPNet embeddings
        self.normalizer = AnsweringAgentNormalizer(tokenizer, config, generate_mpnet_embeddings=True)
        
        # Storage for mined data
        self.visual_features = {}
        self.text_features = {}
        self.episode_data = {}
        
        # K-NN model for visual similarity
        self.visual_knn = None
        self.visual_indices = []
        
        # Clustering for diverse negatives
        self.visual_clusters = None
        self.cluster_labels = None
        
        # Answer quality filtering - Full blacklist (always available for semantic filtering)
        self.full_blacklist = {
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
        
        # Working blacklist (can be overridden for lenient filtering)
        self.answer_blacklist = self.full_blacklist.copy()
        
        # Semantic filtering setup
        self.blacklist_embeddings = {}
        self.semantic_similarity_threshold = 0.88
        
        # Answer embedding cache for performance optimization
        self.answer_embedding_cache = {}
        
        # Phrase diversity tracking
        self.used_phrases = {}
        self.max_phrase_reuse = 3
        
        # Debug mode
        self.debug_mode = False
    
    def _initialize_blacklist_embeddings(self):
        """Initialize embeddings for blacklisted phrases for semantic similarity checking."""
        if not hasattr(self, 'normalizer') or not self.normalizer.generate_mpnet_embeddings:
            print("⚠️ MPNet embeddings not available, using string-based filtering only")
            return
        
        print("🔍 Initializing blacklist embeddings for semantic filtering...")
        
        cache_path = os.path.join(os.path.dirname(__file__), 'blacklist_embeds.pkl')

        # Calculate expected number of embeddings from full blacklist
        expected_count = sum(len(phrases) for phrases in self.full_blacklist.values())

        # Try loading cache first, but validate it has the full blacklist
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_embeddings = pickle.load(f)
                
                # Validate cache has the full blacklist, not just lenient version
                if len(cached_embeddings) >= expected_count:
                    self.blacklist_embeddings = cached_embeddings
                    print(f"✅ Loaded cached blacklist embeddings from {cache_path}")
                    print(f"   {len(self.blacklist_embeddings)} phrases available for semantic filtering")
                    return
                else:
                    print(f"⚠️ Cache has only {len(cached_embeddings)} embeddings, expected {expected_count}. Regenerating...")
            except Exception as e:
                print(f"⚠️ Failed to load cached blacklist embeddings: {e}. Recomputing...")

        # Generate embeddings for full blacklist (not just current working blacklist)
        print(f"🔄 Generating embeddings for {expected_count} phrases from full blacklist...")
        for category, phrases in self.full_blacklist.items():
            for phrase in phrases:
                try:
                    embedding = self.normalizer.generate_mpnet_embedding(phrase)
                    self.blacklist_embeddings[phrase] = embedding
                except Exception as e:
                    print(f"⚠️ Error generating embedding for '{phrase}': {e}")

        # Save cache
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.blacklist_embeddings, f)
            print(f"💾 Cached blacklist embeddings to {cache_path}")
            print(f"   {len(self.blacklist_embeddings)} phrases available for semantic filtering")
        except Exception as e:
            print(f"⚠️ Could not cache blacklist embeddings: {e}")
    
    def _check_semantic_similarity_to_blacklist(self, answer: str) -> bool:
        """Check if answer is semantically similar to any blacklisted phrase."""
        if not self.blacklist_embeddings or not hasattr(self, 'normalizer'):
            return False
        
        # Check cache first
        if answer in self.answer_embedding_cache:
            answer_embedding = self.answer_embedding_cache[answer]
        else:
            try:
                answer_embedding = self.normalizer.generate_mpnet_embedding(answer)
                self.answer_embedding_cache[answer] = answer_embedding
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error generating embedding for '{answer}': {e}")
                return False
        
        max_similarity = 0.0
        most_similar_phrase = ""
        
        for blacklisted_phrase, blacklist_embedding in self.blacklist_embeddings.items():
            similarity = np.dot(answer_embedding, blacklist_embedding)
            
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_phrase = blacklisted_phrase
            
            # Debug mode: show similarity scores >= 0.70
            if similarity >= 0.7 and self.debug_mode:
                print(f"    ↪ sim({similarity:.2f}) to blacklist phrase '{blacklisted_phrase}'")
            
            if similarity > self.semantic_similarity_threshold:
                if self.debug_mode:
                    print(f"    🚫 Semantic filter triggered: sim({similarity:.2f}) > {self.semantic_similarity_threshold} for '{blacklisted_phrase}'")
                return True
        
        # Show the highest similarity even if below threshold (for debugging)
        if self.debug_mode and max_similarity > 0.5:
            print(f"    📊 Highest semantic similarity: {max_similarity:.2f} to '{most_similar_phrase}' (threshold: {self.semantic_similarity_threshold})")
        
        return False
    
    def is_good_answer(self, answer: str, track_rejections: bool = False) -> bool:
        """
        Check if an answer is good enough for negative mining.
        Uses hybrid approach: fast direct matching first, then semantic similarity for edge cases.
        
        Args:
            answer: The answer text to check
            track_rejections: If True, returns (bool, rejection_reason)
        """
        if not answer or not isinstance(answer, str):
            return (False, 'empty_answer') if track_rejections else False
        
        answer_clean = answer.strip()
        
        # Check minimum length
        if len(answer_clean) < self.min_answer_length:
            if self.debug_mode:
                print(f"    ❌ Filtered: too short ({len(answer_clean)} chars)")
            return (False, 'too_short') if track_rejections else False
        
        # Fast direct blacklist check for obvious cases
        answer_lower = answer.lower()
        for category, phrases in self.answer_blacklist.items():
            for phrase in phrases:
                pattern = rf"\b{re.escape(phrase)}\b"
                if re.search(pattern, answer_lower):
                    if self.debug_mode:
                        print(f"    ❌ Filtered: contains blacklisted phrase '{phrase}' (direct)")
                    return (False, 'blacklist_direct') if track_rejections else False
        
        # Only use expensive semantic similarity for longer answers that passed direct check
        # This is where we need to catch semantically similar but differently worded answers
        if len(answer_clean) > 30:  # Only check semantic similarity for longer answers
            if self._check_semantic_similarity_to_blacklist(answer):
                if self.debug_mode:
                    print(f"    ❌ Filtered: semantically similar to blacklisted phrase")
                return (False, 'blacklist_semantic') if track_rejections else False
        
        return (True, 'passed') if track_rejections else True
    
    def extract_visual_features(self, current_view: torch.Tensor) -> np.ndarray:
        """Extract visual features from current view using a more robust CNN approach."""
        if current_view.device != self.device:
            current_view = current_view.to(self.device)
        
        # Use multiple pooling scales to capture more visual information
        features_list = []
        
        # Global average pooling at different scales
        for pool_size in [(4, 4), (8, 8), (16, 16)]:
            features = F.adaptive_avg_pool2d(current_view.unsqueeze(0), pool_size)
            features = features.view(-1)
            features_list.append(features)
        
        # Concatenate multi-scale features
        combined_features = torch.cat(features_list, dim=0)
        
        # Normalize to unit vector
        combined_features = combined_features / (torch.norm(combined_features) + 1e-8)
        
        return combined_features.cpu().numpy()
    
    def extract_text_features(self, dialog_context: str) -> np.ndarray:
        """Extract text features from dialog context using MPNet embeddings or fallback."""
        if hasattr(self, 'normalizer') and self.normalizer.generate_mpnet_embeddings:
            try:
                return self.normalizer.generate_mpnet_embedding(dialog_context)
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error generating MPNet embedding: {e}")
        
        # Fallback: Simple TF-IDF like features
        from collections import Counter
        
        text = dialog_context.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        word_freq = Counter()
        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_freq[word] += 1
        
        if not word_freq:
            return np.zeros(100, dtype=np.float32)
        
        unique_words = list(word_freq.keys())[:1000]
        features = np.zeros(len(unique_words))
        for i, word in enumerate(unique_words):
            features[i] = word_freq[word]
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def build_visual_knn(self, dataset: Dict[int, Dict[str, Any]]):
        """Build K-NN model for visual similarity."""
        print("🔍 Building visual K-NN model...")
        
        visual_features_list = []
        self.visual_indices = []
        
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
                except Exception as e:
                    if self.debug_mode:
                        print(f"⚠️ Error preparing item {idx}: {e}")
                    continue
            
            if not batch_tensors:
                continue
                
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            try:
                features = F.adaptive_avg_pool2d(batch_tensor, (8, 8))
                features = features.view(len(batch_tensors), -1)
                features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
                
                for i, idx in enumerate(batch_indices):
                    feature_np = features[i].cpu().numpy()
                    visual_features_list.append(feature_np)
                    self.visual_indices.append(idx)
                    self.visual_features[idx] = feature_np
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"⚠️ Error processing batch {batch_start}-{batch_end}: {e}")
                continue
        
        if visual_features_list:
            visual_features_array = np.array(visual_features_list)
            k_neighbors = min(max(self.k_nn + 1, 50), len(visual_features_array))  # Increased minimum from 20 to 50
            self.visual_knn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
            self.visual_knn.fit(visual_features_array)
            print(f"✅ Built K-NN model with {len(visual_features_list)} samples (K={k_neighbors})")
        else:
            print("❌ No visual features extracted!")
    
    def build_visual_clusters(self, dataset: Dict[int, Dict[str, Any]], n_clusters: int = 30):
        """Build visual clusters for diverse negative sampling."""
        print("🔍 Building visual clusters for diverse negative sampling...")
        
        if not self.visual_features:
            print("❌ No visual features available for clustering!")
            return
        
        visual_features_list = []
        self.visual_indices = []
        for idx, features in self.visual_features.items():
            visual_features_list.append(features)
            self.visual_indices.append(idx)
        
        if visual_features_list:
            visual_features_array = np.array(visual_features_list)
            
            # For small datasets, ensure we have meaningful clustering
            min_samples_per_cluster = 2
            max_possible_clusters = len(visual_features_array) // min_samples_per_cluster
            n_clusters = min(n_clusters, max_possible_clusters)
            
            # Need at least 2 clusters for diverse negatives to work
            if n_clusters < 2:
                print(f"⚠️ Too few samples ({len(visual_features_array)}) for meaningful clustering. Using random diverse selection.")
                # Create artificial clusters by random assignment
                self.cluster_labels = np.random.randint(0, 2, size=len(visual_features_array))
                self.visual_clusters = None  # Flag that we're using random clustering
                print(f"✅ Using random cluster assignment with 2 artificial clusters")
            else:
                self.visual_clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                self.cluster_labels = self.visual_clusters.fit_predict(visual_features_array)
                print(f"✅ Built {n_clusters} visual clusters with {len(visual_features_list)} samples")
            
            unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
            print(f"📊 Cluster distribution: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
            
            # Show cluster assignments for small datasets
            if len(visual_features_array) <= 20:
                cluster_assignments = {}
                for i, cluster_id in enumerate(self.cluster_labels):
                    if cluster_id not in cluster_assignments:
                        cluster_assignments[cluster_id] = []
                    cluster_assignments[cluster_id].append(self.visual_indices[i])
                
                print("📋 Cluster assignments:")
                for cluster_id, sample_indices in cluster_assignments.items():
                    print(f"  Cluster {cluster_id}: samples {sample_indices}")
        else:
            print("❌ No visual features extracted for clustering!")
    
    def find_hard_negative(self, anchor_idx: int, dataset: Dict[int, Dict[str, Any]], 
                          rejection_stats: Dict[str, int] = None) -> Optional[tuple]:
        """Find a hard negative for the given anchor using visual K-NN and answer text dissimilarity."""
        if rejection_stats is None:
            rejection_stats = {}
            
        if anchor_idx not in self.visual_features or self.visual_knn is None:
            rejection_stats['no_visual_neighbors'] = rejection_stats.get('no_visual_neighbors', 0) + 1
            return None
        
        anchor_item = dataset[anchor_idx]
        anchor_features = self.visual_features[anchor_idx]
        anchor_answer = anchor_item.get('answer', '')  # Use answer instead of context
        anchor_first_instruction = anchor_item.get('first_instruction', '')
        
        distances, indices = self.visual_knn.kneighbors([anchor_features])
        neighbor_indices = indices[0][1:]  # Skip self
        neighbor_distances = distances[0][1:]
        
        best_negative_idx = None
        lowest_text_similarity = float('inf')
        best_visual_similarity = None
        best_threshold_used = None
        
        # Count filtering for debugging
        total_neighbors = len(neighbor_indices)
        processed_neighbors = 0
        
        thresholds_to_try = [0.2, 0.35, 0.5, 0.65, 0.8]  # More aggressive thresholds
        
        # Only show debug info for very small datasets
        show_debug = self.debug_mode and total_neighbors <= 3
        
        if show_debug:
            print(f"    🔍 Searching through {total_neighbors} visual neighbors...")
        
        for threshold in thresholds_to_try:
            found_at_threshold = False
            candidates_at_threshold = 0
            
            for i, pos in enumerate(neighbor_indices):
                sample_idx = self.visual_indices[pos]
                if sample_idx not in dataset:
                    continue
                    
                neighbor_item = dataset[sample_idx]
                neighbor_answer = neighbor_item.get('answer', '')  # Use answer instead of context
                neighbor_first_instruction = neighbor_item.get('first_instruction', '')
                
                processed_neighbors += 1
                
                # Skip if same goal
                if anchor_first_instruction == neighbor_first_instruction:
                    rejection_stats['same_goal'] = rejection_stats.get('same_goal', 0) + 1
                    continue
                
                # Skip if answer is not good enough
                is_good, rejection_reason = self.is_good_answer(neighbor_answer, track_rejections=True)
                if not is_good:
                    if rejection_reason == 'too_short':
                        rejection_stats['bad_answer_length'] = rejection_stats.get('bad_answer_length', 0) + 1
                    elif rejection_reason == 'blacklist_direct':
                        rejection_stats['bad_answer_blacklist'] = rejection_stats.get('bad_answer_blacklist', 0) + 1
                    elif rejection_reason == 'blacklist_semantic':
                        rejection_stats['bad_answer_semantic'] = rejection_stats.get('bad_answer_semantic', 0) + 1
                    continue
                
                # Check phrase diversity
                if not self._is_phrase_diverse(neighbor_answer):
                    rejection_stats['phrase_diversity_fail'] = rejection_stats.get('phrase_diversity_fail', 0) + 1
                    continue
                
                # Calculate ANSWER-level text similarity (key fix)
                anchor_text_features = self.extract_text_features(anchor_answer)
                neighbor_text_features = self.extract_text_features(neighbor_answer)
                text_similarity = np.dot(anchor_text_features, neighbor_text_features)
                
                # Convert cosine distance to similarity (1 - distance = similarity)
                visual_distance = neighbor_distances[i]
                visual_similarity = 1.0 - visual_distance
                
                candidates_at_threshold += 1
                
                # Find the least similar text (lowest cosine similarity)
                if text_similarity < lowest_text_similarity and text_similarity < threshold:
                    lowest_text_similarity = text_similarity
                    best_negative_idx = sample_idx
                    best_visual_similarity = visual_similarity
                    best_threshold_used = threshold
                    found_at_threshold = True
                else:
                    # Count as no text similarity match if not accepted
                    rejection_stats['no_text_similarity_match'] = rejection_stats.get('no_text_similarity_match', 0) + 1
            
            if show_debug:
                print(f"    📊 Threshold {threshold}: {candidates_at_threshold} candidates, found: {found_at_threshold}")
            
            # Continue to try higher thresholds even if we found something
            # This gives us the globally best candidate across all thresholds
        
        # Debug statistics for small datasets only
        if show_debug:
            print(f"    📊 Processed {processed_neighbors} neighbors")
            if best_negative_idx is not None:
                print(f"    ✅ Found negative with text_sim={lowest_text_similarity:.3f}, visual_sim={best_visual_similarity:.3f} at threshold={best_threshold_used}")
        
        if best_negative_idx is not None:
            return (best_negative_idx, lowest_text_similarity, best_visual_similarity)

        # Final fallback: take any valid neighbor with lowest similarity regardless of threshold
        if show_debug:
            print(f"    🔄 Fallback: trying any valid neighbor...")
        
        fallback_best_idx = None
        fallback_best_sim = float('inf')
        fallback_best_vis = None

        for i, pos in enumerate(neighbor_indices):
            sample_idx = self.visual_indices[pos]
            if sample_idx not in dataset:
                continue
                
            neighbor_item = dataset[sample_idx]
            neighbor_first_instruction = neighbor_item.get('first_instruction', '')
            neighbor_answer = neighbor_item.get('answer', '')

            if (anchor_first_instruction != neighbor_first_instruction and 
                self.is_good_answer(neighbor_answer) and 
                self._is_phrase_diverse(neighbor_answer)):
                
                # Use answer-level similarity
                anchor_text_features = self.extract_text_features(anchor_answer)
                neighbor_text_features = self.extract_text_features(neighbor_answer)
                text_sim = np.dot(anchor_text_features, neighbor_text_features)
                
                visual_distance = neighbor_distances[i]
                visual_sim = 1.0 - visual_distance

                if text_sim < fallback_best_sim:
                    fallback_best_sim = text_sim
                    fallback_best_idx = sample_idx
                    fallback_best_vis = visual_sim
        
        if fallback_best_idx is not None:
            if show_debug:
                print(f"    ✅ Fallback found negative with text_sim={fallback_best_sim:.3f}, visual_sim={fallback_best_vis:.3f}")
            return (fallback_best_idx, fallback_best_sim, fallback_best_vis)
        
        return None
    
    def find_diverse_negative(self, anchor_idx: int, dataset: Dict[int, Dict[str, Any]], 
                             rejection_stats: Dict[str, int] = None) -> Optional[tuple]:
        """Find a diverse negative from outside the anchor's visual cluster."""
        if rejection_stats is None:
            rejection_stats = {}
            
        if anchor_idx not in self.visual_features or self.cluster_labels is None:
            rejection_stats['no_valid_clusters'] = rejection_stats.get('no_valid_clusters', 0) + 1
            return None
        
        anchor_item = dataset[anchor_idx]
        anchor_first_instruction = anchor_item.get('first_instruction', '')
        anchor_features = self.visual_features[anchor_idx]
        
        # Find anchor's cluster
        anchor_idx_in_array = self.visual_indices.index(anchor_idx)
        anchor_cluster = self.cluster_labels[anchor_idx_in_array]
        
        # Find candidates from different clusters first
        different_cluster_candidates = []
        same_cluster_candidates = []
        
        for i, cluster_label in enumerate(self.cluster_labels):
            sample_idx = self.visual_indices[i]
            if sample_idx in dataset and sample_idx != anchor_idx:
                neighbor_item = dataset[sample_idx]
                neighbor_first_instruction = neighbor_item.get('first_instruction', '')
                neighbor_answer = neighbor_item.get('answer', '')
                
                # Track rejections for same goal
                if anchor_first_instruction == neighbor_first_instruction:
                    rejection_stats['same_goal'] = rejection_stats.get('same_goal', 0) + 1
                    continue
                    
                # Track rejections for bad answers
                is_good, rejection_reason = self.is_good_answer(neighbor_answer, track_rejections=True)
                if not is_good:
                    if rejection_reason == 'too_short':
                        rejection_stats['bad_answer_length'] = rejection_stats.get('bad_answer_length', 0) + 1
                    elif rejection_reason == 'blacklist_direct':
                        rejection_stats['bad_answer_blacklist'] = rejection_stats.get('bad_answer_blacklist', 0) + 1
                    elif rejection_reason == 'blacklist_semantic':
                        rejection_stats['bad_answer_semantic'] = rejection_stats.get('bad_answer_semantic', 0) + 1
                    continue
                    
                # Track rejections for phrase diversity
                if not self._is_phrase_diverse(neighbor_answer):
                    rejection_stats['phrase_diversity_fail'] = rejection_stats.get('phrase_diversity_fail', 0) + 1
                    continue
                
                neighbor_features = self.visual_features[sample_idx]
                visual_similarity = np.dot(anchor_features, neighbor_features)
                
                # Prioritize different clusters
                if cluster_label != anchor_cluster:
                    different_cluster_candidates.append((sample_idx, cluster_label, visual_similarity))
                else:
                    same_cluster_candidates.append((sample_idx, cluster_label, visual_similarity))
        
        # Try different clusters first, then same cluster as fallback
        if different_cluster_candidates:
            # Sort by visual similarity (ascending for more diverse)
            different_cluster_candidates.sort(key=lambda x: x[2])
            selected_idx, selected_cluster, visual_similarity = different_cluster_candidates[0]
            return (selected_idx, anchor_cluster, selected_cluster, visual_similarity)
        elif same_cluster_candidates:
            # Fallback to same cluster but lowest visual similarity
            same_cluster_candidates.sort(key=lambda x: x[2])
            selected_idx, selected_cluster, visual_similarity = same_cluster_candidates[0]
            return (selected_idx, anchor_cluster, selected_cluster, visual_similarity)
        
        # No valid candidates found
        rejection_stats['no_valid_clusters'] = rejection_stats.get('no_valid_clusters', 0) + 1
        return None
    
    def _is_phrase_diverse(self, answer: str) -> bool:
        """Check if answer phrase is diverse enough (not overused)."""
        if not answer:
            return False
        
        # Normalize answer for comparison
        normalized_answer = answer.lower().strip()
        
        # For very short answers, be extra strict
        if len(normalized_answer) < 30:
            max_reuse = 1  # Very short answers can only be used once
        elif len(normalized_answer) < 60:
            max_reuse = 2  # Short answers can be used twice
        else:
            max_reuse = self.max_phrase_reuse  # Longer answers can be reused more
        
        # Check if this phrase has been used too many times
        if normalized_answer in self.used_phrases:
            if self.used_phrases[normalized_answer] >= max_reuse:
                if self.debug_mode:
                    print(f"    ❌ Phrase overused ({self.used_phrases[normalized_answer]} times): '{answer[:40]}{'...' if len(answer) > 40 else ''}'")
                return False
        
        # Additional quality check: avoid very generic short phrases
        generic_short_phrases = {
            'your goal is the big building right in front of you',
            'go south to black lot',
            'turn left at the intersection',
            'the destination is ahead',
            'go straight ahead',
            'turn right at the corner'
        }
        
        if normalized_answer in generic_short_phrases:
            # Allow these only once each
            if normalized_answer in self.used_phrases and self.used_phrases[normalized_answer] >= 1:
                if self.debug_mode:
                    print(f"    ❌ Generic phrase already used: '{answer[:40]}{'...' if len(answer) > 40 else ''}'")
                return False
        
        return True
    
    def _track_phrase_usage(self, answer: str):
        """Track phrase usage for diversity."""
        if not answer:
            return
        
        normalized_answer = answer.lower().strip()
        self.used_phrases[normalized_answer] = self.used_phrases.get(normalized_answer, 0) + 1
    
    def precompute_answer_embeddings(self, dataset: Dict[int, Dict[str, Any]]):
        """Pre-compute embeddings for all answers in the dataset to improve performance."""
        print("🔄 Pre-computing answer embeddings for performance...")
        
        unique_answers = set()
        for item in dataset.values():
            answer = item.get('answer', '')
            if answer and len(answer.strip()) > 30:  # Only pre-compute for answers that will need semantic checking
                unique_answers.add(answer.strip())
        
        if not unique_answers:
            print("  No answers need semantic checking")
            return
            
        print(f"  Computing embeddings for {len(unique_answers)} unique answers...")
        
        # Batch process embeddings
        from tqdm import tqdm
        for answer in tqdm(unique_answers, desc="Computing embeddings", disable=not self.debug_mode):
            if answer not in self.answer_embedding_cache:
                try:
                    embedding = self.normalizer.generate_mpnet_embedding(answer)
                    self.answer_embedding_cache[answer] = embedding
                except Exception as e:
                    if self.debug_mode:
                        print(f"⚠️ Error computing embedding for '{answer[:50]}...': {e}")
        
        print(f"✅ Pre-computed {len(self.answer_embedding_cache)} answer embeddings")

    def mine_hard_negatives(self, dataset: Dict[int, Dict[str, Any]], 
                           max_samples: Optional[int] = None, debug_mode: bool = False) -> Dict[int, Dict[str, Any]]:
        """Mine hard negatives for the entire dataset."""
        print("⛏️ Mining hard negatives...")
        
        self.debug_mode = debug_mode
        
        # Initialize semantic filtering with full blacklist
        if not debug_mode:
            # Suppress initialization prints for cleaner output
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                self._initialize_blacklist_embeddings()
            print(f"🔍 Semantic filtering: {len(self.blacklist_embeddings)} embeddings loaded")
        else:
            self._initialize_blacklist_embeddings()
        
        # Adjust direct string filtering for small datasets
        if len(dataset) < 150:  # Increased threshold to apply lenient filtering more often
            if debug_mode:
                print("📊 Small dataset detected, using semantic-first filtering...")
            self.min_answer_length = 12  # More lenient for small datasets
            # Use minimal blacklist for direct string matching - rely mainly on semantic filtering
            self.answer_blacklist = {
                'short_affirmative': ['yes'],  # Only the most problematic phrase
            }
            # Lower semantic threshold to make it more effective
            self.semantic_similarity_threshold = 0.75  # Lowered from 0.88 to 0.75
            # Increase phrase reuse for small datasets
            self.max_phrase_reuse = 10  # Increased further
            if debug_mode:
                print(f"  Adjusted min_answer_length to {self.min_answer_length}")
                print(f"  Lowered semantic_similarity_threshold to {self.semantic_similarity_threshold}")
                print(f"  Increased max_phrase_reuse to {self.max_phrase_reuse}")
                print(f"  Using minimal direct blacklist with {sum(len(phrases) for phrases in self.answer_blacklist.values())} phrases")
                print(f"  Primary filtering: MPNet semantic similarity with {len(self.blacklist_embeddings)} phrases")
        else:
            # For larger datasets, use balanced approach
            self.semantic_similarity_threshold = 0.80  # Lowered from 0.88
            if debug_mode:
                print(f"📊 Large dataset: using balanced filtering with semantic threshold {self.semantic_similarity_threshold}")
        
        # Pre-compute answer embeddings for performance
        self.precompute_answer_embeddings(dataset)
        
        # Build visual models
        self.build_visual_knn(dataset)
        if self.use_diverse_negatives:
            self.build_visual_clusters(dataset)
        
        # Mine negatives
        negatives = {}
        samples_to_process = list(dataset.keys())[:max_samples] if max_samples else list(dataset.keys())
        
        validation_stats = {
            'total_attempts': 0,
            'hard_attempts': 0,
            'diverse_attempts': 0,
            'hard_success': 0,
            'diverse_success': 0,
            'fallback_used': 0,
            'no_candidates_found': 0
        }
        
        # Add comprehensive rejection tracking - FIXED
        rejection_stats = {
            'same_goal': 0,
            'bad_answer_length': 0,
            'bad_answer_blacklist': 0,
            'bad_answer_semantic': 0,
            'phrase_diversity_fail': 0,
            'no_text_similarity_match': 0,
            'no_visual_neighbors': 0,
            'no_valid_clusters': 0
        }
        
        # Only show detailed debug for very small datasets
        show_detailed_debug = debug_mode and len(samples_to_process) <= 10
        
        progress_bar = tqdm(samples_to_process, desc="Mining negatives", disable=False)
        for anchor_idx in progress_bar:
            validation_stats['total_attempts'] += 1

            # Decide mining strategy
            if self.use_diverse_negatives and random.random() < self.diverse_ratio:
                strategy_order = ["diverse", "hard"]
            else:
                strategy_order = ["hard"] if not self.use_diverse_negatives else ["hard", "diverse"]

            if show_detailed_debug and validation_stats['total_attempts'] <= 3:
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
                    negative_result = self.find_hard_negative(anchor_idx, dataset, rejection_stats)
                else:
                    validation_stats['diverse_attempts'] += 1
                    negative_result = self.find_diverse_negative(anchor_idx, dataset, rejection_stats)

                if negative_result is not None:
                    negative_type = strategy
                    if strategy != strategy_order[0]:
                        validation_stats['fallback_used'] += 1
                    break

            if negative_result is None:
                validation_stats['no_candidates_found'] += 1
                continue
            
            # Track success
            if negative_type == "hard":
                validation_stats['hard_success'] += 1
            else:
                validation_stats['diverse_success'] += 1

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
        
        # Print comprehensive statistics
        print(f"📊 Validation Statistics:")
        print(f"  Total attempts: {validation_stats['total_attempts']}")
        print(f"  Hard attempts: {validation_stats['hard_attempts']} (success: {validation_stats['hard_success']})")
        print(f"  Diverse attempts: {validation_stats['diverse_attempts']} (success: {validation_stats['diverse_success']})")
        print(f"  Fallback used: {validation_stats['fallback_used']}")
        print(f"  No candidates found: {validation_stats['no_candidates_found']}")
        print(f"  Success rate: {len(negatives)}/{validation_stats['total_attempts']} ({len(negatives)/validation_stats['total_attempts']*100:.1f}%)")
        
        # Print rejection breakdown - FIXED calculation
        print(f"\n🚫 Rejection Breakdown:")
        total_rejections = sum(rejection_stats.values())
        if total_rejections > 0:
            for reason, count in rejection_stats.items():
                if count > 0:  # Only show categories with actual rejections
                    percentage = (count / total_rejections * 100)
                    print(f"  {reason.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            print(f"  Total rejections analyzed: {total_rejections}")
        else:
            print("  No rejections tracked")
        
        hard_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'hard')
        diverse_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'diverse')
        
        print(f"\n✅ Mined {len(negatives)} negatives total")
        print(f"📊 Hard negatives: {hard_count}, Diverse negatives: {diverse_count}")
        
        if negatives:
            unique_phrases = len(self.used_phrases)
            avg_reuse = sum(self.used_phrases.values()) / len(self.used_phrases) if self.used_phrases else 0
            print(f"📈 Phrase diversity: {unique_phrases} unique phrases, avg reuse: {avg_reuse:.2f}")
            
            answer_lengths = [len(data['negative_text_2']) for data in negatives.values()]
            avg_length = sum(answer_lengths) / len(answer_lengths)
            print(f"📏 Answer quality: avg length {avg_length:.1f} chars")
            
            hard_sims = [data['validation_metadata_2']['text_similarity'] 
                        for data in negatives.values() 
                        if data.get('negative_type_2') == 'hard' and 'text_similarity' in data['validation_metadata_2']]
            if hard_sims:
                avg_hard_sim = sum(hard_sims) / len(hard_sims)
                print(f"🎯 Hard negative quality: avg text similarity {avg_hard_sim:.3f}")
            
            # Report visual similarity statistics with better analysis
            hard_visual_sims = [data['validation_metadata_2']['visual_similarity'] 
                               for data in negatives.values() 
                               if data.get('negative_type_2') == 'hard' and 'visual_similarity' in data['validation_metadata_2']]
            diverse_visual_sims = [data['validation_metadata_2']['visual_similarity'] 
                                  for data in negatives.values() 
                                  if data.get('negative_type_2') == 'diverse' and 'visual_similarity' in data['validation_metadata_2']]
            
            if hard_visual_sims:
                avg_hard_visual_sim = sum(hard_visual_sims) / len(hard_visual_sims)
                min_hard_vis = min(hard_visual_sims)
                max_hard_vis = max(hard_visual_sims)
                print(f"👁️ Hard visual similarity: avg {avg_hard_visual_sim:.3f} (range: {min_hard_vis:.3f} to {max_hard_vis:.3f})")
            
            if diverse_visual_sims:
                avg_diverse_visual_sim = sum(diverse_visual_sims) / len(diverse_visual_sims)
                min_diverse_vis = min(diverse_visual_sims)
                max_diverse_vis = max(diverse_visual_sims)
                print(f"🌈 Diverse visual similarity: avg {avg_diverse_visual_sim:.3f} (range: {min_diverse_vis:.3f} to {max_diverse_vis:.3f})")
            
            if not debug_mode:
                print(f"🔍 Semantic filtering: {len(self.blacklist_embeddings)} blacklist embeddings")
                print(f"   Similarity threshold: {self.semantic_similarity_threshold}")
        
        return negatives
    
    def add_hard_negatives_to_dataset(self, dataset: Dict[int, Dict[str, Any]], 
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
        from dataset import AnsweringDataset
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
        from dataset import AnsweringDataset
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
                       help='GPU ID to use')
    parser.add_argument('--num-shards', type=int, default=1,
                       help='Total number of dataset shards')
    parser.add_argument('--shard-id', type=int, default=0,
                       help='Shard index for this process')
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"🚀 Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    dataset = load_dataset(config, args.split)

    if args.num_shards > 1:
        original_size = len(dataset)
        dataset = {k: v for k, v in dataset.items() if (k % args.num_shards) == args.shard_id}
        print(f"🔀 Sharded dataset: keeping {len(dataset)} / {original_size} samples for shard {args.shard_id} of {args.num_shards}")
    
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
    
    miner.batch_size = args.batch_size
    miner.num_workers = args.num_workers
    if torch.cuda.is_available():
        miner.device = torch.device(f'cuda:{args.gpu_id}')
    
    hard_negatives = miner.mine_hard_negatives(dataset, max_samples=args.max_samples)
    updated_dataset = miner.add_hard_negatives_to_dataset(dataset, hard_negatives)
    save_dataset(updated_dataset, config, args.split)
    
    print(f"🎉 Successfully added {len(hard_negatives)} negatives to {args.split} dataset!")

if __name__ == '__main__':
    main() 