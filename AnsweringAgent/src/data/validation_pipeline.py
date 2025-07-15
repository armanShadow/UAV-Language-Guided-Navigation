#!/usr/bin/env python3
"""
Pipeline 2: Validation Pipeline
Comprehensive validation of paraphrases for spatial accuracy and quality.
Separate from generation pipeline for modular architecture.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationPipeline:
    """
    Pipeline 2: Validate paraphrases for spatial accuracy and quality.
    Focus: Embedding-based and rule-based validation approaches.
    """
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Spatial feature definitions for validation
        self.spatial_features = {
            'directions': {
                'regex_patterns': [
                    r'\d+\s*o\'?clock', r'one\s+o\'?clock', r'two\s+o\'?clock', r'three\s+o\'?clock', 
                    r'four\s+o\'?clock', r'five\s+o\'?clock', r'six\s+o\'?clock', r'seven\s+o\'?clock',
                    r'eight\s+o\'?clock', r'nine\s+o\'?clock', r'ten\s+o\'?clock', r'eleven\s+o\'?clock',
                    r'twelve\s+o\'?clock'
                ],
                'string_patterns': [
                    'north', 'south', 'east', 'west', 
                    'northwest', 'northeast', 'southwest', 'southeast',
                    'northern', 'southern', 'eastern', 'western',  # Added compass variations
                    'northeastern', 'northwestern', 'southeastern', 'southwestern',  # Added compound variations
                    'left', 'right', 'forward', 'ahead', 'straight', 'backwards', 'backward', 'reverse'
                ],
                'synonyms': {
                    'forward': ['forward', 'ahead', 'straight', 'front'],
                    'backward': ['backward', 'backwards', 'reverse', 'back', 'behind'],
                    'left': ['left', 'port'],
                    'right': ['right', 'starboard'],
                    'north': ['north', 'northern', 'northward'],  # Added compass synonyms
                    'south': ['south', 'southern', 'southward'],
                    'east': ['east', 'eastern', 'eastward'],
                    'west': ['west', 'western', 'westward'],
                    'northeast': ['northeast', 'northeastern', 'north-east'],
                    'northwest': ['northwest', 'northwestern', 'north-west'],
                    'southeast': ['southeast', 'southeastern', 'south-east'],
                    'southwest': ['southwest', 'southwestern', 'south-west']
                }
            },
            'landmarks': {
                'string_patterns': [
                    'building', 'structure', 'road', 'street', 'highway', 'house',
                    'parking', 'lot', 'area', 'destination', 'target', 'goal', 'construction', 'edifice'
                ],
                'synonyms': {
                    'building': ['building', 'structure', 'house', 'edifice', 'construction'],
                    'road': ['road', 'street', 'highway', 'path'],
                    'destination': ['destination', 'target', 'goal', 'endpoint']
                }
            },
            'movement_verbs': {
                'string_patterns': ['move', 'go', 'turn', 'head', 'fly', 'navigate', 'reverse', 'pivot', 'proceed', 'advance'],
                'synonyms': {
                    'move': ['move', 'go', 'head', 'proceed', 'travel', 'navigate', 'advance'],
                    'turn': ['turn', 'rotate', 'pivot', 'swing', 'veer'],
                    'reverse': ['reverse', 'back', 'backwards', 'backward'],
                    'fly': ['fly', 'soar', 'hover', 'pilot']
                }
            },
            'spatial_relations': {
                'string_patterns': [
                    'next to', 'beside', 'near', 'in front of', 
                    'across', 'over', 'through', 'around'
                ],
                'synonyms': {}
            }
        }
        
        # Clock number mappings for equivalence checking
        self.clock_mappings = {
            '1': ['1', 'one'], '2': ['2', 'two'], '3': ['3', 'three'], '4': ['4', 'four'],
            '5': ['5', 'five'], '6': ['6', 'six'], '7': ['7', 'seven'], '8': ['8', 'eight'],
            '9': ['9', 'nine'], '10': ['10', 'ten'], '11': ['11', 'eleven'], '12': ['12', 'twelve']
        }
        
        logger.info(f"Initializing Validation Pipeline on {self.device}")
        
        # Automatically load embedding model
        self.load_embedding_model()
    
    def load_embedding_model(self) -> bool:
        """Load sentence embedding model for semantic similarity."""
        try:
            logger.info("Loading embedding tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
            
            logger.info("Loading embedding model...")
            self.model = AutoModel.from_pretrained(self.embedding_model_name).to(self.device)
            self.model.eval()
            
            logger.info("Validation embedding model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return False
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate sentence embedding for text."""
        try:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token or mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([])
    
    def compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts using embeddings."""
        try:
            embedding1 = self.generate_embedding(text1)
            embedding2 = self.generate_embedding(text2)
            
            if len(embedding1) == 0 or len(embedding2) == 0:
                return 0.0
            
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing embedding similarity: {e}")
            return 0.0
    
    def extract_spatial_features(self, text: str) -> Dict[str, List[str]]:
        """Extract spatial features from text for preservation analysis."""
        text_lower = text.lower()
        features = {}
        
        for category, category_data in self.spatial_features.items():
            found_features = []
            
            # Process regex patterns
            if 'regex_patterns' in category_data:
                for pattern in category_data['regex_patterns']:
                    matches = re.findall(pattern, text_lower)
                    found_features.extend(matches)
            
            # Process string patterns
            if 'string_patterns' in category_data:
                for pattern in category_data['string_patterns']:
                    if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                        found_features.append(pattern)
            
            if found_features:
                features[category] = list(set(found_features))
        
        return features
    
    def validate_positive_paraphrase(self, original: str, paraphrase: str) -> Dict[str, any]:
        """
        Validate positive paraphrase for spatial accuracy preservation.
        
        Returns:
            Dictionary with validation results and scores
        """
        # Extract spatial features
        orig_features = self.extract_spatial_features(original)
        para_features = self.extract_spatial_features(paraphrase)
        
        # Compute embedding similarity
        embedding_similarity = self.compute_embedding_similarity(original, paraphrase)
        
        # Feature preservation analysis
        feature_preservation = self._analyze_feature_preservation(orig_features, para_features, is_positive=True)
        
        # Direction preservation (should be similar for positives)
        direction_similarity = self._compute_direction_similarity(
            orig_features.get('directions', []), 
            para_features.get('directions', [])
        )
        
        # Landmark preservation
        landmark_similarity = self._compute_landmark_similarity(
            orig_features.get('landmarks', []), 
            para_features.get('landmarks', [])
        )
        
        # Combined spatial score
        combined_score = (direction_similarity + landmark_similarity) / 2
        
        # REALISTIC POSITIVE VALIDATION THRESHOLDS (Updated based on test results)
        # For positive paraphrases, we want high semantic similarity AND spatial preservation
        is_valid = (
            embedding_similarity > 0.5 and  # Significantly relaxed from 0.65 - more realistic for quality paraphrases
            (direction_similarity > 0.7 or landmark_similarity > 0.5 or combined_score > 0.6)  # More lenient spatial preservation
        )
        
        return {
            'is_valid': is_valid,
            'embedding_similarity': embedding_similarity,
            'direction_similarity': direction_similarity,
            'landmark_similarity': landmark_similarity,
            'combined_score': combined_score,
            'feature_preservation': feature_preservation,
            'original_features': orig_features,
            'paraphrase_features': para_features
        }

    def validate_negative_paraphrase(self, original: str, paraphrase: str) -> Dict[str, any]:
        """
        Validate negative paraphrase for appropriate spatial changes.
        
        Returns:
            Dictionary with validation results and scores
        """
        # Extract spatial features
        orig_features = self.extract_spatial_features(original)
        para_features = self.extract_spatial_features(paraphrase)
        
        # Compute embedding similarity
        embedding_similarity = self.compute_embedding_similarity(original, paraphrase)
        
        # Feature change analysis
        feature_changes = self._analyze_feature_changes(orig_features, para_features)
        
        # Direction change (should be different for negatives)
        direction_similarity = self._compute_direction_similarity(
            orig_features.get('directions', []), 
            para_features.get('directions', [])
        )
        
        # Landmark change
        landmark_similarity = self._compute_landmark_similarity(
            orig_features.get('landmarks', []), 
            para_features.get('landmarks', [])
        )
        
        # REALISTIC NEGATIVE VALIDATION THRESHOLDS (Updated based on AVDN results)
        # For negatives: moderate semantic similarity BUT clear spatial changes
        direction_changed = direction_similarity < 0.7  # More lenient - allows more moderate changes
        landmark_changed = landmark_similarity < 0.7    # More lenient - allows more moderate changes
        spatial_changed = direction_changed or landmark_changed
        
        # Validation for negative paraphrases (Updated thresholds)
        is_valid = (
            embedding_similarity > 0.3 and  # Lower bound - still navigation-related
            embedding_similarity < 0.92 and  # Slightly relaxed upper bound from 0.90
            spatial_changed                  # Clear spatial differences required
        )
        
        # Debug logging for negative validation
        logger.debug(f"Negative validation - Embedding: {embedding_similarity:.3f}, "
                    f"Direction changed: {direction_changed}, Landmark changed: {landmark_changed}, "
                    f"Spatial changed: {spatial_changed}, Valid: {is_valid}")
        
        return {
            'is_valid': is_valid,
            'embedding_similarity': embedding_similarity,
            'direction_similarity': direction_similarity,
            'landmark_similarity': landmark_similarity,
            'direction_changed': direction_changed,
            'landmark_changed': landmark_changed,
            'spatial_changed': spatial_changed,
            'feature_changes': feature_changes,
            'original_features': orig_features,
            'paraphrase_features': para_features
        }
    
    def validate_paraphrase_batch(self, original: str, positives: List[str], negatives: List[str]) -> Dict[str, any]:
        """
        Validate a batch of paraphrases (positives and negatives).
        
        Returns:
            Comprehensive validation results for the entire batch
        """
        results = {
            'original': original,
            'positive_results': [],
            'negative_results': [],
            'summary': {}
        }
        
        # Validate positive paraphrases
        for i, positive in enumerate(positives):
            pos_result = self.validate_positive_paraphrase(original, positive)
            pos_result['index'] = i
            pos_result['text'] = positive
            results['positive_results'].append(pos_result)
        
        # Validate negative paraphrases
        for i, negative in enumerate(negatives):
            neg_result = self.validate_negative_paraphrase(original, negative)
            neg_result['index'] = i
            neg_result['text'] = negative
            results['negative_results'].append(neg_result)
        
        # Generate summary statistics
        valid_positives = sum(1 for r in results['positive_results'] if r['is_valid'])
        valid_negatives = sum(1 for r in results['negative_results'] if r['is_valid'])
        
        results['summary'] = {
            'total_positives': len(positives),
            'valid_positives': valid_positives,
            'positive_validity_rate': valid_positives / len(positives) if positives else 0,
            'total_negatives': len(negatives),
            'valid_negatives': valid_negatives,
            'negative_validity_rate': valid_negatives / len(negatives) if negatives else 0,
            'overall_validity_rate': (valid_positives + valid_negatives) / (len(positives) + len(negatives)) if (positives or negatives) else 0
        }
        
        return results
    
    def _analyze_feature_preservation(self, orig_features: Dict, para_features: Dict, is_positive: bool) -> Dict[str, any]:
        """Analyze how well spatial features are preserved."""
        preservation = {}
        
        for category in orig_features.keys():
            if category not in para_features:
                preservation[category] = {'preserved': False, 'score': 0.0}
                continue
            
            if category == 'directions':
                preserved = self._check_direction_preservation(
                    orig_features[category], 
                    para_features[category]
                )
            elif category == 'landmarks':
                preserved = self._check_landmark_preservation(
                    orig_features[category], 
                    para_features[category]
                )
            else:
                # Default preservation check
                preserved = self._check_default_preservation(
                    orig_features[category], 
                    para_features[category],
                    category
                )
            
            preservation[category] = preserved
        
        return preservation
    
    def _analyze_feature_changes(self, orig_features: Dict, para_features: Dict) -> Dict[str, any]:
        """Analyze spatial feature changes for negative validation."""
        changes = {}
        
        for category in orig_features.keys():
            if category not in para_features:
                changes[category] = {'changed': True, 'type': 'removed'}
                continue
            
            orig_set = set(orig_features[category])
            para_set = set(para_features[category])
            
            if orig_set != para_set:
                changes[category] = {
                    'changed': True,
                    'type': 'modified',
                    'added': list(para_set - orig_set),
                    'removed': list(orig_set - para_set)
                }
            else:
                changes[category] = {'changed': False, 'type': 'preserved'}
        
        return changes
    
    def _compute_direction_similarity(self, orig_dirs: List[str], para_dirs: List[str]) -> float:
        """Compute direction similarity considering synonyms and equivalences."""
        if not orig_dirs and not para_dirs:
            return 1.0
        if not orig_dirs or not para_dirs:
            return 0.0
        
        # Extract clock hours for comparison
        orig_clock_hours = set()
        para_clock_hours = set()
        
        for direction in orig_dirs:
            clock_hour = self._extract_clock_hour(direction)
            if clock_hour:
                orig_clock_hours.add(clock_hour)
        
        for direction in para_dirs:
            clock_hour = self._extract_clock_hour(direction)
            if clock_hour:
                para_clock_hours.add(clock_hour)
        
        # Clock direction similarity
        if orig_clock_hours and para_clock_hours:
            # Both have clock directions - compare them
            clock_similarity = 1.0 if orig_clock_hours == para_clock_hours else 0.0
        elif not orig_clock_hours and not para_clock_hours:
            # Neither has clock directions - no clock information to compare
            clock_similarity = 0.0
        else:
            # One has clock directions, other doesn't - different
            clock_similarity = 0.0
        
        # Synonym-based similarity for other directions
        synonym_matches = 0
        total_directions = len(orig_dirs)
        
        for orig_dir in orig_dirs:
            if self._find_direction_synonym_match(orig_dir, para_dirs):
                synonym_matches += 1
        
        synonym_similarity = synonym_matches / total_directions if total_directions > 0 else 0.0
        
        # Combined similarity
        return max(clock_similarity, synonym_similarity)
    
    def _compute_landmark_similarity(self, orig_landmarks: List[str], para_landmarks: List[str]) -> float:
        """Compute landmark similarity considering synonyms and multi-word landmarks."""
        if not orig_landmarks and not para_landmarks:
            return 1.0
        if not orig_landmarks or not para_landmarks:
            return 0.0
        
        # Create combined strings for multi-word landmark detection
        orig_combined = ' '.join(sorted(orig_landmarks)).lower()
        para_combined = ' '.join(sorted(para_landmarks)).lower()
        
        # Check for exact match first (handles "parking lot" cases)
        if orig_combined == para_combined:
            return 1.0
        
        # Check for multi-word landmark combinations
        # e.g., ["parking", "lot"] should match "parking lot"
        orig_compound = orig_combined.replace(' ', '')
        para_compound = para_combined.replace(' ', '')
        if orig_compound == para_compound:
            return 1.0
        
        # Check if one is subset of other (e.g., "lot" in "parking lot")
        if orig_combined in para_combined or para_combined in orig_combined:
            return 0.8  # High similarity for subset matches
        
        # Traditional synonym-based matching
        synonym_matches = 0
        total_landmarks = len(orig_landmarks)
        
        for orig_landmark in orig_landmarks:
            if self._find_landmark_synonym_match(orig_landmark, para_landmarks):
                synonym_matches += 1
        
        return synonym_matches / total_landmarks if total_landmarks > 0 else 0.0
    
    def _extract_clock_hour(self, direction_text: str) -> Optional[str]:
        """Extract clock hour from direction text."""
        # Match numeric clock (e.g., "5 o'clock")
        numeric_match = re.search(r'(\d+)\s*o\'?clock', direction_text.lower())
        if numeric_match:
            return numeric_match.group(1)
        
        # Match word form clock (e.g., "five o'clock")
        for hour, variants in self.clock_mappings.items():
            for variant in variants:
                if re.search(rf'\b{variant}\s+o\'?clock', direction_text.lower()):
                    return hour
        return None
    
    def _find_direction_synonym_match(self, orig_dir: str, para_dirs: List[str]) -> bool:
        """Find if original direction has synonym match in paraphrase directions."""
        synonyms = self.spatial_features['directions']['synonyms']
        
        # Normalize the original direction
        orig_dir_lower = orig_dir.lower()
        
        # Check direct match first
        for para_dir in para_dirs:
            if orig_dir_lower == para_dir.lower():
                return True
        
        # Check synonym groups
        for base_dir, synonym_list in synonyms.items():
            if orig_dir_lower in [syn.lower() for syn in synonym_list]:
                for para_dir in para_dirs:
                    para_dir_lower = para_dir.lower()
                    # Check if paraphrase direction is in the same synonym group
                    if para_dir_lower in [syn.lower() for syn in synonym_list]:
                        return True
                    # Check if paraphrase direction contains the synonym (for "northeastern direction" cases)
                    if any(syn.lower() in para_dir_lower for syn in synonym_list):
                        return True
        
        return False
    
    def _find_landmark_synonym_match(self, orig_landmark: str, para_landmarks: List[str]) -> bool:
        """Find if original landmark has synonym match in paraphrase landmarks."""
        synonyms = self.spatial_features['landmarks']['synonyms']
        
        for base_landmark, synonym_list in synonyms.items():
            if orig_landmark.lower() in synonym_list:
                for para_landmark in para_landmarks:
                    if any(syn in para_landmark.lower() for syn in synonym_list):
                        return True
        return False
    
    def _check_direction_preservation(self, orig_dirs: List[str], para_dirs: List[str]) -> Dict[str, any]:
        """Check direction preservation with synonym awareness."""
        similarity = self._compute_direction_similarity(orig_dirs, para_dirs)
        return {
            'preserved': similarity > 0.8,
            'score': similarity,
            'original': orig_dirs,
            'paraphrase': para_dirs
        }
    
    def _check_landmark_preservation(self, orig_landmarks: List[str], para_landmarks: List[str]) -> Dict[str, any]:
        """Check landmark preservation with synonym awareness."""
        similarity = self._compute_landmark_similarity(orig_landmarks, para_landmarks)
        return {
            'preserved': similarity > 0.7,
            'score': similarity,
            'original': orig_landmarks,
            'paraphrase': para_landmarks
        }
    
    def _check_default_preservation(self, orig_features: List[str], para_features: List[str], category: str) -> Dict[str, any]:
        """Default preservation check for other feature categories."""
        if not orig_features:
            return {'preserved': True, 'score': 1.0}
        
        # Simple overlap-based preservation
        orig_set = set(f.lower() for f in orig_features)
        para_set = set(f.lower() for f in para_features)
        
        overlap = len(orig_set & para_set)
        score = overlap / len(orig_set) if orig_set else 1.0
        
        return {
            'preserved': score > 0.5,
            'score': score,
            'original': orig_features,
            'paraphrase': para_features
        }

if __name__ == "__main__":
    # Simple import test - comprehensive testing is handled by comprehensive_avdn_pipeline.py
    logger.info("ValidationPipeline can be imported successfully")
    logger.info("Use comprehensive_avdn_pipeline.py for full testing and processing") 