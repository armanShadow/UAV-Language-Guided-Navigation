#!/usr/bin/env python3
"""
Comprehensive AVDN Pipeline
==========================
Single comprehensive pipeline for AVDN dataset augmentation with paraphrases.

FEATURES:
✅ Mixtral-8x7B-Instruct paraphrase generation (GPUs 0-8)
✅ Comprehensive spatial validation (GPU 9) with:
   - Embedding similarity validation
   - Spatial feature analysis (directions, landmarks, movement verbs)
   - Clock direction recognition (1 o'clock, 2 o'clock, etc.)
   - Synonym-aware validation (north/northern, building/structure)
   - Multi-word landmark handling (parking lot)
   - UAV navigation terminology awareness
✅ AVDN dataset structure preservation (keeps Turn 0, adds paraphrases to answers)
✅ Memory optimization for 10-GPU setup
✅ Comprehensive validation reports and statistics

ARCHITECTURE:
- Generation: ParaphraseGenerationPipeline (GPUs 0-8, Mixtral distributed)
- Validation: ValidationPipeline (GPU 9, comprehensive spatial analysis)
- Processing: Sequential episode processing with GPU memory management
- Output: Augmented AVDN dataset with paraphrases field added to dialog turns

USAGE:
    python comprehensive_avdn_pipeline.py

This replaces all previous separate pipeline components (avdn_dataset_augmenter, 
paraphrase_validator, paraphrase_generator) with a single comprehensive solution.
"""

import os
import json
import time
import logging
import torch
import gc
from typing import List, Dict, Optional
from pathlib import Path

# Import only the essential components
from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
from validation_pipeline import ValidationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveAVDNPipeline:
    """
    COMPREHENSIVE AVDN Dataset Augmentation Pipeline
    ================================================
    
    Single unified pipeline that combines:
    1. Mixtral-8x7B-Instruct paraphrase generation (GPUs 0-8)
    2. Comprehensive spatial validation with UAV awareness (GPU 9)
    3. AVDN dataset structure preservation
    4. Memory-optimized processing for 10-GPU setup
    
    This replaces all previous separate components with one comprehensive solution.
    """
    
    def __init__(self):
        self.generation_pipeline = None
        self.validation_pipeline = None
        
        # Dataset paths
        self.dataset_path = "processed_data/train_data.json"
        self.output_path = "augmented_data/train_data_with_paraphrases.json"
        
        # Statistics
        self.stats = {
            'total_episodes': 0,
            'total_dialog_turns_with_answers': 0,
            'successful_paraphrases': 0,
            'failed_paraphrases': 0
        }
        
        logger.info(f"🚀 Comprehensive AVDN Pipeline initialized")
        logger.info(f"🎯 Goal: Add paraphrases to dialog turns with answers")
        logger.info(f"📂 Input: {self.dataset_path}")
        logger.info(f"💾 Output: {self.output_path}")
        logger.info(f"🔧 Features: Mixtral generation + comprehensive spatial validation")
        logger.info(f"🎮 Hardware: 10-GPU setup (GPUs 0-8: Mixtral, GPU 9: validation)")
    
    def initialize(self) -> bool:
        """Initialize pipelines with proper GPU placement."""
        try:
            logger.info("🚀 Initializing pipeline...")
            
            # Initialize generation pipeline (GPUs 0-8)
            logger.info("📝 Loading ParaphraseGenerationPipeline (GPUs 0-8)...")
            self.generation_pipeline = ParaphraseGenerationPipeline()
            
            if not self.generation_pipeline.load_model():
                logger.error("❌ Failed to load generation model")
                return False
            
            # Initialize validation pipeline (GPU 9)
            logger.info("✅ Loading ValidationPipeline (GPU 9)...")
            self.validation_pipeline = ValidationPipeline()
            
            # Move validation model to GPU 9
            if torch.cuda.is_available() and torch.cuda.device_count() > 9:
                logger.info("📍 Moving validation model to GPU 9...")
                self.validation_pipeline.model = self.validation_pipeline.model.to('cuda:9')
                self.validation_pipeline.device = 'cuda:9'
                logger.info("✅ Validation model moved to GPU 9")
            else:
                logger.warning("⚠️ GPU 9 not available, using default device")
            
            self._log_memory_status("After initialization")
            logger.info("✅ Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def load_avdn_dataset(self, max_episodes: Optional[int] = None) -> List[Dict]:
        """
        Load AVDN dataset keeping original structure.
        
        Args:
            max_episodes: Maximum number of episodes to process (None for all)
            
        Returns:
            List of episodes in original AVDN format
        """
        try:
            logger.info(f"📂 Loading AVDN dataset from {self.dataset_path}...")
            
            with open(self.dataset_path, 'r') as f:
                episodes = json.load(f)
            
            # Limit episodes if specified
            if max_episodes:
                episodes = episodes[:max_episodes]
            
            logger.info(f"📊 Loaded {len(episodes)} episodes")
            
            return episodes
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            return []
    
    def augment_episode(self, episode: Dict) -> Dict:
        """
        Augment a single episode by adding paraphrases to dialog turns with answers.
        
        Args:
            episode: Original episode in AVDN format
            
        Returns:
            Augmented episode with paraphrases added to applicable dialog turns
        """
        logger.info(f"🔄 Processing episode {episode['episode_id']}...")
        
        # Create a copy to avoid modifying the original
        augmented_episode = episode.copy()
        dialogs = episode.get('dialogs', [])
        
        if not dialogs:
            logger.info(f"  ⏭️ No dialogs in episode {episode['episode_id']}, skipping...")
            return augmented_episode
        
        # Process each dialog turn
        augmented_dialogs = []
        
        for turn_idx, dialog in enumerate(dialogs):
            # Copy the original dialog
            augmented_dialog = dialog.copy()
            
            # Check if this dialog turn has an answer
            answer = dialog.get('answer')
            
            if answer and answer.strip() and len(answer.strip().split()) >= 3:
                # This dialog turn has an answer - add paraphrases
                logger.info(f"  📝 Turn {turn_idx}: Processing answer '{answer[:50]}...'")
                
                paraphrases_result = self._generate_and_validate_paraphrases(answer)
                
                if paraphrases_result['success']:
                    # Add paraphrases field to this dialog turn
                    augmented_dialog['paraphrases'] = {
                        'positives': paraphrases_result['positives'],
                        'negatives': paraphrases_result['negatives'],
                        'valid_positives': paraphrases_result['valid_positives'],
                        'valid_negatives': paraphrases_result['valid_negatives'],
                        'validation_analysis': paraphrases_result['validation_report']
                    }
                    
                    self.stats['successful_paraphrases'] += 1
                    logger.info(f"  ✅ Turn {turn_idx}: Added paraphrases ({len(paraphrases_result['valid_positives'])} positives, {len(paraphrases_result['valid_negatives'])} negatives)")
                else:
                    self.stats['failed_paraphrases'] += 1
                    logger.error(f"  ❌ Turn {turn_idx}: Failed to generate paraphrases")
                
                self.stats['total_dialog_turns_with_answers'] += 1
            else:
                # No answer or answer too short - keep dialog turn as-is
                if turn_idx == 0:
                    logger.info(f"  ⏭️ Turn {turn_idx}: No answer (normal for turn 0)")
                else:
                    logger.info(f"  ⏭️ Turn {turn_idx}: No valid answer, skipping paraphrases")
            
            augmented_dialogs.append(augmented_dialog)
        
        # Update the episode with augmented dialogs
        augmented_episode['dialogs'] = augmented_dialogs
        self.stats['total_episodes'] += 1
        
        logger.info(f"✅ Episode {episode['episode_id']} processed")
        return augmented_episode
    
    def _generate_and_validate_paraphrases(self, answer: str) -> Dict:
        """
        Generate paraphrases for an answer and validate them.
        
        Args:
            answer: Dialog answer to paraphrase
            
        Returns:
            Dictionary with paraphrases and validation results
        """
        try:
            # Step 1: Generate paraphrases (GPUs 0-8)
            self._cleanup_generation_gpus()
            
            generation_result = self.generation_pipeline.generate_paraphrases(
                answer, 
                strategy="combined"
            )
            
            positives = generation_result.get('positives', [])
            negatives = generation_result.get('negatives', [])
            
            if not positives and not negatives:
                return {'success': False, 'error': 'No paraphrases generated'}
            
            # Step 2: Validate paraphrases (GPU 9)
            validation_result = self._validate_paraphrases(answer, positives, negatives)
            
            if not validation_result['success']:
                return {'success': False, 'error': validation_result.get('error', 'Validation failed')}
            
            return {
                'success': True,
                'positives': positives,
                'negatives': negatives,
                'valid_positives': validation_result['valid_positives'],
                'valid_negatives': validation_result['valid_negatives'],
                'validation_report': validation_result['validation_report']
            }
            
        except Exception as e:
            logger.error(f"Error generating paraphrases: {e}")
            return {'success': False, 'error': str(e)}
    
    def _validate_paraphrases(self, answer: str, positives: List[str], negatives: List[str]) -> Dict:
        """
        Validate paraphrases using GPU 9.
        Note: This is sequential processing, not parallel.
        
        Args:
            answer: Original dialog answer
            positives: List of positive paraphrases
            negatives: List of negative paraphrases
            
        Returns:
            Validation results
        """
        try:
            # Ensure we're using GPU 9
            if torch.cuda.is_available() and torch.cuda.device_count() > 9:
                torch.cuda.set_device(9)
                torch.cuda.empty_cache()
            
            validation_report = {
                'original_answer': answer,
                'valid_positives': [],
                'valid_negatives': [],
                'validation_details': {
                    'positive_results': [],
                    'negative_results': []
                }
            }
            
            # Validate positives (sequential, not parallel)
            for positive in positives:
                result = self.validation_pipeline.validate_positive_paraphrase(answer, positive)
                validation_report['validation_details']['positive_results'].append(result)
                
                if result['is_valid']:
                    validation_report['valid_positives'].append(positive)
            
            # Validate negatives (sequential, not parallel)
            for negative in negatives:
                result = self.validation_pipeline.validate_negative_paraphrase(answer, negative)
                validation_report['validation_details']['negative_results'].append(result)
                
                if result['is_valid']:
                    validation_report['valid_negatives'].append(negative)
            
            # Cleanup GPU 9
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'success': True,
                'valid_positives': validation_report['valid_positives'],
                'valid_negatives': validation_report['valid_negatives'],
                'validation_report': validation_report
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_episodes(self, episodes: List[Dict]) -> List[Dict]:
        """
        Process multiple episodes by adding paraphrases.
        This is just a simple loop, not parallel processing.
        
        Args:
            episodes: List of episodes to process
            
        Returns:
            List of augmented episodes
        """
        logger.info(f"🚀 Processing {len(episodes)} episodes...")
        
        augmented_episodes = []
        
        for i, episode in enumerate(episodes, 1):
            logger.info(f"📝 Processing episode {i}/{len(episodes)}: {episode['episode_id']}")
            
            # Process single episode
            augmented_episode = self.augment_episode(episode)
            augmented_episodes.append(augmented_episode)
            
            # Log progress
            success_rate = self.stats['successful_paraphrases'] / max(1, self.stats['total_dialog_turns_with_answers'])
            logger.info(f"📊 Progress: {i}/{len(episodes)} episodes | Paraphrase success rate: {success_rate:.2%}")
            
            # Cleanup between episodes
            self._cleanup_all_gpus()
        
        return augmented_episodes
    
    def save_augmented_dataset(self, augmented_episodes: List[Dict]) -> bool:
        """
        Save augmented dataset maintaining original AVDN structure.
        
        Args:
            augmented_episodes: List of augmented episodes
            
        Returns:
            Success status
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"💾 Saving {len(augmented_episodes)} augmented episodes to {self.output_path}...")
            
            # Save in exact same format as original AVDN dataset
            with open(self.output_path, 'w') as f:
                json.dump(augmented_episodes, f, indent=2)
            
            logger.info(f"✅ Augmented dataset saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving dataset: {e}")
            return False
    
    def _cleanup_generation_gpus(self):
        """Cleanup GPUs 0-8 (generation GPUs)."""
        if torch.cuda.is_available():
            for i in range(9):  # GPUs 0-8
                try:
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                except Exception:
                    pass
    
    def _cleanup_all_gpus(self):
        """Cleanup all GPUs."""
        gc.collect()
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                except Exception:
                    pass
    
    def _log_memory_status(self, stage: str):
        """Log memory status for all GPUs."""
        if not torch.cuda.is_available():
            return
        
        logger.info(f"📊 Memory status at {stage}:")
        for i in range(torch.cuda.device_count()):
            try:
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                free = total - allocated
                usage_pct = (allocated / total) * 100
                logger.info(f"  GPU {i}: {allocated:.2f}GB/{total:.2f}GB ({usage_pct:.1f}%)")
            except Exception:
                pass
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()

def main():
    """Run the comprehensive AVDN pipeline with first few episodes for testing."""
    
    pipeline = ComprehensiveAVDNPipeline()
    
    try:
        # Initialize pipeline
        if not pipeline.initialize():
            logger.error("❌ Pipeline initialization failed")
            return
        
        # Load dataset (first 2 episodes for testing)
        logger.info("📂 Loading first few AVDN episodes...")
        episodes = pipeline.load_avdn_dataset(max_episodes=2)  # Small test
        
        if not episodes:
            logger.error("❌ No episodes loaded")
            return
        
        # Show what we're processing
        logger.info(f"📊 Episodes to process:")
        for episode in episodes:
            dialogs = episode.get('dialogs', [])
            answers_count = sum(1 for d in dialogs if d.get('answer') and d['answer'].strip())
            logger.info(f"  {episode['episode_id']}: {len(dialogs)} dialog turns, {answers_count} with answers")
        
        # Process episodes
        logger.info("\n=== Processing Episodes ===")
        start_time = time.time()
        
        augmented_episodes = pipeline.process_episodes(episodes)
        
        total_time = time.time() - start_time
        
        # Save results
        logger.info("\n=== Saving Results ===")
        if pipeline.save_augmented_dataset(augmented_episodes):
            logger.info("✅ Dataset saved successfully")
        else:
            logger.error("❌ Failed to save dataset")
        
        # Show final statistics
        stats = pipeline.get_statistics()
        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"Episodes processed: {stats['total_episodes']}")
        logger.info(f"Dialog turns with answers: {stats['total_dialog_turns_with_answers']}")
        logger.info(f"Successful paraphrases: {stats['successful_paraphrases']}")
        logger.info(f"Failed paraphrases: {stats['failed_paraphrases']}")
        if stats['total_dialog_turns_with_answers'] > 0:
            success_rate = stats['successful_paraphrases'] / stats['total_dialog_turns_with_answers']
            logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
    finally:
        # Final cleanup
        pipeline._cleanup_all_gpus()

if __name__ == "__main__":
    main() 