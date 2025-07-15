#!/usr/bin/env python3
"""
Correct AVDN Pipeline
Focuses on dialog answers (turn 1+) for AnsweringAgent training.
- Skips first instruction (no paraphrasing needed)
- Paraphrases dialog answers for turns 1+
- Augments dataset for AnsweringAgent training
"""

import os
import json
import time
import logging
import torch
import gc
from typing import List, Dict, Optional
from pathlib import Path
import random

# Import only the essential components
from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
from validation_pipeline import ValidationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorrectAVDNPipeline:
    """
    Correct pipeline for AVDN dataset processing.
    - Generation: GPUs 0-8 (Mixtral distributed)
    - Validation: GPU 9 (dedicated)
    - Focus: Dialog answers (turn 1+) for AnsweringAgent training
    """
    
    def __init__(self, validation_batch_size: int = 4):
        self.validation_batch_size = validation_batch_size
        self.generation_pipeline = None
        self.validation_pipeline = None
        
        # Dataset paths
        self.dataset_path = "processed_data/train_data.json"
        self.output_path = "augmented_data/avdn_dialog_answers_augmented.json"
        
        # Statistics
        self.stats = {
            'total_episodes': 0,
            'total_dialog_turns': 0,
            'total_answers_processed': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'validation_success_rate': 0.0
        }
        
        logger.info(f"🔧 Correct AVDN Pipeline initialized")
        logger.info(f"🎯 Focus: Dialog answers (turn 1+) for AnsweringAgent training")
        logger.info(f"📊 Validation batch size: {validation_batch_size}")
        logger.info(f"📂 Dataset: {self.dataset_path}")
        logger.info(f"💾 Output: {self.output_path}")
    
    def initialize(self) -> bool:
        """Initialize pipelines with proper GPU placement."""
        try:
            logger.info("🚀 Initializing correct pipeline...")
            
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
            logger.info("✅ Correct pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def load_avdn_dataset(self, max_episodes: Optional[int] = None) -> List[Dict]:
        """
        Load AVDN dataset and extract dialog answers for paraphrasing.
        
        Args:
            max_episodes: Maximum number of episodes to process (None for all)
            
        Returns:
            List of dialog answers that need paraphrasing
        """
        try:
            logger.info(f"📂 Loading AVDN dataset from {self.dataset_path}...")
            
            with open(self.dataset_path, 'r') as f:
                episodes = json.load(f)
            
            # Process episodes to extract dialog answers
            dialog_answers = []
            
            for episode in episodes:
                episode_id = episode['episode_id']
                dialogs = episode.get('dialogs', [])
                
                # Skip episodes without dialogs
                if not dialogs:
                    continue
                
                # Process each dialog turn (skip turn 0 - no question/answer)
                for turn_idx, dialog in enumerate(dialogs):
                    if turn_idx == 0:
                        # Skip first turn (no question/answer)
                        continue
                    
                    question = dialog.get('question')
                    answer = dialog.get('answer')
                    
                    # Skip if no question or answer
                    if not question or not answer:
                        continue
                    
                    # Skip if answer is too short
                    if len(answer.strip().split()) < 3:
                        continue
                    
                    # Create dialog answer entry
                    dialog_answer = {
                        'episode_id': episode_id,
                        'turn_id': turn_idx,
                        'question': question.strip(),
                        'answer': answer.strip(),
                        'first_instruction': episode.get('first_instruction', '').strip(),
                        'map_name': episode.get('map_name', ''),
                        'observation': dialog.get('observation', {}),
                        'original_episode': episode  # Keep reference
                    }
                    
                    dialog_answers.append(dialog_answer)
                    self.stats['total_dialog_turns'] += 1
                
                self.stats['total_episodes'] += 1
                
                # Limit episodes if specified
                if max_episodes and self.stats['total_episodes'] >= max_episodes:
                    break
            
            logger.info(f"📊 Loaded {len(dialog_answers)} dialog answers from {self.stats['total_episodes']} episodes")
            logger.info(f"📊 Average {len(dialog_answers)/self.stats['total_episodes']:.1f} dialog turns per episode")
            
            return dialog_answers
            
        except Exception as e:
            logger.error(f"❌ Error loading dataset: {e}")
            return []
    
    def process_dialog_answer(self, dialog_answer: Dict) -> Dict:
        """
        Process a single dialog answer: generate paraphrases and validate.
        
        Args:
            dialog_answer: Dialog answer dictionary
            
        Returns:
            Augmented dialog answer with paraphrases and validation analysis
        """
        start_time = time.time()
        answer = dialog_answer['answer']
        
        logger.info(f"🔄 Processing {dialog_answer['episode_id']}_T{dialog_answer['turn_id']}: {answer[:50]}...")
        
        try:
            # Step 1: Generate paraphrases (GPUs 0-8)
            logger.info("📝 Generating paraphrases...")
            self._cleanup_generation_gpus()
            
            generation_result = self.generation_pipeline.generate_paraphrases(
                answer, 
                strategy="combined"
            )
            
            positives = generation_result.get('positives', [])
            negatives = generation_result.get('negatives', [])
            
            if not positives and not negatives:
                logger.error("❌ Generation failed: No paraphrases generated")
                return self._create_failed_result(dialog_answer, "Generation failed")
            
            logger.info(f"✅ Generated {len(positives)} positives, {len(negatives)} negatives")
            
            # Step 2: Validate paraphrases (GPU 9)
            logger.info("🔍 Validating paraphrases...")
            validation_result = self._validate_paraphrases(answer, positives, negatives)
            
            if not validation_result['success']:
                logger.error(f"❌ Validation failed: {validation_result.get('error', 'Unknown error')}")
                return self._create_failed_result(dialog_answer, validation_result.get('error', 'Validation failed'))
            
            # Step 3: Create augmented dialog answer
            processing_time = time.time() - start_time
            
            augmented_dialog_answer = {
                **dialog_answer,  # Keep original dialog data
                'paraphrases': {
                    'positives': positives,
                    'negatives': negatives,
                    'valid_positives': validation_result['valid_positives'],
                    'valid_negatives': validation_result['valid_negatives']
                },
                'validation_analysis': validation_result['validation_report'],
                'processing_metadata': {
                    'processing_time': processing_time,
                    'generation_success': True,
                    'validation_success': validation_result['success'],
                    'timestamp': time.time()
                }
            }
            
            # Update statistics
            self.stats['successful_generations'] += 1
            self.stats['total_answers_processed'] += 1
            
            logger.info(f"✅ Dialog answer processed successfully in {processing_time:.2f}s")
            logger.info(f"📊 Valid: {len(validation_result['valid_positives'])} positives, {len(validation_result['valid_negatives'])} negatives")
            
            return augmented_dialog_answer
            
        except Exception as e:
            logger.error(f"❌ Error processing dialog answer: {e}")
            return self._create_failed_result(dialog_answer, str(e))
    
    def _validate_paraphrases(self, answer: str, positives: List[str], negatives: List[str]) -> Dict:
        """
        Validate paraphrases using GPU 9.
        
        Args:
            answer: Original dialog answer
            positives: List of positive paraphrases
            negatives: List of negative paraphrases
            
        Returns:
            Validation results with detailed analysis
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
                },
                'summary': {
                    'positive_success_rate': 0.0,
                    'negative_success_rate': 0.0,
                    'overall_quality_score': 0.0
                }
            }
            
            # Validate positives
            logger.info(f"🔍 Validating {len(positives)} positive paraphrases...")
            for i, positive in enumerate(positives, 1):
                result = self.validation_pipeline.validate_positive_paraphrase(answer, positive)
                validation_report['validation_details']['positive_results'].append(result)
                
                if result['is_valid']:
                    validation_report['valid_positives'].append(positive)
                    logger.info(f"  ✅ Positive {i}/{len(positives)} valid")
                else:
                    logger.info(f"  ❌ Positive {i}/{len(positives)} invalid")
            
            # Validate negatives
            logger.info(f"🔍 Validating {len(negatives)} negative paraphrases...")
            for i, negative in enumerate(negatives, 1):
                result = self.validation_pipeline.validate_negative_paraphrase(answer, negative)
                validation_report['validation_details']['negative_results'].append(result)
                
                if result['is_valid']:
                    validation_report['valid_negatives'].append(negative)
                    logger.info(f"  ✅ Negative {i}/{len(negatives)} valid")
                else:
                    logger.info(f"  ❌ Negative {i}/{len(negatives)} invalid")
            
            # Calculate summary statistics
            if positives:
                validation_report['summary']['positive_success_rate'] = len(validation_report['valid_positives']) / len(positives)
            if negatives:
                validation_report['summary']['negative_success_rate'] = len(validation_report['valid_negatives']) / len(negatives)
            
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
            logger.error(f"❌ Validation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_failed_result(self, dialog_answer: Dict, error_message: str) -> Dict:
        """Create a failed result entry."""
        self.stats['failed_generations'] += 1
        self.stats['total_answers_processed'] += 1
        
        return {
            **dialog_answer,
            'paraphrases': {
                'positives': [],
                'negatives': [],
                'valid_positives': [],
                'valid_negatives': []
            },
            'validation_analysis': {},
            'processing_metadata': {
                'processing_time': 0.0,
                'generation_success': False,
                'validation_success': False,
                'error': error_message,
                'timestamp': time.time()
            }
        }
    
    def process_batch(self, dialog_answers: List[Dict]) -> List[Dict]:
        """
        Process a batch of dialog answers.
        
        Args:
            dialog_answers: List of dialog answers to process
            
        Returns:
            List of augmented dialog answers
        """
        logger.info(f"🚀 Processing batch of {len(dialog_answers)} dialog answers...")
        
        results = []
        
        for i, dialog_answer in enumerate(dialog_answers, 1):
            logger.info(f"📝 Processing {i}/{len(dialog_answers)}: {dialog_answer['episode_id']}_T{dialog_answer['turn_id']}")
            
            # Process single dialog answer
            result = self.process_dialog_answer(dialog_answer)
            results.append(result)
            
            # Log progress
            if self.stats['total_answers_processed'] > 0:
                success_rate = self.stats['successful_generations'] / self.stats['total_answers_processed']
                logger.info(f"📊 Progress: {i}/{len(dialog_answers)} | Success rate: {success_rate:.2%}")
            
            # Cleanup between dialog answers
            self._cleanup_all_gpus()
        
        return results
    
    def save_augmented_dataset(self, augmented_dialog_answers: List[Dict]) -> bool:
        """
        Save augmented dataset to file.
        
        Args:
            augmented_dialog_answers: List of augmented dialog answers
            
        Returns:
            Success status
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(self.output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"💾 Saving {len(augmented_dialog_answers)} augmented dialog answers to {self.output_path}...")
            
            # Save with metadata
            output_data = {
                'metadata': {
                    'total_dialog_answers': len(augmented_dialog_answers),
                    'processing_stats': self.stats,
                    'timestamp': time.time(),
                    'pipeline_version': 'CorrectAVDNPipeline_v1.0',
                    'description': 'Augmented AVDN dialog answers for AnsweringAgent training'
                },
                'dialog_answers': augmented_dialog_answers
            }
            
            with open(self.output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
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
        if self.stats['total_answers_processed'] > 0:
            self.stats['validation_success_rate'] = self.stats['successful_generations'] / self.stats['total_answers_processed']
        
        return self.stats.copy()

def main():
    """Test the correct AVDN pipeline with first batch of dialog answers."""
    
    pipeline = CorrectAVDNPipeline(validation_batch_size=4)
    
    try:
        # Initialize pipeline
        if not pipeline.initialize():
            logger.error("❌ Pipeline initialization failed")
            return
        
        # Load dataset (first 4 episodes for testing)
        logger.info("📂 Loading first batch of AVDN dialog answers...")
        dialog_answers = pipeline.load_avdn_dataset(max_episodes=4)  # First 4 episodes
        
        if not dialog_answers:
            logger.error("❌ No dialog answers loaded")
            return
        
        logger.info(f"📊 Loaded {len(dialog_answers)} dialog answers for processing:")
        for i, dialog_answer in enumerate(dialog_answers, 1):
            logger.info(f"  {i}. {dialog_answer['episode_id']}_T{dialog_answer['turn_id']}: Q='{dialog_answer['question'][:40]}...' A='{dialog_answer['answer'][:40]}...'")
        
        # Process batch
        logger.info("\n=== Processing Dialog Answers ===")
        start_time = time.time()
        
        augmented_dialog_answers = pipeline.process_batch(dialog_answers)
        
        total_time = time.time() - start_time
        
        # Save results
        logger.info("\n=== Saving Results ===")
        if pipeline.save_augmented_dataset(augmented_dialog_answers):
            logger.info("✅ Dataset saved successfully")
        else:
            logger.error("❌ Failed to save dataset")
        
        # Show final statistics
        stats = pipeline.get_statistics()
        logger.info(f"\n📊 Final Statistics:")
        logger.info(f"Total episodes processed: {stats['total_episodes']}")
        logger.info(f"Total dialog answers processed: {stats['total_answers_processed']}")
        logger.info(f"Successful generations: {stats['successful_generations']}")
        logger.info(f"Failed generations: {stats['failed_generations']}")
        logger.info(f"Success rate: {stats['validation_success_rate']:.2%}")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Average time per dialog answer: {total_time/len(dialog_answers):.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
    finally:
        # Final cleanup
        pipeline._cleanup_all_gpus()

if __name__ == "__main__":
    main() 