#!/usr/bin/env python3
"""
Iterative Contrastive Pipeline
Combines paraphrase generation and validation pipelines with iterative refinement.
Complete two-pipeline architecture for high-quality contrastive learning samples.
"""

import json
import time
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
from validation_pipeline import ValidationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IterativeContrastivePipeline:
    """
    Complete iterative pipeline combining generation and validation.
    Implements refinement loop until high-quality samples are obtained.
    """
    
    def __init__(self, 
                 generation_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 validation_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_iterations: int = 3,
                 target_positives: int = 2,
                 target_negatives: int = 1):
        
        self.max_iterations = max_iterations
        self.target_positives = target_positives
        self.target_negatives = target_negatives
        
        # Initialize pipelines
        self.generation_pipeline = ParaphraseGenerationPipeline(model_name=generation_model)
        self.validation_pipeline = ValidationPipeline(embedding_model=validation_model)
        
        # Iteration statistics
        self.stats = {
            'total_instructions_processed': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'average_iterations': 0,
            'validation_scores': []
        }
        
        logger.info("Initializing Iterative Contrastive Pipeline")
    
    def initialize(self) -> bool:
        """Initialize both pipelines."""
        logger.info("Loading generation and validation models...")
        
        # Load generation model
        if not self.generation_pipeline.load_model():
            logger.error("Failed to load generation model")
            return False
        
        # Load validation model
        if not self.validation_pipeline.load_embedding_model():
            logger.error("Failed to load validation model")
            return False
        
        logger.info("✅ Both pipelines initialized successfully")
        return True
    
    def generate_contrastive_samples(self, instruction: str) -> Dict[str, any]:
        """
        Generate high-quality contrastive samples with iterative refinement.
        
        Args:
            instruction: Original navigation instruction
            
        Returns:
            Dictionary with validated positive and negative samples
        """
        logger.info(f"Generating contrastive samples for: {instruction}")
        
        best_result = {
            'original': instruction,
            'positives': [],
            'negatives': [],
            'validation_results': {},
            'iterations_used': 0,
            'success': False,
            'generation_time': 0,
            'validation_time': 0
        }
        
        generation_start = time.time()
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"--- Iteration {iteration}/{self.max_iterations} ---")
            
            # Step 1: Generate paraphrases
            iteration_start = time.time()
            generation_results = self.generation_pipeline.generate_paraphrases(
                instruction, strategy="combined"
            )
            
            if not generation_results['positives'] and not generation_results['negatives']:
                logger.warning(f"No paraphrases generated in iteration {iteration}")
                continue
            
            # Step 2: Validate paraphrases
            validation_start = time.time()
            validation_results = self.validation_pipeline.validate_paraphrase_batch(
                instruction,
                generation_results['positives'],
                generation_results['negatives']
            )
            validation_time = time.time() - validation_start
            
            # Step 3: Check if we have sufficient valid samples
            valid_positives = [
                result for result in validation_results['positive_results'] 
                if result['is_valid']
            ]
            valid_negatives = [
                result for result in validation_results['negative_results'] 
                if result['is_valid']
            ]
            
            logger.info(f"Valid samples: {len(valid_positives)} positives, {len(valid_negatives)} negatives")
            
            # Update best result if this iteration is better
            if (len(valid_positives) >= len(best_result['positives']) and 
                len(valid_negatives) >= len(best_result['negatives'])):
                
                best_result.update({
                    'positives': [r['text'] for r in valid_positives[:self.target_positives]],
                    'negatives': [r['text'] for r in valid_negatives[:self.target_negatives]],
                    'validation_results': validation_results,
                    'iterations_used': iteration,
                    'validation_time': best_result['validation_time'] + validation_time
                })
            
            # Check success criteria
            if (len(valid_positives) >= self.target_positives and 
                len(valid_negatives) >= self.target_negatives):
                
                best_result['success'] = True
                logger.info(f"✅ Success! Generated {len(valid_positives)} valid positives and {len(valid_negatives)} valid negatives")
                break
            
            logger.info(f"Iteration {iteration} incomplete: need {self.target_positives - len(valid_positives)} more positives, {self.target_negatives - len(valid_negatives)} more negatives")
        
        best_result['generation_time'] = time.time() - generation_start
        
        # Update statistics
        self._update_stats(best_result)
        
        return best_result
    
    def process_instruction_batch(self, instructions: List[str]) -> List[Dict[str, any]]:
        """
        Process a batch of instructions through the iterative pipeline.
        
        Args:
            instructions: List of navigation instructions
            
        Returns:
            List of results for each instruction
        """
        logger.info(f"Processing batch of {len(instructions)} instructions")
        
        batch_results = []
        batch_start = time.time()
        
        for i, instruction in enumerate(instructions, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing instruction {i}/{len(instructions)}")
            logger.info(f"{'='*60}")
            
            result = self.generate_contrastive_samples(instruction)
            batch_results.append(result)
            
            # Log progress
            if result['success']:
                logger.info(f"✅ Instruction {i} successful in {result['iterations_used']} iterations")
            else:
                logger.warning(f"❌ Instruction {i} failed after {self.max_iterations} iterations")
        
        batch_time = time.time() - batch_start
        
        # Batch summary
        successful = sum(1 for r in batch_results if r['success'])
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total instructions: {len(instructions)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Success rate: {successful/len(instructions)*100:.1f}%")
        logger.info(f"Total time: {batch_time:.1f}s")
        logger.info(f"Average time per instruction: {batch_time/len(instructions):.1f}s")
        
        return batch_results
    
    def process_dataset(self, 
                       dataset_path: str, 
                       output_path: str, 
                       max_samples: Optional[int] = None,
                       sample_randomly: bool = True) -> Dict[str, any]:
        """
        Process an entire dataset through the iterative pipeline.
        
        Args:
            dataset_path: Path to input dataset
            output_path: Path to save augmented dataset
            max_samples: Maximum number of samples to process (None for all)
            sample_randomly: Whether to sample randomly or sequentially
            
        Returns:
            Dataset processing summary
        """
        logger.info(f"Processing dataset: {dataset_path}")
        
        # Load dataset
        try:
            with open(dataset_path, 'r') as f:
                episodes = json.load(f)
            logger.info(f"Loaded {len(episodes)} episodes from dataset")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return {'success': False, 'error': str(e)}
        
        # Extract instructions
        all_instructions = []
        instruction_metadata = []
        
        for episode_idx, episode in enumerate(episodes):
            dialogs = episode.get('dialogs', [])
            for dialog_idx, dialog in enumerate(dialogs):
                if dialog and dialog.get('answer'):
                    answer = dialog['answer'].strip()
                    if answer and len(answer.split()) >= 3:
                        all_instructions.append(answer)
                        instruction_metadata.append({
                            'episode_idx': episode_idx,
                            'dialog_idx': dialog_idx,
                            'episode_id': episode.get('episode_id', f'ep_{episode_idx}')
                        })
        
        logger.info(f"Extracted {len(all_instructions)} valid instructions")
        
        # Sample instructions if needed
        if max_samples and max_samples < len(all_instructions):
            if sample_randomly:
                import random
                indices = random.sample(range(len(all_instructions)), max_samples)
                selected_instructions = [all_instructions[i] for i in indices]
                selected_metadata = [instruction_metadata[i] for i in indices]
            else:
                selected_instructions = all_instructions[:max_samples]
                selected_metadata = instruction_metadata[:max_samples]
            
            logger.info(f"Selected {len(selected_instructions)} instructions for processing")
        else:
            selected_instructions = all_instructions
            selected_metadata = instruction_metadata
        
        # Process instructions
        dataset_start = time.time()
        results = self.process_instruction_batch(selected_instructions)
        processing_time = time.time() - dataset_start
        
        # Create augmented dataset
        augmented_episodes = []
        for i, (instruction, metadata, result) in enumerate(zip(selected_instructions, selected_metadata, results)):
            if result['success']:
                # Create augmented episode
                augmented_episode = {
                    'original_episode_id': metadata['episode_id'],
                    'instruction_index': i,
                    'original_instruction': instruction,
                    'positive_paraphrases': result['positives'],
                    'negative_paraphrases': result['negatives'],
                    'validation_summary': result['validation_results']['summary'],
                    'iterations_used': result['iterations_used'],
                    'generation_time': result['generation_time'],
                    'validation_time': result['validation_time']
                }
                augmented_episodes.append(augmented_episode)
        
        # Save augmented dataset
        try:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(augmented_episodes, f, indent=2)
            
            logger.info(f"Saved {len(augmented_episodes)} augmented episodes to {output_path}")
        except Exception as e:
            logger.error(f"Error saving augmented dataset: {e}")
            return {'success': False, 'error': str(e)}
        
        # Dataset summary
        successful_count = len(augmented_episodes)
        success_rate = successful_count / len(selected_instructions) * 100
        
        summary = {
            'success': True,
            'input_dataset': dataset_path,
            'output_dataset': output_path,
            'total_instructions_processed': len(selected_instructions),
            'successful_augmentations': successful_count,
            'success_rate': success_rate,
            'processing_time': processing_time,
            'average_time_per_instruction': processing_time / len(selected_instructions),
            'pipeline_stats': self.get_statistics()
        }
        
        logger.info(f"Dataset processing complete: {success_rate:.1f}% success rate")
        return summary
    
    def get_statistics(self) -> Dict[str, any]:
        """Get pipeline performance statistics."""
        if self.stats['total_instructions_processed'] > 0:
            avg_iterations = self.stats['average_iterations'] / self.stats['total_instructions_processed']
        else:
            avg_iterations = 0
        
        return {
            'total_instructions_processed': self.stats['total_instructions_processed'],
            'successful_generations': self.stats['successful_generations'],
            'failed_generations': self.stats['failed_generations'],
            'success_rate': self.stats['successful_generations'] / max(self.stats['total_instructions_processed'], 1) * 100,
            'average_iterations_per_instruction': avg_iterations,
            'average_validation_score': sum(self.stats['validation_scores']) / max(len(self.stats['validation_scores']), 1)
        }
    
    def _update_stats(self, result: Dict[str, any]) -> None:
        """Update pipeline statistics."""
        self.stats['total_instructions_processed'] += 1
        
        if result['success']:
            self.stats['successful_generations'] += 1
        else:
            self.stats['failed_generations'] += 1
        
        self.stats['average_iterations'] += result['iterations_used']
        
        if 'validation_results' in result and 'summary' in result['validation_results']:
            validity_rate = result['validation_results']['summary'].get('overall_validity_rate', 0)
            self.stats['validation_scores'].append(validity_rate)

# Test function
def test_iterative_pipeline():
    """Test the complete iterative contrastive pipeline."""
    pipeline = IterativeContrastivePipeline(max_iterations=2)
    
    print("=== Iterative Contrastive Pipeline Test ===")
    
    # Test 1: Initialize pipelines
    print("\n1. Initializing pipelines...")
    if pipeline.initialize():
        print("✅ Pipelines initialized successfully")
    else:
        print("❌ Pipeline initialization failed")
        return
    
    # Test 2: Single instruction processing
    print("\n2. Testing single instruction processing...")
    test_instruction = "Turn right and fly over the white building at 3 o'clock"
    result = pipeline.generate_contrastive_samples(test_instruction)
    
    print(f"Original: {result['original']}")
    print(f"Success: {result['success']}")
    print(f"Iterations: {result['iterations_used']}")
    print(f"Positives: {result['positives']}")
    print(f"Negatives: {result['negatives']}")
    
    # Test 3: Batch processing
    print("\n3. Testing batch processing...")
    test_instructions = [
        "Turn right and fly over the white building at 3 o'clock",
        "Go straight ahead towards the gray road near the parking area",
        "Navigate to the brown house at 6 o'clock position"
    ]
    
    batch_results = pipeline.process_instruction_batch(test_instructions)
    print(f"Batch processed: {len(batch_results)} instructions")
    
    successful = sum(1 for r in batch_results if r['success'])
    print(f"Success rate: {successful}/{len(batch_results)} ({successful/len(batch_results)*100:.1f}%)")
    
    # Test 4: Statistics
    print("\n4. Pipeline statistics:")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n=== Iterative Pipeline Test Complete ===")

if __name__ == "__main__":
    test_iterative_pipeline() 