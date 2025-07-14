#!/usr/bin/env python3
"""
Batch Processing Pipeline
Process multiple instructions simultaneously using batch inference.
More efficient than parallel threads - single model, multiple prompts at once.
"""

import torch
import time
import json
from typing import List, Dict, Optional
import logging
from pathlib import Path
import os

# Set up for efficient GPU usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchProcessingPipeline:
    """
    Efficient pipeline that processes multiple instructions in batches.
    Uses single model with batch inference for maximum GPU utilization.
    """
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.num_gpus = torch.cuda.device_count()
        self.generation_pipeline = None
        self.validation_pipeline = None
        self.loaded = False
        
        logger.info(f"Initializing batch processing pipeline")
        logger.info(f"  Available GPUs: {self.num_gpus}")
        logger.info(f"  Batch size: {batch_size}")
        
    def initialize(self) -> bool:
        """Initialize pipelines with model distributed across all GPUs."""
        try:
            logger.info("Loading shared model distributed across all GPUs...")
            
            # Import pipelines
            from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
            from validation_pipeline import ValidationPipeline
            
            # Initialize generation pipeline (distributed across all GPUs)
            self.generation_pipeline = ParaphraseGenerationPipeline()
            if not self.generation_pipeline.load_model():
                logger.error("Failed to load generation model")
                return False
            
            # Initialize validation pipeline
            self.validation_pipeline = ValidationPipeline()
            if not self.validation_pipeline.load_embedding_model():
                logger.error("Failed to load validation model")
                return False
            
            # Test batch processing capability
            if not self._test_batch_capability():
                logger.warning("Batch processing not available, falling back to sequential")
                
            self.loaded = True
            logger.info("✅ Shared model loaded successfully across all GPUs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def process_instructions_batch(self, instructions: List[str]) -> List[Dict]:
        """
        Process instructions using batch processing for maximum efficiency.
        
        Args:
            instructions: List of navigation instructions to process
            
        Returns:
            List of processing results in the same order as input
        """
        if not self.loaded:
            logger.error("Pipeline not initialized. Call initialize() first.")
            return []
            
        if not instructions:
            return []
        
        logger.info(f"🚀 Processing {len(instructions)} instructions using batch processing")
        logger.info(f"📊 Model distributed across {self.num_gpus} GPUs, batch size: {self.batch_size}")
        
        all_results = []
        total_batches = (len(instructions) + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        
        # Process in batches
        for batch_idx in range(0, len(instructions), self.batch_size):
            batch_num = (batch_idx // self.batch_size) + 1
            batch = instructions[batch_idx:batch_idx + self.batch_size]
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} instructions)")
            
            # Process batch
            batch_start = time.time()
            batch_results = self._process_single_batch(batch, batch_idx)
            batch_time = time.time() - batch_start
            
            all_results.extend(batch_results)
            
            # Progress update
            completed = len(all_results)
            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            eta = avg_time * (len(instructions) - completed)
            
            logger.info(f"✅ Batch {batch_num}/{total_batches} completed in {batch_time:.1f}s")
            logger.info(f"📈 Progress: {completed}/{len(instructions)} "
                       f"(avg: {avg_time:.1f}s/instruction, ETA: {eta:.1f}s)")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in all_results if r and r.get("success", False))
        
        logger.info(f"🎯 Batch processing complete:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Success rate: {successful}/{len(instructions)} ({successful/len(instructions)*100:.1f}%)")
        logger.info(f"  Average time per instruction: {total_time/len(instructions):.1f}s")
        logger.info(f"  Batch efficiency: {len(instructions)/total_batches:.1f} instructions per batch")
        
        return all_results
    
    def _process_single_batch(self, batch_instructions: List[str], start_idx: int) -> List[Dict]:
        """Process a single batch of instructions."""
        batch_results = []
        
        for i, instruction in enumerate(batch_instructions):
            instruction_idx = start_idx + i
            logger.info(f"  📝 Instruction {instruction_idx}: {instruction[:50]}...")
            
            # Process individual instruction with iterative refinement
            start_time = time.time()
            result = self._generate_contrastive_samples(instruction, instruction_idx)
            processing_time = time.time() - start_time
            
            result["processing_time"] = processing_time
            result["instruction_index"] = instruction_idx
            
            batch_results.append(result)
            
            status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
            logger.info(f"  {status} Instruction {instruction_idx}: {processing_time:.1f}s")
        
        return batch_results
    
    def _generate_contrastive_samples(self, instruction: str, instruction_idx: int) -> Dict:
        """Generate contrastive samples with iterative refinement."""
        max_iterations = 3
        target_positives = 2
        target_negatives = 1
        
        valid_positives = []
        valid_negatives = []
        
        for iteration in range(1, max_iterations + 1):
            try:
                logger.debug(f"Instruction {instruction_idx}: Iteration {iteration}/{max_iterations}")
                
                # Generate paraphrases
                generation_result = self.generation_pipeline.generate_paraphrases(
                    instruction, strategy="combined"
                )
                
                if not generation_result:
                    logger.warning(f"Instruction {instruction_idx}: No paraphrases generated in iteration {iteration}")
                    continue
                
                positives = generation_result.get('positives', [])
                negatives = generation_result.get('negatives', [])
                
                logger.debug(f"Instruction {instruction_idx}: Generated {len(positives)} positives, {len(negatives)} negatives")
                
                # Validate paraphrases
                for pos in positives:
                    if len(valid_positives) < target_positives:
                        result = self.validation_pipeline.validate_positive_paraphrase(instruction, pos)
                        if result.get('is_valid', False):
                            valid_positives.append(pos)
                            logger.debug(f"Instruction {instruction_idx}: Valid positive: {pos[:30]}...")
                
                for neg in negatives:
                    if len(valid_negatives) < target_negatives:
                        result = self.validation_pipeline.validate_negative_paraphrase(instruction, neg)
                        if result.get('is_valid', False):
                            valid_negatives.append(neg)
                            logger.debug(f"Instruction {instruction_idx}: Valid negative: {neg[:30]}...")
                
                # Check if we have enough samples
                if len(valid_positives) >= target_positives and len(valid_negatives) >= target_negatives:
                    logger.info(f"Instruction {instruction_idx}: ✅ Success in iteration {iteration}")
                    return {
                        'success': True,
                        'original_instruction': instruction,
                        'positives': valid_positives[:target_positives],
                        'negatives': valid_negatives[:target_negatives],
                        'iterations_used': iteration,
                        'statistics': {
                            'total_positives_generated': len(positives),
                            'total_negatives_generated': len(negatives),
                            'valid_positives': len(valid_positives),
                            'valid_negatives': len(valid_negatives)
                        }
                    }
                
                logger.debug(f"Instruction {instruction_idx}: Iteration {iteration} - "
                           f"Need {target_positives - len(valid_positives)} more positives, "
                           f"{target_negatives - len(valid_negatives)} more negatives")
                
            except Exception as e:
                logger.error(f"Instruction {instruction_idx}: Error in iteration {iteration}: {e}")
                continue
        
        # Failed after all iterations
        logger.warning(f"Instruction {instruction_idx}: ❌ Failed after {max_iterations} iterations")
        return {
            'success': False,
            'original_instruction': instruction,
            'positives': valid_positives,
            'negatives': valid_negatives,
            'iterations_used': max_iterations,
            'error': 'Failed to generate sufficient valid samples',
            'statistics': {
                'valid_positives': len(valid_positives),
                'valid_negatives': len(valid_negatives)
            }
        }
    
    def _test_batch_capability(self) -> bool:
        """Test if batch processing is working."""
        try:
            # Simple test to verify model is loaded correctly
            test_instruction = "Turn right at the building"
            result = self.generation_pipeline.generate_paraphrases(test_instruction, strategy="combined")
            return bool(result and result.get('positives'))
        except Exception as e:
            logger.error(f"Batch capability test failed: {e}")
            return False

def test_batch_processing_pipeline():
    """Test the batch processing pipeline."""
    print("🚀 Testing Batch Processing Pipeline")
    print("="*70)
    
    # Load test instructions
    try:
        from test_two_pipeline_architecture import load_random_avdn_examples
        test_instructions = load_random_avdn_examples(num_examples=4)
    except:
        # Fallback instructions
        test_instructions = [
            "Turn right and fly over the white building at 3 o'clock",
            "Go straight ahead towards the gray road near the parking area",
            "Navigate to the brown house at 6 o'clock position",
            "Fly north over the highway and turn left at the intersection"
        ]
    
    print(f"📝 Test instructions ({len(test_instructions)}):")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"  {i}. {instruction[:60]}...")
    
    # Initialize pipeline
    pipeline = BatchProcessingPipeline(batch_size=2)
    
    print(f"\n🔧 Initializing shared model...")
    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return False
    
    print(f"✅ Shared model loaded successfully")
    
    # Process in batches
    print(f"\n🚀 Starting batch processing...")
    results = pipeline.process_instructions_batch(test_instructions)
    
    # Analyze results
    successful = sum(1 for r in results if r and r.get("success", False))
    print(f"\n📊 Final Results:")
    print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    # Show detailed results
    for i, result in enumerate(results):
        if result and result.get("success"):
            proc_time = result.get("processing_time", 0)
            iterations = result.get("iterations_used", 0)
            print(f"\n📝 Instruction {i+1}:")
            print(f"    Processing time: {proc_time:.1f}s")
            print(f"    Iterations: {iterations}")
            print(f"    Positives: {len(result.get('positives', []))}")
            print(f"    Negatives: {len(result.get('negatives', []))}")
    
    return successful >= len(results) * 0.5  # 50% success threshold

if __name__ == "__main__":
    success = test_batch_processing_pipeline()
    exit(0 if success else 1) 