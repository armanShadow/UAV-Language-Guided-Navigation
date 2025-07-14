#!/usr/bin/env python3
"""
Efficient Parallel Pipeline
Single model distributed across all GPUs with parallel inference using threading.
Much more efficient than loading separate models per GPU.
"""

import torch
import threading
from threading import Thread, Lock
import time
import json
from typing import List, Dict, Optional
import logging
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Set up for efficient GPU usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientParallelPipeline:
    """
    Efficient pipeline that loads model once across all GPUs and runs parallel inference.
    Uses threading for parallel instruction processing with shared model.
    """
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.num_gpus = torch.cuda.device_count()
        self.generation_pipeline = None
        self.validation_pipeline = None
        self.inference_lock = Lock()  # Protect model during inference
        self.loaded = False
        
        logger.info(f"Initializing efficient parallel pipeline")
        logger.info(f"  Available GPUs: {self.num_gpus}")
        logger.info(f"  Max concurrent instructions: {max_concurrent}")
        
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
            
            self.loaded = True
            logger.info("✅ Shared model loaded successfully across all GPUs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def process_instructions_parallel(self, instructions: List[str]) -> List[Dict]:
        """
        Process instructions in parallel using shared model with threading.
        
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
        
        logger.info(f"🚀 Processing {len(instructions)} instructions in parallel")
        logger.info(f"📊 Shared model across {self.num_gpus} GPUs, {self.max_concurrent} concurrent threads")
        
        # Results storage with thread safety
        results = [None] * len(instructions)
        results_lock = Lock()
        
        def process_single_instruction(index: int, instruction: str):
            """Process a single instruction with shared model."""
            try:
                logger.info(f"🔄 Thread {index}: Starting '{instruction[:50]}...'")
                start_time = time.time()
                
                # Use iterative pipeline logic but with shared models
                result = self._generate_contrastive_samples_shared(instruction, index)
                
                processing_time = time.time() - start_time
                result["processing_time"] = processing_time
                result["thread_id"] = index
                
                # Thread-safe result storage
                with results_lock:
                    results[index] = result
                
                status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
                logger.info(f"{status} Thread {index}: Completed in {processing_time:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ Thread {index}: Error processing instruction: {e}")
                with results_lock:
                    results[index] = {
                        "success": False,
                        "error": str(e),
                        "thread_id": index,
                        "original_instruction": instruction
                    }
        
        # Execute parallel processing
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all tasks
            futures = []
            for i, instruction in enumerate(instructions):
                future = executor.submit(process_single_instruction, i, instruction)
                futures.append(future)
            
            # Wait for completion with progress tracking
            completed = 0
            for future in as_completed(futures):
                completed += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                eta = avg_time * (len(instructions) - completed)
                
                logger.info(f"📈 Progress: {completed}/{len(instructions)} "
                           f"(avg: {avg_time:.1f}s, ETA: {eta:.1f}s)")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r and r.get("success", False))
        
        # Theoretical vs actual speedup
        avg_sequential_time = sum(r.get("processing_time", 0) for r in results if r) / len(results)
        theoretical_sequential_time = avg_sequential_time * len(instructions)
        actual_speedup = theoretical_sequential_time / total_time if total_time > 0 else 1
        
        logger.info(f"🎯 Parallel processing complete:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Success rate: {successful}/{len(instructions)} ({successful/len(instructions)*100:.1f}%)")
        logger.info(f"  Average time per instruction: {total_time/len(instructions):.1f}s")
        logger.info(f"  Actual speedup: {actual_speedup:.1f}x vs sequential")
        logger.info(f"  Efficiency: {actual_speedup/self.max_concurrent*100:.1f}% of theoretical maximum")
        
        return results
    
    def _generate_contrastive_samples_shared(self, instruction: str, thread_id: int) -> Dict:
        """
        Generate contrastive samples using shared model with thread safety.
        Implements iterative refinement logic with shared model access.
        """
        max_iterations = 3
        target_positives = 2
        target_negatives = 1
        
        valid_positives = []
        valid_negatives = []
        
        for iteration in range(1, max_iterations + 1):
            try:
                logger.debug(f"Thread {thread_id}: Iteration {iteration}/{max_iterations}")
                
                # Generate paraphrases with thread-safe model access
                with self.inference_lock:
                    # Only one thread can use model at a time for generation
                    generation_result = self.generation_pipeline.generate_paraphrases(
                        instruction, strategy="combined"
                    )
                
                if not generation_result:
                    logger.warning(f"Thread {thread_id}: No paraphrases generated in iteration {iteration}")
                    continue
                
                positives = generation_result.get('positives', [])
                negatives = generation_result.get('negatives', [])
                
                # Validate paraphrases (validation model is smaller, can be concurrent)
                for pos in positives:
                    if len(valid_positives) < target_positives:
                        result = self.validation_pipeline.validate_positive_paraphrase(instruction, pos)
                        if result.get('is_valid', False):
                            valid_positives.append(pos)
                
                for neg in negatives:
                    if len(valid_negatives) < target_negatives:
                        result = self.validation_pipeline.validate_negative_paraphrase(instruction, neg)
                        if result.get('is_valid', False):
                            valid_negatives.append(neg)
                
                # Check if we have enough samples
                if len(valid_positives) >= target_positives and len(valid_negatives) >= target_negatives:
                    logger.info(f"Thread {thread_id}: ✅ Success in iteration {iteration}")
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
                
                logger.debug(f"Thread {thread_id}: Iteration {iteration} - "
                           f"Need {target_positives - len(valid_positives)} more positives, "
                           f"{target_negatives - len(valid_negatives)} more negatives")
                
            except Exception as e:
                logger.error(f"Thread {thread_id}: Error in iteration {iteration}: {e}")
                continue
        
        # Failed after all iterations
        logger.warning(f"Thread {thread_id}: ❌ Failed after {max_iterations} iterations")
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

def test_efficient_parallel_pipeline():
    """Test the efficient parallel pipeline."""
    print("🚀 Testing Efficient Parallel Pipeline")
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
    pipeline = EfficientParallelPipeline(max_concurrent=4)
    
    print(f"\n🔧 Initializing shared model...")
    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return False
    
    print(f"✅ Shared model loaded successfully")
    
    # Process in parallel
    print(f"\n🚀 Starting efficient parallel processing...")
    results = pipeline.process_instructions_parallel(test_instructions)
    
    # Analyze results
    successful = sum(1 for r in results if r and r.get("success", False))
    print(f"\n📊 Final Results:")
    print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    # Show detailed results
    for i, result in enumerate(results):
        if result and result.get("success"):
            thread_id = result.get("thread_id", i)
            proc_time = result.get("processing_time", 0)
            iterations = result.get("iterations_used", 0)
            print(f"\n📝 Instruction {i+1} (Thread {thread_id}):")
            print(f"    Processing time: {proc_time:.1f}s")
            print(f"    Iterations: {iterations}")
            print(f"    Positives: {len(result.get('positives', []))}")
            print(f"    Negatives: {len(result.get('negatives', []))}")
    
    return successful >= len(results) * 0.5  # 50% success threshold

if __name__ == "__main__":
    success = test_efficient_parallel_pipeline()
    exit(0 if success else 1) 