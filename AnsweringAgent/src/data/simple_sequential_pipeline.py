#!/usr/bin/env python3
"""
Simple Sequential Pipeline
Reliable sequential processing with optimized GPU utilization per instruction.
Focus: High success rate and reliability over parallelization.
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

class SimpleSequentialPipeline:
    """
    Simple, reliable pipeline that processes instructions sequentially.
    Optimizes for success rate and GPU utilization per instruction.
    """
    
    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        self.iterative_pipeline = None
        self.loaded = False
        
        logger.info(f"Initializing Simple Sequential Pipeline")
        logger.info(f"  Available GPUs: {self.num_gpus}")
        logger.info(f"  Strategy: Sequential with optimized GPU utilization")
        
    def initialize(self) -> bool:
        """Initialize the iterative pipeline."""
        try:
            logger.info(f"Loading model distributed across {self.num_gpus} GPUs...")
            
            # Import iterative pipeline
            from iterative_contrastive_pipeline import IterativeContrastivePipeline
            
            # Initialize pipeline
            self.iterative_pipeline = IterativeContrastivePipeline()
            
            if not self.iterative_pipeline.initialize():
                logger.error("Failed to initialize iterative pipeline")
                return False
                
            self.loaded = True
            logger.info(f"✅ Pipeline loaded across {self.num_gpus} GPUs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def process_instructions_sequential(self, instructions: List[str]) -> List[Dict]:
        """
        Process instructions sequentially with optimized GPU utilization.
        
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
        
        logger.info(f"🚀 SEQUENTIAL PROCESSING: {len(instructions)} instructions")
        logger.info(f"📊 Model distributed across {self.num_gpus} GPUs")
        logger.info(f"⚡ Optimized GPU utilization per instruction")
        
        results = []
        start_time = time.time()
        
        for i, instruction in enumerate(instructions):
            logger.info(f"\n🔄 Processing instruction {i+1}/{len(instructions)}")
            logger.info(f"📝 Instruction: {instruction[:50]}...")
            
            # Process single instruction with full GPU utilization
            instruction_start = time.time()
            result = self.iterative_pipeline.generate_contrastive_samples(instruction)
            instruction_time = time.time() - instruction_start
            
            # Add metadata
            result["processing_time"] = instruction_time
            result["instruction_index"] = i
            result["original_instruction"] = instruction
            
            results.append(result)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            eta = avg_time * (len(instructions) - i - 1)
            
            status = "✅ SUCCESS" if result.get("success", False) else "❌ FAILED"
            logger.info(f"{status} Instruction {i+1}: {instruction_time:.1f}s")
            logger.info(f"📈 Progress: {i+1}/{len(instructions)} (avg: {avg_time:.1f}s, ETA: {eta:.1f}s)")
            
            # Show success details
            if result.get("success"):
                positives = len(result.get('positives', []))
                negatives = len(result.get('negatives', []))
                iterations = result.get('iterations_used', 0)
                logger.info(f"  📊 Generated: {positives} positives, {negatives} negatives in {iterations} iterations")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get("success", False))
        avg_per_instruction = total_time / len(instructions)
        
        logger.info(f"\n🎯 SEQUENTIAL PROCESSING COMPLETE:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Success rate: {successful}/{len(instructions)} ({successful/len(instructions)*100:.1f}%)")
        logger.info(f"  Average time per instruction: {avg_per_instruction:.1f}s")
        logger.info(f"  GPU utilization: {self.num_gpus} GPUs per instruction")
        
        # Success analysis
        if successful > 0:
            successful_results = [r for r in results if r.get("success")]
            avg_successful_time = sum(r.get("processing_time", 0) for r in successful_results) / len(successful_results)
            avg_iterations = sum(r.get("iterations_used", 0) for r in successful_results) / len(successful_results)
            
            logger.info(f"  Successful instructions:")
            logger.info(f"    Average time: {avg_successful_time:.1f}s")
            logger.info(f"    Average iterations: {avg_iterations:.1f}")
        
        return results

def test_simple_sequential_pipeline():
    """Test the simple sequential pipeline."""
    print("🚀 Testing Simple Sequential Pipeline")
    print("="*60)
    
    # Load test instructions
    from test_two_pipeline_architecture import load_random_avdn_examples
    test_instructions = load_random_avdn_examples(num_examples=4)
    
    print(f"📝 Test instructions:")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"  {i}. {instruction[:60]}...")
    
    # Initialize simple sequential pipeline
    pipeline = SimpleSequentialPipeline()
    
    print(f"\n🔧 Initializing pipeline across {pipeline.num_gpus} GPUs...")
    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return False
    
    print(f"✅ Pipeline loaded across {pipeline.num_gpus} GPUs")
    
    # Process sequentially
    print(f"\n⚡ Starting SEQUENTIAL PROCESSING...")
    print(f"📊 Each instruction uses ALL {pipeline.num_gpus} GPUs")
    
    start_time = time.time()
    results = pipeline.process_instructions_sequential(test_instructions)
    total_time = time.time() - start_time
    
    if not results:
        print("❌ Sequential processing failed")
        return False
    
    # Analyze results
    successful = sum(1 for r in results if r.get('success', False))
    avg_time = total_time / len(results)
    
    print(f"\n📊 SEQUENTIAL PROCESSING Results:")
    print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average per instruction: {avg_time:.1f}s")
    print(f"  GPU utilization: ALL {pipeline.num_gpus} GPUs per instruction")
    
    # Show sample results
    for i, result in enumerate(results[:2]):
        if result.get("success"):
            print(f"\n📝 Instruction {i+1}:")
            print(f"  Processing time: {result.get('processing_time', 0):.1f}s")
            print(f"  Iterations: {result.get('iterations_used', 0)}")
            print(f"  Positives: {len(result.get('positives', []))}")
            print(f"  Negatives: {len(result.get('negatives', []))}")
            if result.get('positives'):
                print(f"  Sample positive: {result['positives'][0][:50]}...")
    
    return successful >= len(results) * 0.5

if __name__ == "__main__":
    test_simple_sequential_pipeline() 