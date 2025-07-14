#!/usr/bin/env python3
"""
Parallel GPU Pipeline
Processes instructions simultaneously across multiple GPUs for maximum throughput.
Each GPU handles one instruction independently.
"""

import torch
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
import time
import json
from typing import List, Dict, Optional
import logging
from pathlib import Path
import os

# Set up for multiprocessing with CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParallelGPUPipeline:
    """
    Parallel pipeline that processes instructions across multiple GPUs simultaneously.
    Each GPU gets its own model instance and processes one instruction at a time.
    """
    
    def __init__(self, num_gpus: int = None):
        self.num_gpus = num_gpus or torch.cuda.device_count()
        logger.info(f"Initializing parallel pipeline with {self.num_gpus} GPUs")
        
    def process_instructions_parallel(self, instructions: List[str], max_concurrent: int = None) -> List[Dict]:
        """
        Process instructions in parallel across multiple GPUs.
        
        Args:
            instructions: List of navigation instructions to process
            max_concurrent: Maximum number of concurrent GPU processes (default: num_gpus)
            
        Returns:
            List of processing results in the same order as input
        """
        if not instructions:
            return []
            
        max_concurrent = max_concurrent or self.num_gpus
        logger.info(f"Processing {len(instructions)} instructions across {max_concurrent} GPUs")
        
        # Prepare work queue and results
        work_queue = Queue()
        result_queue = Queue()
        
        # Add all instructions to work queue with indices for ordering
        for i, instruction in enumerate(instructions):
            work_queue.put((i, instruction))
        
        # Add sentinel values to signal workers to stop
        for _ in range(max_concurrent):
            work_queue.put(None)
        
        # Start worker processes
        processes = []
        for gpu_id in range(max_concurrent):
            p = Process(
                target=self._gpu_worker,
                args=(gpu_id, work_queue, result_queue)
            )
            p.start()
            processes.append(p)
        
        # Collect results
        results = {}
        completed = 0
        start_time = time.time()
        
        while completed < len(instructions):
            try:
                index, result = result_queue.get(timeout=300)  # 5 minute timeout per instruction
                results[index] = result
                completed += 1
                
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                eta = avg_time * (len(instructions) - completed)
                
                logger.info(f"Completed {completed}/{len(instructions)} instructions "
                           f"(avg: {avg_time:.1f}s/instruction, ETA: {eta:.1f}s)")
                           
            except Exception as e:
                logger.error(f"Error collecting result: {e}")
                break
        
        # Wait for all processes to complete
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.warning(f"Terminating unresponsive process {p.pid}")
                p.terminate()
                p.join()
        
        # Return results in original order
        ordered_results = [results.get(i, {"success": False, "error": "No result received"}) 
                          for i in range(len(instructions))]
        
        total_time = time.time() - start_time
        successful = sum(1 for r in ordered_results if r.get("success", False))
        
        logger.info(f"Parallel processing complete:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Success rate: {successful}/{len(instructions)} ({successful/len(instructions)*100:.1f}%)")
        logger.info(f"  Average time per instruction: {total_time/len(instructions):.1f}s")
        logger.info(f"  Theoretical speedup: {max_concurrent}x (if GPU-bound)")
        
        return ordered_results
    
    def _gpu_worker(self, gpu_id: int, work_queue: Queue, result_queue: Queue):
        """
        Worker process that runs on a specific GPU.
        Processes instructions from the work queue until None is received.
        """
        try:
            # Set CUDA device for this worker
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
            
            logger.info(f"GPU {gpu_id}: Worker started on device {device}")
            
            # Import and initialize pipelines in worker process
            # This ensures each process gets its own model instances
            from iterative_contrastive_pipeline import IterativeContrastivePipeline
            
            # Initialize pipeline for this GPU
            pipeline = IterativeContrastivePipeline()
            
            # Load models with specific GPU configuration
            logger.info(f"GPU {gpu_id}: Loading models...")
            if not self._initialize_pipeline_for_gpu(pipeline, gpu_id):
                logger.error(f"GPU {gpu_id}: Failed to initialize pipeline")
                return
            
            logger.info(f"GPU {gpu_id}: Models loaded successfully")
            
            # Process instructions from queue
            instructions_processed = 0
            while True:
                try:
                    work_item = work_queue.get(timeout=5)
                    if work_item is None:  # Sentinel value to stop
                        logger.info(f"GPU {gpu_id}: Received stop signal, processed {instructions_processed} instructions")
                        break
                    
                    index, instruction = work_item
                    logger.info(f"GPU {gpu_id}: Processing instruction {index}: {instruction[:50]}...")
                    
                    # Process the instruction
                    start_time = time.time()
                    result = pipeline.generate_contrastive_samples(instruction)
                    processing_time = time.time() - start_time
                    
                    # Add metadata
                    result["gpu_id"] = gpu_id
                    result["processing_time"] = processing_time
                    result["instruction_index"] = index
                    
                    # Send result back
                    result_queue.put((index, result))
                    instructions_processed += 1
                    
                    logger.info(f"GPU {gpu_id}: Completed instruction {index} in {processing_time:.1f}s "
                              f"(success: {result.get('success', False)})")
                    
                    # Clear GPU cache after each instruction
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"GPU {gpu_id}: Error processing instruction: {e}")
                    result_queue.put((index, {
                        "success": False, 
                        "error": str(e),
                        "gpu_id": gpu_id
                    }))
                    
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Worker failed: {e}")
        finally:
            logger.info(f"GPU {gpu_id}: Worker shutting down")
    
    def _initialize_pipeline_for_gpu(self, pipeline, gpu_id: int) -> bool:
        """Initialize pipeline components for specific GPU."""
        try:
            # Set conservative memory limits for single GPU
            max_memory = {gpu_id: "8GB", "cpu": "10GB"}
            
            # Initialize with single GPU configuration
            from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
            from validation_pipeline import ValidationPipeline
            
            # Create single-GPU generation pipeline
            generation_pipeline = ParaphraseGenerationPipeline()
            
            # Monkey patch for single GPU
            original_load = generation_pipeline.load_model
            def single_gpu_load():
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
                    import torch
                    
                    generation_pipeline.tokenizer = AutoTokenizer.from_pretrained(generation_pipeline.model_name)
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                    
                    generation_pipeline.model = AutoModelForCausalLM.from_pretrained(
                        generation_pipeline.model_name,
                        torch_dtype=torch.float16,
                        device_map={0: gpu_id},  # Single GPU mapping
                        trust_remote_code=True,
                        quantization_config=quantization_config,
                        max_memory=max_memory,
                    )
                    
                    generation_pipeline.device = f"cuda:{gpu_id}"
                    return True
                except Exception as e:
                    logger.error(f"GPU {gpu_id}: Error loading generation model: {e}")
                    return False
                    
            generation_pipeline.load_model = single_gpu_load
            
            # Initialize validation pipeline
            validation_pipeline = ValidationPipeline()
            
            # Replace pipeline components
            pipeline.generation_pipeline = generation_pipeline
            pipeline.validation_pipeline = validation_pipeline
            
            # Initialize both pipelines
            return pipeline.initialize()
            
        except Exception as e:
            logger.error(f"GPU {gpu_id}: Pipeline initialization failed: {e}")
            return False

def test_parallel_pipeline():
    """Test the parallel GPU pipeline."""
    print("🚀 Testing Parallel GPU Pipeline")
    print("="*60)
    
    # Load test instructions
    from test_two_pipeline_architecture import load_random_avdn_examples
    test_instructions = load_random_avdn_examples(num_examples=4)
    
    print(f"📝 Test instructions:")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"  {i}. {instruction[:60]}...")
    
    # Initialize parallel pipeline
    parallel_pipeline = ParallelGPUPipeline()
    
    # Process in parallel
    print(f"\n🚀 Starting parallel processing across {parallel_pipeline.num_gpus} GPUs...")
    results = parallel_pipeline.process_instructions_parallel(test_instructions)
    
    # Analyze results
    successful = sum(1 for r in results if r.get("success", False))
    print(f"\n📊 Results:")
    print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    
    # Show sample results
    for i, result in enumerate(results[:2]):
        if result.get("success"):
            print(f"\n📝 Instruction {i+1} (GPU {result.get('gpu_id', 'unknown')}):")
            print(f"  Processing time: {result.get('processing_time', 0):.1f}s")
            print(f"  Iterations: {result.get('iterations_used', 0)}")
            print(f"  Positives: {len(result.get('positives', []))}")
            print(f"  Negatives: {len(result.get('negatives', []))}")
    
    return successful >= len(results) * 0.5  # 50% success threshold

if __name__ == "__main__":
    success = test_parallel_pipeline()
    exit(0 if success else 1) 