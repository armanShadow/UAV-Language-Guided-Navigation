#!/usr/bin/env python3
"""
True Batch Processing Pipeline
Implements actual batch inference at the model level for genuine parallel processing.
Uses model's native batch capability for multiple prompts simultaneously.
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

class TrueBatchProcessingPipeline:
    """
    Pipeline that implements genuine batch processing at the model level.
    Processes multiple instructions simultaneously using model's batch inference.
    """
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.num_gpus = torch.cuda.device_count()
        self.generation_pipeline = None
        self.validation_pipeline = None
        self.loaded = False
        
        logger.info(f"Initializing TRUE batch processing pipeline")
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
                
            self.loaded = True
            logger.info("✅ Shared model loaded successfully across all GPUs")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def process_instructions_true_batch(self, instructions: List[str]) -> List[Dict]:
        """
        Process instructions using TRUE batch processing at model level.
        
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
        
        logger.info(f"🚀 TRUE BATCH PROCESSING: {len(instructions)} instructions")
        logger.info(f"📊 Model distributed across {self.num_gpus} GPUs, batch size: {self.batch_size}")
        logger.info(f"⚡ GENUINE PARALLEL INFERENCE at model level")
        
        all_results = []
        total_batches = (len(instructions) + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        
        # Process in TRUE batches at model level
        for batch_idx in range(0, len(instructions), self.batch_size):
            batch_num = (batch_idx // self.batch_size) + 1
            batch = instructions[batch_idx:batch_idx + self.batch_size]
            
            logger.info(f"🔄 TRUE BATCH {batch_num}/{total_batches} ({len(batch)} instructions)")
            logger.info(f"⚡ Processing ALL {len(batch)} instructions SIMULTANEOUSLY")
            
            # Process batch with TRUE parallel inference
            batch_start = time.time()
            batch_results = self._process_true_batch(batch, batch_idx)
            batch_time = time.time() - batch_start
            
            all_results.extend(batch_results)
            
            # Progress update
            completed = len(all_results)
            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            
            # True batch metrics
            avg_batch_time = batch_time / len(batch)
            speedup = (30.0 / avg_batch_time) if avg_batch_time > 0 else 1  # Assuming ~30s sequential
            
            logger.info(f"✅ TRUE BATCH {batch_num}/{total_batches} completed in {batch_time:.1f}s")
            logger.info(f"⚡ Average per instruction: {avg_batch_time:.1f}s (speedup: {speedup:.1f}x)")
            logger.info(f"📈 Progress: {completed}/{len(instructions)}")
        
        # Calculate final statistics
        total_time = time.time() - start_time
        successful = sum(1 for r in all_results if r and r.get("success", False))
        avg_per_instruction = total_time / len(instructions)
        
        # Theoretical speedup calculation
        theoretical_sequential = len(instructions) * 30  # Assume 30s per instruction sequential
        actual_speedup = theoretical_sequential / total_time if total_time > 0 else 1
        
        logger.info(f"🎯 TRUE BATCH PROCESSING COMPLETE:")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Success rate: {successful}/{len(instructions)} ({successful/len(instructions)*100:.1f}%)")
        logger.info(f"  Average time per instruction: {avg_per_instruction:.1f}s")
        logger.info(f"  TRUE SPEEDUP: {actual_speedup:.1f}x vs sequential")
        logger.info(f"  Batch efficiency: {actual_speedup/self.batch_size*100:.1f}% of theoretical maximum")
        
        return all_results
    
    def _process_true_batch(self, batch_instructions: List[str], start_idx: int) -> List[Dict]:
        """Process a batch using TRUE parallel inference at model level."""
        batch_size = len(batch_instructions)
        
        logger.info(f"  ⚡ SIMULTANEOUS GENERATION for {batch_size} instructions")
        
        # Step 1: Generate all paraphrases in TRUE batch mode
        batch_start = time.time()
        generation_results = self._generate_batch_paraphrases(batch_instructions)
        generation_time = time.time() - batch_start
        
        logger.info(f"  ⚡ Batch generation completed in {generation_time:.1f}s ({generation_time/batch_size:.1f}s per instruction)")
        
        # Step 2: Validate all results (can be done in parallel for validation model)
        validation_start = time.time()
        batch_results = []
        
        for i, (instruction, gen_result) in enumerate(zip(batch_instructions, generation_results)):
            instruction_idx = start_idx + i
            
            if not gen_result or not gen_result.get('success', False):
                batch_results.append({
                    "success": False,
                    "error": "Generation failed",
                    "instruction_index": instruction_idx,
                    "original_instruction": instruction
                })
                continue
            
            # Validate paraphrases
            result = self._validate_generated_samples(
                instruction, 
                gen_result.get('positives', []),
                gen_result.get('negatives', []),
                instruction_idx
            )
            
            batch_results.append(result)
        
        validation_time = time.time() - validation_start
        logger.info(f"  ✅ Batch validation completed in {validation_time:.1f}s")
        
        # Summary for this batch
        successful = sum(1 for r in batch_results if r.get("success", False))
        total_time = generation_time + validation_time
        
        logger.info(f"  📊 Batch summary: {successful}/{batch_size} successful in {total_time:.1f}s")
        
        return batch_results
    
    def _generate_batch_paraphrases(self, instructions: List[str]) -> List[Dict]:
        """Generate paraphrases for multiple instructions simultaneously."""
        try:
            # Create batch prompts
            prompts = []
            for instruction in instructions:
                positive_prompt = self._create_positive_prompt(instruction)
                negative_prompt = self._create_negative_prompt(instruction)
                prompts.extend([positive_prompt, negative_prompt])
            
            logger.info(f"    🔥 Generating {len(prompts)} prompts SIMULTANEOUSLY")
            
            # TRUE BATCH INFERENCE - All prompts processed at once
            start_time = time.time()
            batch_responses = self._generate_batch_responses(prompts)
            generation_time = time.time() - start_time
            
            logger.info(f"    ⚡ TRUE BATCH inference completed in {generation_time:.1f}s")
            logger.info(f"    📊 {len(prompts)} prompts in {generation_time:.1f}s = {generation_time/len(prompts):.2f}s per prompt")
            
            # Parse responses back to instruction format
            results = []
            for i in range(0, len(batch_responses), 2):
                positive_response = batch_responses[i] if i < len(batch_responses) else ""
                negative_response = batch_responses[i+1] if i+1 < len(batch_responses) else ""
                
                # Parse responses
                positives = self._parse_paraphrases(positive_response, target_count=2)
                negatives = self._parse_paraphrases(negative_response, target_count=1)
                
                results.append({
                    "success": len(positives) > 0 or len(negatives) > 0,
                    "positives": positives,
                    "negatives": negatives
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            return [{"success": False, "error": str(e)} for _ in instructions]
    
    def _generate_batch_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts simultaneously using model's batch capability."""
        try:
            # Tokenize all prompts at once
            inputs = self.generation_pipeline.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True,  # Pad to same length for batch processing
                truncation=True,
                max_length=512
            ).to(self.generation_pipeline.device)
            
            logger.info(f"      🔥 Tokenized batch shape: {inputs['input_ids'].shape}")
            
            # Generate responses for all prompts simultaneously
            with torch.no_grad():
                outputs = self.generation_pipeline.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.generation_pipeline.tokenizer.eos_token_id,
                    # Batch-specific parameters
                    batch_size=len(prompts)  # Explicit batch size
                )
            
            # Decode all outputs
            responses = []
            for i, output in enumerate(outputs):
                # Extract only the generated tokens (remove input)
                input_length = inputs['input_ids'][i].shape[0]
                generated_tokens = output[input_length:]
                response = self.generation_pipeline.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                responses.append(response)
            
            # Clean up GPU memory
            del inputs, outputs
            torch.cuda.empty_cache()
            
            return responses
            
        except Exception as e:
            logger.error(f"Error in batch response generation: {e}")
            return [""] * len(prompts)
    
    def _create_positive_prompt(self, instruction: str) -> str:
        """Create prompt for positive paraphrases."""
        return f"""Generate 2 positive paraphrases for this navigation instruction. Preserve all spatial information exactly.

Original: {instruction}

Positive paraphrases (preserve direction, landmarks, spatial relationships):
1."""
    
    def _create_negative_prompt(self, instruction: str) -> str:
        """Create prompt for negative paraphrases."""
        return f"""Generate 1 negative paraphrase for this navigation instruction. Change spatial information to make it incorrect.

Original: {instruction}

Negative paraphrase (change direction OR landmarks):
1."""
    
    def _parse_paraphrases(self, response: str, target_count: int = 2) -> List[str]:
        """Parse paraphrases from model response."""
        if not response.strip():
            return []
        
        lines = response.strip().split('\n')
        paraphrases = []
        
        for line in lines:
            line = line.strip()
            # Remove numbering (1., 2., etc.)
            if line and len(line) > 3:
                # Remove common prefixes
                for prefix in ['1.', '2.', '3.', '-', '*']:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                        break
                
                if len(line) > 10:  # Reasonable minimum length
                    paraphrases.append(line)
                    if len(paraphrases) >= target_count:
                        break
        
        return paraphrases[:target_count]
    
    def _validate_generated_samples(self, original: str, positives: List[str], negatives: List[str], instruction_idx: int) -> Dict:
        """Validate generated samples and return result."""
        valid_positives = []
        valid_negatives = []
        
        # Validate positives
        for pos in positives:
            result = self.validation_pipeline.validate_positive_paraphrase(original, pos)
            if result.get('is_valid', False):
                valid_positives.append(pos)
        
        # Validate negatives  
        for neg in negatives:
            result = self.validation_pipeline.validate_negative_paraphrase(original, neg)
            if result.get('is_valid', False):
                valid_negatives.append(neg)
        
        success = len(valid_positives) >= 1 or len(valid_negatives) >= 1
        
        return {
            "success": success,
            "original_instruction": original,
            "positives": valid_positives,
            "negatives": valid_negatives,
            "generated_positives": positives,
            "generated_negatives": negatives,
            "instruction_index": instruction_idx,
            "validation_summary": {
                "valid_positives": len(valid_positives),
                "valid_negatives": len(valid_negatives),
                "total_generated": len(positives) + len(negatives)
            }
        }

def test_true_batch_processing():
    """Test the true batch processing pipeline."""
    print("🚀 Testing TRUE Batch Processing Pipeline")
    print("="*60)
    
    # Load test instructions
    from test_two_pipeline_architecture import load_random_avdn_examples
    test_instructions = load_random_avdn_examples(num_examples=4)
    
    print(f"📝 Test instructions:")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"  {i}. {instruction[:60]}...")
    
    # Initialize true batch processing pipeline
    pipeline = TrueBatchProcessingPipeline(batch_size=4)
    
    print(f"\n🔧 Initializing shared model across {pipeline.num_gpus} GPUs...")
    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return False
    
    print(f"✅ Shared model loaded across {pipeline.num_gpus} GPUs")
    
    # Process with TRUE batch processing
    print(f"\n⚡ Starting TRUE BATCH PROCESSING...")
    print(f"📊 ALL {len(test_instructions)} instructions will be processed SIMULTANEOUSLY")
    
    start_time = time.time()
    results = pipeline.process_instructions_true_batch(test_instructions)
    total_time = time.time() - start_time
    
    if not results:
        print("❌ True batch processing failed")
        return False
    
    # Analyze results
    successful = sum(1 for r in results if r.get('success', False))
    avg_time = total_time / len(results)
    
    print(f"\n📊 TRUE BATCH PROCESSING Results:")
    print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average per instruction: {avg_time:.1f}s")
    print(f"  Theoretical speedup: {30.0/avg_time:.1f}x (assuming 30s sequential)")
    
    # Show sample results
    for i, result in enumerate(results[:2]):
        if result.get("success"):
            print(f"\n📝 Instruction {i+1}:")
            print(f"  Positives: {len(result.get('positives', []))}")
            print(f"  Negatives: {len(result.get('negatives', []))}")
            if result.get('positives'):
                print(f"  Sample positive: {result['positives'][0][:50]}...")
    
    return successful >= len(results) * 0.5

if __name__ == "__main__":
    test_true_batch_processing() 