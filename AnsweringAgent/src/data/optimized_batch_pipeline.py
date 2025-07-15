#!/usr/bin/env python3
"""
Optimized Batch Processing Pipeline
Focused implementation for true parallel batch inference with batch size 4.
Optimized for 10x RTX 2080 Ti GPU setup.
"""

import torch
import time
import logging
from typing import List, Dict, Optional
import os
import re

# Configure environment
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedBatchPipeline:
    """
    Optimized batch processing pipeline for true parallel inference.
    Batch size 4, optimized for 10 GPU setup.
    """
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.num_gpus = torch.cuda.device_count()
        self.generation_pipeline = None
        self.validation_pipeline = None
        self.loaded = False
        
        logger.info(f"🚀 Initializing Optimized Batch Pipeline")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Available GPUs: {self.num_gpus}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
    
    def initialize(self) -> bool:
        """Initialize generation and validation pipelines."""
        try:
            logger.info("📦 Loading generation pipeline...")
            from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
            
            self.generation_pipeline = ParaphraseGenerationPipeline()
            if not self.generation_pipeline.load_model():
                logger.error("❌ Failed to load generation model")
                return False
            
            logger.info("📦 Loading validation pipeline...")
            from validation_pipeline import ValidationPipeline
            
            self.validation_pipeline = ValidationPipeline()
            # ValidationPipeline auto-loads in __init__
            
            self.loaded = True
            logger.info("✅ All pipelines loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize pipelines: {e}")
            return False
    
    def process_batch(self, instructions: List[str]) -> List[Dict]:
        """
        Process a batch of instructions with true parallel inference.
        
        Args:
            instructions: List of navigation instructions (up to batch_size)
            
        Returns:
            List of processing results
        """
        if not self.loaded:
            logger.error("❌ Pipeline not initialized")
            return []
        
        batch_size = len(instructions)
        logger.info(f"🔄 Processing batch of {batch_size} instructions")
        
        # Step 1: Create prompts for batch processing
        prompts = []
        for i, instruction in enumerate(instructions):
            prompt = self._create_combined_prompt(instruction)
            prompts.append(prompt)
            logger.info(f"   📝 {i+1}. {instruction[:50]}...")
        
        # Step 2: True parallel batch inference
        logger.info("⚡ Starting TRUE PARALLEL BATCH INFERENCE...")
        batch_start = time.time()
        
        try:
            # Generate all responses simultaneously
            responses = self._generate_batch_responses(prompts)
            generation_time = time.time() - batch_start
            
            logger.info(f"✅ Batch generation completed in {generation_time:.1f}s")
            logger.info(f"   Average per instruction: {generation_time/batch_size:.1f}s")
            
            # Step 3: Parse and validate results
            results = []
            validation_start = time.time()
            
            for i, (instruction, response) in enumerate(zip(instructions, responses)):
                try:
                    # Parse response
                    parsed = self._parse_response(response)
                    
                    # Validate paraphrases
                    validation_result = self._validate_paraphrases(
                        instruction, parsed['positives'], parsed['negatives']
                    )
                    
                    # Create result
                    result = {
                        'success': validation_result['success'],
                        'original_instruction': instruction,
                        'positives': parsed['positives'],
                        'negatives': parsed['negatives'],
                        'valid_positives': validation_result['valid_positives'],
                        'valid_negatives': validation_result['valid_negatives'],
                        'processing_time': generation_time / batch_size,
                        'validation_summary': validation_result['summary']
                    }
                    
                    results.append(result)
                    
                    status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
                    logger.info(f"   {i+1}. {status} - {len(result['valid_positives'])}P + {len(result['valid_negatives'])}N")
                    
                except Exception as e:
                    logger.error(f"   {i+1}. ❌ ERROR: {e}")
                    results.append({
                        'success': False,
                        'original_instruction': instruction,
                        'error': str(e),
                        'processing_time': generation_time / batch_size
                    })
            
            validation_time = time.time() - validation_start
            total_time = generation_time + validation_time
            
            # Batch summary
            successful = sum(1 for r in results if r.get('success', False))
            logger.info(f"📊 Batch summary: {successful}/{batch_size} successful in {total_time:.1f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return []
    
    def process_instructions(self, instructions: List[str]) -> List[Dict]:
        """
        Process multiple instructions in batches.
        
        Args:
            instructions: List of navigation instructions
            
        Returns:
            List of all processing results
        """
        if not instructions:
            return []
        
        all_results = []
        total_batches = (len(instructions) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"🚀 Processing {len(instructions)} instructions in {total_batches} batches")
        
        start_time = time.time()
        
        for batch_idx in range(0, len(instructions), self.batch_size):
            batch_num = (batch_idx // self.batch_size) + 1
            batch = instructions[batch_idx:batch_idx + self.batch_size]
            
            logger.info(f"\n📦 BATCH {batch_num}/{total_batches} ({len(batch)} instructions)")
            
            # Process batch
            batch_results = self.process_batch(batch)
            all_results.extend(batch_results)
            
            # Progress update
            completed = len(all_results)
            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            
            logger.info(f"✅ Batch {batch_num} completed - Progress: {completed}/{len(instructions)}")
            logger.info(f"   Average time per instruction: {avg_time:.1f}s")
        
        # Final summary
        total_time = time.time() - start_time
        successful = sum(1 for r in all_results if r.get('success', False))
        
        logger.info(f"\n🎯 FINAL RESULTS:")
        logger.info(f"   Total instructions: {len(instructions)}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Success rate: {successful/len(instructions)*100:.1f}%")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info(f"   Average per instruction: {total_time/len(instructions):.1f}s")
        
        return all_results
    
    def _create_combined_prompt(self, instruction: str) -> str:
        """Create optimized prompt for both positive and negative paraphrases."""
        return f"""<s>[INST] Generate paraphrases for this UAV navigation instruction:

Original: "{instruction}"

Generate EXACTLY:
- 2 positive paraphrases that maintain the same spatial meaning
- 1 negative paraphrase that changes spatial meaning strategically

Format:
POSITIVE 1: [paraphrase]
POSITIVE 2: [paraphrase]
NEGATIVE 1: [paraphrase]

Requirements:
- Preserve spatial terms for positives (landmarks, directions, clock positions)
- Change spatial elements for negative (direction OR landmark changes)
- Use natural language
- NO explanations or notes

[/INST]"""
    
    def _generate_batch_responses(self, prompts: List[str]) -> List[str]:
        """Generate responses for multiple prompts simultaneously."""
        try:
            logger.info(f"    🔥 Batch inference: {len(prompts)} prompts simultaneously")
            
            # Clear memory
            torch.cuda.empty_cache()
            
            # Set up tokenizer for batch processing
            if self.generation_pipeline.tokenizer.pad_token is None:
                self.generation_pipeline.tokenizer.pad_token = self.generation_pipeline.tokenizer.eos_token
            
            # Tokenize all prompts together
            inputs = self.generation_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1000
            )
            
            # Move to model device
            model_device = next(self.generation_pipeline.model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Generate responses
            with torch.no_grad():
                outputs = self.generation_pipeline.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.generation_pipeline.tokenizer.pad_token_id,
                    eos_token_id=self.generation_pipeline.tokenizer.eos_token_id,
                    use_cache=False
                )
            
            # Decode responses
            responses = []
            for i, output in enumerate(outputs):
                input_length = inputs['input_ids'][i].shape[0]
                generated_tokens = output[input_length:]
                response = self.generation_pipeline.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()
                responses.append(response)
            
            # Cleanup
            del inputs, outputs
            torch.cuda.empty_cache()
            
            return responses
            
        except Exception as e:
            logger.error(f"❌ Batch generation failed: {e}")
            return [""] * len(prompts)
    
    def _parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse response to extract positives and negatives."""
        positives = []
        negatives = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('POSITIVE 1:'):
                positives.append(line.replace('POSITIVE 1:', '').strip())
            elif line.startswith('POSITIVE 2:'):
                positives.append(line.replace('POSITIVE 2:', '').strip())
            elif line.startswith('NEGATIVE 1:'):
                negatives.append(line.replace('NEGATIVE 1:', '').strip())
        
        return {'positives': positives, 'negatives': negatives}
    
    def _validate_paraphrases(self, original: str, positives: List[str], negatives: List[str]) -> Dict:
        """Validate paraphrases using validation pipeline."""
        valid_positives = []
        valid_negatives = []
        
        # Validate positives
        for positive in positives:
            if positive:
                result = self.validation_pipeline.validate_positive_paraphrase(original, positive)
                if result['is_valid']:
                    valid_positives.append(positive)
        
        # Validate negatives
        for negative in negatives:
            if negative:
                result = self.validation_pipeline.validate_negative_paraphrase(original, negative)
                if result['is_valid']:
                    valid_negatives.append(negative)
        
        # Determine success
        success = len(valid_positives) >= 2 and len(valid_negatives) >= 1
        
        return {
            'success': success,
            'valid_positives': valid_positives,
            'valid_negatives': valid_negatives,
            'summary': {
                'total_positives': len(positives),
                'valid_positives': len(valid_positives),
                'total_negatives': len(negatives),
                'valid_negatives': len(valid_negatives)
            }
        }
    
    def cleanup(self):
        """Clean up resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ Pipeline cleanup complete")

def main():
    """Test the optimized batch pipeline."""
    # Test with sample instructions
    test_instructions = [
        "Turn right and fly over the white building at 3 o'clock",
        "Head north towards the red house near the highway",
        "Navigate left around the tall structure and proceed straight",
        "Fly northeast to the long gray building"
    ]
    
    pipeline = OptimizedBatchPipeline(batch_size=4)
    
    if pipeline.initialize():
        results = pipeline.process_instructions(test_instructions)
        
        print(f"\n📊 Test Results:")
        successful = sum(1 for r in results if r.get('success', False))
        print(f"Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        
        pipeline.cleanup()
    else:
        print("❌ Failed to initialize pipeline")

if __name__ == "__main__":
    main() 