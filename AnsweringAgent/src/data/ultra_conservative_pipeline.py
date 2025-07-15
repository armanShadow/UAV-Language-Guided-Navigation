#!/usr/bin/env python3
"""
Ultra-Conservative Memory Pipeline
Designed for systems where Mixtral model is already using 97% of GPU memory.
- Single instruction processing only
- Minimal memory overhead
- Aggressive cleanup after each operation
- Reduced generation parameters
"""

import os
import json
import time
import logging
import torch
import gc
from typing import List, Dict
from pathlib import Path

# Import the core pipeline components
from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
from validation_pipeline import ValidationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraConservativeMemoryPipeline:
    """
    Ultra-conservative pipeline designed for memory-constrained environments.
    - Single instruction processing only
    - Minimal memory footprint
    - Aggressive cleanup between operations
    """
    
    def __init__(self):
        self.generation_pipeline = None
        self.validation_pipeline = None
        
        # Ultra-conservative generation parameters
        self.generation_params = {
            'max_new_tokens': 100,  # Reduced from 150
            'max_length': 400,      # Reduced from 600
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9
        }
        
        logger.info("🔧 Ultra-Conservative Memory Pipeline initialized")
        logger.info("⚠️  Single instruction processing only - no batching")
    
    def initialize(self) -> bool:
        """Initialize pipelines with memory monitoring."""
        try:
            logger.info("🚀 Initializing pipelines...")
            
            # Check initial memory
            self._log_memory_status("Before initialization")
            
            # Initialize generation pipeline (Mixtral already loaded)
            logger.info("📝 Loading ParaphraseGenerationPipeline...")
            self.generation_pipeline = ParaphraseGenerationPipeline()
            
            if not self.generation_pipeline.load_model():
                logger.error("❌ Failed to load generation model")
                return False
            
            # Aggressive cleanup after model loading
            self._aggressive_cleanup()
            self._log_memory_status("After generation model")
            
            # Initialize validation pipeline (lightweight)
            logger.info("✅ Loading ValidationPipeline...")
            self.validation_pipeline = ValidationPipeline()
            
            self._aggressive_cleanup()
            self._log_memory_status("After validation model")
            
            logger.info("✅ All pipelines initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def process_single_instruction(self, instruction: str) -> Dict:
        """
        Process a single instruction with ultra-conservative memory usage.
        """
        start_time = time.time()
        logger.info(f"🔄 Processing: '{instruction[:50]}...'")
        
        try:
            # Pre-processing cleanup
            self._aggressive_cleanup()
            self._log_memory_status("Before processing")
            
            # Generate paraphrases with minimal parameters
            logger.info("📝 Generating paraphrases (conservative mode)...")
            generation_result = self._generate_with_cleanup(instruction)
            
            if not generation_result['success']:
                return {
                    'success': False,
                    'error': generation_result.get('error', 'Generation failed'),
                    'processing_time': time.time() - start_time
                }
            
            positives = generation_result.get('positives', [])
            negatives = generation_result.get('negatives', [])
            
            logger.info(f"📊 Generated: {len(positives)} positives, {len(negatives)} negatives")
            
            # Validate with cleanup
            logger.info("🔍 Validating paraphrases...")
            validation_result = self._validate_with_cleanup(instruction, positives, negatives)
            
            if not validation_result['success']:
                return {
                    'success': False,
                    'error': validation_result.get('error', 'Validation failed'),
                    'processing_time': time.time() - start_time
                }
            
            # Final cleanup
            self._aggressive_cleanup()
            
            processing_time = time.time() - start_time
            logger.info(f"✅ Processing complete in {processing_time:.2f}s")
            
            return {
                'success': True,
                'instruction': instruction,
                'positives': positives,
                'negatives': negatives,
                'validation_report': validation_result['validation_report'],
                'processing_time': processing_time
            }
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"💥 CUDA OOM during processing: {e}")
            self._emergency_cleanup()
            return {
                'success': False,
                'error': f'CUDA Out of Memory: {str(e)}',
                'processing_time': time.time() - start_time
            }
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            self._aggressive_cleanup()
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _generate_with_cleanup(self, instruction: str) -> Dict:
        """Generate paraphrases with aggressive memory management."""
        try:
            # Pre-generation cleanup
            self._aggressive_cleanup()
            
            # Override generation parameters for minimal memory usage
            original_generate = self.generation_pipeline._generate_response
            
            def conservative_generate(prompt, max_length=400):
                # Clear cache before generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Use minimal parameters
                inputs = self.generation_pipeline.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=400,  # Reduced
                    truncation=True
                ).to(self.generation_pipeline.device)
                
                with torch.no_grad():
                    outputs = self.generation_pipeline.model.generate(
                        **inputs,
                        max_new_tokens=100,  # Reduced
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.generation_pipeline.tokenizer.eos_token_id,
                        use_cache=False,  # Disable cache to save memory
                        output_attentions=False,
                        output_hidden_states=False
                    )
                
                # Extract generated text
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                generated_text = self.generation_pipeline.tokenizer.decode(
                    generated_tokens, 
                    skip_special_tokens=True
                ).strip()
                
                # Immediate cleanup
                del inputs, outputs, generated_tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return generated_text
            
            # Temporarily replace generation method
            self.generation_pipeline._generate_response = conservative_generate
            
            # Generate paraphrases
            result = self.generation_pipeline.generate_paraphrases(
                instruction, 
                strategy="combined"
            )
            
            # Restore original method
            self.generation_pipeline._generate_response = original_generate
            
            # Post-generation cleanup
            self._aggressive_cleanup()
            
            return {
                'success': True,
                'positives': result.get('positives', []),
                'negatives': result.get('negatives', [])
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            self._emergency_cleanup()
            return {'success': False, 'error': str(e)}
    
    def _validate_with_cleanup(self, instruction: str, positives: List[str], negatives: List[str]) -> Dict:
        """Validate paraphrases with memory management."""
        try:
            # Pre-validation cleanup
            self._aggressive_cleanup()
            
            validation_report = {
                'original_instruction': instruction,
                'valid_positives': [],
                'valid_negatives': [],
                'validation_details': {
                    'positive_results': [],
                    'negative_results': []
                }
            }
            
            # Validate positives one by one
            for positive in positives:
                self._aggressive_cleanup()  # Cleanup before each validation
                
                result = self.validation_pipeline.validate_positive_paraphrase(instruction, positive)
                validation_report['validation_details']['positive_results'].append(result)
                
                if result['is_valid']:
                    validation_report['valid_positives'].append(positive)
                    logger.info(f"✅ Positive validated: {positive[:30]}...")
                else:
                    logger.warning(f"❌ Positive failed: {positive[:30]}...")
            
            # Validate negatives one by one
            for negative in negatives:
                self._aggressive_cleanup()  # Cleanup before each validation
                
                result = self.validation_pipeline.validate_negative_paraphrase(instruction, negative)
                validation_report['validation_details']['negative_results'].append(result)
                
                if result['is_valid']:
                    validation_report['valid_negatives'].append(negative)
                    logger.info(f"✅ Negative validated: {negative[:30]}...")
                else:
                    logger.warning(f"❌ Negative failed: {negative[:30]}...")
            
            # Final cleanup
            self._aggressive_cleanup()
            
            return {
                'success': True,
                'validation_report': validation_report
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            self._emergency_cleanup()
            return {'success': False, 'error': str(e)}
    
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup across all GPUs."""
        try:
            # Python garbage collection
            gc.collect()
            
            # CUDA cleanup on all devices
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    def _emergency_cleanup(self):
        """Emergency cleanup for OOM situations."""
        logger.warning("🚨 Emergency cleanup triggered")
        
        try:
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear all CUDA caches
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except Exception:
                        pass
            
            # Clear any model caches
            if hasattr(self.generation_pipeline, 'model') and self.generation_pipeline.model:
                if hasattr(self.generation_pipeline.model, 'config'):
                    self.generation_pipeline.model.config.use_cache = False
            
            logger.info("🔧 Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    def _log_memory_status(self, stage: str):
        """Log memory status for all GPUs."""
        if not torch.cuda.is_available():
            return
        
        logger.info(f"📊 Memory status at {stage}:")
        for i in range(torch.cuda.device_count()):
            try:
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                free = total - allocated
                logger.info(f"  GPU {i}: {allocated:.2f}GB allocated, {free:.2f}GB free, {total:.2f}GB total")
            except Exception:
                pass
    
    def process_multiple_instructions(self, instructions: List[str]) -> List[Dict]:
        """Process multiple instructions sequentially with cleanup between each."""
        results = []
        
        logger.info(f"🔄 Processing {len(instructions)} instructions sequentially...")
        
        for i, instruction in enumerate(instructions, 1):
            logger.info(f"📝 Processing instruction {i}/{len(instructions)}")
            
            # Process single instruction
            result = self.process_single_instruction(instruction)
            results.append(result)
            
            # Log progress
            if result['success']:
                logger.info(f"✅ Instruction {i} completed successfully")
            else:
                logger.error(f"❌ Instruction {i} failed: {result.get('error', 'Unknown error')}")
            
            # Cleanup between instructions
            self._aggressive_cleanup()
            
            # Brief pause to allow memory to stabilize
            time.sleep(0.5)
        
        return results

def main():
    """Test the ultra-conservative pipeline."""
    
    # Sample instructions
    test_instructions = [
        "Turn right and fly over the white building at 3 o'clock",
        "Head north towards the red house near the highway"
    ]
    
    pipeline = UltraConservativeMemoryPipeline()
    
    try:
        # Initialize
        if not pipeline.initialize():
            logger.error("❌ Pipeline initialization failed")
            return
        
        # Test single instruction
        logger.info("\n=== Testing Single Instruction ===")
        result = pipeline.process_single_instruction(test_instructions[0])
        
        if result['success']:
            logger.info("✅ Single instruction test passed")
            logger.info(f"Positives: {result['positives']}")
            logger.info(f"Negatives: {result['negatives']}")
        else:
            logger.error(f"❌ Single instruction test failed: {result['error']}")
        
        # Test multiple instructions
        logger.info("\n=== Testing Multiple Instructions ===")
        results = pipeline.process_multiple_instructions(test_instructions)
        
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"📊 Results: {success_count}/{len(results)} successful")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    finally:
        # Final cleanup
        pipeline._aggressive_cleanup()

if __name__ == "__main__":
    main() 