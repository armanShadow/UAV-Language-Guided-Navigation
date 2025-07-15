#!/usr/bin/env python3
"""
Simple Batch Processing Pipeline
Clean, straightforward implementation focused on core functionality.
Based on the working enhanced_mixtral_paraphraser.py approach.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel
import re
import json
import time
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleBatchPipeline:
    """
    Simple, clean batch processing pipeline.
    Uses the proven approach from enhanced_mixtral_paraphraser.py.
    """
    
    def __init__(self, batch_size: int = 2):
        self.batch_size = batch_size
        self.num_gpus = torch.cuda.device_count()
        
        # Generation model
        self.generation_model = None
        self.generation_tokenizer = None
        
        # Validation model
        self.validation_model = None
        self.validation_tokenizer = None
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Simple Batch Pipeline")
        logger.info(f"  Available GPUs: {self.num_gpus}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Device: {self.device}")
    
    def load_models(self) -> bool:
        """Load both generation and validation models."""
        try:
            # Load generation model (Mixtral) - using the EXACT working configuration
            logger.info("Loading Mixtral for generation...")
            self.generation_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
            
            # EXACT configuration from working enhanced_mixtral_paraphraser.py
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            self.generation_model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mixtral-8x7B-Instruct-v0.1",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            
            logger.info("✅ Mixtral loaded successfully")
            
            # Load validation model (sentence transformer)
            logger.info("Loading validation model...")
            self.validation_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.validation_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)
            self.validation_model.eval()
            
            logger.info("✅ Validation model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def create_combined_prompt(self, instruction: str) -> str:
        """Create a combined prompt for both positive and negative paraphrases."""
        prompt = f"""<s>[INST] Generate paraphrases for this UAV navigation instruction:

Original: "{instruction}"

Generate EXACTLY:
1. 2 positive paraphrases that maintain the same spatial meaning
2. 1 negative paraphrase that changes spatial meaning strategically

Format your response as:
POSITIVE 1: [paraphrase]
POSITIVE 2: [paraphrase]
NEGATIVE 1: [paraphrase]

NO EXPLANATIONS OR NOTES - ONLY THE PARAPHRASES. [/INST]"""
        
        return prompt
    
    def generate_single_response(self, prompt: str) -> str:
        """Generate response for a single prompt."""
        try:
            inputs = self.generation_tokenizer(prompt, return_tensors="pt")
            
            # Move to model device
            model_device = next(self.generation_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.generation_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.generation_tokenizer.eos_token_id
                )
            
            # Extract generated text
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.generation_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def parse_response(self, response: str) -> Dict[str, List[str]]:
        """Parse the generated response into positives and negatives."""
        positives = []
        negatives = []
        
        # Look for the structured format
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('POSITIVE 1:'):
                positives.append(line.replace('POSITIVE 1:', '').strip())
            elif line.startswith('POSITIVE 2:'):
                positives.append(line.replace('POSITIVE 2:', '').strip())
            elif line.startswith('NEGATIVE 1:'):
                negatives.append(line.replace('NEGATIVE 1:', '').strip())
        
        # Clean up any notes or explanations
        def clean_paraphrase(text):
            # Remove common note patterns
            text = re.sub(r'\(.*?\)', '', text)
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'Note:.*', '', text, flags=re.IGNORECASE)
            return text.strip()
        
        positives = [clean_paraphrase(p) for p in positives if clean_paraphrase(p)]
        negatives = [clean_paraphrase(n) for n in negatives if clean_paraphrase(n)]
        
        return {'positives': positives, 'negatives': negatives}
    
    def simple_validation(self, original: str, positives: List[str], negatives: List[str]) -> Dict:
        """Simple validation with proper quality scoring and consistent logic."""
        
        # Enhanced validation criteria
        def is_valid_paraphrase(text, is_positive=True):
            if not text or len(text.strip()) < 10:
                return False, "Too short"
            
            words = text.lower().split()
            if len(words) < 3:
                return False, "Insufficient words"
            
            # Check for navigation words
            navigation_words = ['turn', 'go', 'head', 'move', 'fly', 'navigate', 'proceed', 'advance', 'steer', 'pivot', 'reverse']
            has_navigation = any(word in text.lower() for word in navigation_words)
            
            # Check for spatial words
            spatial_words = ['left', 'right', 'north', 'south', 'east', 'west', 'forward', 'backward', 'building', 'road', 'o\'clock', 'clock', 'parking', 'destination', 'target', 'direction']
            has_spatial = any(word in text.lower() for word in spatial_words)
            
            if not has_navigation:
                return False, "No navigation words"
            if not has_spatial:
                return False, "No spatial words"
                
            return True, "Valid"
        
        # Calculate quality scores (0.0 to 1.0)
        def calculate_quality_score(text, original_text):
            if not text:
                return 0.0
            
            # Basic quality factors
            length_score = min(len(text.split()) / 10, 1.0)  # Normalize to 10 words
            spatial_words = ['left', 'right', 'north', 'south', 'east', 'west', 'forward', 'building', 'road', 'o\'clock', 'destination']
            spatial_score = min(sum(1 for word in spatial_words if word in text.lower()) / 3, 1.0)
            
            # Combine scores
            return (length_score * 0.4 + spatial_score * 0.6)
        
        # Validate positives
        valid_positives = []
        positive_quality_scores = []
        positive_failures = []
        
        for i, pos in enumerate(positives):
            is_valid, reason = is_valid_paraphrase(pos, is_positive=True)
            quality_score = calculate_quality_score(pos, original)
            positive_quality_scores.append(quality_score)
            
            if is_valid:
                valid_positives.append(pos)
            else:
                positive_failures.append(f"Positive {i+1}: {reason}")
        
        # Validate negatives
        valid_negatives = []
        negative_quality_scores = []
        negative_failures = []
        
        for i, neg in enumerate(negatives):
            is_valid, reason = is_valid_paraphrase(neg, is_positive=False)
            quality_score = calculate_quality_score(neg, original)
            negative_quality_scores.append(quality_score)
            
            if is_valid:
                valid_negatives.append(neg)
            else:
                negative_failures.append(f"Negative {i+1}: {reason}")
        
        # Calculate averages
        avg_positive_quality = sum(positive_quality_scores) / len(positive_quality_scores) if positive_quality_scores else 0.0
        avg_negative_quality = sum(negative_quality_scores) / len(negative_quality_scores) if negative_quality_scores else 0.0
        
        # Determine success - require at least 1 valid positive AND 1 valid negative
        success = len(valid_positives) >= 1 and len(valid_negatives) >= 1
        
        return {
            'success': success,
            'valid_positives': valid_positives,
            'valid_negatives': valid_negatives,
            'validation_summary': {
                'valid_positives': len(valid_positives),
                'valid_negatives': len(valid_negatives)
            },
            'quality_assessment': {
                'avg_positive_quality': avg_positive_quality,
                'avg_negative_quality': avg_negative_quality,
                'positive_scores': positive_quality_scores,
                'negative_scores': negative_quality_scores
            },
            'failure_reasons': {
                'positive_failures': positive_failures,
                'negative_failures': negative_failures
            }
        }
    
    def process_instructions(self, instructions: List[str]) -> List[Dict]:
        """Process a list of instructions with simple batch processing."""
        if not self.generation_model or not self.validation_model:
            logger.error("Models not loaded. Call load_models() first.")
            return []
        
        results = []
        
        logger.info(f"🚀 Processing {len(instructions)} instructions")
        
        # Process instructions in batches
        for i in range(0, len(instructions), self.batch_size):
            batch = instructions[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(instructions) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} instructions)")
            
            batch_results = []
            
            for j, instruction in enumerate(batch):
                logger.info(f"  📝 Processing instruction {j+1}/{len(batch)}")
                
                # Generate combined prompt
                prompt = self.create_combined_prompt(instruction)
                
                # Generate response
                response = self.generate_single_response(prompt)
                
                # Parse response
                parsed = self.parse_response(response)
                
                # Simple validation
                validation = self.simple_validation(instruction, parsed['positives'], parsed['negatives'])
                
                # Create result - use only VALID paraphrases, not all generated ones
                result = {
                    'success': validation['success'],
                    'original_instruction': instruction,
                    'generated_positives': parsed['positives'],  # All generated
                    'generated_negatives': parsed['negatives'],  # All generated
                    'positives': validation['valid_positives'],  # Only valid ones
                    'negatives': validation['valid_negatives'],  # Only valid ones
                    'validation_summary': validation['validation_summary'],
                    'quality_assessment': validation['quality_assessment'],
                    'failure_reasons': validation['failure_reasons']
                }
                
                batch_results.append(result)
                
                status = "✅ SUCCESS" if validation['success'] else "❌ FAILED"
                valid_p = len(validation['valid_positives'])
                valid_n = len(validation['valid_negatives'])
                total_p = len(parsed['positives'])
                total_n = len(parsed['negatives'])
                logger.info(f"    {status}: {valid_p}/{total_p}P + {valid_n}/{total_n}N valid")
            
            results.extend(batch_results)
            logger.info(f"✅ Batch {batch_num}/{total_batches} completed")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        success_rate = (successful / len(results)) * 100 if results else 0
        
        logger.info(f"🎯 Processing complete: {successful}/{len(results)} successful ({success_rate:.1f}%)")
        
        return results

# For compatibility with existing test code
class TrueBatchProcessingPipeline:
    """Compatibility wrapper for existing test code."""
    
    def __init__(self, batch_size: int = 2):
        self.pipeline = SimpleBatchPipeline(batch_size)
        self.num_gpus = self.pipeline.num_gpus
    
    def initialize(self) -> bool:
        """Initialize the pipeline."""
        return self.pipeline.load_models()
    
    def process_instructions_true_batch(self, instructions: List[str]) -> List[Dict]:
        """Process instructions using the simple pipeline."""
        return self.pipeline.process_instructions(instructions) 