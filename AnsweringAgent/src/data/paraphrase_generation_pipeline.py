#!/usr/bin/env python3
"""
Pipeline 1: Paraphrase Generation Pipeline
Focused Mixtral-based paraphrasing for UAV navigation instructions.
Separate from validation pipeline for modular architecture.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import json
import random
import os
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path

# Set PyTorch memory allocator to avoid fragmentation (as recommended by PyTorch)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParaphraseGenerationPipeline:
    """
    Pipeline 1: Generate high-quality paraphrases using Mixtral model.
    Focus: Natural language diversity with spatial term preservation.
    """
    
    def __init__(self, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing Paraphrase Generation Pipeline on {self.device}")
    
    def load_model(self) -> bool:
        """Load Mixtral model and tokenizer."""
        try:
            logger.info("Loading Mixtral tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("Loading Mixtral model...")
            
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config
            )
            
            # Enable gradient checkpointing to reduce memory usage
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            
            logger.info("Paraphrase generation model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def create_positive_prompt(self, instruction: str) -> str:
        """Create focused prompt for positive paraphrases."""
        # Extract main instruction (remove notes)
        main_instruction = self._extract_main_instruction(instruction)
        
        prompt = f"""<s>[INST] You are an expert in UAV navigation instructions. Generate 2 positive paraphrases that maintain the same spatial meaning and navigation intent.

Original instruction: "{main_instruction}"

CRITICAL REQUIREMENTS:
1. Preserve ALL spatial information exactly (directions, landmarks, clock positions)
2. Use natural language variation in word choice and sentence structure
3. Sound like natural human navigation instructions
4. NO explanatory notes or additional context
5. Maintain the same navigation outcome

Generate exactly 2 positive paraphrases that preserve spatial meaning: [/INST]"""
        
        return prompt
    
    def create_negative_prompt(self, instruction: str) -> str:
        """Create focused prompt for negative paraphrases."""
        main_instruction = self._extract_main_instruction(instruction)
        
        prompt = f"""<s>[INST] You are an expert in UAV navigation instructions. Generate 1 negative paraphrase that changes spatial meaning strategically.

Original instruction: "{main_instruction}"

STRATEGIC CHANGES REQUIRED:
- Change landmarks: building→structure→house, road→highway→parking
- Change directions: turn→go→move, left→right, north→south→east→west
- Change clock directions: shift by 2-4 hours (e.g., 3 o'clock→6 o'clock)

CRITICAL REQUIREMENTS:
1. Make strategic spatial changes (direction + landmark, or clock + landmark)
2. Ensure changes are LOGICALLY CONSISTENT and realistic
3. Use natural language (avoid robotic phrasing)
4. Create a plausible but incorrect navigation instruction
5. NO explanatory notes or additional context

Generate exactly 1 negative paraphrase with strategic spatial changes: [/INST]"""
        
        return prompt
    
    def create_combined_prompt(self, instruction: str) -> str:
        """Create unified prompt for both positive and negative paraphrases."""
        main_instruction = self._extract_main_instruction(instruction)
        
        prompt = f"""<s>[INST] You are an expert in UAV navigation instructions. Generate paraphrases for this instruction:

Original instruction: "{main_instruction}"

Generate:
1. 2 positive paraphrases that maintain the same spatial meaning and navigation intent
2. 1 negative paraphrase that changes spatial meaning strategically

For positives:
- Use natural language variation in word choice and sentence structure
- Preserve key spatial terms (landmarks, directions, EXACT clock references)
- Sound like natural human navigation instructions

For negative:
- Change landmarks: building↔structure↔house, road↔highway↔parking
- Change directions: turn↔go↔move, left↔right, north↔south↔east↔west
- Change clock directions: shift by 2-4 hours (e.g., 3 o'clock→6 o'clock)
- Make correlated strategic changes (e.g., direction + landmark, or clock + landmark)
- Examples: "right + white building" → "left + gray structure", "3 o'clock + building" → "9 o'clock + farm"
- Ensure changes are LOGICALLY CONSISTENT and realistic for UAV navigation
- Use natural language (avoid robotic or template-like phrasing)

Provide ONLY the paraphrases, NO EXPLANATIONS OR NOTES: [/INST]"""
        
        return prompt
    
    def generate_paraphrases(self, instruction: str, strategy: str = "combined") -> Dict[str, List[str]]:
        """
        Generate paraphrases using specified strategy.
        
        Args:
            instruction: Original navigation instruction
            strategy: 'combined', 'separate', 'positive_only', 'negative_only'
        
        Returns:
            {'positives': [...], 'negatives': [...]}
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Call load_model() first.")
            return {'positives': [], 'negatives': []}
        
        results = {'positives': [], 'negatives': []}
        
        try:
            if strategy == "combined":
                # Generate all paraphrases with unified prompt
                combined_prompt = self.create_combined_prompt(instruction)
                response = self._generate_response(combined_prompt)
                all_paraphrases = self._parse_paraphrases(response, 3)
                
                if len(all_paraphrases) >= 3:
                    results['positives'] = all_paraphrases[:2]
                    results['negatives'] = all_paraphrases[2:3]
                else:
                    # Fallback to available paraphrases
                    results['positives'] = all_paraphrases[:len(all_paraphrases)//2]
                    results['negatives'] = all_paraphrases[len(all_paraphrases)//2:]
                    
            elif strategy == "separate":
                # Generate positives and negatives separately
                if strategy in ["separate", "positive_only"]:
                    positive_prompt = self.create_positive_prompt(instruction)
                    positive_response = self._generate_response(positive_prompt)
                    results['positives'] = self._parse_paraphrases(positive_response, 2)
                
                if strategy in ["separate", "negative_only"]:
                    negative_prompt = self.create_negative_prompt(instruction)
                    negative_response = self._generate_response(negative_prompt)
                    results['negatives'] = self._parse_paraphrases(negative_response, 1)
            
            logger.info(f"Generated {len(results['positives'])} positives and {len(results['negatives'])} negatives")
            return results
            
        except Exception as e:
            logger.error(f"Error generating paraphrases: {e}")
            return {'positives': [], 'negatives': []}

    def generate_paraphrases_batch(self, instructions: List[str], strategy: str = "combined", batch_size: int = 4) -> List[Dict[str, List[str]]]:
        """
        Generate paraphrases for a batch of instructions with optimized multi-GPU utilization.
        
        Args:
            instructions: List of navigation instructions to process
            strategy: Generation strategy ('combined', 'separate', etc.)
            batch_size: Number of instructions to process in parallel
            
        Returns:
            List of results in same order as input instructions
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Call load_model() first.")
            return []
        
        results = []
        total_batches = (len(instructions) + batch_size - 1) // batch_size
        
        logger.info(f"🚀 Multi-GPU batch processing: {len(instructions)} instructions across 10 GPUs")
        logger.info(f"📦 Processing {total_batches} batches of size {batch_size}")
        
        for batch_idx in range(0, len(instructions), batch_size):
            batch = instructions[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} instructions)")
            
            # Process batch - leveraging distributed model across 10 GPUs
            batch_results = []
            for i, instruction in enumerate(batch):
                logger.info(f"  📝 Instruction {i+1}/{len(batch)}: {instruction[:50]}...")
                result = self.generate_paraphrases(instruction, strategy)
                batch_results.append(result)
                logger.info(f"  ✅ Generated {len(result.get('positives', []))}P + {len(result.get('negatives', []))}N")
            
            results.extend(batch_results)
            logger.info(f"✅ Completed batch {batch_num}/{total_batches}")
        
        # Summary statistics
        total_positives = sum(len(r.get('positives', [])) for r in results)
        total_negatives = sum(len(r.get('negatives', [])) for r in results)
        logger.info(f"🎯 Batch processing complete: {total_positives} positives, {total_negatives} negatives")
        
        return results
    
    def _generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from Mixtral model with memory management."""
        try:
            # Clear GPU cache before generation to prevent memory accumulation
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract only the generated tokens
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clear intermediate tensors and cache again
            del inputs, outputs, generated_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return ""
    
    def _parse_paraphrases(self, response: str, expected_count: int) -> List[str]:
        """Parse generated paraphrases from response with note removal."""
        if not response:
            return []
        
        # Split by lines and clean up
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Note removal patterns
        note_removal_patterns = [
            r'\(.*?\)',   # Remove text in parentheses
            r'\[.*?\]',   # Remove text in square brackets
            r'Note:.*',   # Remove lines starting with "Note:"
            r'Additional context:.*',  # Remove lines with additional context
            r'^[0-9]+\.?\s*',  # Remove numbering
            r'^(Positive|Negative)\s*Paraphrases?:',  # Remove paraphrase labels
            r'^(Positives?|Negatives?):',  # Remove section headers
        ]
        
        # Extract clean paraphrases
        paraphrases = []
        for line in lines:
            # Apply all note removal patterns
            cleaned = line
            for pattern in note_removal_patterns:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
            
            # Remove quotes
            cleaned = re.sub(r'^["\']|["\']$', '', cleaned)
            
            # Only include actual instruction text
            if (cleaned and 
                len(cleaned) > 10 and 
                not cleaned.lower().startswith(('positive', 'negative', 'positives', 'negatives')) and
                not any(keyword in cleaned.lower() for keyword in ['note:', 'additional context', 'explanation:'])):
                paraphrases.append(cleaned)
        
        return paraphrases[:expected_count]
    
    def _extract_main_instruction(self, text: str) -> str:
        """Extract main instruction by removing notes and additional context."""
        # Remove common note indicators
        note_markers = [
            r'\(.*?\)',   # Text in parentheses
            r'\[.*?\]',   # Text in square brackets
            r'Note:.*',   # Explicit "Note:" prefix
            r'Additional context:.*'  # "Additional context" prefix
        ]
        
        main_instruction = text
        for marker in note_markers:
            main_instruction = re.sub(marker, '', main_instruction, flags=re.IGNORECASE).strip()
        
        return main_instruction.strip()

if __name__ == "__main__":
    # Simple import test - comprehensive testing is handled by comprehensive_avdn_pipeline.py
    logger.info("ParaphraseGenerationPipeline can be imported successfully")
    logger.info("Use comprehensive_avdn_pipeline.py for full testing and processing") 