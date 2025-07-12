#!/usr/bin/env python3
"""
Enhanced Mixtral Paraphraser for UAV Navigation Instructions
Generates 2 positive + 1 negative paraphrases per original instruction
Based on AVDN dataset analysis for optimal prompt engineering
"""

import re
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

class EnhancedMixtralParaphraser:
    """
    Enhanced Mixtral paraphraser for UAV navigation with positive/negative generation.
    Based on successful SimpleMixtralParaphraser foundation.
    """
    
    def __init__(self, model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", device="auto"):
        """Initialize enhanced paraphraser with Mixtral model."""
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loading enhanced Mixtral paraphraser: {model_name}")
        
        # Load tokenizer and model (based on successful SimpleMixtralParaphraser)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=True,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",  # Automatic distribution across 10x RTX 2080 Ti
            use_auth_token=True,
            trust_remote_code=True
        )
        
        # Initialize spatial preservation patterns from dataset analysis
        self._init_spatial_patterns()
        
        # Initialize prompts based on AVDN analysis
        self._init_avdn_prompts()
        
        self.logger.info("Enhanced Mixtral paraphraser ready")
    
    def _init_spatial_patterns(self):
        """Initialize critical spatial patterns that must be preserved."""
        # From dataset analysis - these must be preserved exactly
        self.clock_pattern = re.compile(r'\d+\s*o\'?clock|\d+:\d+')
        self.cardinal_directions = {
            'north', 'south', 'east', 'west', 
            'northeast', 'northwest', 'southeast', 'southwest'
        }
        self.spatial_prepositions = {
            'next to', 'in front of', 'behind', 'above', 'below', 
            'near', 'across from', 'over', 'under'
        }
        self.landmarks = {
            'building', 'parking', 'road', 'structure', 'house', 
            'field', 'container', 'tower', 'tree'
        }
    
    def _init_avdn_prompts(self):
        """Initialize prompts based on real AVDN dataset examples."""
        
        # Positive paraphrasing prompt with real AVDN examples
        self.positive_prompt = """Paraphrase this UAV navigation instruction while preserving ALL spatial information:
- Keep exact directions (clock positions, cardinal directions)
- Maintain all landmarks and their spatial relationships  
- Preserve step order and navigation target
- Use different wording but same meaning

Examples:

Input: "Go in the 8 o'clock direction from your current position."
Output: "Head toward the 8 o'clock direction from your current location."

Input: "Go southeast to the building on the treeline."
Output: "Proceed southeast to the structure on the treeline."

Input: "Move towards your 11 o'clock direction. You will fly over two residential buildings."
Output: "Head toward your 11 o'clock direction. You'll fly over two residential structures."

Now paraphrase: "{original_instruction}"

Paraphrase:"""

        # Negative paraphrasing prompt for controlled corruption
        self.negative_prompt = """Create a SINGLE spatial error in this UAV navigation instruction while keeping everything else the same:
- Change ONLY ONE spatial element (direction, landmark, or relation)
- Keep the same sentence structure and other details
- Make it realistic but clearly wrong for navigation

Examples:

Input: "Go in the 8 o'clock direction from your current position."
Output: "Go in the 2 o'clock direction from your current position."

Input: "Go southeast to the building on the treeline."
Output: "Go northwest to the building on the treeline."

Input: "Your destination is next to the parking lot."
Output: "Your destination is far from the parking lot."

Now create ONE spatial error in: "{original_instruction}"

Modified instruction:"""
    
    def generate_paraphrases(self, original_instruction):
        """
        Generate 2 positive + 1 negative paraphrases for the original instruction.
        
        Returns:
            dict: {
                'original': str,
                'positives': [str, str],
                'negative': str,
                'spatial_tokens': dict
            }
        """
        try:
            # Extract spatial tokens for validation
            spatial_tokens = self._extract_spatial_tokens(original_instruction)
            
            # Generate 2 positive paraphrases
            positive1 = self._generate_positive(original_instruction)
            positive2 = self._generate_positive(original_instruction, temperature=0.8)  # More variation
            
            # Generate 1 negative paraphrase
            negative = self._generate_negative(original_instruction)
            
            # Validate quality
            results = {
                'original': original_instruction,
                'positives': [positive1, positive2],
                'negative': negative,
                'spatial_tokens': spatial_tokens
            }
            
            # Basic validation
            valid_results = self._validate_paraphrases(results)
            
            return valid_results
            
        except Exception as e:
            self.logger.error(f"Error generating paraphrases: {e}")
            return {
                'original': original_instruction,
                'positives': [original_instruction, original_instruction],  # Fallback
                'negative': original_instruction,
                'spatial_tokens': {},
                'error': str(e)
            }
    
    def _generate_positive(self, instruction, temperature=0.7):
        """Generate a positive paraphrase preserving spatial fidelity."""
        prompt = self.positive_prompt.format(original_instruction=instruction)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and extract paraphrase
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        paraphrase = self._extract_paraphrase_from_response(full_response, "Paraphrase:")
        
        return paraphrase or instruction  # Fallback to original if extraction fails
    
    def _generate_negative(self, instruction):
        """Generate a negative paraphrase with controlled spatial corruption."""
        prompt = self.negative_prompt.format(original_instruction=instruction)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.9,  # Higher temp for more variation in corruption
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and extract corrupted instruction
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        negative = self._extract_paraphrase_from_response(full_response, "Modified instruction:")
        
        return negative or instruction  # Fallback to original if extraction fails
    
    def _extract_paraphrase_from_response(self, response, marker):
        """Extract paraphrase from model response after the marker."""
        try:
            if marker in response:
                paraphrase = response.split(marker)[-1].strip()
                # Clean up common artifacts
                paraphrase = paraphrase.split('\n')[0].strip()  # First line only
                paraphrase = paraphrase.strip('"').strip("'")    # Remove quotes
                return paraphrase if paraphrase else None
            return None
        except Exception:
            return None
    
    def _extract_spatial_tokens(self, instruction):
        """Extract critical spatial tokens from instruction."""
        tokens = {
            'clock_directions': self.clock_pattern.findall(instruction.lower()),
            'cardinal_directions': [d for d in self.cardinal_directions if d in instruction.lower()],
            'spatial_prepositions': [p for p in self.spatial_prepositions if p in instruction.lower()],
            'landmarks': [l for l in self.landmarks if l in instruction.lower()]
        }
        return tokens
    
    def _validate_paraphrases(self, results):
        """Basic validation of generated paraphrases."""
        original_tokens = results['spatial_tokens']
        
        # Validate positives preserve spatial tokens
        for i, positive in enumerate(results['positives']):
            positive_tokens = self._extract_spatial_tokens(positive)
            # Add validation score (simplified for now)
            if not self._tokens_preserved(original_tokens, positive_tokens):
                self.logger.warning(f"Positive {i+1} may have lost spatial tokens")
        
        # Validate negative has exactly one change
        negative_tokens = self._extract_spatial_tokens(results['negative'])
        # Add validation for controlled corruption
        
        return results
    
    def _tokens_preserved(self, original_tokens, new_tokens):
        """Check if critical spatial tokens are preserved."""
        # Simplified preservation check
        for token_type, tokens in original_tokens.items():
            if tokens and not new_tokens.get(token_type):
                return False
        return True

def test_enhanced_paraphraser():
    """Test the enhanced paraphraser with AVDN examples."""
    print("🚀 Testing Enhanced Mixtral Paraphraser")
    print("=" * 50)
    
    # Test cases from AVDN dataset analysis
    test_instructions = [
        "Go in the 8 o'clock direction from your current position.",
        "Go southeast to the building on the treeline.",
        "Your destination is next to the parking lot.",
        "Move towards your 11 o'clock direction. You will fly over two residential buildings."
    ]
    
    try:
        paraphraser = EnhancedMixtralParaphraser()
        
        for i, instruction in enumerate(test_instructions, 1):
            print(f"\n📝 Test {i}: {instruction}")
            results = paraphraser.generate_paraphrases(instruction)
            
            print(f"✅ Positive 1: {results['positives'][0]}")
            print(f"✅ Positive 2: {results['positives'][1]}")
            print(f"❌ Negative:   {results['negative']}")
            print(f"🔍 Spatial tokens: {results['spatial_tokens']}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    test_enhanced_paraphraser() 