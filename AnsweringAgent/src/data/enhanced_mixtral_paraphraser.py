#!/usr/bin/env python3
# Enhanced Mixtral Paraphraser for UAV Navigation Instructions
# Focus: High-quality, fewer paraphrases with natural language diversity

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMixtralParaphraser:
    """
    Enhanced Mixtral-based paraphraser for UAV navigation instructions.
    Focus: High-quality, natural diversity with spatial accuracy preservation.
    """
    
    def __init__(self, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Spatial terms to preserve (from AVDN dataset analysis)
        self.spatial_terms = {
            'landmarks': ['building', 'road', 'parking', 'field', 'house', 'highway', 'structure'],
            'directions': ['turn', 'forward', 'right', 'left', 'north', 'south', 'east', 'west', 'straight'],
            'spatial_relations': ['over', 'near', 'in front of', 'next to', 'around', 'through', 'behind'],
            'clock_directions': ['o\'clock', 'clock', '1 o\'clock', '2 o\'clock', '3 o\'clock', '4 o\'clock', 
                               '5 o\'clock', '6 o\'clock', '7 o\'clock', '8 o\'clock', '9 o\'clock', 
                               '10 o\'clock', '11 o\'clock', '12 o\'clock']
        }
        
        logger.info(f"Initializing Enhanced Mixtral Paraphraser on {self.device}")
        
    def load_model(self):
        """Load Mixtral model and tokenizer."""
        try:
            logger.info("Loading Mixtral tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("Loading Mixtral model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def extract_spatial_terms(self, instruction: str) -> Dict[str, List[str]]:
        """Extract spatial terms from instruction for preservation."""
        instruction_lower = instruction.lower()
        extracted_terms = {}
        
        for category, terms in self.spatial_terms.items():
            found_terms = []
            for term in terms:
                if term in instruction_lower:
                    found_terms.append(term)
            if found_terms:
                extracted_terms[category] = found_terms
                
        return extracted_terms
    
    def create_positive_prompt(self, instruction: str) -> str:
        """
        Create minimal but strategic prompt for positive paraphrases.
        Focus: Natural diversity while preserving spatial accuracy.
        """
        
        prompt = f"""<s>[INST] You are an expert in UAV navigation instructions. Paraphrase the following instruction naturally while maintaining spatial accuracy and key navigation terms.

Original instruction: "{instruction}"

Generate 2 high-quality paraphrases that:
- Maintain the same spatial meaning and navigation intent
- Use natural language variation in word choice and sentence structure
- Preserve key spatial terms (landmarks, directions, clock references)
- Sound like natural human navigation instructions

Provide only the paraphrases, one per line: [/INST]"""
        
        return prompt
    
    def create_negative_prompt(self, instruction: str) -> str:
        """
        Create comprehensive prompt for negative paraphrases.
        Focus: Strategic term changes while maintaining realistic navigation language.
        """
        spatial_terms = self.extract_spatial_terms(instruction)
        
        # Build term substitution guidance based on AVDN frequency analysis
        substitution_guidance = ""
        if 'landmarks' in spatial_terms:
            substitution_guidance += "- Change landmarks: building↔structure↔house, road↔highway↔parking\n"
        if 'directions' in spatial_terms:
            substitution_guidance += "- Change directions: turn↔go↔move, left↔right, north↔south↔east↔west\n"
        if 'clock_directions' in spatial_terms:
            substitution_guidance += "- Change clock directions: shift by 2-4 hours (e.g., 3 o'clock→6 o'clock)\n"
        
        prompt = f"""<s>[INST] You are an expert in UAV navigation instructions. Create a negative paraphrase that changes key spatial terms while maintaining realistic navigation language.

Original instruction: "{instruction}"

Generate 1 negative paraphrase that:
{substitution_guidance}- Changes spatial meaning (wrong direction, landmark, or location)
- Maintains realistic UAV navigation vocabulary and sentence structure
- Uses natural language (avoid robotic or template-like phrasing)
- Creates a plausible but incorrect navigation instruction

Provide only the negative paraphrase: [/INST]"""
        
        return prompt
    
    def generate_paraphrases(self, instruction: str, num_positives: int = 2, num_negatives: int = 1) -> Dict[str, List[str]]:
        """
        Generate high-quality paraphrases with natural diversity.
        Returns: {'positives': [...], 'negatives': [...]}
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Call load_model() first.")
            return {'positives': [], 'negatives': []}
        
        results = {'positives': [], 'negatives': []}
        
        try:
            # Generate positive paraphrases
            logger.info("Generating positive paraphrases...")
            positive_prompt = self.create_positive_prompt(instruction)
            positive_response = self._generate_response(positive_prompt)
            positive_paraphrases = self._parse_paraphrases(positive_response, num_positives)
            results['positives'] = positive_paraphrases
            
            # Generate negative paraphrases
            logger.info("Generating negative paraphrases...")
            negative_prompt = self.create_negative_prompt(instruction)
            negative_response = self._generate_response(negative_prompt)
            negative_paraphrases = self._parse_paraphrases(negative_response, num_negatives)
            results['negatives'] = negative_paraphrases
            
            logger.info(f"Generated {len(results['positives'])} positives and {len(results['negatives'])} negatives")
            return results
            
        except Exception as e:
            logger.error(f"Error generating paraphrases: {e}")
            return {'positives': [], 'negatives': []}
    
    def _generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response from Mixtral model."""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,  # Balanced creativity
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # More robust extraction of generated text
            # Look for the instruction part after the prompt
            if prompt in response:
                generated_text = response.split(prompt, 1)[1].strip()
            else:
                # Fallback: try to find the generated part by looking for common patterns
                generated_text = response.strip()
                # Remove common prompt artifacts
                for artifact in ['[/INST]', '[INST]', '<s>', '</s>']:
                    generated_text = generated_text.replace(artifact, '').strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return ""
    
    def _parse_paraphrases(self, response: str, expected_count: int) -> List[str]:
        """Parse generated paraphrases from response."""
        if not response:
            return []
        
        # Split by lines and clean up
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Extract paraphrases (remove numbering, quotes, etc.)
        paraphrases = []
        for line in lines:
            # Remove common prefixes
            cleaned = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. ", "2. " etc.
            cleaned = re.sub(r'^["\']|["\']$', '', cleaned)  # Remove quotes
            cleaned = cleaned.strip()
            
            if cleaned and len(cleaned) > 10:  # Minimum meaningful length
                paraphrases.append(cleaned)
        
        # Return expected number or all if fewer
        return paraphrases[:expected_count]
    
    def validate_spatial_accuracy(self, original: str, paraphrase: str) -> Dict[str, bool]:
        """
        Basic validation of spatial accuracy.
        Returns: {'spatial_terms_preserved': bool, 'meaning_changed': bool}
        """
        original_terms = self.extract_spatial_terms(original)
        paraphrase_terms = self.extract_spatial_terms(paraphrase)
        
        # Check if key spatial terms are preserved (for positives) or changed (for negatives)
        spatial_terms_preserved = len(original_terms) > 0 and len(paraphrase_terms) > 0
        
        # Simple meaning change detection (for negatives)
        meaning_changed = len(original_terms) != len(paraphrase_terms) or \
                        any(cat not in paraphrase_terms for cat in original_terms)
        
        return {
            'spatial_terms_preserved': spatial_terms_preserved,
            'meaning_changed': meaning_changed
        }

# Test function
def test_enhanced_paraphraser():
    """Test the enhanced paraphraser with real AVDN examples."""
    
    # Real AVDN examples from dataset analysis
    test_instructions = [
        "Turn right and fly over the white building at 3 o'clock",
        "Go straight ahead towards the gray road near the parking area",
        "Navigate to the brown house at 6 o'clock position",
        "Fly north over the highway and turn left at the intersection"
    ]
    
    paraphraser = EnhancedMixtralParaphraser()
    
    print("=== Enhanced Mixtral Paraphraser Test ===")
    print(f"Device: {paraphraser.device}")
    
    # Test 1: Model loading
    print("\n1. Testing model loading...")
    if paraphraser.load_model():
        print("✅ Model loaded successfully")
    else:
        print("❌ Model loading failed")
        return
    
    # Test 2: Spatial term extraction
    print("\n2. Testing spatial term extraction...")
    for instruction in test_instructions[:2]:
        terms = paraphraser.extract_spatial_terms(instruction)
        print(f"Instruction: {instruction}")
        print(f"Extracted terms: {terms}")
        print()
    
    # Test 3: Prompt generation
    print("\n3. Testing prompt generation...")
    test_instruction = test_instructions[0]
    positive_prompt = paraphraser.create_positive_prompt(test_instruction)
    negative_prompt = paraphraser.create_negative_prompt(test_instruction)
    
    print("Positive prompt preview:")
    print(positive_prompt)
    print("\nNegative prompt preview:")
    print(negative_prompt)
    
    # Test 4: Full paraphrase generation (if model loaded)
    print("\n4. Testing full paraphrase generation...")
    results = paraphraser.generate_paraphrases(test_instruction, num_positives=2, num_negatives=1)
    
    print(f"Original: {test_instruction}")
    print(f"Positives: {results['positives']}")
    print(f"Negatives: {results['negatives']}")
    
    # Test 5: Validation
    print("\n5. Testing validation...")
    if results['positives']:
        validation = paraphraser.validate_spatial_accuracy(test_instruction, results['positives'][0])
        print(f"Validation results: {validation}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_enhanced_paraphraser() 