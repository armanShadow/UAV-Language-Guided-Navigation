#!/usr/bin/env python3
# Enhanced Mixtral Paraphraser for UAV Navigation Instructions
# Focus: High-quality, fewer paraphrases with natural language diversity

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import json
from typing import List, Dict, Tuple, Optional
import logging
import random
from pathlib import Path

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
    
    def load_random_avdn_examples(self, num_examples: int = 4) -> List[str]:
        """
        Load random examples from the processed AVDN dataset.
        Returns a list of navigation instructions.
        """
        # Possible dataset paths (prioritize processed data)
        dataset_paths = [
            "processed_data/train_data.json",
            "src/data/processed_data/train_data.json",
            "AnsweringAgent/src/data/processed_data/train_data.json",
            "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json",
            "../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json"
        ]
        
        for path in dataset_paths:
            if Path(path).exists():
                logger.info(f"📂 Loading dataset from: {path}")
                try:
                    with open(path, 'r') as f:
                        episodes = json.load(f)
                    
                    # Extract all answers from dialogs
                    all_instructions = []
                    for episode in episodes:
                        dialogs = episode.get('dialogs', [])
                        for dialog in dialogs:
                            if dialog and dialog.get('answer'):
                                answer = dialog['answer'].strip()
                                if answer and len(answer.split()) >= 3:
                                    all_instructions.append(answer)
                    
                    # Get random examples
                    if len(all_instructions) >= num_examples:
                        # Use current time for true randomness
                        random.seed()  # Remove fixed seed for true randomness
                        random_examples = random.sample(all_instructions, num_examples)
                        logger.info(f"📊 Selected {num_examples} random examples from {len(all_instructions)} total")
                        return random_examples
                    else:
                        logger.warning(f"Only {len(all_instructions)} examples available, returning all")
                        return all_instructions
                        
                except Exception as e:
                    logger.error(f"Error loading dataset from {path}: {e}")
                    continue
        
        logger.error("❌ Could not find AVDN dataset. Using fallback examples.")
        # Fallback to original hardcoded examples
        return [
            "Turn right and fly over the white building at 3 o'clock",
            "Go straight ahead towards the gray road near the parking area",
            "Navigate to the brown house at 6 o'clock position",
            "Fly north over the highway and turn left at the intersection"
        ]
        
    def load_model(self):
        """Load Mixtral model and tokenizer."""
        try:
            logger.info("Loading Mixtral tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("Loading Mixtral model...")
            
            # Configure quantization for better memory efficiency
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
    
    def create_combined_prompt(self, instruction: str) -> str:
        """
        Create unified prompt for both positive and negative paraphrases.
        Focus: Cohesive generation with better contrast understanding.
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
        
        prompt = f"""<s>[INST] You are an expert in UAV navigation instructions. Generate paraphrases for this instruction:

Original instruction: "{instruction}"

Generate:
1. 2 positive paraphrases that maintain the same spatial meaning and navigation intent
2. 1 negative paraphrase that changes spatial meaning strategically

For positives:
- Use natural language variation in word choice and sentence structure
- Preserve key spatial terms (landmarks, directions, clock references)
- Sound like natural human navigation instructions

For negative:
{substitution_guidance}- Make EXACTLY TWO correlated strategic changes (e.g., direction + landmark, or clock + landmark)
- Examples: "right + white building" → "left + gray structure", "3 o'clock + building" → "9 o'clock + structure"
- Ensure changes are LOGICALLY CONSISTENT and realistic for UAV navigation
- Use natural language (avoid robotic or template-like phrasing)
- Create a plausible but incorrect navigation instruction
- Focus on spatial accuracy changes that would lead to different navigation outcomes
- Ensure both changes work together coherently (e.g., "turn left at the gray building" not "turn left at the blue sky")

Provide only the paraphrases, no explanations: [/INST]"""
        
        return prompt
    
    def generate_paraphrases(self, instruction: str, num_positives: int = 2, num_negatives: int = 1) -> Dict[str, List[str]]:
        """
        Generate high-quality paraphrases with natural diversity using combined prompt.
        Returns: {'positives': [...], 'negatives': [...]}
        """
        if not self.model or not self.tokenizer:
            logger.error("Model not loaded. Call load_model() first.")
            return {'positives': [], 'negatives': []}
        
        results = {'positives': [], 'negatives': []}
        
        try:
            # Generate all paraphrases using combined prompt
            logger.info("Generating paraphrases with combined prompt...")
            combined_prompt = self.create_combined_prompt(instruction)
            response = self._generate_response(combined_prompt)
            
            # Parse all paraphrases from response
            all_paraphrases = self._parse_paraphrases(response, num_positives + num_negatives)
            
            # Split into positives and negatives
            if len(all_paraphrases) >= num_positives + num_negatives:
                results['positives'] = all_paraphrases[:num_positives]
                results['negatives'] = all_paraphrases[num_positives:num_positives + num_negatives]
            else:
                # Fallback: use all available paraphrases
                results['positives'] = all_paraphrases[:len(all_paraphrases)//2]
                results['negatives'] = all_paraphrases[len(all_paraphrases)//2:]
            
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
            
            # Extract only the generated tokens (not the input tokens)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
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
        
        # Extract paraphrases (remove numbering, quotes, labels, etc.)
        paraphrases = []
        for line in lines:
            # Remove common prefixes and labels
            cleaned = re.sub(r'^\d+\.\s*', '', line)  # Remove "1. ", "2. " etc.
            cleaned = re.sub(r'^["\']|["\']$', '', cleaned)  # Remove quotes
            cleaned = re.sub(r'^(Positive|Negative)\s+Paraphrases?:\s*', '', cleaned, flags=re.IGNORECASE)  # Remove labels
            cleaned = re.sub(r'^(Positives?|Negatives?):\s*', '', cleaned, flags=re.IGNORECASE)  # Remove section headers
            cleaned = cleaned.strip()
            
            # Only include actual instruction text (not labels or empty lines)
            if cleaned and len(cleaned) > 10 and not cleaned.lower().startswith(('positive', 'negative', 'positives', 'negatives')):
                paraphrases.append(cleaned)
        
        # Return expected number or all if fewer
        return paraphrases[:expected_count]
    
    def validate_spatial_accuracy(self, original: str, paraphrase: str, is_positive: bool = True) -> Dict[str, bool]:
        """
        Enhanced validation of spatial accuracy with comprehensive logging and precise detection.
        Returns: {'spatial_terms_preserved': bool, 'meaning_changed': bool, 'validation_type': str}
        """
        # Configure logging for detailed tracking
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        # Detailed logging handler
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        def log_validation_details(message):
            """Log detailed validation information."""
            logger.debug(f"[VALIDATION] {message}")
        
        # Comprehensive spatial element extraction
        def extract_comprehensive_elements(text):
            """
            Extract a wide range of spatial elements with detailed categorization.
            """
            text_lower = text.lower()
            return {
                'clock_directions': re.findall(r'(\d+)\s*o\'?clock', text_lower),
                'cardinal_directions': re.findall(r'\b(north|south|east|west)\b', text_lower),
                'relative_directions': re.findall(r'\b(left|right|forward|backward|up|down)\b', text_lower),
                'landmarks': re.findall(r'\b(building|road|parking|field|house|highway|structure|roof|rooftop|lot|aircraft|airplane|hangar)\b', text_lower),
                'colors': re.findall(r'\b(white|gray|grey|brown|tan|tawny|drab|blue|green|red|black|light|dark)\b', text_lower),
                'location_descriptors': re.findall(r'\b(middle|center|edge|side|north|south|east|west)\b', text_lower),
                'movement_verbs': re.findall(r'\b(turn|go|move|head|steer|navigate|cross|fly)\b', text_lower)
            }
        
        def compute_spatial_similarity(orig_elements, para_elements):
            """
            Compute a similarity score based on spatial elements.
            Lower score indicates more significant changes.
            """
            similarity_score = 0
            total_categories = 0
            
            # Check each category
            for category in ['clock_directions', 'cardinal_directions', 'relative_directions', 
                             'landmarks', 'colors', 'location_descriptors', 'movement_verbs']:
                orig_set = set(orig_elements.get(category, []))
                para_set = set(para_elements.get(category, []))
                
                # Compute Jaccard similarity
                if orig_set or para_set:
                    total_categories += 1
                    intersection = len(orig_set.intersection(para_set))
                    union = len(orig_set.union(para_set))
                    similarity_score += intersection / union if union > 0 else 0
            
            # Normalize similarity score
            normalized_similarity = similarity_score / total_categories if total_categories > 0 else 0
            
            log_validation_details(f"Similarity Score: {normalized_similarity:.2f}")
            return normalized_similarity
        
        def detect_meaning_change(original, paraphrase):
            """
            Detect spatial meaning change with comprehensive analysis.
            """
            orig_elements = extract_comprehensive_elements(original)
            para_elements = extract_comprehensive_elements(paraphrase)
            
            # Log extracted elements for debugging
            log_validation_details(f"Original Elements: {orig_elements}")
            log_validation_details(f"Paraphrase Elements: {para_elements}")
            
            # Compute similarity
            similarity = compute_spatial_similarity(orig_elements, para_elements)
            
            # Determine meaning change based on similarity threshold
            # Positive paraphrases should have high similarity
            # Negative paraphrases should have low similarity
            if is_positive:
                meaning_changed = similarity < 0.6  # Allow some variation
                log_validation_details(f"Positive Paraphrase - Meaning Changed: {meaning_changed}")
            else:
                meaning_changed = similarity < 0.3  # Require significant change
                log_validation_details(f"Negative Paraphrase - Meaning Changed: {meaning_changed}")
            
            return meaning_changed
        
        def check_term_preservation(original, paraphrase):
            """
            Check if key spatial terms are preserved.
            More flexible and comprehensive approach.
            """
            orig_elements = extract_comprehensive_elements(original)
            para_elements = extract_comprehensive_elements(paraphrase)
            
            preserved_categories = 0
            total_categories = len(orig_elements)
            
            for category, orig_terms in orig_elements.items():
                para_terms = para_elements.get(category, [])
                
                # Check for any overlap or semantic similarity
                if orig_terms and para_terms:
                    overlap = len(set(orig_terms).intersection(set(para_terms)))
                    if overlap > 0:
                        preserved_categories += 1
            
            # Require preservation of most categories
            terms_preserved = preserved_categories >= max(1, total_categories - 1)
            
            log_validation_details(f"Term Preservation - Preserved: {terms_preserved}")
            log_validation_details(f"Preserved Categories: {preserved_categories}/{total_categories}")
            
            return terms_preserved
        
        # Main validation logic
        spatial_terms_preserved = check_term_preservation(original, paraphrase)
        meaning_changed = detect_meaning_change(original, paraphrase)
        
        # Remove the logging handler to prevent duplicate logs
        logger.removeHandler(handler)
        
        return {
            'spatial_terms_preserved': spatial_terms_preserved,
            'meaning_changed': meaning_changed,
            'validation_type': 'positive' if is_positive else 'negative'
        }

# Test function
def test_enhanced_paraphraser():
    """Test the enhanced paraphraser with real AVDN examples."""
    
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
    
    # Test 2: Load random AVDN examples
    print("\n2. Loading random AVDN examples...")
    avdn_examples = paraphraser.load_random_avdn_examples(num_examples=4)
    print(f"✅ Loaded {len(avdn_examples)} examples from AVDN dataset")
    for i, example in enumerate(avdn_examples, 1):
        print(f"   {i}. {example}")
    
    # Test 3: Spatial term extraction
    print("\n3. Testing spatial term extraction...")
    for instruction in avdn_examples[:2]:
        terms = paraphraser.extract_spatial_terms(instruction)
        print(f"Instruction: {instruction}")
        print(f"Extracted terms: {terms}")
        print()
    
    # Test 4: Combined prompt generation
    print("\n4. Testing combined prompt generation...")
    test_instruction = avdn_examples[0]
    combined_prompt = paraphraser.create_combined_prompt(test_instruction)
    
    print("Combined prompt preview:")
    print(combined_prompt)
    
    # Test 5: Full paraphrase generation for all examples
    print("\n5. Testing full paraphrase generation for all examples...")
    
    for i, instruction in enumerate(avdn_examples, 1):
        print(f"\n--- Example {i} ---")
        print(f"Original: {instruction}")
        
        # Generate paraphrases
        results = paraphraser.generate_paraphrases(instruction, num_positives=2, num_negatives=1)
        
        print(f"Positives:")
        for j, positive in enumerate(results['positives'], 1):
            print(f"  {j}. {positive}")
        
        print(f"Negatives:")
        for j, negative in enumerate(results['negatives'], 1):
            print(f"  {j}. {negative}")
        
        # Enhanced validation for positives and negatives
        if results['positives']:
            positive_validation = paraphraser.validate_spatial_accuracy(instruction, results['positives'][0], is_positive=True)
            print(f"Positive validation: {positive_validation}")
        
        if results['negatives']:
            negative_validation = paraphraser.validate_spatial_accuracy(instruction, results['negatives'][0], is_positive=False)
            print(f"Negative validation: {negative_validation}")
        
        print("-" * 50)
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_enhanced_paraphraser() 