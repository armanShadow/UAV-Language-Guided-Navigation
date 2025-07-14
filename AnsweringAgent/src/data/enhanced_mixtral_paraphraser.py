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
        """Extract spatial terms from instruction for preservation with flexible matching."""
        instruction_lower = instruction.lower()
        extracted_terms = {}
        
        # Enhanced matching with regex and word boundary checks
        for category, terms in self.spatial_terms.items():
            found_terms = []
            for term in terms:
                # Use word boundary regex to prevent partial matches
                pattern = r'\b' + re.escape(term) + r'\b'
                if re.search(pattern, instruction_lower):
                    found_terms.append(term)
                
                # Handle plural and slight variations
                if term.endswith('ing'):
                    # Check for -ing forms (e.g., "building" → "buildings")
                    plural_pattern = r'\b' + re.escape(term) + r's?\b'
                    if re.search(plural_pattern, instruction_lower):
                        found_terms.append(term)
            
            if found_terms:
                extracted_terms[category] = list(set(found_terms))  # Remove duplicates
        
        return extracted_terms
    
    def create_combined_prompt(self, instruction: str) -> str:
        """
        Create unified prompt for both positive and negative paraphrases.
        Focus: Cohesive generation with better contrast understanding and note handling.
        """
        # Separate main instruction from any notes
        def extract_main_instruction_and_notes(text):
            # Look for common note indicators
            note_markers = [
                r'\(.*?\)',   # Text in parentheses
                r'\[.*?\]',   # Text in square brackets
                r'Note:.*',   # Explicit "Note:" prefix
                r'Additional context:.*'  # "Additional context" prefix
            ]
            
            # Remove note markers and extract main instruction
            main_instruction = text
            notes = []
            
            for marker in note_markers:
                note_matches = re.findall(marker, text, re.IGNORECASE)
                if note_matches:
                    notes.extend(note_matches)
                    main_instruction = re.sub(marker, '', main_instruction, flags=re.IGNORECASE).strip()
            
            return main_instruction.strip(), notes
        
        # Extract main instruction and notes
        main_instruction, notes = extract_main_instruction_and_notes(instruction)
        
        spatial_terms = self.extract_spatial_terms(main_instruction)
        
        # Build term substitution guidance based on AVDN frequency analysis
        substitution_guidance = ""
        if 'landmarks' in spatial_terms:
            substitution_guidance += "- Change landmarks: building↔structure↔house, road↔highway↔parking\n"
        if 'directions' in spatial_terms:
            substitution_guidance += "- Change directions: turn↔go↔move, left↔right, north↔south↔east↔west\n"
        if 'clock_directions' in spatial_terms:
            substitution_guidance += "- Change clock directions: shift by 2-4 hours (e.g., 3 o'clock→6 o'clock)\n"
        
        prompt = f"""<s>[INST] You are an expert in UAV navigation instructions. Generate paraphrases for this instruction:

Original instruction: "{main_instruction}"

CRITICAL NOTE GENERATION RULES:
1. DO NOT generate explanatory notes or additional context
2. FOCUS SOLELY on generating the paraphrased navigation instruction
5. Preserve the core spatial and navigational meaning

Generate:
1. 2 positive paraphrases that maintain the same spatial meaning and navigation intent
2. 1 negative paraphrase that changes spatial meaning strategically

For positives:
- Use natural language variation in word choice and sentence structure
- Preserve key spatial terms (landmarks, directions, EXACT clock references)
- Sound like natural human navigation instructions
- IMPORTANT: NO ADDITIONAL NOTES OR CONTEXT ALLOWED

For negative:
{substitution_guidance}- Make correlated strategic changes (e.g., direction + landmark, or clock + landmark)
- Examples: "right + white building" → "left + gray structure", "3 o'clock + building" → "9 o'clock + farm"
- Ensure changes are LOGICALLY CONSISTENT and realistic for UAV navigation
- Use natural language (avoid robotic or template-like phrasing)
- Create a plausible but incorrect navigation instruction
- Focus on spatial accuracy changes that would lead to different navigation outcomes
- Ensure both changes work together coherently (e.g., "turn left at the gray building" not "turn left at the blue sky")

Provide ONLY the paraphrases, NO EXPLANATIONS, NO NOTES: [/INST]"""
        
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
            
            # Extract main instruction and notes
            def extract_main_instruction_and_notes(text):
                # Look for common note indicators
                note_markers = [
                    r'\(.*?\)',   # Text in parentheses
                    r'\[.*?\]',   # Text in square brackets
                    r'Note:.*',   # Explicit "Note:" prefix
                    r'Additional context:.*'  # "Additional context" prefix
                ]
                
                # Remove note markers and extract main instruction
                main_instruction = text
                notes = []
                
                for marker in note_markers:
                    note_matches = re.findall(marker, text, re.IGNORECASE)
                    if note_matches:
                        notes.extend(note_matches)
                        main_instruction = re.sub(marker, '', main_instruction, flags=re.IGNORECASE).strip()
                
                return main_instruction.strip(), notes
            
            main_instruction, notes = extract_main_instruction_and_notes(instruction)
            
            combined_prompt = self.create_combined_prompt(instruction)
            response = self._generate_response(combined_prompt)
            
            # Parse all paraphrases from response
            all_paraphrases = self._parse_paraphrases(response, num_positives + num_negatives, notes)
            
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
    
    def _parse_paraphrases(self, response: str, expected_count: int, original_notes: List[str] = None) -> List[str]:
        """
        Parse generated paraphrases from response with strict note removal.
        
        Args:
            response (str): Generated text from the model
            expected_count (int): Number of paraphrases to extract
            original_notes (List[str], optional): Original notes (not used in this version)
        
        Returns:
            List[str]: Extracted paraphrases with all notes removed
        """
        if not response:
            return []
        
        # Split by lines and clean up
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Regex patterns to remove various types of notes or explanatory text
        note_removal_patterns = [
            r'\(.*?\)',   # Remove text in parentheses
            r'\[.*?\]',   # Remove text in square brackets
            r'Note:.*',   # Remove lines starting with "Note:"
            r'Additional context:.*',  # Remove lines with additional context
            r'^[0-9]+\.?\s*',  # Remove numbering
            r'^(Positive|Negative)\s*Paraphrases?:',  # Remove paraphrase labels
            r'^(Positives?|Negatives?):',  # Remove section headers
        ]
        
        # Extract paraphrases (remove numbering, quotes, labels, notes, etc.)
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
        
        # Return expected number or all if fewer
        return paraphrases[:expected_count]
    
    def calibrate_weights(self, feature_similarities):
        """
        Dynamically adjust weights based on actual feature preservation.
        Enhanced version with more nuanced weight distribution and dynamic feature handling.
        """
        # Default base weights with prioritized spatial features
        base_weights = {
            'clock_directions': 0.4,   # Highest importance for precise spatial reference
            'cardinal_directions': 0.25,  # Critical for navigation intent
            'landmarks': 0.2,           # Important for spatial context
            'movement_verbs': 0.1,      # Moderate importance for action preservation
            'spatial_relations': 0.05   # Least critical, but still relevant
        }
        
        # Dynamically add any new features with a default low weight
        default_new_feature_weight = 0.05
        
        # Adaptive weight adjustment
        for feature, similarity in feature_similarities.items():
            # Add feature to base_weights if not already present
            if feature not in base_weights:
                base_weights[feature] = default_new_feature_weight
            
            # More aggressive weight scaling
            # Reward high similarity with exponential boost
            base_weights[feature] *= (1 + (similarity ** 2))
        
        # Normalize weights to ensure they sum to 1
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}
        
        return normalized_weights
    
    def validate_spatial_accuracy(self, original: str, paraphrase: str, is_positive: bool = True) -> Dict[str, bool]:
        def clean_text(text):
            """Clean text by removing punctuation and converting to lowercase"""
            return re.sub(r'[^\w\s]', '', text.lower()).strip()
        
        def extract_spatial_features(text):
            """
            Enhanced spatial feature extraction with more comprehensive matching
            """
            # Expanded feature dictionaries with synonyms and related terms
            direction_synonyms = {
                'directions': [
                    r'\d+\s*o\'?clock', r'one\s+o\'?clock', r'two\s+o\'?clock', r'three\s+o\'?clock', 
                    r'four\s+o\'?clock', r'five\s+o\'?clock', r'six\s+o\'?clock', r'seven\s+o\'?clock',
                    r'eight\s+o\'?clock', r'nine\s+o\'?clock', r'ten\s+o\'?clock', r'eleven\s+o\'?clock',
                    r'twelve\s+o\'?clock', 'north', 'south', 'east', 'west', 
                    'northwest', 'northeast', 'southwest', 'southeast',
                    'left', 'right', 'forward', 'ahead', 'straight'
                ],
                'movement_verbs': ['move', 'go', 'turn', 'head', 'fly', 'navigate']
            }
            
            landmark_categories = {
                'landmarks': [
                    'building', 'structure', 'road', 'street', 'highway', 
                    'parking', 'lot', 'area', 'destination', 'target'
                ]
            }
            
            spatial_relation_synonyms = {
                'spatial_relations': [
                    'next to', 'beside', 'near', 'in front of', 
                    'across', 'over', 'through', 'around'
                ]
            }
            
            features = {
                'directions': [],
                'landmarks': [],
                'movement_verbs': [],
                'spatial_relations': []
            }
            
            # Comprehensive feature matching
            for category, synonyms in direction_synonyms.items():
                for syn in synonyms:
                    matches = re.findall(syn, text.lower())
                    if matches:
                        features[category].extend(matches)
            
            for category, synonyms in landmark_categories.items():
                for syn in synonyms:
                    matches = re.findall(r'\b' + syn + r'\b', text.lower())
                    if matches:
                        features[category].extend(matches)
            
            for category, synonyms in spatial_relation_synonyms.items():
                for syn in synonyms:
                    if syn in text.lower():
                        features[category].append(syn)
            
            return features
        
        def compute_similarity(orig_text, para_text):
            """
            Advanced similarity computation with multiple metrics
            Focuses on preserving spatial and navigational semantics
            """
            # Clean texts
            orig_clean = clean_text(orig_text).split()
            para_clean = clean_text(para_text).split()
            
            # Compute word overlap
            common_words = set(orig_clean) & set(para_clean)
            total_words = set(orig_clean) | set(para_clean)
            
            # Overlap metrics
            overlap_ratio = len(common_words) / len(total_words) if total_words else 0
            
            # Length similarity
            length_similarity = 1 - abs(len(orig_clean) - len(para_clean)) / max(len(orig_clean), len(para_clean))
            
            # Semantic keyword preservation with more flexible matching
            semantic_keywords = {
                'navigation': ['destination', 'target', 'goal', 'location'],
                'direction': ['left', 'right', 'forward', 'back', 'ahead', 'return'],
                'movement': ['go', 'move', 'turn', 'head', 'navigate']
            }
            
            keyword_preservation = {}
            for category, keywords in semantic_keywords.items():
                keyword_preservation[category] = any(
                    keyword in orig_clean or keyword in para_clean for keyword in keywords
                )
            
            # Compute overall similarity score with weighted components
            similarity_score = (
                0.4 * overlap_ratio + 
                0.3 * length_similarity + 
                0.3 * sum(keyword_preservation.values()) / len(keyword_preservation)
            )
            
            return {
                'similarity_score': similarity_score,
                'overlap_ratio': overlap_ratio,
                'length_similarity': length_similarity,
                'keyword_preservation': keyword_preservation
            }
        
        def compute_advanced_similarity(orig_features, para_features, similarity_metrics):
            """
            Advanced similarity computation focusing on spatial semantic preservation
            """
            # Directional similarity
            def compute_direction_similarity(orig_dirs, para_dirs):
                # Handle cardinal, compound directions, clock directions, and spatial synonyms
                cardinal_directions = {
                    'north': ['north', 'n'],
                    'south': ['south', 's'],
                    'east': ['east', 'e'],
                    'west': ['west', 'w'],
                    'northwest': ['northwest', 'nw'],
                    'northeast': ['northeast', 'ne'],
                    'southwest': ['southwest', 'sw'],
                    'southeast': ['southeast', 'se']
                }
                
                # Spatial direction synonyms - CRITICAL for semantic equivalence
                direction_synonyms = {
                    'forward': ['forward', 'ahead', 'straight', 'front'],
                    'backward': ['backward', 'back', 'behind'],
                    'left': ['left', 'port'],
                    'right': ['right', 'starboard'],
                    'up': ['up', 'above', 'upward'],
                    'down': ['down', 'below', 'downward']
                }
                
                # Clock direction mappings (both numeric and word forms)
                clock_mappings = {
                    '1': ['1', 'one'],
                    '2': ['2', 'two'],
                    '3': ['3', 'three'],
                    '4': ['4', 'four'],
                    '5': ['5', 'five'],
                    '6': ['6', 'six'],
                    '7': ['7', 'seven'],
                    '8': ['8', 'eight'],
                    '9': ['9', 'nine'],
                    '10': ['10', 'ten'],
                    '11': ['11', 'eleven'],
                    '12': ['12', 'twelve']
                }
                
                def extract_clock_hour(direction_text):
                    """Extract clock hour from direction text"""
                    import re
                    # Match numeric clock (e.g., "5 o'clock")
                    numeric_match = re.search(r'(\d+)\s*o\'?clock', direction_text.lower())
                    if numeric_match:
                        return numeric_match.group(1)
                    
                    # Match word form clock (e.g., "five o'clock")
                    for hour, variants in clock_mappings.items():
                        for variant in variants:
                            if re.search(rf'\b{variant}\s+o\'?clock', direction_text.lower()):
                                return hour
                    return None
                
                # Create sets of direction components
                orig_dir_set = set()
                para_dir_set = set()
                orig_clock_hours = set()
                para_clock_hours = set()
                
                # Process original directions
                for orig_dir in orig_dirs:
                    # Check for cardinal directions
                    for cardinal, variants in cardinal_directions.items():
                        if any(var in orig_dir.lower() for var in variants):
                            orig_dir_set.add(cardinal)
                    
                    # Check for spatial direction synonyms
                    for base_dir, synonyms in direction_synonyms.items():
                        if any(syn in orig_dir.lower() for syn in synonyms):
                            orig_dir_set.add(base_dir)
                    
                    # Check for clock directions
                    clock_hour = extract_clock_hour(orig_dir)
                    if clock_hour:
                        orig_clock_hours.add(clock_hour)
                        orig_dir_set.add(f"clock_{clock_hour}")
                
                # Process paraphrase directions
                for para_dir in para_dirs:
                    # Check for cardinal directions
                    for cardinal, variants in cardinal_directions.items():
                        if any(var in para_dir.lower() for var in variants):
                            para_dir_set.add(cardinal)
                    
                    # Check for spatial direction synonyms
                    for base_dir, synonyms in direction_synonyms.items():
                        if any(syn in para_dir.lower() for syn in synonyms):
                            para_dir_set.add(base_dir)
                    
                    # Check for clock directions
                    clock_hour = extract_clock_hour(para_dir)
                    if clock_hour:
                        para_clock_hours.add(clock_hour)
                        para_dir_set.add(f"clock_{clock_hour}")
                
                # Compute direction similarity
                if not orig_dir_set and not para_dir_set:
                    return 1.0  # No directions in both
                
                if not orig_dir_set or not para_dir_set:
                    return 0.0  # One has directions, other doesn't
                
                common_dirs = orig_dir_set & para_dir_set
                total_dirs = orig_dir_set | para_dir_set
                
                return len(common_dirs) / len(total_dirs)
            
            # Landmark similarity
            def compute_landmark_similarity(orig_landmarks, para_landmarks):
                landmark_synonyms = {
                    'building': ['structure', 'house', 'edifice'],
                    'road': ['street', 'highway', 'path'],
                    'destination': ['target', 'goal', 'endpoint']
                }
                
                # Create sets of landmarks with synonyms
                orig_landmark_set = set()
                para_landmark_set = set()
                
                for landmark in orig_landmarks:
                    orig_landmark_set.add(landmark)
                    orig_landmark_set.update(
                        syn for category, synonyms in landmark_synonyms.items()
                        for syn in synonyms if landmark in category
                    )
                
                for landmark in para_landmarks:
                    para_landmark_set.add(landmark)
                    para_landmark_set.update(
                        syn for category, synonyms in landmark_synonyms.items()
                        for syn in synonyms if landmark in category
                    )
                
                # Compute landmark similarity
                if not orig_landmark_set and not para_landmark_set:
                    return 1.0  # No landmarks in both
                
                common_landmarks = orig_landmark_set & para_landmark_set
                return len(common_landmarks) / max(len(orig_landmark_set), len(para_landmark_set))
            
            # Compute individual similarities
            direction_similarity = compute_direction_similarity(
                orig_features.get('directions', []), 
                para_features.get('directions', [])
            )
            
            landmark_similarity = compute_landmark_similarity(
                orig_features.get('landmarks', []), 
                para_features.get('landmarks', [])
            )
            
            # Combine similarities with original similarity metrics
            combined_similarity = (
                0.4 * similarity_metrics['similarity_score'] +
                0.3 * direction_similarity +
                0.3 * landmark_similarity
            )
            
            return {
                'combined_similarity': combined_similarity,
                'direction_similarity': direction_similarity,
                'landmark_similarity': landmark_similarity
            }
        
        # Extract spatial features for original and paraphrased instructions
        orig_features = extract_spatial_features(original)
        para_features = extract_spatial_features(paraphrase)
        
        # Compute similarity metrics
        similarity_metrics = compute_similarity(original, paraphrase)
        
        # Modify feature preservation and validation
        feature_preservation = {}
        for category in orig_features.keys():
            # Ensure the category exists in para_features
            if category not in para_features or not para_features[category]:
                feature_preservation[category] = False
                continue
                
            # More flexible feature matching
            if category == 'directions':
                # Enhanced direction matching with clock direction equivalence
                preserved = False
                
                def extract_clock_number(direction_text):
                    """Extract clock number from direction text (handles both numeric and word forms)"""
                    import re
                    # Numeric form
                    numeric_match = re.search(r'(\d+)\s*o\'?clock', direction_text.lower())
                    if numeric_match:
                        return int(numeric_match.group(1))
                    
                    # Word form mapping
                    word_to_num = {
                        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                        'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
                    }
                    for word, num in word_to_num.items():
                        if re.search(rf'\b{word}\s+o\'?clock', direction_text.lower()):
                            return num
                    return None
                
                # Direction synonym mappings for feature preservation
                direction_synonyms_feature = {
                    'forward': ['forward', 'ahead', 'straight', 'front'],
                    'backward': ['backward', 'back', 'behind'],
                    'left': ['left', 'port'],
                    'right': ['right', 'starboard'],
                    'up': ['up', 'above', 'upward'],
                    'down': ['down', 'below', 'downward']
                }
                
                for orig_term in orig_features[category]:
                    # Direct term matching
                    if orig_term in para_features[category]:
                        preserved = True
                        break
                    
                    # Cardinal direction matching
                    elif orig_term in ['north', 'south', 'east', 'west']:
                        for p in para_features[category]:
                            if orig_term in p.lower():
                                preserved = True
                                break
                        if preserved:
                            break
                    
                    # Direction synonym matching (CRITICAL FIX)
                    else:
                        for base_dir, synonyms in direction_synonyms_feature.items():
                            if orig_term in synonyms:
                                # Check if any synonym appears in paraphrase directions
                                for para_term in para_features[category]:
                                    if any(syn in para_term.lower() for syn in synonyms):
                                        preserved = True
                                        break
                                if preserved:
                                    break
                        
                        # Clock direction equivalence matching (if not already preserved)
                        if not preserved:
                            orig_clock = extract_clock_number(orig_term)
                            if orig_clock:
                                for para_term in para_features[category]:
                                    para_clock = extract_clock_number(para_term)
                                    if para_clock and orig_clock == para_clock:
                                        preserved = True
                                        break
                                if preserved:
                                    break
                            
            elif category == 'landmarks':
                # Semantic landmark matching
                preserved = False
                landmark_synonyms_feature = {
                    'building': ['building', 'structure', 'house', 'edifice'],
                    'road': ['road', 'street', 'highway', 'path'],
                    'destination': ['destination', 'target', 'goal', 'endpoint']
                }
                
                for orig_term in orig_features[category]:
                    if orig_term in para_features[category]:
                        preserved = True
                        break
                    
                    # Landmark synonym matching
                    for base_landmark, synonyms in landmark_synonyms_feature.items():
                        if orig_term in synonyms:
                            for para_term in para_features[category]:
                                if any(syn in para_term.lower() for syn in synonyms):
                                    preserved = True
                                    break
                            if preserved:
                                break
                    if preserved:
                        break
                        
            else:
                # Enhanced feature preservation with synonyms
                preserved = False
                
                # Movement verb synonyms for better semantic matching
                if category == 'movement_verbs':
                    movement_synonyms_feature = {
                        'move': ['move', 'go', 'head', 'proceed', 'travel', 'navigate'],
                        'turn': ['turn', 'rotate', 'pivot', 'swing'],
                        'fly': ['fly', 'soar', 'hover', 'pilot']
                    }
                    
                    for orig_term in orig_features[category]:
                        if orig_term in para_features[category]:
                            preserved = True
                            break
                        
                        # Movement verb synonym matching
                        for base_verb, synonyms in movement_synonyms_feature.items():
                            if orig_term in synonyms:
                                for para_term in para_features[category]:
                                    if para_term in synonyms:
                                        preserved = True
                                        break
                                if preserved:
                                    break
                        if preserved:
                            break
                else:
                    # Standard feature preservation for other categories
                    preserved = any(
                        orig_term in para_features[category]
                        for orig_term in orig_features[category]
                    )
            
            feature_preservation[category] = preserved
        
        # Compute advanced similarity
        advanced_similarity = compute_advanced_similarity(
            orig_features, para_features, similarity_metrics
        )
        
        # Validation logic for positive paraphrases
        if is_positive:
            # More flexible validation for positive paraphrases
            # Focus on overall semantic preservation rather than strict feature matching
            spatial_terms_preserved = (
                (advanced_similarity['combined_similarity'] > 0.4 and
                 advanced_similarity['landmark_similarity'] > 0.7) or
                (advanced_similarity['direction_similarity'] > 0.8 and
                 advanced_similarity['landmark_similarity'] > 0.5) or
                (advanced_similarity['combined_similarity'] > 0.6)
            )
            validation_result = spatial_terms_preserved
            spatial_validation_score = spatial_terms_preserved
        
        # Validation logic for negative paraphrases
        else:
            # For negatives, prioritize spatial feature changes over overall text similarity
            # Text can be very similar, but spatial elements must change
            direction_changed = advanced_similarity['direction_similarity'] < 0.5
            landmark_changed = advanced_similarity['landmark_similarity'] < 0.5
            
            # At least one major spatial component must change
            spatial_terms_changed = direction_changed or landmark_changed
            validation_result = spatial_terms_changed
            spatial_validation_score = spatial_terms_changed
        
        # Detailed logging
        logger.info(f"\n--- Spatial Accuracy Validation ---")
        logger.info(f"Original: {original}")
        logger.info(f"Paraphrase: {paraphrase}")
        logger.info(f"Is Positive Paraphrase: {is_positive}")
        
        logger.info("Original Spatial Features:")
        for category, terms in orig_features.items():
            logger.info(f"  {category}: {terms}")
        
        logger.info("Paraphrase Spatial Features:")
        for category, terms in para_features.items():
            logger.info(f"  {category}: {terms}")
        
        logger.info("Similarity Metrics:")
        for metric, value in similarity_metrics.items():
            logger.info(f"  {metric}: {value}")
        
        logger.info("Feature Preservation:")
        for category, preserved in feature_preservation.items():
            logger.info(f"  {category}: {preserved}")
        
        logger.info(f"Validation Result: {validation_result}")

        return {
            'spatial_validation_score': spatial_validation_score,
            'validation_result': validation_result,
            'similarity_metrics': similarity_metrics,
            'feature_preservation': feature_preservation,
            'advanced_similarity': advanced_similarity
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