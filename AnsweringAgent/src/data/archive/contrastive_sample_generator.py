#!/usr/bin/env python3
"""
Clean Contrastive Sample Generator for UAV Navigation
Enhanced with 4-strategy pipeline for comprehensive positive generation.
"""

import torch
import torch.nn.functional as F
import json
import random
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import re
import os
from typing import List, Dict, Any

class ContrastiveSampleGenerator:
    """
    Enhanced contrastive sample generator for UAV navigation tasks.
    Implements 4-strategy pipeline for positive generation.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu", 
                 llm_model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """Initialize the generator with essential components only."""
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Load sentence embedding model
        self.logger.info(f"Loading sentence embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Initialize UAV-specific terminology (dataset-driven)
        self._init_navigation_terminology()
        
        # For negative generation
        self.alternative_answers = []
        
        # Initialize LLM for Strategy 2 (try Mixtral, fallback to alternatives)
        self.llm_tokenizer = None
        self.llm_model = None
        self.llm_model_name = llm_model
        
        # Try loading the specified model, with fallbacks
        fallback_models = [
            llm_model,  # User specified (e.g., Mixtral)
            "microsoft/DialoGPT-medium",  # Alternative 1: Good for dialogue
            "google/flan-t5-large"  # Alternative 2: Reliable fallback
        ]
        
        for model_attempt in fallback_models:
            try:
                self.logger.info(f"Attempting to load model: {model_attempt}")
                
                if "flan-t5" in model_attempt.lower():
                    # Handle T5 models differently
                    from transformers import T5ForConditionalGeneration
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(model_attempt)
                    self.llm_model = T5ForConditionalGeneration.from_pretrained(
                        model_attempt,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
                    self.llm_model_name = model_attempt
                    self.is_t5_model = True
                else:
                    # Handle causal LMs (Mixtral, DialoGPT, etc.)
                    self.llm_tokenizer = AutoTokenizer.from_pretrained(model_attempt)
                    
                    # Set pad token if not available
                    if self.llm_tokenizer.pad_token is None:
                        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
                    
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        model_attempt, 
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        device_map="auto" if device == "cuda" else None,
                        trust_remote_code=True
                    )
                    self.llm_model_name = model_attempt
                    self.is_t5_model = False
                
            if device == "cpu":
                    self.llm_model = self.llm_model.to(device)
                self.llm_model.eval()
                self.logger.info(f"Successfully loaded: {model_attempt}")
                break
                
        except Exception as e:
                self.logger.warning(f"Failed to load {model_attempt}: {e}")
                continue
        
        if self.llm_model is None:
            self.logger.error("Failed to load any LLM model")
            self.logger.warning("Strategy 2 will use simple fallback method")
    
    def _init_navigation_terminology(self):
        """Initialize UAV navigation terminology based on AVDN dataset analysis."""
        
        # Direction terms (frequency-ordered from dataset)
        self.direction_terms = [
            "turn", "forward", "right", "left", "north", "south", "straight", "west", "east",
            "ahead", "behind", "northwest", "southeast", "southwest", "northeast"
        ]
        
        # Landmark terms (frequency-ordered from dataset analysis)
        self.landmark_terms = [
            "building", "road", "parking", "field", "area", "house", "highway", 
            "structure", "section", "intersection", "tree", "container", "truck", "yard", "tower"
        ]
        
        # Color terms (frequency-ordered)
        self.color_terms = [
            "white", "gray", "brown", "grey", "red", "blue", "green", "black", "dark", "light"
        ]
        
        # Spatial relations (frequency-ordered)
        self.spatial_relation_terms = [
            "over", "near", "in front of", "next to", "around", "through", "behind", 
            "across", "along", "between", "above", "below", "under"
        ]
        
        # Size terms
        self.size_terms = ["large", "small", "big", "tiny", "huge", "medium", "long", "short", "wide", "narrow"]
        
        # Shape terms
        self.shape_terms = ["square", "rectangular", "round", "circular", "oval", "triangular", "flat", "tall"]
    
    def generate_embedding(self, text):
        """Generate embedding for text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        return F.normalize(embeddings.squeeze(0), p=2, dim=0)
    
    def calculate_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts."""
        emb1 = self.generate_embedding(text1)
        emb2 = self.generate_embedding(text2)
        return float(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item())
    
    def extract_spatial_tokens(self, text):
        """Extract spatial tokens that must be preserved."""
        spatial_tokens = set()
        text_lower = text.lower()
        
        # Extract clock directions
        clock_matches = re.findall(r'\d+\s*o\'?clock', text_lower)
        spatial_tokens.update(clock_matches)
        
        # Extract cardinal directions
        cardinal_matches = re.findall(r'\b(north|south|east|west|northeast|northwest|southeast|southwest)\b', text_lower)
        spatial_tokens.update(cardinal_matches)
        
        # Extract spatial prepositions
        spatial_prep_matches = re.findall(r'\b(next to|in front of|behind|above|below|near|across from|over|under)\b', text_lower)
        spatial_tokens.update(spatial_prep_matches)
        
        # Extract landmarks
        landmark_matches = [term for term in self.landmark_terms if term in text_lower]
        spatial_tokens.update(landmark_matches)
        
        # Extract colors (for landmark descriptions)
        color_matches = [term for term in self.color_terms if term in text_lower]
        spatial_tokens.update(color_matches)
        
        return spatial_tokens
    
    def validate_spatial_fidelity(self, original_text, candidate_text):
        """Validate that spatial information is preserved."""
        original_tokens = self.extract_spatial_tokens(original_text)
        candidate_tokens = self.extract_spatial_tokens(candidate_text)
        
        # Check that all original spatial tokens are preserved
        preserved_ratio = len(original_tokens & candidate_tokens) / max(len(original_tokens), 1)
        
        # Additional checks
        repetition_penalty = self._check_repetition(candidate_text)
        reference_consistency = self._check_reference_consistency(original_text, candidate_text)
        
        return {
            'spatial_preservation': preserved_ratio,
            'repetition_score': repetition_penalty,
            'reference_consistency': reference_consistency,
            'is_valid': preserved_ratio >= 0.9 and repetition_penalty > 0.8 and reference_consistency > 0.8
        }
    
    def _check_repetition(self, text):
        """Check for repetitive phrases."""
        words = text.lower().split()
        if len(words) < 3:
            return 1.0
        
        # Check for repeated 3-grams
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        
        return unique_ratio
    
    def _check_reference_consistency(self, original, candidate):
        """Check that spatial references remain consistent."""
        # Simple heuristic: count landmark mentions
        original_landmarks = [term for term in self.landmark_terms if term in original.lower()]
        candidate_landmarks = [term for term in self.landmark_terms if term in candidate.lower()]
        
        if not original_landmarks:
            return 1.0
        
        # Check that landmark count is preserved
        landmark_consistency = min(len(candidate_landmarks) / len(original_landmarks), 1.0)
        
        return landmark_consistency
    
    # =================================================================
    # POSITIVE GENERATION - 4-Strategy Pipeline
    # =================================================================
    
    def generate_positive_examples(self, original_answer, n=2):
        """Generate spatial-aware positive examples using a 4-strategy pipeline.
        
        Strategy Pipeline:
        1. Combined Current Approaches (spatial synonyms + structure + clock formats)
        2. Enhanced LLM Paraphrasing with Mixtral-8×7B
        3. Hierarchical Spatial Decomposition (to be implemented)
        4. Trajectory-Aware Positives (to be implemented)
        
        Each strategy builds upon the previous ones to create comprehensive paraphrases.
        """
        self.logger.info(f"Generating {n} spatial-aware positive examples using pipeline approach")
        
        if len(original_answer.split()) < 3:
            return self._generate_simple_variations(original_answer, n)
        
        # Strategy 1: Combined Current Approaches Pipeline
        strategy1_positives = self._strategy1_combined_current_approaches(original_answer)
        strategy1_positives = sorted(strategy1_positives, key=lambda x: x["similarity"], reverse=False)

        
        # Strategy 2: Enhanced LLM Paraphrasing with Mixtral
        strategy2_positives = self._strategy2_enhanced_llm_paraphrasing(original_answer)
        strategy2_positives = sorted(strategy2_positives, key=lambda x: x["similarity"], reverse=True)
        # TODO: Strategy 3: Hierarchical Spatial Decomposition (awaiting implementation permission)  
        # TODO: Strategy 4: Trajectory-Aware Positives (awaiting implementation permission)
        
        # Combine all strategy results
        all_positives = strategy1_positives[:1] + strategy2_positives[:1]
        
        # If not enough, add simple fallback
        if len(all_positives) < n:
            remaining = n - len(all_positives)
            simple_variations = self._generate_simple_variations(original_answer, remaining, all_positives)
            all_positives.extend(simple_variations)
        
        return all_positives[:n]
    
    def _strategy1_combined_current_approaches(self, original_answer):
        """Strategy 1: Apply all current approaches as a unified pipeline.
        
        This combines:
        - Spatial synonym substitution
        - Spatial structure variation
        - Clock direction format variations
        
        Each transformation is applied to create comprehensive variations.
        """
        positives = []
        extracted_info = self._extract_navigation_info(original_answer)
        
        # Step 1a: Generate spatial synonym variations
        synonym_variations = self._generate_spatial_synonym_positives(original_answer, extracted_info)
        
        # Step 1b: Apply structure variations to both original AND synonym variations
        structure_variations = []
        if synonym_variations:
            for item in synonym_variations:
                structure_variations.extend(self._generate_spatial_structure_positives(item["text"], extracted_info)) 
        
        else:
            structure_variations = self._generate_spatial_structure_positives(original_answer, extracted_info)
        
        # Step 3: Apply clock format variations to structure variants
        
        if extracted_info.get("clock_directions"):
            clock_variations = []
            if structure_variations:
                for item in structure_variations:
                    clock_variations.extend(self._generate_clock_format_positives(item["text"], extracted_info["clock_directions"]))
            else:
                clock_variations = self._generate_clock_format_positives(original_answer, extracted_info["clock_directions"])
            
            positives.extend(clock_variations)
        else:
            positives.extend(structure_variations)
        
        # Mark all as Strategy 1 outputs
        for pos in positives:
            pos["strategy"] = "strategy1_combined_current"
        
        return positives
    
    def _strategy2_enhanced_llm_paraphrasing(self, original_answer):
        """Strategy 2: Enhanced LLM paraphrasing with Mixtral-8×7B and spatial constraints.
        
        Uses Mixtral-8×7B-Instruct to generate spatially-faithful paraphrases with:
        - AVDN-specific few-shot prompting
        - Spatial token forcing through constrained decoding
        - Enhanced validation pipeline
        """
        positives = []
        
        if self.llm_model is None:
            # Fallback to simple method
            return self._generate_simple_fallback(original_answer)
        
        # Use enhanced Mixtral paraphrasing
        positives = self._generate_mixtral_paraphrases(original_answer)
        
        # Mark all as Strategy 2 outputs
        for pos in positives:
            pos["strategy"] = "strategy2_enhanced_llm"
        
        return positives
    
    def _generate_mixtral_paraphrases(self, original_answer):
        """Generate paraphrases using Mixtral-8×7B with AVDN-specific prompting and spatial constraints."""
        positives = []
        
        # AVDN-specific prompt based on dataset analysis
        system_prompt = """You are an expert UAV navigation instructor. Paraphrase navigation instructions while preserving ALL spatial information exactly.

CRITICAL RULES:
- Keep exact directions (3 o'clock, 6 o'clock, north, south, etc.)
- Maintain all landmarks and their spatial relationships
- Preserve step order and navigation target
- Use different wording but identical spatial meaning
- Do NOT add or remove spatial elements

Examples:
Input: Move towards your 3 o'clock direction. You will pass two red rooftops and reach a small white building.
Output: Head in the 3 o'clock direction. You'll go past two red-roofed structures and arrive at a small white building.

Input: Turn right and go northeast. Look for a square building next to a water tower.
Output: Make a right turn and head northeast. Find the square structure beside the water tower.

Input: Head to 6 o'clock. Cross the parking lot and find the triangular building near the large tree.
Output: Go toward 6 o'clock. Traverse the parking area and locate the triangle-roofed structure beside the tall tree."""

        user_prompt = f"Now paraphrase: {original_answer}"
        
        # Extract spatial tokens for validation
        spatial_tokens = self.extract_spatial_tokens(original_answer)
        
        try:
            # Format as Mixtral chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Apply chat template
            formatted_prompt = self.llm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.llm_tokenizer(
                formatted_prompt,
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            # Generate with enhanced parameters for spatial preservation
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=150,
                    num_return_sequences=3,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.llm_tokenizer.pad_token_id
                )
            
            # Process outputs
            for output in outputs:
                # Decode and extract response
                generated_text = self.llm_tokenizer.decode(output, skip_special_tokens=True)
                
                # Extract the paraphrase (after the last assistant response)
                if "[/INST]" in generated_text:
                    paraphrase = generated_text.split("[/INST]")[-1].strip()
                else:
                    paraphrase = generated_text.replace(formatted_prompt, "").strip()
                
                if paraphrase and paraphrase != original_answer and len(paraphrase.split()) > 3:
                    # Validate spatial fidelity
                    validation = self.validate_spatial_fidelity(original_answer, paraphrase)
                    
                    if validation['is_valid']:
                    similarity = self.calculate_similarity(original_answer, paraphrase)
                    
                        if 0.7 <= similarity <= 0.95:
                        positives.append({
                            "text": paraphrase,
                            "similarity": similarity,
                                "type": "mixtral_paraphrase",
                                "validation": validation
                        })
                
        except Exception as e:
            self.logger.error(f"Mixtral generation error: {e}")
            # Fall back to simple method
            positives = self._generate_simple_fallback(original_answer)
        
        return positives
    
    def _generate_simple_fallback(self, original_answer):
        """Simple fallback when LLM generation fails."""
        positives = []
        
        # Basic synonym replacement
        fallback_variants = []
        
        simple_subs = {
            'move': 'go',
            'turn': 'rotate',
            'destination': 'target',
            'building': 'structure'
        }
        
        for original, replacement in simple_subs.items():
            if original in original_answer.lower():
                variant = original_answer.replace(original, replacement)
                if variant != original_answer:
                    similarity = self.calculate_similarity(original_answer, variant)
                    if similarity > 0.8:
                        fallback_variants.append({
                            "text": variant,
                            "similarity": similarity,
                            "type": "simple_fallback"
                        })
                        break
        
        return fallback_variants[:2]
    
    def _generate_spatial_synonym_positives(self, original_answer, extracted_info):
        """Generate positives using spatial-aware synonym substitution (EXPANDED with dataset patterns)."""
        positives = []
        
        # EXPANDED UAV-specific spatial synonyms based on AVDN dataset analysis
        spatial_synonyms = {
            # High-frequency movement verbs from dataset
            "turn": ["rotate", "pivot", "shift"],
            "move": ["proceed", "travel", "navigate", "go"],
            "head": ["go", "proceed", "travel", "move"],
            "fly": ["navigate", "travel", "move"],
            "go": ["move", "proceed", "travel", "navigate", "continue"],
            
            # Destination terms - very common in dataset
            "destination": ["target", "goal", "location"],
            "goal": ["destination", "target", "location"],
            "target": ["destination", "goal", "location"],
            
            # Building/landmark terms - 402 occurrences
            "building": ["structure", "facility", "construction"],
            "structure": ["building", "facility"],
            "facility": ["building", "structure"],
            
            # Direction refinements  
            "road": ["highway", "street"],
            "parking": ["lot", "area"],
            "field": ["area", "ground"],
            
            # Spatial descriptors
            "near": ["close to", "next to"],
            "over": ["above"],
            "large": ["big", "huge"],
            "small": ["little", "tiny"],
            
            # Direction refinements from dataset
            "toward": ["towards", "in the direction of"],
            "towards": ["toward", "in the direction of"],
            
            # Common typo fixes found in dataset (high frequency!)
            "foward": ["forward"],
            "forwards": ["forward"],
            "grey": ["gray"],
            
            # Informal starters (high frequency)
            "please": [""],
            "just": [""],
        }
        
        for original_term, synonyms in spatial_synonyms.items():
            if original_term in original_answer.lower():
                for synonym in synonyms:
                    positive_text = self._replace_preserving_case(original_answer, original_term, synonym)
                    
                    if positive_text != original_answer:
                        similarity = self.calculate_similarity(original_answer, positive_text)
                        positives.append({
                            "text": positive_text,
                            "similarity": similarity,
                            "type": "spatial_synonym",
                        })
        
        return positives
    
    def _generate_spatial_structure_positives(self, original_answer, extracted_info):
        """Generate positives by varying sentence structure (EXPANDED with dataset patterns)."""
        positives = []
        
        # EXPANDED structure transformations based on AVDN dataset frequency analysis
        structure_patterns = [
            # Original patterns (keep existing)
            (r"head\s+(\w+)\s+to\s+(.+)", r"go \1 and you'll reach \2"),
            (r"turn\s+(\w+)\s+and\s+proceed\s+(.+)", r"make a \1 turn and continue \2"),
            (r"the destination is\s+(.+)", r"your target is located \1"),
            (r"move towards\s+(.+)", r"navigate in the direction of \1"),
            
            # NEW HIGH-FREQUENCY PATTERNS from comprehensive analysis
            
            # Clock direction variants (very high frequency)
            # "turn on X o'clock" - 57 matches  
            (r"turn\s+on\s+(\d+)\s*o'?clock", r"rotate to \1 o'clock"),
            (r"turn\s+on\s+(\d+)\s*o'?clock", r"go towards \1 o'clock"),
            
            # Typo variants - 43 matches for "X'o clock"
            (r"(\d+)'o\s+clock", r"\1 o'clock"),
            (r"(\d+)'\s*o\s*clock", r"\1 o'clock"),
            
            # Informal starters (high frequency)
            # "please turn X" - 73 matches
            (r"please\s+turn\s+(.+)", r"turn \1"),
            (r"please\s+turn\s+(.+)", r"rotate \1"),
            
            # "just turn X" - 29 matches  
            (r"just\s+turn\s+(.+)", r"turn \1"),
            (r"just\s+turn\s+(.+)", r"rotate \1"),
            
            # "just move X" - 15 matches
            (r"just\s+move\s+(.+)", r"move \1"),
            (r"just\s+move\s+(.+)", r"proceed \1"),
            
            # "you should go to X" - 42 matches
            (r"you\s+should\s+go\s+to\s+(.+)", r"go to \1"),
            (r"you\s+should\s+go\s+to\s+(.+)", r"move to \1"),
            
            # Spatial navigation patterns
            # "cross X" - 186 matches (very high!)
            (r"cross\s+(.+)", r"go across \1"),
            (r"cross\s+(.+)", r"traverse \1"),
            
            # "pass X" - 48 matches
            (r"pass\s+(.+)", r"go past \1"),
            (r"pass\s+(.+)", r"move beyond \1"),
            
            # "go through X" - 8 matches
            (r"go\s+through\s+(.+)", r"pass through \1"),
            (r"go\s+through\s+(.+)", r"traverse \1"),
            
            # Positional descriptions (high frequency)
            # "looks like X" - 95 matches
            (r"looks\s+like\s+(.+)", r"appears to be \1"),
            (r"looks\s+like\s+(.+)", r"resembles \1"),
            
            # "destination looks like X" - 65 matches
            (r"destination\s+looks\s+like\s+(.+)", r"your target appears to be \1"),
            (r"destination\s+looks\s+like\s+(.+)", r"the goal resembles \1"),
            
            # "you will see X" - 82 matches
            (r"you\s+will\s+see\s+(.+)", r"you'll observe \1"),
            (r"you\s+will\s+see\s+(.+)", r"you can spot \1"),
            
            # Clock direction enhanced patterns
            # "at your X o'clock" - 29 matches
            (r"at\s+your\s+(\w+)\s+o'?clock", r"in your \1 o'clock direction"),
            (r"at\s+your\s+(\w+)\s+o'?clock", r"towards your \1 o'clock"),
            
            # "in your X o'clock" - 19 matches  
            (r"in\s+your\s+(\w+)\s+o'?clock", r"at your \1 o'clock"),
            (r"in\s+your\s+(\w+)\s+o'?clock", r"towards \1 o'clock"),
            
            # Common typo fixes
            # "go foward" - 19 matches  
            (r"go\s+foward", r"go forward"),
            (r"move\s+foward", r"move forward"),
            
            # Existing patterns from original analysis
            # "turn to your X" - 36 matches
            (r"turn\s+to\s+your\s+(\w+)", r"rotate to your \1"),
            (r"turn\s+to\s+your\s+(\w+)", r"pivot to your \1"),
            
            # "go to your X" - 56 matches  
            (r"go\s+to\s+your\s+(\w+)", r"move to your \1"),
            (r"go\s+to\s+your\s+(\w+)", r"proceed to your \1"),
            
            # "your destination is X" - 82 matches
            (r"your\s+destination\s+is\s+(.+)", r"your target is \1"),
            (r"your\s+destination\s+is\s+(.+)", r"your goal is \1"),
            
            # "your goal is X" - 17 matches
            (r"your\s+goal\s+is\s+(.+)", r"your target is \1"),
            (r"your\s+goal\s+is\s+(.+)", r"your destination is \1"),
            
            # "X is your destination" - 56 matches
            (r"(\w+)\s+is\s+your\s+destination", r"\1 is your target"),
            (r"(\w+)\s+is\s+your\s+destination", r"\1 is your goal"),
            
            # "turn X and go Y" - 12 matches
            (r"turn\s+(\w+)\s+and\s+go\s+(.+)", r"rotate \1 and proceed \2"),
            (r"turn\s+(\w+)\s+and\s+go\s+(.+)", r"pivot \1 and move \2"),
            
            # "turn X and move Y" - 6 matches
            (r"turn\s+(\w+)\s+and\s+move\s+(.+)", r"pivot \1 and travel \2"),
            (r"turn\s+(\w+)\s+and\s+move\s+(.+)", r"rotate \1 and proceed \2"),
            
            # "move to the X" - 8 matches
            (r"move\s+to\s+the\s+(\w+)", r"proceed to the \1"),
            (r"move\s+to\s+the\s+(\w+)", r"travel to the \1"),
            
            # "that is your X" - 22 matches
            (r"that\s+is\s+your\s+(destination|goal|target)", r"that is your target"),
            
            # "it is in your X direction" - 86 matches
            (r"it\s+is\s+in\s+your\s+(.+)\s+direction", r"it is located in your \1 direction"),
            (r"it\s+is\s+in\s+your\s+(.+)\s+direction", r"it can be found in your \1 direction"),
            
            # Clock direction patterns - 48 matches for "toward/towards X o'clock"
            (r"towards?\s+(\d+)\s*o'?clock", r"in the direction of \1 o'clock"),
            (r"towards?\s+(\d+)\s*o'?clock\s+direction", r"toward \1 o'clock"),
            
            # "X direction" - 377 matches (very high frequency!)
            (r"(\d+)\s*o'?clock\s+direction", r"\1 o'clock"),
            (r"in\s+the\s+(\d+)\s*o'?clock\s+direction", r"towards \1 o'clock"),
            
            # General direction patterns
            (r"go\s+in\s+the\s+(.+)\s+direction", r"move towards \1"),
            (r"head\s+in\s+the\s+(.+)\s+direction", r"go towards \1"),
        ]
        
        for pattern, replacement in structure_patterns:
            positive_text = re.sub(pattern, replacement, original_answer, flags=re.IGNORECASE)
            
            if positive_text != original_answer:
                positive_text = positive_text[0].upper() + positive_text[1:] if positive_text else positive_text
                similarity = self.calculate_similarity(original_answer, positive_text)
                positives.append({
                    "text": positive_text,
                    "similarity": similarity,
                    "type": "spatial_structure"
                })
        
        return positives
    
    def _generate_clock_format_positives(self, original_answer, clock_directions):
        """Generate positives by varying clock direction formats."""
        positives = []
        
        for clock_dir in clock_directions:
            hour_match = re.search(r'(\d+)', clock_dir)
            if not hour_match:
                continue
                
            hour = hour_match.group(1)
            
            # Format variations that preserve exact direction
            if "o'clock" in clock_dir:
                variations = [f"{hour} oclock", f"{hour} o'clock direction"]
            elif ":" in clock_dir and ":30" in clock_dir:
                variations = [f"{hour}.30", f"{hour}:30 direction"]
            else:
                continue
            
            for variation in variations:
                if variation != clock_dir:
                    positive_text = original_answer.replace(clock_dir, variation)
                    
                    if positive_text != original_answer:
                        similarity = self.calculate_similarity(original_answer, positive_text)
                        positives.append({
                            "text": positive_text,
                            "similarity": similarity,
                            "type": "clock_format_variation"
                        })
        
        return positives
    
    def _generate_simple_variations(self, original_answer, n, existing_positives=None):
        """Simple fallback variations."""
        variations = [
            f"Based on what I can see, {original_answer.lower()}",
            f"From the drone's view, {original_answer.lower()}",
            f"{original_answer} That's your destination."
        ]
        
        positives = []
        existing_texts = set()
        
        if existing_positives:
            existing_texts = {pos["text"] for pos in existing_positives}
        
        for variation in variations:
            if len(positives) >= n:
                break
            
            if variation not in existing_texts and variation != original_answer:
                similarity = self.calculate_similarity(original_answer, variation)
                if similarity > 0.7:
                    positives.append({
                        "text": variation,
                        "similarity": similarity,
                        "type": "simple_variation",
                        "strategy": "simple_fallback"
                    })
        
        return positives
    
    # =================================================================
    # NEGATIVE GENERATION - Enhanced Spatial Strategies
    # =================================================================
    
    def generate_negative_examples(self, original_answer, n=3):
        """Generate enhanced spatial negatives using AVDN dataset insights."""
        negatives = []
        extracted_info = self._extract_navigation_info(original_answer)
        
        # Strategy 1: Enhanced clock direction shifting (UAV-critical)
        if extracted_info.get("clock_directions"):
            clock_negatives = self._generate_enhanced_clock_shift_negatives(original_answer, extracted_info["clock_directions"])
            negatives.extend(clock_negatives)
        
        # Strategy 2: Enhanced direction reversal with AVDN patterns
        if extracted_info["directions"]:
            direction_negatives = self._generate_enhanced_direction_negatives(original_answer, extracted_info["directions"])
            negatives.extend(direction_negatives)
            
        # Strategy 3: AVDN-specific landmark substitution
        if extracted_info["landmarks"]:
            landmark_negatives = self._generate_avdn_landmark_negatives(original_answer, extracted_info["landmarks"])
            negatives.extend(landmark_negatives)
        
        # Strategy 4: Spatial relation perturbation with validation
        if extracted_info["spatial_relations"]:
            spatial_negatives = self._generate_enhanced_spatial_relation_negatives(original_answer, extracted_info["spatial_relations"])
            negatives.extend(spatial_negatives)
        
        # Filter and rank using enhanced validation
        quality_negatives = self._filter_and_rank_negatives(negatives, original_answer)
        
        return quality_negatives[:n]
    
    def _generate_enhanced_clock_shift_negatives(self, original_answer, clock_directions):
        """Generate enhanced clock direction negatives using AVDN patterns."""
        negatives = []
        
        for clock_dir in clock_directions:
            try:
                # Handle both "X o'clock" and "X:30" formats
                hour_match = re.search(r'(\d+)\s*(?:o\'?clock|:\d+)', clock_dir.lower())
                if hour_match:
                    hour = int(hour_match.group(1))
                    
                    # AVDN-specific strategic shifts based on frequency analysis
                    # From dataset: most common are 90°, 180° shifts
                    shifts = [3, 6, 9]  # hours (90°, 180°, 270°)
                    
                    for shift in shifts:
                        new_hour = ((hour - 1 + shift) % 12) + 1
                        
                        # Preserve original format 
                        if "o'clock" in clock_dir:
                        new_clock = f"{new_hour} o'clock"
                        elif ":" in clock_dir:
                            new_clock = f"{new_hour}:30"
                        else:
                            continue
                        
                        negative_text = original_answer.replace(clock_dir, new_clock)
                        
                        if negative_text != original_answer:
                            # Validate the negative maintains spatial structure
                            validation = self.validate_spatial_fidelity(original_answer, negative_text)
                            
                            # We want exactly one error (clock direction change)
                            if validation['spatial_preservation'] >= 0.8:  # Most tokens preserved
                            similarity = self.calculate_similarity(original_answer, negative_text)
                            
                            if 0.85 <= similarity <= 0.99:
                                negatives.append({
                                    "text": negative_text,
                                    "similarity": similarity,
                                        "type": "enhanced_clock_shift",
                                    "shift_degrees": shift * 30,
                                    "original_hour": hour,
                                        "new_hour": new_hour,
                                        "validation": validation
                                })
                                
            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Error processing clock direction {clock_dir}: {e}")
                continue
        
        return negatives
    
    def _generate_enhanced_direction_negatives(self, original_answer, directions):
        """Generate enhanced direction negatives based on AVDN frequency patterns."""
        negatives = []
        
        # AVDN-specific direction opposites based on dataset analysis
        avdn_direction_opposites = {
            "north": ["south", "southeast", "southwest"],  # Most common in dataset
            "south": ["north", "northeast", "northwest"], 
            "east": ["west", "northwest", "southwest"],
            "west": ["east", "northeast", "southeast"],
            "northeast": ["southwest", "south", "west"],
            "northwest": ["southeast", "south", "east"],
            "southeast": ["northwest", "north", "west"],
            "southwest": ["northeast", "north", "east"],
            "left": ["right"],
            "right": ["left"],
            "forward": ["backward", "behind"]  # Common in AVDN
        }
        
        for direction in directions:
            direction_lower = direction.lower()
            
            if direction_lower in avdn_direction_opposites:
                alternatives = avdn_direction_opposites[direction_lower]
                
                for alt_direction in alternatives:
                    negative_text = original_answer.replace(direction, alt_direction)
                    
                    if negative_text != original_answer:
                        # Validate spatial fidelity
                        validation = self.validate_spatial_fidelity(original_answer, negative_text)
                        
                        if validation['spatial_preservation'] >= 0.8:
                        similarity = self.calculate_similarity(original_answer, negative_text)
                        
                        if 0.8 <= similarity <= 0.95:
                            negatives.append({
                                "text": negative_text,
                                "similarity": similarity,
                                "type": "enhanced_direction_reversal",
                                "original_direction": direction,
                                    "new_direction": alt_direction,
                                    "validation": validation
                            })
        
        return negatives
    
    def _generate_avdn_landmark_negatives(self, original_answer, landmarks):
        """Generate landmark negatives based on AVDN dataset frequency analysis."""
        negatives = []
        
        # AVDN-specific landmark substitutions based on 300-sample analysis
        avdn_landmark_substitutions = {
            # High-frequency landmarks from dataset
            "building": ["structure", "house", "facility"],  # 158 matches in dataset
            "structure": ["building", "house", "facility"],
            "parking": ["road", "field", "area"],  # 19 matches
            "road": ["parking", "field", "area"],  # 17 matches  
            "field": ["area", "parking", "road"],
            "house": ["building", "structure"],  # 4 matches
            "container": ["building", "structure", "house"],
            "tree": ["building", "structure"],
            "tower": ["building", "structure"]
        }
        
        for landmark in landmarks:
            landmark_lower = landmark.lower()
            
            if landmark_lower in avdn_landmark_substitutions:
                alternatives = avdn_landmark_substitutions[landmark_lower]
                
                for alt_landmark in alternatives:
                    negative_text = original_answer.replace(landmark, alt_landmark)
                    
                    if negative_text != original_answer:
                        validation = self.validate_spatial_fidelity(original_answer, negative_text)
                        
                        if validation['spatial_preservation'] >= 0.8:
                        similarity = self.calculate_similarity(original_answer, negative_text)
                        
                        if 0.85 <= similarity <= 0.99:
                            negatives.append({
                                "text": negative_text,
                                "similarity": similarity,
                                    "type": "avdn_landmark_substitution",
                                "original_landmark": landmark,
                                    "new_landmark": alt_landmark,
                                    "validation": validation
                            })
        
        return negatives
    
    def _generate_enhanced_spatial_relation_negatives(self, original_answer, spatial_relations):
        """Generate spatial relation negatives with enhanced validation."""
        negatives = []
        
        # AVDN-specific spatial relation opposites
        spatial_opposites = {
            "above": ["below", "under"],
            "below": ["above", "over"],
            "over": ["below", "under"],  # 16 matches in dataset
            "near": ["far from", "distant from"],  # 9 matches
            "in front of": ["behind"],  # 16 matches
            "behind": ["in front of"],  # 2 matches
            "next to": ["far from", "across from"]  # 25 matches
        }
        
        for relation in spatial_relations:
            relation_lower = relation.lower()
            
            if relation_lower in spatial_opposites:
                alternatives = spatial_opposites[relation_lower]
                
                for alt_relation in alternatives:
                    negative_text = original_answer.replace(relation, alt_relation)
                    
                    if negative_text != original_answer:
                        validation = self.validate_spatial_fidelity(original_answer, negative_text)
                        
                        if validation['spatial_preservation'] >= 0.8:
                        similarity = self.calculate_similarity(original_answer, negative_text)
                        
                        if 0.8 <= similarity <= 0.95:
                            negatives.append({
                                "text": negative_text,
                                "similarity": similarity,
                                    "type": "enhanced_spatial_relation",
                                "original_relation": relation,
                                    "new_relation": alt_relation,
                                    "validation": validation
                            })
        
        return negatives
    
    # =================================================================
    # UTILITY METHODS
    # =================================================================
    
    def _extract_navigation_info(self, text):
        """Extract navigation information from text."""
        text_lower = text.lower()
        
        # Extract directions
        directions = [term for term in self.direction_terms if term in text_lower]
        
        # Extract clock directions
        clock_directions = []
        for i in range(1, 13):
            if f"{i} o'clock" in text_lower:
                clock_directions.append(f"{i} o'clock")
            if f"{i}:30" in text_lower:
                clock_directions.append(f"{i}:30")
        
        # Extract other spatial elements
        landmarks = [term for term in self.landmark_terms if term in text_lower]
        colors = [term for term in self.color_terms if term in text_lower]
        shapes = [term for term in self.shape_terms if term in text_lower]
        spatial_relations = [term for term in self.spatial_relation_terms if term in text_lower]
        sizes = [term for term in self.size_terms if term in text_lower]
        
        return {
            "directions": directions,
            "clock_directions": clock_directions,
            "landmarks": landmarks,
            "colors": colors,
            "shapes": shapes,
            "spatial_relations": spatial_relations,
            "sizes": sizes
        }
    
    def _filter_and_rank_positives(self, positives, original_answer):
        """Filter and rank positive examples for quality and diversity."""
        if not positives:
            return []
        
        # Remove duplicates
        unique_positives = []
        seen_texts = set()
        
        for pos in positives:
            if pos["text"] not in seen_texts and pos["text"] != original_answer:
                seen_texts.add(pos["text"])
                unique_positives.append(pos)
        
        # Sort by similarity (descending)
        unique_positives.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Select diverse positives
        selected_positives = []
        used_similarities = []
        
        for pos in unique_positives:
            # Check diversity
            similarity_ok = True
            for used_sim in used_similarities:
                if abs(pos["similarity"] - used_sim) < 0.05:
                    similarity_ok = False
                    break
            
            if similarity_ok and 0.65 <= pos["similarity"] <= 0.95:
                selected_positives.append(pos)
                used_similarities.append(pos["similarity"])
        
        return selected_positives
    
    def _filter_and_rank_negatives(self, negatives, original_answer):
        """Filter and rank negative examples for quality and diversity."""
        if not negatives:
            return []
        
        # Remove duplicates
        unique_negatives = []
        seen_texts = set()
        
        for neg in negatives:
            if neg["text"] not in seen_texts and neg["text"] != original_answer:
                seen_texts.add(neg["text"])
                unique_negatives.append(neg)
        
        # Sort by similarity (descending) - prioritize hard negatives
        unique_negatives.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Select diverse negatives
        selected_negatives = []
        used_similarities = []
        
        for neg in unique_negatives:
            # Check diversity
            similarity_ok = True
            for used_sim in used_similarities:
                if abs(neg["similarity"] - used_sim) < 0.1:
                    similarity_ok = False
                    break
        
            if similarity_ok and 0.5 <= neg["similarity"] <= 0.95:
                selected_negatives.append(neg)
                used_similarities.append(neg["similarity"])
        
        return selected_negatives
    
    def _replace_preserving_case(self, text, original_term, replacement):
        """Replace term while preserving case pattern."""
        import re
        
        pattern = re.compile(re.escape(original_term), re.IGNORECASE)
        
        def replace_func(match):
            matched_text = match.group(0)
            
            if matched_text.isupper():
                return replacement.upper()
            elif matched_text.islower():
                return replacement.lower()
            elif matched_text[0].isupper():
                return replacement.capitalize()
            else:
                return replacement
        
        return pattern.sub(replace_func, text) 