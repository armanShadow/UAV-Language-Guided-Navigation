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
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
import re
import os
from typing import List, Dict, Any

class ContrastiveSampleGenerator:
    """
    Enhanced contrastive sample generator for UAV navigation tasks.
    Implements 4-strategy pipeline for positive generation.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu", 
                 flan_t5_model="google/flan-t5-large"):
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
        
        # Initialize FLAN-T5 for Strategy 2
        self.flan_t5_tokenizer = None
        self.flan_t5_model = None
        
        try:
            self.logger.info(f"Loading FLAN-T5 model: {flan_t5_model}")
            
            # Use AutoTokenizer instead of T5Tokenizer (avoids SentencePiece dependency)
            self.flan_t5_tokenizer = AutoTokenizer.from_pretrained(flan_t5_model)
            
            self.flan_t5_model = T5ForConditionalGeneration.from_pretrained(
                flan_t5_model, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            if device == "cpu":
                self.flan_t5_model = self.flan_t5_model.to(device)
            self.flan_t5_model.eval()
            self.logger.info("FLAN-T5 model loaded successfully with AutoTokenizer")
        except Exception as e:
            self.logger.error(f"Failed to load FLAN-T5: {e}")
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
    
    # =================================================================
    # POSITIVE GENERATION - 4-Strategy Pipeline
    # =================================================================
    
    def generate_positive_examples(self, original_answer, n=2):
        """Generate spatial-aware positive examples using a 4-strategy pipeline.
        
        Strategy Pipeline:
        1. Combined Current Approaches (spatial synonyms + structure + clock formats)
        2. LLM Paraphrasing (to be implemented)
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

        
        # Strategy 2: LLM Paraphrasing
        strategy2_positives = self._strategy2_llm_paraphrasing(original_answer)
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
    
    def _strategy2_llm_paraphrasing(self, original_answer):
        """Strategy 2: Generate paraphrases using LLM with UAV navigation context.
        
        Uses OpenAI API to generate contextually appropriate paraphrases that:
        - Preserve spatial semantics and navigation accuracy
        - Maintain UAV/drone perspective language
        - Generate diverse linguistic variations
        """
        positives = []
        
        # Use FLAN-T5 for high-quality paraphrasing
        positives = self._generate_flan_t5_paraphrases(original_answer)
        
        # Mark all as Strategy 2 outputs
        for pos in positives:
            pos["strategy"] = "strategy2_llm_paraphrasing"
        
        return positives
    
    def _generate_flan_t5_paraphrases(self, original_answer):
        """Generate paraphrases using FLAN-T5 with UAV navigation prompting."""
        positives = []
        
        # Simplified instruction format that works better with FLAN-T5
        instruction_prompt = f"Paraphrase this UAV navigation instruction: {original_answer}"

        try:
            # Tokenize input
            inputs = self.flan_t5_tokenizer(
                instruction_prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=256
            ).to(self.device)
            
            # Generate paraphrases with corrected parameters
            with torch.no_grad():
                outputs = self.flan_t5_model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    num_return_sequences=3,  # Generate multiple variations
                    do_sample=True,          # Enable sampling
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.flan_t5_tokenizer.pad_token_id or self.flan_t5_tokenizer.eos_token_id
                )
            
            # Decode all outputs
            for output in outputs:
                generated_text = self.flan_t5_tokenizer.decode(output, skip_special_tokens=True)
                
                # Remove the input prompt from the output
                if instruction_prompt in generated_text:
                    paraphrase = generated_text.replace(instruction_prompt, "").strip()
                else:
                    paraphrase = generated_text.strip()
                
                if paraphrase and paraphrase != original_answer and len(paraphrase.split()) > 3:
                    similarity = self.calculate_similarity(original_answer, paraphrase)
                    
                    # Accept paraphrases with reasonable semantic similarity
                    if 0.6 <= similarity <= 0.95:
                        positives.append({
                            "text": paraphrase,
                            "similarity": similarity,
                            "type": "flan_t5_paraphrase"
                        })
            
            # If no good paraphrases, try alternative prompts
            if len(positives) < 2:
                positives.extend(self._generate_flan_t5_variants(original_answer))
                
        except Exception as e:
            self.logger.error(f"FLAN-T5 generation error: {e}")
            # Fall back to simple rule-based method
            positives = self._generate_simple_fallback(original_answer)
        
        return positives
    
    def _generate_flan_t5_variants(self, original_answer):
        """Generate additional variants using different FLAN-T5 prompting strategies."""
        positives = []
        
        # Simpler, more effective prompting strategies
        prompt_variants = [
            f"Rewrite: {original_answer}",
            f"Rephrase: {original_answer}",
            f"Say differently: {original_answer}"
        ]
        
        for prompt in prompt_variants:
            try:
                inputs = self.flan_t5_tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.flan_t5_model.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.6,
                        top_p=0.8,
                        pad_token_id=self.flan_t5_tokenizer.pad_token_id or self.flan_t5_tokenizer.eos_token_id
                    )
                
                generated_text = self.flan_t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the generated text
                if prompt in generated_text:
                    clean_text = generated_text.replace(prompt, "").strip()
                else:
                    clean_text = generated_text.strip()
                
                if clean_text and clean_text != original_answer and len(clean_text.split()) > 3:
                    similarity = self.calculate_similarity(original_answer, clean_text)
                    
                    if 0.5 <= similarity <= 0.95:
                        positives.append({
                            "text": clean_text,
                            "similarity": similarity,
                            "type": "flan_t5_variant"
                        })
                        
            except Exception as e:
                self.logger.warning(f"FLAN-T5 variant generation failed: {e}")
                continue
        
        return positives
    
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
                        "type": "simple_variation"
                    })
        
        return positives
    
    # =================================================================
    # NEGATIVE GENERATION - Enhanced Spatial Strategies
    # =================================================================
    
    def generate_negative_examples(self, original_answer, n=3):
        """Generate enhanced spatial negatives."""
        negatives = []
        extracted_info = self._extract_navigation_info(original_answer)
        
        # Strategy 1: Clock direction shifting (UAV-critical)
        if extracted_info.get("clock_directions"):
            clock_negatives = self._generate_clock_shift_negatives(original_answer, extracted_info["clock_directions"])
            negatives.extend(clock_negatives)
        
        # Strategy 2: Enhanced direction reversal
        if extracted_info["directions"]:
            direction_negatives = self._generate_enhanced_direction_negatives(original_answer, extracted_info["directions"])
            negatives.extend(direction_negatives)
            
        # Strategy 3: Contextual landmark substitution
        if extracted_info["landmarks"]:
            landmark_negatives = self._generate_contextual_landmark_negatives(original_answer, extracted_info["landmarks"])
            negatives.extend(landmark_negatives)
        
        # Strategy 4: Spatial relation perturbation
        if extracted_info["spatial_relations"]:
            spatial_negatives = self._generate_spatial_relation_negatives(original_answer, extracted_info["spatial_relations"])
            negatives.extend(spatial_negatives)
        
        # Filter and rank
        quality_negatives = self._filter_and_rank_negatives(negatives, original_answer)
        
        return quality_negatives[:n]
    
    def _generate_clock_shift_negatives(self, original_answer, clock_directions):
        """Generate negatives by shifting clock directions (UAV-critical)."""
        negatives = []
        
        for clock_dir in clock_directions:
            try:
                hour_match = re.search(r'(\d+)\s*o\'?clock', clock_dir.lower())
                if hour_match:
                    hour = int(hour_match.group(1))
                    
                    # Strategic shifts: 90°, 180°, 270°
                    shifts = [3, 6, 9]  # hours
                    
                    for shift in shifts:
                        new_hour = ((hour - 1 + shift) % 12) + 1
                        new_clock = f"{new_hour} o'clock"
                        
                        negative_text = original_answer.replace(clock_dir, new_clock)
                        
                        if negative_text != original_answer:
                            similarity = self.calculate_similarity(original_answer, negative_text)
                            
                            if 0.85 <= similarity <= 0.99:
                                negatives.append({
                                    "text": negative_text,
                                    "similarity": similarity,
                                    "type": "clock_shift",
                                    "shift_degrees": shift * 30,
                                    "original_hour": hour,
                                    "new_hour": new_hour
                                })
                                
            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Error processing clock direction {clock_dir}: {e}")
                continue
        
        return negatives
    
    def _generate_enhanced_direction_negatives(self, original_answer, directions):
        """Generate enhanced direction negatives."""
        negatives = []
        
        # Enhanced opposites
        enhanced_opposites = {
            "north": ["south", "southeast", "southwest"],
            "south": ["north", "northeast", "northwest"], 
            "east": ["west", "northwest", "southwest"],
            "west": ["east", "northeast", "southeast"],
            "left": ["right"],
            "right": ["left"],
            "forward": ["backward", "behind"]
        }
        
        for direction in directions:
            direction_lower = direction.lower()
            
            if direction_lower in enhanced_opposites:
                alternatives = enhanced_opposites[direction_lower]
                
                for alt_direction in alternatives:
                    negative_text = original_answer.replace(direction, alt_direction)
                    
                    if negative_text != original_answer:
                        similarity = self.calculate_similarity(original_answer, negative_text)
                        
                        if 0.8 <= similarity <= 0.95:
                            negatives.append({
                                "text": negative_text,
                                "similarity": similarity,
                                "type": "enhanced_direction_reversal",
                                "original_direction": direction,
                                "new_direction": alt_direction
                            })
        
        return negatives
    
    def _generate_contextual_landmark_negatives(self, original_answer, landmarks):
        """Generate landmark negatives based on dataset frequency."""
        negatives = []
        
        # Dataset-driven landmark substitutions
        landmark_substitutions = {
            "building": ["structure", "house", "area", "facility"],
            "road": ["highway", "parking", "field"],
            "parking": ["road", "field", "area"],
            "field": ["area", "parking", "road"],
            "area": ["field", "section", "building"],
            "house": ["building", "structure"],
            "tree": ["building", "structure"]
        }
        
        for landmark in landmarks:
            landmark_lower = landmark.lower()
            
            if landmark_lower in landmark_substitutions:
                alternatives = landmark_substitutions[landmark_lower]
                
                for alt_landmark in alternatives:
                    negative_text = original_answer.replace(landmark, alt_landmark)
                    
                    if negative_text != original_answer:
                        similarity = self.calculate_similarity(original_answer, negative_text)
                        
                        if 0.85 <= similarity <= 0.99:
                            negatives.append({
                                "text": negative_text,
                                "similarity": similarity,
                                "type": "contextual_landmark",
                                "original_landmark": landmark,
                                "new_landmark": alt_landmark
                            })
        
        return negatives
    
    def _generate_spatial_relation_negatives(self, original_answer, spatial_relations):
        """Generate spatial relation negatives."""
        negatives = []
        
        spatial_opposites = {
            "above": ["below", "under"],
            "below": ["above", "over"],
            "over": ["below", "under"],
            "near": ["far from", "distant from"],
            "in front of": ["behind"],
            "behind": ["in front of"],
            "next to": ["far from", "across from"]
        }
        
        for relation in spatial_relations:
            relation_lower = relation.lower()
            
            if relation_lower in spatial_opposites:
                alternatives = spatial_opposites[relation_lower]
                
                for alt_relation in alternatives:
                    negative_text = original_answer.replace(relation, alt_relation)
                    
                    if negative_text != original_answer:
                        similarity = self.calculate_similarity(original_answer, negative_text)
                        
                        if 0.8 <= similarity <= 0.95:
                            negatives.append({
                                "text": negative_text,
                                "similarity": similarity,
                                "type": "spatial_relation_perturbation",
                                "original_relation": relation,
                                "new_relation": alt_relation
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