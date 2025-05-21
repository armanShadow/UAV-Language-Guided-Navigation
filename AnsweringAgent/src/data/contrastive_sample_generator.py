import torch
import torch.nn.functional as F
import json
import random
import numpy as np
import os
import tqdm
import logging
from transformers import AutoTokenizer, AutoModel, pipeline

class ContrastiveSampleGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 paraphrase_model_name="prithivida/parrot_paraphraser_on_T5",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the contrastive sample generator with a sentence embedding model."""
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loading sentence embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Initialize paraphrasing model
        self.logger.info(f"Loading paraphrasing model: {paraphrase_model_name}")
        try:
            # Using device_map='auto' for better memory management
            if torch.cuda.is_available() and device == "cuda":
                self.logger.info("Using CUDA for paraphraser with device_map='auto'")
                self.paraphraser = pipeline(
                    "text2text-generation", 
                    model=paraphrase_model_name,
                    device_map="auto"  # Automatically choose best device layout
                )
            else:
                self.logger.info("Using CPU for paraphraser")
                self.paraphraser = pipeline(
                    "text2text-generation", 
                    model=paraphrase_model_name,
                    device=-1  # Use CPU
                )
            self.has_paraphraser = True
            self.logger.info("Successfully loaded paraphrasing model")
            
            # Try to load T5 model for generating negatives
            try:
                self.logger.info("Loading T5 model for negative example generation")
                negative_model_name = "t5-base"  # T5 base model
                if torch.cuda.is_available() and device == "cuda":
                    self.negative_generator = pipeline(
                        "text2text-generation",
                        model=negative_model_name,
                        device_map="auto"
                    )
                else:
                    self.negative_generator = pipeline(
                        "text2text-generation", 
                        model=negative_model_name,
                        device=-1
                    )
                self.has_negative_generator = True
                self.logger.info("Successfully loaded T5 model for negatives")
            except Exception as e:
                self.logger.warning(f"Could not load T5 model for negatives: {str(e)}")
                self.has_negative_generator = False
                
        except Exception as e:
            self.logger.warning(f"Could not load paraphrasing model: {str(e)}")
            self.logger.warning("Will fall back to template-based paraphrasing")
            self.has_paraphraser = False
            self.has_negative_generator = False
        
        # Enhanced navigation-specific terminology
        self.direction_terms = [
            "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
            "left", "right", "forward", "backward", "ahead", "behind", "clockwise",
            "counterclockwise", "turn", "rotate", "face", "head", "proceed", "continue", "go", "move",
            "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock", "5 o'clock", "6 o'clock", 
            "7 o'clock", "8 o'clock", "9 o'clock", "10 o'clock", "11 o'clock", "12 o'clock"
        ]
        
        self.landmark_terms = [
            "building", "house", "structure", "tower", "road", "bridge", "highway", 
            "hill", "mountain", "tree", "field", "river", "lake", "landfill",
            "rooftop", "parking lot", "intersection", "corner", "entrance", "exit",
            "area", "zone", "region", "block", "neighborhood", "complex", "facility",
            "stadium", "park", "garden", "campus", "fence", "wall", "path", "trail"
        ]
        
        self.color_terms = [
            "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
            "brown", "purple", "orange", "beige", "pink", "tan", "golden", "silver",
            "dark", "light", "bright", "dull", "vibrant", "colorful"
        ]
        
        self.shape_terms = [
            "square", "rectangular", "round", "circular", "oval", "triangular", 
            "dome", "L-shaped", "U-shaped", "flat", "tall", "short", "wide", "narrow",
            "curved", "straight", "zigzag", "angled", "sloped"
        ]
        
        self.spatial_relation_terms = [
            "above", "below", "under", "over", "on top of", "beneath", "adjacent to",
            "next to", "beside", "alongside", "in front of", "behind", "between",
            "among", "surrounding", "inside", "outside", "within", "near", "far",
            "close to", "distant from", "across from", "opposite to", "parallel to",
            "perpendicular to", "diagonal from", "at the edge of", "in the center of",
            "in the middle of", "at the corner of", "at the intersection of"
        ]
        
        self.size_terms = [
            "large", "small", "big", "tiny", "huge", "massive", "enormous", "giant",
            "little", "medium", "medium-sized", "compact", "expansive", "long", "short",
            "wide", "narrow", "thick", "thin"
        ]
        
    def generate_embedding(self, text):
        """Generate embedding for a text using the sentence embedding model."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling to get sentence embedding
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
    
    def generate_lm_paraphrases(self, original_answer, n=2):
        """
        Generate paraphrases using a language model.
        
        Args:
            original_answer: Original answer to paraphrase
            n: Number of paraphrases to generate
            
        Returns:
            List of paraphrases with similarity scores
        """
        paraphrases = []
        
        # Skip if answer is too short or if paraphraser is not available
        if len(original_answer.split()) < 3 or not self.has_paraphraser:
            return self.generate_template_paraphrases(original_answer, n)
        
        try:
            # Format input for the T5 paraphraser (parrot model expects a 'paraphrase: ' prefix)
            paraphrase_input = f"paraphrase: {original_answer}"
            
            # Use the paraphrasing model to generate multiple paraphrases
            self.logger.info(f"Generating paraphrases for text of length {len(original_answer.split())} words")
            model_outputs = self.paraphraser(
                paraphrase_input,
                max_length=min(128, len(original_answer.split()) * 2),  # Reasonable max length 
                num_return_sequences=n+2,  # Generate extra in case some are filtered
                num_beams=5,  # Increased beam count for better diversity with T5
                temperature=1.0,  # Adjusted for T5 model
                do_sample=True  # Enable sampling for diversity
            )
            
            # Process the generated paraphrases
            for output in model_outputs:
                paraphrase = output['generated_text']
                
                # Skip if paraphrase is empty or too similar to original
                if not paraphrase or paraphrase.strip() == original_answer.strip():
                    continue
                
                # Calculate similarity
                similarity = self.calculate_similarity(original_answer, paraphrase)
                
                # Only keep if reasonably similar (to avoid completely unrelated paraphrases)
                if similarity > 0.6:
                    paraphrases.append({
                        "text": paraphrase,
                        "similarity": similarity,
                        "type": "lm_paraphrase"
                    })
                    
                    if len(paraphrases) >= n:
                        break
        except Exception as e:
            self.logger.warning(f"Error generating LM paraphrases: {str(e)}")
            import traceback
            self.logger.debug(f"Exception details: {traceback.format_exc()}")
            self.logger.warning("Falling back to template-based paraphrasing")
            return self.generate_template_paraphrases(original_answer, n)
        
        # If we couldn't generate enough valid paraphrases, add template-based ones
        if len(paraphrases) < n:
            template_paraphrases = self.generate_template_paraphrases(original_answer, n - len(paraphrases))
            paraphrases.extend(template_paraphrases)
        
        return paraphrases[:n]
    
    def generate_template_paraphrases(self, original_answer, n=2):
        """Generate positive examples (paraphrases) using templates based on navigation dialog patterns."""
        positives = []
        
        # Extract key navigation information using the more comprehensive method
        extracted_info = self._extract_navigation_info(original_answer)
        directions = extracted_info["directions"]
        landmarks = extracted_info["landmarks"]
        colors = extracted_info["colors"] 
        shapes = extracted_info["shapes"]
        spatial_relations = extracted_info["spatial_relations"]
        sizes = extracted_info["sizes"]
        
        # Create a more diverse set of UAV navigation-specific templates
        templates = [
            # Direction-focused templates
            "The destination can be found {direction}. It's {shape} and {color}.",
            "If you look {direction}, you'll see the destination which is {shape} with {color} color.",
            "Your destination is {direction} from your position. Look for a {color} {shape} structure.",
            "Head {direction} to reach your destination. It's {color} and {shape}.",
            "{direction} is where you'll find your destination. It's a {color} {shape} building.",
            "Fly {direction} to find your target. The {color} {shape} structure is your destination.",
            "Move your drone {direction} and look for a {color} {shape} landmark.",
            
            # Landmark-focused templates
            "Look for a {color} {landmark} {direction} from your current position.",
            "Your target is the {size} {color} {landmark} that you can see {direction}.",
            "The {landmark} with the {color} {feature} is your destination, located {direction}.",
            "Fly {direction} toward the {color} {landmark}.",
            "Navigate to the {size} {landmark} {spatial_relation} the area.",
            "The {color} {landmark} is your destination. It's {direction} from where you are now.",
            "Your destination is the {size} {landmark} with {color} features that you can see {direction}.",
            
            # Clock direction templates
            "Your destination is at {clock_direction}, it's the {color} {shape} {landmark}.",
            "If you turn to {clock_direction}, you'll see a {size} {color} {landmark}.",
            "The {color} {landmark} at your {clock_direction} is the destination.",
            "Look towards your {clock_direction} for a {color} {shape} {landmark}.",
            "Rotate to {clock_direction} and you'll find the {color} {landmark}.",
            "The {size} {landmark} at {clock_direction} position is your target.",
            "If you orient yourself to {clock_direction}, you'll see your destination - a {color} {shape} structure.",
            
            # Combined attribute templates
            "The {size} {landmark} with {color} {feature} {spatial_relation} is your target.",
            "Head to the {size} {color} {shape} structure {direction} of your position.",
            "Your goal is the {color} {landmark} with {shape} features {spatial_relation}.",
            "Navigate to the {color} {landmark} that appears {spatial_relation} your view area.",
            "Fly towards the {size} {landmark} with {color} exteriors located {direction}.",
            "Your destination is a {size} {color} structure with {shape} architecture {spatial_relation}.",
            "Move the drone to the {color} {landmark} that has {shape} features {direction} of your current position."
        ]
        
        # UAV-specific action verbs to create more variety
        action_verbs = [
            "fly", "navigate", "move", "head", "proceed", "travel", "hover near", 
            "go toward", "approach", "make your way to", "steer toward", "direct yourself to"
        ]
        
        # Clock direction references
        clock_directions = [
            "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock", "5 o'clock", "6 o'clock",
            "7 o'clock", "8 o'clock", "9 o'clock", "10 o'clock", "11 o'clock", "12 o'clock"
        ]
        
        # Features that can describe landmarks
        features = ["roof", "exterior", "walls", "edge", "perimeter", "facade", "structure", "design"]
        
        # Determine which templates we can use based on available information
        usable_templates = []
        for template in templates:
            can_use = True
            
            # Check if template requires direction
            if "{direction}" in template and (not directions or len(directions) == 0):
                can_use = False
                
            # Check if template requires landmark
            if "{landmark}" in template and (not landmarks or len(landmarks) == 0):
                can_use = False
                
            # Check if template requires color
            if "{color}" in template and (not colors or len(colors) == 0):
                can_use = False
                
            # Check if template requires shape
            if "{shape}" in template and (not shapes or len(shapes) == 0):
                can_use = False
                
            # Check if template requires spatial relation
            if "{spatial_relation}" in template and (not spatial_relations or len(spatial_relations) == 0):
                can_use = False
                
            # Check if template requires size
            if "{size}" in template and (not sizes or len(sizes) == 0):
                can_use = False
                
            # Check if template requires clock direction
            if "{clock_direction}" in template:
                # For clock directions, we can generate them even if not present
                if not directions or len(directions) == 0:
                    can_use = False
                    
            if can_use:
                usable_templates.append(template)
        
        # Add generic templates that don't require specific extracted elements
        generic_templates = [
            "Your destination is located {generic_direction}.",
            "Head {generic_direction} to find your target.",
            "The target is {generic_direction} from your current position.",
            "Navigate {generic_direction} to reach your destination.",
            "{generic_action} {generic_direction} to arrive at your destination.",
            "Your target can be found if you go {generic_direction}.",
            "To reach the destination, {generic_action} {generic_direction}."
        ]
        usable_templates.extend(generic_templates)
        
        # Generate paraphrases using templates
        paraphrase_count = 0
        max_attempts = min(30, len(usable_templates) * 3)  # Limit attempts to avoid infinite loops
        attempts = 0
        
        while paraphrase_count < n and attempts < max_attempts:
            attempts += 1
            
            if not usable_templates:
                break
                
            template = random.choice(usable_templates)
            
            try:
                # Fill in the template
                paraphrase = template
                
                # Replace direction placeholder
                if "{direction}" in paraphrase:
                    if directions and len(directions) > 0:
                        paraphrase = paraphrase.replace("{direction}", random.choice(directions))
                        
                # Replace landmark placeholder
                if "{landmark}" in paraphrase:
                    if landmarks and len(landmarks) > 0:
                        paraphrase = paraphrase.replace("{landmark}", random.choice(landmarks))
                    else:
                        paraphrase = paraphrase.replace("{landmark}", random.choice(self.landmark_terms))
                        
                # Replace color placeholder
                if "{color}" in paraphrase:
                    if colors and len(colors) > 0:
                        paraphrase = paraphrase.replace("{color}", random.choice(colors))
                    else:
                        paraphrase = paraphrase.replace("{color}", random.choice(self.color_terms))
                        
                # Replace shape placeholder
                if "{shape}" in paraphrase:
                    if shapes and len(shapes) > 0:
                        paraphrase = paraphrase.replace("{shape}", random.choice(shapes))
                    else:
                        paraphrase = paraphrase.replace("{shape}", random.choice(self.shape_terms))
                        
                # Replace spatial relation placeholder
                if "{spatial_relation}" in paraphrase:
                    if spatial_relations and len(spatial_relations) > 0:
                        paraphrase = paraphrase.replace("{spatial_relation}", random.choice(spatial_relations))
                    else:
                        paraphrase = paraphrase.replace("{spatial_relation}", random.choice(self.spatial_relation_terms))
                        
                # Replace size placeholder
                if "{size}" in paraphrase:
                    if sizes and len(sizes) > 0:
                        paraphrase = paraphrase.replace("{size}", random.choice(sizes))
                    else:
                        paraphrase = paraphrase.replace("{size}", random.choice(self.size_terms))
                        
                # Replace clock direction placeholder
                if "{clock_direction}" in paraphrase:
                    paraphrase = paraphrase.replace("{clock_direction}", random.choice(clock_directions))
                    
                # Replace feature placeholder
                if "{feature}" in paraphrase:
                    paraphrase = paraphrase.replace("{feature}", random.choice(features))
                    
                # Replace generic placeholders
                if "{generic_direction}" in paraphrase:
                    generic_direction = random.choice(self.direction_terms)
                    paraphrase = paraphrase.replace("{generic_direction}", generic_direction)
                    
                if "{generic_action}" in paraphrase:
                    generic_action = random.choice(action_verbs)
                    paraphrase = paraphrase.replace("{generic_action}", generic_action)
                    
                # Calculate similarity with original answer
                similarity = self.calculate_similarity(original_answer, paraphrase)
                
                # Only include if similar enough to original but not identical
                if similarity > 0.5 and paraphrase != original_answer:
                    # Check if this paraphrase is significantly different from previously generated ones
                    is_unique = True
                    for existing in positives:
                        existing_similarity = self.calculate_similarity(existing["text"], paraphrase)
                        if existing_similarity > 0.85:  # Very similar to existing paraphrase
                            is_unique = False
                            break
                            
                    if is_unique:
                        positives.append({
                            "text": paraphrase,
                            "similarity": similarity,
                            "type": "template_paraphrase"
                        })
                        paraphrase_count += 1
                        
            except Exception as e:
                self.logger.warning(f"Error generating template paraphrase: {str(e)}")
                continue
            
        # If we couldn't generate enough paraphrases, add simple variations
        if len(positives) < n:
            simple_variations = [
                f"I believe {original_answer}",
                f"{original_answer} That's your destination.",
                f"Based on what I can see, {original_answer.lower()}",
                f"From the drone's view, {original_answer.lower()}"
            ]
            
            for variation in simple_variations:
                if len(positives) >= n:
                    break
                    
                similarity = self.calculate_similarity(original_answer, variation)
                if similarity > 0.7 and variation != original_answer:
                    positives.append({
                        "text": variation,
                        "similarity": similarity,
                        "type": "simple_variation"
                    })
        
        return positives[:n]
    
    def generate_positive_examples(self, original_answer, n=3):
        """
        Generate positive examples (paraphrases) using a hybrid approach.
        
        Args:
            original_answer: The original navigation instruction to paraphrase
            n: Number of positive examples to generate (default: 3)
            
        Returns:
            List of positive examples with similarity scores and type labels
        """
        self.logger.info(f"Generating {n} positive examples using hybrid approach")
        positives = []
        
        # Skip short answers - they're difficult to paraphrase meaningfully
        if len(original_answer.split()) < 3:
            self.logger.warning(f"Answer too short ({len(original_answer.split())} words), using simple variations")
            simple_variations = [
                f"I believe {original_answer}",
                f"{original_answer} That's your destination.",
                f"Based on what I can see, {original_answer.lower()}"
            ]
            
            for variation in simple_variations[:n]:
                similarity = self.calculate_similarity(original_answer, variation)
                positives.append({
                    "text": variation,
                    "similarity": similarity,
                    "type": "simple_variation"
                })
            
            return positives[:n]
        
        # Hybrid approach:
        # 1. Try to generate one LM-based paraphrase (higher quality but more resource-intensive)
        # 2. Generate the rest using template-based approach (more diverse and lightweight)
        
        try:
            # First attempt to generate a high-quality LM-based paraphrase
            if self.has_paraphraser:
                self.logger.info("Generating LM-based paraphrase")
                lm_paraphrases = []
                
                try:
                    # Use a prompt format for T5 paraphraser
                    paraphrase_input = f"paraphrase: {original_answer}"
                    
                    # Generate a paraphrase
                    outputs = self.paraphraser(
                        paraphrase_input,
                        max_length=min(128, len(original_answer.split()) * 2),
                        num_return_sequences=2,  # Try a couple to increase chances of success
                        temperature=1.0,
                        do_sample=True
                    )
                    
                    for output in outputs:
                        paraphrase = output['generated_text'].strip()
                        
                        # Skip if paraphrase is empty or too similar to original
                        if not paraphrase or paraphrase == original_answer:
                            continue
                        
                        # Calculate similarity
                        similarity = self.calculate_similarity(original_answer, paraphrase)
                        
                        # Only keep if reasonably similar (to avoid completely unrelated paraphrases)
                        if similarity > 0.7:
                            lm_paraphrases.append({
                                "text": paraphrase,
                                "similarity": similarity,
                                "type": "lm_paraphrase"
                            })
                    
                    # Add best LM paraphrase to our collection
                    if lm_paraphrases:
                        # Sort by descending similarity to find the best one
                        lm_paraphrases.sort(key=lambda x: x["similarity"], reverse=True)
                        positives.append(lm_paraphrases[0])
                        self.logger.info(f"Successfully generated LM paraphrase (similarity: {lm_paraphrases[0]['similarity']:.3f})")
                    else:
                        self.logger.warning("LM paraphraser didn't generate valid paraphrases")
                
                except Exception as e:
                    self.logger.warning(f"Error generating LM paraphrase: {str(e)}")
                    import traceback
                    self.logger.debug(f"LM paraphrase exception details: {traceback.format_exc()}")
            else:
                self.logger.warning("LM paraphraser not available, using only template approach")
        
        except Exception as e:
            self.logger.warning(f"Error in LM-based phase: {str(e)}")
        
        # Generate the remaining examples using template-based approach
        remaining = n - len(positives)
        if remaining > 0:
            self.logger.info(f"Generating {remaining} template-based paraphrases")
            template_paraphrases = self.generate_template_paraphrases(original_answer, remaining)
            positives.extend(template_paraphrases)
        
        # Check if we have enough examples
        if len(positives) < n:
            self.logger.warning(f"Generated only {len(positives)}/{n} positive examples, adding simple variations")
            
            # Add simple variations to reach the desired count
            simple_variations = [
                f"I believe {original_answer}",
                f"{original_answer} That's your destination.",
                f"Based on what I can see, {original_answer.lower()}",
                f"From the drone's view, {original_answer.lower()}"
            ]
            
            for variation in simple_variations:
                if len(positives) >= n:
                    break
                    
                # Skip if we already have something very similar
                is_unique = True
                for existing in positives:
                    if self.calculate_similarity(existing["text"], variation) > 0.9:
                        is_unique = False
                        break
                
                if is_unique:
                    similarity = self.calculate_similarity(original_answer, variation)
                    positives.append({
                        "text": variation,
                        "similarity": similarity,
                        "type": "simple_variation"
                    })
        
        # Return the requested number of positives
        return positives[:n]
    
    def generate_negative_examples(self, original_answer, n=3):
        """Generate diverse negative examples using multiple strategies.
        
        Uses a mixed approach with LM-based and rule-based negative examples.
        """
        # If n is less than 3, adjust the counts accordingly
        lm_count = 1 if n >= 1 else 0
        rule_count = n - lm_count
        
        # Get LM-based negative examples if paraphraser is available
        lm_negatives = []
        if lm_count > 0 and self.has_paraphraser:
            try:
                lm_negatives = self.generate_lm_negatives(original_answer, lm_count)
            except Exception as e:
                self.logger.warning(f"Error generating LM negatives: {str(e)}")
                # If LM generation fails, use rule-based for all
                rule_count = n
        
        # Get rule-based negative examples
        rule_negatives = self.generate_rule_based_negatives(original_answer, rule_count)
        
        # Combine the results
        all_negatives = lm_negatives + rule_negatives
        
        return all_negatives[:n]
    
    def generate_lm_negatives(self, original_answer, n=1):
        """
        Generate negative examples using a language model.
        
        Args:
            original_answer: Original answer to negate
            n: Number of negatives to generate
            
        Returns:
            List of negative examples with similarity scores
        """
        negatives = []
        
        # Skip if answer is too short
        if len(original_answer.split()) < 3:
            return self._generate_random_negative(original_answer)
        
        try:
            if self.has_negative_generator:
                self.logger.info(f"Generating T5-based negatives for text of length {len(original_answer.split())} words")
                
                # Extract navigation information to create better contradiction prompts
                nav_info = self._extract_navigation_info(original_answer)
                
                # Create specialized prompts for T5
                prompts = [
                    f"contradict: {original_answer}",
                    f"opposite: {original_answer}",
                    f"negate: {original_answer}"
                ]
                
                # If we have specific direction/landmark information, add more targeted prompts
                if nav_info["directions"]:
                    direction_terms = ", ".join(nav_info["directions"][:2])  # Use first couple directions
                    prompts.append(f"change direction {direction_terms} in: {original_answer}")
                
                if nav_info["landmarks"]:
                    landmark_terms = ", ".join(nav_info["landmarks"][:2])  # Use first couple landmarks
                    prompts.append(f"change landmark {landmark_terms} in: {original_answer}")
                    
                if nav_info["spatial_relations"]:
                    spatial_terms = ", ".join(nav_info["spatial_relations"][:2])
                    prompts.append(f"reverse relation {spatial_terms} in: {original_answer}")
                
                # Choose some unique prompts to try
                selected_prompts = list(set(prompts))[:min(3, len(prompts))]
                
                for prompt in selected_prompts:
                    if len(negatives) >= n:
                        break
                        
                    try:
                        # Generate contradiction using T5
                        outputs = self.negative_generator(
                            prompt,
                            max_length=min(128, len(original_answer.split()) * 2),
                            do_sample=True,
                            temperature=1.0,
                            num_return_sequences=1
                        )
                        
                        negative_text = outputs[0]['generated_text'].strip()
                        
                        # Ensure the negative is not empty or too similar to original
                        if not negative_text or negative_text == original_answer:
                            continue
                        
                        # Calculate similarity
                        similarity = self.calculate_similarity(original_answer, negative_text)
                        
                        # Only keep if reasonably different but not entirely unrelated
                        # Ideal range: similar enough to be challenging, different enough to be negative
                        if 0.3 <= similarity <= 0.7:
                            negatives.append({
                                "text": negative_text,
                                "similarity": similarity,
                                "type": "lm_negative"
                            })
                    
                    except Exception as e:
                        self.logger.warning(f"Error generating T5 negative with prompt '{prompt}': {str(e)}")
                        continue
            
            # If we don't have a negative generator or couldn't generate any valid negatives
            if not self.has_negative_generator or not negatives:
                self.logger.warning("Falling back to rule-based negatives (no valid LM negatives generated)")
                return self.generate_rule_based_negatives(original_answer, n)
        
        except Exception as e:
            self.logger.warning(f"Error in LM negative generation: {str(e)}")
            import traceback
            self.logger.debug(f"Exception details: {traceback.format_exc()}")
            self.logger.warning("Falling back to rule-based negatives")
            return self.generate_rule_based_negatives(original_answer, n)
        
        # If we couldn't generate enough valid negatives, add rule-based ones
        if len(negatives) < n:
            rule_based_negatives = self.generate_rule_based_negatives(original_answer, n - len(negatives))
            negatives.extend(rule_based_negatives)
        
        return negatives[:n]
    
    def generate_rule_based_negatives(self, original_answer, n=2):
        """Generate negative examples using rule-based methods focused on navigation elements."""
        negatives = []
        
        # Extract navigation elements from the original answer
        extracted_info = self._extract_navigation_info(original_answer)
        directions = extracted_info["directions"]
        landmarks = extracted_info["landmarks"]
        colors = extracted_info["colors"]
        shapes = extracted_info["shapes"]
        spatial_relations = extracted_info["spatial_relations"]
        sizes = extracted_info["sizes"]
        
        # Prioritize different strategies based on available information
        strategies = []
        
        # Strategy 1: Direction reversal/substitution
        if directions and len(directions) > 0:
            strategies.append(self._generate_direction_reversal_negatives)
        
        # Strategy 2: Landmark substitution  
        if landmarks and len(landmarks) > 0:
            strategies.append(self._generate_landmark_substitution_negatives)
        
        # Strategy 3: Color negation
        if colors and len(colors) > 0:
            strategies.append(self._generate_color_substitution_negatives)
        
        # Strategy 4: Shape negation
        if shapes and len(shapes) > 0:
            strategies.append(self._generate_shape_substitution_negatives)
        
        # Strategy 5: Spatial relation reversal
        if spatial_relations and len(spatial_relations) > 0:
            strategies.append(self._generate_spatial_relation_negatives)
        
        # Strategy 6: Size substitution
        if sizes and len(sizes) > 0:
            strategies.append(self._generate_size_substitution_negatives)
        
        # Strategy 7: Semantic frame alterations (combined contradictions)
        if (directions or landmarks or colors or shapes) and len(strategies) >= 2:
            strategies.append(self._generate_semantic_frame_negatives)
        
        # Strategy 8: Generic opposition (fallback)
        strategies.append(self._generate_generic_opposition_negatives)
        
        # Strategy 9: Random negative (last resort)
        strategies.append(self._generate_random_negative)
        
        # Ensure we have enough strategies
        while len(strategies) < n:
            strategies.append(random.choice(strategies))
        
        # Use different strategies to generate varied negative examples
        random.shuffle(strategies)  # Randomize strategy order
        
        # Keep track of generated texts to avoid duplicates
        generated_texts = set()
        strategy_idx = 0
        max_attempts = 10  # Avoid infinite loop if strategies keep failing
        
        while len(negatives) < n and strategy_idx < len(strategies) and max_attempts > 0:
            strategy = strategies[strategy_idx]
            try:
                # Call the strategy with the original answer and extracted elements
                if strategy == self._generate_semantic_frame_negatives:
                    new_negatives = strategy(original_answer, directions, landmarks, colors, shapes, spatial_relations, sizes)
                elif strategy == self._generate_landmark_substitution_negatives:
                    new_negatives = strategy(original_answer, landmarks)
                elif strategy == self._generate_direction_reversal_negatives:
                    new_negatives = strategy(original_answer, directions)
                elif strategy == self._generate_spatial_relation_negatives:
                    new_negatives = strategy(original_answer, spatial_relations)
                elif strategy == self._generate_color_substitution_negatives:
                    new_negatives = strategy(original_answer, colors)
                elif strategy == self._generate_shape_substitution_negatives:
                    new_negatives = strategy(original_answer, shapes)
                elif strategy == self._generate_size_substitution_negatives:
                    new_negatives = strategy(original_answer, sizes)
                else:
                    new_negatives = strategy(original_answer)
                
                # Add new negatives if they're not duplicates
                for neg in new_negatives:
                    text = neg["text"]
                    if text not in generated_texts and text != original_answer:
                        generated_texts.add(text)
                        negatives.append(neg)
                        
                        if len(negatives) >= n:
                            break
            except Exception as e:
                self.logger.warning(f"Error with negative generation strategy {strategy.__name__}: {str(e)}")
                max_attempts -= 1
            
            strategy_idx += 1
            
        # If we still don't have enough negatives, add random generic ones
        while len(negatives) < n:
            try:
                random_negative = self._generate_random_negative(original_answer)[0]
                if random_negative["text"] not in generated_texts and random_negative["text"] != original_answer:
                    generated_texts.add(random_negative["text"])
                    negatives.append(random_negative)
                else:
                    # If duplicate, try again with different content
                    random_negative["text"] = "The destination is at the opposite location from where you're looking."
                    negatives.append(random_negative)
            except Exception as e:
                self.logger.warning(f"Error generating random negative: {str(e)}")
                # Emergency fallback
                negatives.append({
                    "text": "Go in the complete opposite direction to find an entirely different building.",
                    "similarity": 0.3,
                    "type": "emergency_fallback"
                })
        
        return negatives[:n]
    
    def _generate_color_substitution_negatives(self, original_answer, colors):
        """Generate negatives by substituting colors with opposite/different colors."""
        negatives = []
        
        # Start with original text
        text = original_answer
        
        # Define color opposites and groups
        color_opposites = {
            "red": ["blue", "green"],
            "blue": ["red", "orange"],
            "green": ["red", "purple"],
            "yellow": ["blue", "purple"],
            "black": ["white", "light"],
            "white": ["black", "dark"],
            "gray": ["colorful", "vibrant"],
            "grey": ["colorful", "vibrant"],
            "brown": ["white", "blue"],
            "purple": ["yellow", "green"],
            "orange": ["blue", "green"],
            "beige": ["dark", "black"],
            "pink": ["green", "brown"],
            "tan": ["gray", "blue"],
            "golden": ["silver", "dark"],
            "silver": ["golden", "dark"],
            "dark": ["light", "bright"],
            "light": ["dark", "dim"],
            "bright": ["dull", "dark"],
            "dull": ["bright", "vibrant"],
            "vibrant": ["dull", "gray"]
        }
        
        # Try to replace each color with its opposite
        for color in colors:
            if color in color_opposites:
                opposite_colors = color_opposites[color]
                opposite_color = random.choice(opposite_colors)
                
                # Create a new text with the opposite color
                new_text = text.replace(color, opposite_color)
                
                # Only add if the text actually changed
                if new_text != text:
                    similarity = self.calculate_similarity(original_answer, new_text)
                    negatives.append({
                        "text": new_text,
                        "similarity": similarity,
                        "type": "color_negation"
                    })
                    
                    # Return early if we have at least one negative
                    if negatives:
                        break
        
        # If no color-specific negatives were generated, use a generic approach
        if not negatives:
            # Replace all detected colors with random different ones
            all_colors = list(self.color_terms)
            new_text = text
            
            for color in colors:
                different_colors = [c for c in all_colors if c != color]
                if different_colors:
                    different_color = random.choice(different_colors)
                    new_text = new_text.replace(color, different_color)
            
            # Only add if the text actually changed
            if new_text != text:
                similarity = self.calculate_similarity(original_answer, new_text)
                negatives.append({
                    "text": new_text,
                    "similarity": similarity,
                    "type": "color_substitution"
                })
        
        # If we still don't have any negatives, explicitly state a different color
        if not negatives and colors:
            first_color = colors[0]
            different_colors = [c for c in self.color_terms if c != first_color]
            if different_colors:
                different_color = random.choice(different_colors)
                
                # Create a more explicit contradiction about the color
                if "landmark" in original_answer.lower() or any(landmark in original_answer.lower() for landmark in self.landmark_terms):
                    new_text = f"The landmark is {different_color}, not {first_color}."
                else:
                    new_text = f"The destination is {different_color}, not {first_color}."
                    
                similarity = self.calculate_similarity(original_answer, new_text)
                negatives.append({
                    "text": new_text,
                    "similarity": similarity,
                    "type": "explicit_color_negation"
                })
        
        # Return what we have, or empty list if nothing worked
        return negatives
    
    def _generate_shape_substitution_negatives(self, original_answer, shapes):
        """Generate negatives by substituting shapes with different shapes."""
        negatives = []
        
        # Start with original text
        text = original_answer
        
        # Define shape opposites
        shape_opposites = {
            "square": ["round", "circular", "oval"],
            "rectangular": ["circular", "oval", "round"],
            "round": ["square", "rectangular", "triangular"],
            "circular": ["square", "rectangular", "triangular"],
            "oval": ["square", "rectangular", "triangular"],
            "triangular": ["square", "round", "oval"],
            "dome": ["flat", "square", "rectangular"],
            "L-shaped": ["straight", "rectangular", "square"],
            "U-shaped": ["straight", "rectangular", "triangular"],
            "flat": ["dome", "tall", "curved"],
            "tall": ["short", "flat", "wide"],
            "short": ["tall", "high", "long"],
            "wide": ["narrow", "thin", "tall"],
            "narrow": ["wide", "broad", "expansive"],
            "curved": ["straight", "angled", "flat"],
            "straight": ["curved", "zigzag", "angled"],
            "zigzag": ["straight", "curved", "flat"],
            "angled": ["curved", "straight", "flat"],
            "sloped": ["flat", "level", "straight"]
        }
        
        # Try to replace each shape with its opposite
        for shape in shapes:
            if shape in shape_opposites:
                opposite_shapes = shape_opposites[shape]
                opposite_shape = random.choice(opposite_shapes)
                
                # Create a new text with the opposite shape
                new_text = text.replace(shape, opposite_shape)
                
                # Only add if the text actually changed
                if new_text != text:
                    similarity = self.calculate_similarity(original_answer, new_text)
                    negatives.append({
                        "text": new_text,
                        "similarity": similarity,
                        "type": "shape_negation"
                    })
                    
                    # Return early if we have at least one negative
                    if negatives:
                        break
        
        # If no shape-specific negatives were generated, use a generic approach
        if not negatives:
            # Replace all detected shapes with random different ones
            all_shapes = list(self.shape_terms)
            new_text = text
            
            for shape in shapes:
                different_shapes = [s for s in all_shapes if s != shape]
                if different_shapes:
                    different_shape = random.choice(different_shapes)
                    new_text = new_text.replace(shape, different_shape)
            
            # Only add if the text actually changed
            if new_text != text:
                similarity = self.calculate_similarity(original_answer, new_text)
                negatives.append({
                    "text": new_text,
                    "similarity": similarity,
                    "type": "shape_substitution"
                })
        
        # If we still don't have any negatives, explicitly state a different shape
        if not negatives and shapes:
            first_shape = shapes[0]
            different_shapes = [s for s in self.shape_terms if s != first_shape]
            if different_shapes:
                different_shape = random.choice(different_shapes)
                
                # Create a more explicit contradiction about the shape
                if "landmark" in original_answer.lower() or any(landmark in original_answer.lower() for landmark in self.landmark_terms):
                    new_text = f"The landmark is {different_shape}, not {first_shape}."
                else:
                    new_text = f"The destination is {different_shape}, not {first_shape}."
                    
                similarity = self.calculate_similarity(original_answer, new_text)
                negatives.append({
                    "text": new_text,
                    "similarity": similarity,
                    "type": "explicit_shape_negation"
                })
        
        # Return what we have, or empty list if nothing worked
        return negatives
    
    def _generate_size_substitution_negatives(self, original_answer, sizes):
        """Generate negatives by substituting sizes with opposite sizes."""
        negatives = []
        
        # Start with original text
        text = original_answer
        
        # Define size opposites
        size_opposites = {
            "large": ["small", "tiny", "little"],
            "small": ["large", "big", "huge"],
            "big": ["small", "tiny", "little"],
            "tiny": ["big", "large", "huge"],
            "huge": ["small", "tiny", "little"],
            "massive": ["small", "tiny", "compact"],
            "enormous": ["small", "tiny", "compact"],
            "giant": ["small", "little", "compact"],
            "little": ["big", "large", "huge"],
            "medium": ["tiny", "huge", "enormous"],
            "medium-sized": ["tiny", "huge", "enormous"],
            "compact": ["massive", "enormous", "expansive"],
            "expansive": ["compact", "tiny", "small"],
            "long": ["short", "tiny", "small"],
            "short": ["long", "massive", "enormous"],
            "wide": ["narrow", "thin", "small"],
            "narrow": ["wide", "expansive", "large"],
            "thick": ["thin", "narrow", "small"],
            "thin": ["thick", "wide", "massive"]
        }
        
        # Try to replace each size with its opposite
        for size in sizes:
            if size in size_opposites:
                opposite_sizes = size_opposites[size]
                opposite_size = random.choice(opposite_sizes)
                
                # Create a new text with the opposite size
                new_text = text.replace(size, opposite_size)
                
                # Only add if the text actually changed
                if new_text != text:
                    similarity = self.calculate_similarity(original_answer, new_text)
                    negatives.append({
                        "text": new_text,
                        "similarity": similarity,
                        "type": "size_negation"
                    })
                    
                    # Return early if we have at least one negative
                    if negatives:
                        break
        
        # If no size-specific negatives were generated, use a generic approach
        if not negatives:
            # Replace all detected sizes with random different ones
            all_sizes = list(self.size_terms)
            new_text = text
            
            for size in sizes:
                different_sizes = [s for s in all_sizes if s != size]
                if different_sizes:
                    different_size = random.choice(different_sizes)
                    new_text = new_text.replace(size, different_size)
            
            # Only add if the text actually changed
            if new_text != text:
                similarity = self.calculate_similarity(original_answer, new_text)
                negatives.append({
                    "text": new_text,
                    "similarity": similarity,
                    "type": "size_substitution"
                })
        
        # Return what we have, or empty list if nothing worked
        return negatives
    
    def _extract_navigation_info(self, text):
        """Extract key navigation information from text."""
        text_lower = text.lower()
        
        # Find directions
        directions = []
        for term in self.direction_terms:
            if term in text_lower:
                directions.append(term)
                
        # Find landmarks
        landmarks = []
        for term in self.landmark_terms:
            if term in text_lower:
                landmarks.append(term)
                
        # Find colors
        colors = []
        for term in self.color_terms:
            if term in text_lower:
                colors.append(term)
                
        # Find shapes
        shapes = []
        for term in self.shape_terms:
            if term in text_lower:
                shapes.append(term)
        
        # Find spatial relations
        spatial_relations = []
        for term in self.spatial_relation_terms:
            if term in text_lower:
                spatial_relations.append(term)
        
        # Find sizes
        sizes = []
        for term in self.size_terms:
            if term in text_lower:
                sizes.append(term)
                
        return {
            "directions": directions,
            "landmarks": landmarks,
            "colors": colors,
            "shapes": shapes,
            "spatial_relations": spatial_relations,
            "sizes": sizes
        }
    
    def _generate_semantic_frame_negatives(self, original_answer, directions, landmarks, colors, shapes, spatial_relations, sizes):
        """Generate negatives by transforming semantic frames."""
        negatives = []
        
        # Try direction transformation first
        if directions:
            for direction in directions:
                opposite_directions = self._get_opposite_direction(direction)
                for opposite in opposite_directions:
                    # Replace the direction with its opposite
                    negative_text = original_answer.lower().replace(direction, opposite)
                    
                    # Only add if it's different enough
                    similarity = self.calculate_similarity(original_answer, negative_text)
                    if similarity < 0.95:  # Avoid near-duplicates
                        negatives.append({
                            "text": negative_text,
                            "similarity": similarity,
                            "type": "semantic_frame_direction"
                        })
        
        # Try landmark transformation
        if landmarks:
            for landmark in landmarks:
                # Pick a random different landmark
                other_landmarks = [l for l in self.landmark_terms if l != landmark]
                if other_landmarks:
                    wrong_landmark = random.choice(other_landmarks)
                    negative_text = original_answer.lower().replace(landmark, wrong_landmark)
                    
                    similarity = self.calculate_similarity(original_answer, negative_text)
                    if similarity < 0.95:
                        negatives.append({
                            "text": negative_text,
                            "similarity": similarity,
                            "type": "semantic_frame_landmark"
                        })
        
        # Try color transformation
        if colors:
            for color in colors:
                # Pick a random different color
                other_colors = [c for c in self.color_terms if c != color]
                if other_colors:
                    wrong_color = random.choice(other_colors)
                    negative_text = original_answer.lower().replace(color, wrong_color)
                    
                    similarity = self.calculate_similarity(original_answer, negative_text)
                    if similarity < 0.95:
                        negatives.append({
                            "text": negative_text,
                            "similarity": similarity,
                            "type": "semantic_frame_color"
                        })
        
        return negatives
    
    def _generate_landmark_substitution_negatives(self, original_answer, landmarks):
        """Generate negatives by substituting landmarks with their opposites."""
        negatives = []
        
        if landmarks:
            for landmark in landmarks:
                # Get opposite landmark pairs (e.g., building → lake)
                opposite_landmarks = self._get_opposite_landmark(landmark)
                for opposite in opposite_landmarks:
                    negative_text = original_answer.lower().replace(landmark, opposite)
                    
                    similarity = self.calculate_similarity(original_answer, negative_text)
                    if similarity < 0.95:
                        negatives.append({
                            "text": negative_text,
                            "similarity": similarity,
                            "type": "landmark_substitution"
                        })
        
        return negatives
    
    def _generate_direction_reversal_negatives(self, original_answer, directions):
        """Generate negatives by reversing directions."""
        negatives = []
        
        for direction in directions:
            # Get reversed direction
            reversed_directions = self._get_opposite_direction(direction)
            
            # Generate complex direction reversal by transforming multiple parts
            text_parts = original_answer.split()
            for i, part in enumerate(text_parts):
                part_lower = part.lower().strip('.,!?;:"\'')
                if part_lower == direction or direction in part_lower:
                    for reversed_dir in reversed_directions:
                        # Create a copy and modify just this one part
                        new_parts = text_parts.copy()
                        new_parts[i] = part.replace(direction, reversed_dir)
                        negative_text = ' '.join(new_parts)
                        
                        similarity = self.calculate_similarity(original_answer, negative_text)
                        if similarity < 0.95:
                            negatives.append({
                                "text": negative_text,
                                "similarity": similarity,
                                "type": "direction_reversal"
                            })
        
        return negatives
    
    def _generate_spatial_relation_negatives(self, original_answer, spatial_relations):
        """Generate negatives by swapping spatial relations."""
        negatives = []
        
        for relation in spatial_relations:
            # Get opposite spatial relation
            opposite_relations = self._get_opposite_spatial_relation(relation)
            for opposite in opposite_relations:
                negative_text = original_answer.lower().replace(relation, opposite)
                
                similarity = self.calculate_similarity(original_answer, negative_text)
                if similarity < 0.95:
                    negatives.append({
                        "text": negative_text,
                        "similarity": similarity,
                        "type": "spatial_relation_swap"
                    })
        
        return negatives
    
    def _generate_generic_opposition_negatives(self, original_answer):
        """Generate generic negation by semantic opposition."""
        negatives = []
        
        # Add negation
        if not any(neg in original_answer.lower() for neg in ["not", "don't", "doesn't", "isn't", "aren't", "can't"]):
            # Insert negation before the main verb or directive
            for verb in ["go", "head", "turn", "proceed", "continue", "move", "follow", "take"]:
                if verb in original_answer.lower():
                    negative_text = original_answer.lower().replace(verb, f"don't {verb}")
                    similarity = self.calculate_similarity(original_answer, negative_text)
                    negatives.append({
                        "text": negative_text,
                        "similarity": similarity,
                        "type": "generic_opposition_negation"
                    })
                    break
        
        # Complete reversal of instruction
        negative_text = f"Do the opposite of what was instructed: {original_answer}"
        similarity = self.calculate_similarity(original_answer, negative_text)
        negatives.append({
            "text": negative_text,
            "similarity": similarity,
            "type": "generic_opposition_reversal"
        })
        
        return negatives
    
    def _generate_random_negative(self, original_answer):
        """Generate a random negative if all else fails."""
        # Generate random navigation instruction
        templates = [
            "Head {direction} until you reach the {color} {landmark}.",
            "Make a {direction} turn and go towards the {size} {landmark}.",
            "Fly in the {direction} direction to find a {shape} building.",
            "Go {direction} about {distance} meters and look for a {color} {landmark}.",
            "Your destination is the {color} {landmark} to the {direction}.",
            "Turn to {time} o'clock direction and proceed to the {landmark}."
        ]
        
        template = random.choice(templates)
        direction = random.choice(self.direction_terms)
        landmark = random.choice(self.landmark_terms)
        color = random.choice(self.color_terms)
        shape = random.choice(self.shape_terms)
        size = random.choice(self.size_terms)
        distance = str(random.randint(10, 500))
        time = str(random.randint(1, 12))
        
        negative_text = template.format(
            direction=direction,
            landmark=landmark,
            color=color,
            shape=shape,
            size=size,
            distance=distance,
            time=time
        )
        
        similarity = self.calculate_similarity(original_answer, negative_text)
        
        return {
            "text": negative_text,
            "similarity": similarity,
            "type": "random_negative"
        }
    
    def _get_opposite_direction(self, direction):
        """Get opposite(s) of a direction term."""
        opposites = {
            "north": ["south"],
            "south": ["north"],
            "east": ["west"],
            "west": ["east"],
            "northeast": ["southwest"],
            "northwest": ["southeast"],
            "southeast": ["northwest"],
            "southwest": ["northeast"],
            "left": ["right"],
            "right": ["left"],
            "forward": ["backward", "behind"],
            "backward": ["forward", "ahead"],
            "ahead": ["behind", "backward"],
            "behind": ["ahead", "forward"],
            "clockwise": ["counterclockwise"],
            "counterclockwise": ["clockwise"],
            "1 o'clock": ["7 o'clock"],
            "2 o'clock": ["8 o'clock"],
            "3 o'clock": ["9 o'clock"],
            "4 o'clock": ["10 o'clock"],
            "5 o'clock": ["11 o'clock"],
            "6 o'clock": ["12 o'clock"],
            "7 o'clock": ["1 o'clock"],
            "8 o'clock": ["2 o'clock"],
            "9 o'clock": ["3 o'clock"],
            "10 o'clock": ["4 o'clock"],
            "11 o'clock": ["5 o'clock"],
            "12 o'clock": ["6 o'clock"]
        }
        
        # Handle shortened forms
        for hour in range(1, 13):
            opposites[f"{hour}o'clock"] = [f"{(hour + 6) % 12 or 12}o'clock"]
            opposites[f"{hour} o'clock"] = [f"{(hour + 6) % 12 or 12} o'clock"]
        
        return opposites.get(direction.lower(), ["opposite direction"])
    
    def _get_opposite_landmark(self, landmark):
        """Get semantically opposite landmark types."""
        opposites = {
            "building": ["field", "lake", "river", "mountain"],
            "house": ["park", "lake", "forest"],
            "tower": ["pond", "field", "ground"],
            "road": ["river", "lake", "field"],
            "bridge": ["tunnel", "valley", "river"],
            "hill": ["valley", "plain", "hole"],
            "mountain": ["valley", "plain", "lake"],
            "tree": ["rock", "building", "structure"],
            "field": ["city", "building", "structure"],
            "river": ["road", "path", "building"],
            "lake": ["mountain", "hill", "building"]
        }
        
        return opposites.get(landmark.lower(), ["different place"])
    
    def _get_opposite_spatial_relation(self, relation):
        """Get opposite spatial relation."""
        opposites = {
            "above": ["below", "under", "beneath"],
            "below": ["above", "over"],
            "under": ["above", "over"],
            "over": ["under", "below"],
            "on top of": ["under", "beneath"],
            "beneath": ["above", "on top of"],
            "adjacent to": ["far from", "distant from"],
            "next to": ["far from", "away from"],
            "beside": ["far from", "separate from"],
            "in front of": ["behind"],
            "behind": ["in front of"],
            "inside": ["outside"],
            "outside": ["inside"],
            "near": ["far from", "distant from"],
            "far": ["near", "close to"],
            "close to": ["far from", "distant from"],
            "across from": ["beside", "next to"],
            "opposite to": ["alongside", "next to"]
        }
        
        return opposites.get(relation.lower(), ["differently located"])
    
    def analyze_complexity(self, instruction, answer):
        """Analyze the complexity of a dialog turn for curriculum learning."""
        # Count spatial terms
        spatial_terms = ["left", "right", "north", "south", "east", "west", 
                         "above", "below", "behind", "in front", "next to",
                         "between", "across", "along", "through", "around",
                         "clockwise", "counterclockwise", "o'clock"]
        
        question = instruction or ""
        answer = answer or ""
        
        spatial_count = sum(1 for term in spatial_terms if term in question.lower() or term in answer.lower())
        spatial_level = min(3, spatial_count)
        
        # Count landmarks
        landmark_count = sum(1 for term in self.landmark_terms if term in question.lower() or term in answer.lower())
        
        # Assess directional complexity
        directional_complexity = 0
        if any(term in question.lower() or term in answer.lower() for term in ["turn", "rotate", "face"]):
            directional_complexity += 1
        if any(term in question.lower() or term in answer.lower() for term in ["proceed", "continue", "go", "move"]):
            directional_complexity += 1
        if any(term in question.lower() or term in answer.lower() for term in ["after", "before", "then", "until", "while"]):
            directional_complexity += 1
        
        # Overall complexity score (0-1)
        overall_complexity = (0.4 * (spatial_level / 3) + 
                             0.3 * min(1.0, landmark_count / 3) + 
                             0.3 * (directional_complexity / 3))
        
        return {
            "spatial_reasoning_level": spatial_level,
            "landmark_references": landmark_count,
            "directional_complexity": directional_complexity,
            "overall_complexity": float(overall_complexity)  # Ensure it's a native Python float for JSON serialization
        }

    def augment_dialog_turn(self, question, answer, num_positives=2, num_negatives=3):
        """Generate contrastive samples for a dialog turn."""
        if not answer:
            return None
        
        contrastive_samples = {
            "positive_examples": self.generate_positive_examples(answer, num_positives),
            "negative_examples": self.generate_negative_examples(answer, num_negatives)
        }
        
        complexity_metadata = self.analyze_complexity(question, answer)
        
        return {
            "contrastive_samples": contrastive_samples,
            "complexity_metadata": complexity_metadata
        }
    
    def augment_dataset(self, input_json_path, output_json_path, num_positives=2, num_negatives=3):
        """Augment an entire dataset with contrastive samples."""
        # Load dataset
        self.logger.info(f"Loading dataset from {input_json_path}")
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"Processing {len(data)} episodes")
        
        augmented_count = 0
        total_dialogs = 0
        
        # Process each episode
        for episode in tqdm.tqdm(data, desc="Augmenting episodes"):
            for dialog in episode["dialogs"]:
                total_dialogs += 1
                if dialog.get("question") is not None and dialog.get("answer") is not None:
                    augmentation = self.augment_dialog_turn(
                        dialog["question"], 
                        dialog["answer"],
                        num_positives=num_positives,
                        num_negatives=num_negatives
                    )
                    
                    if augmentation:
                        dialog.update(augmentation)
                        augmented_count += 1
        
        self.logger.info(f"Augmented {augmented_count} out of {total_dialogs} dialog turns")
        
        # Save augmented dataset
        self.logger.info(f"Saving augmented dataset to {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(data, f)
        
        return augmented_count 