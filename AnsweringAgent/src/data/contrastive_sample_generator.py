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
                 paraphrase_model_name="tuner007/pegasus_paraphrase",
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
            self.paraphraser = pipeline(
                "text2text-generation", 
                model=paraphrase_model_name,
                device=0 if device == "cuda" else -1
            )
            self.has_paraphraser = True
            self.logger.info("Successfully loaded paraphrasing model")
        except Exception as e:
            self.logger.warning(f"Could not load paraphrasing model: {str(e)}")
            self.logger.warning("Will fall back to template-based paraphrasing")
            self.has_paraphraser = False
        
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
            # Use the paraphrasing model to generate multiple paraphrases
            model_outputs = self.paraphraser(
                original_answer,
                max_length=100,
                num_return_sequences=n+2,  # Generate extra in case some are filtered
                num_beams=10,
                temperature=1.5  # Higher temperature for more diverse outputs
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
            
            # Landmark-focused templates
            "Look for a {color} {landmark} {direction} from your current position.",
            "Your target is the {size} {color} {landmark} that you can see {direction}.",
            "The {landmark} with the {color} {feature} is your destination, located {direction}.",
            "Fly {direction} toward the {color} {landmark}.",
            "Navigate to the {size} {landmark} {spatial_relation} the area.",
            
            # Clock direction templates
            "Your destination is at {clock_direction}, it's the {color} {shape} {landmark}.",
            "If you turn to {clock_direction}, you'll see a {size} {color} {landmark}.",
            "The {color} {landmark} at your {clock_direction} is the destination.",
            "Look towards your {clock_direction} for a {color} {shape} {landmark}.",
            
            # Combined attribute templates
            "The {size} {landmark} with {color} {feature} {spatial_relation} is your target.",
            "Head to the {size} {color} {shape} structure {direction} of your position.",
            "Your goal is the {color} {landmark} with {shape} features {spatial_relation}.",
            "Navigate to the {color} {landmark} that appears {spatial_relation} your view area."
        ]
        
        # Determine which templates we can use based on available information
        usable_templates = []
        for template in templates:
            can_use = True
            
            # Check if we have the needed information for this template
            if "{direction}" in template and not directions:
                can_use = False
            if "{landmark}" in template and not landmarks:
                can_use = False
            if "{color}" in template and not colors:
                can_use = False
            if "{shape}" in template and not shapes:
                can_use = False
            if "{spatial_relation}" in template and not spatial_relations:
                can_use = False
            if "{size}" in template and not sizes:
                can_use = False
            if "{clock_direction}" in template and not any("o'clock" in d for d in directions):
                can_use = False
                
            if can_use:
                usable_templates.append(template)
        
        # If we have usable templates, generate paraphrases
        if usable_templates:
            # Select random templates, avoiding duplicates
            selected_templates = random.sample(usable_templates, min(n, len(usable_templates)))
            
            for template in selected_templates:
                # Select random values from available categories
                direction = random.choice(directions) if directions else ""
                landmark = random.choice(landmarks) if landmarks else "building"
                color = random.choice(colors) if colors else "regular"
                shape = random.choice(shapes) if shapes else "regular"
                spatial_relation = random.choice(spatial_relations) if spatial_relations else "nearby"
                size = random.choice(sizes) if sizes else "regular"
                
                # Handle clock direction specially
                clock_direction = next((d for d in directions if "o'clock" in d), "")
                
                # Additional possible fillers
                feature = "roof" if random.random() > 0.5 else "structure"
                
                # Format the template
                paraphrase = template.format(
                    direction=direction,
                    landmark=landmark,
                    color=color,
                    shape=shape,
                    spatial_relation=spatial_relation,
                    size=size,
                    clock_direction=clock_direction,
                    feature=feature
                )
                
                # Calculate similarity
                similarity = self.calculate_similarity(original_answer, paraphrase)
                
                positives.append({
                    "text": paraphrase,
                    "similarity": similarity,
                    "type": "template_paraphrase"
                })
        
        # If we couldn't generate enough paraphrases from templates, add word-level variations
        if len(positives) < n:
            # Use word-level variations as a fallback
            words = original_answer.split()
            if len(words) >= 4:
                for _ in range(min(n - len(positives), 3)):
                    # Copy the original words
                    new_words = words.copy()
                    
                    # Make a small number of targeted substitutions
                    num_substitutions = random.randint(1, min(3, len(words) // 3))
                    for _ in range(num_substitutions):
                        idx = random.randint(0, len(words) - 1)
                        word = words[idx].lower().strip('.,!?;:"\'')
                        
                        # Word-specific substitutions
                        if word in ["the", "a", "an"]:
                            new_words[idx] = random.choice(["the", "a", "this", "that", "your"])
                        elif word in ["is", "are"]:
                            new_words[idx] = random.choice(["is", "appears to be", "looks like", "seems to be"])
                        elif word in ["go", "head", "move", "navigate"]:
                            new_words[idx] = random.choice(["go", "head", "move", "navigate", "proceed", "travel"])
                        elif word in ["towards", "toward", "to"]:
                            new_words[idx] = random.choice(["towards", "toward", "to", "in the direction of"])
                        elif word in directions:
                            # Keep the same direction - we're generating positives
                            continue
                        elif word in landmarks:
                            # Keep the same landmark - we're generating positives
                            continue
                    
                    paraphrase = " ".join(new_words)
                    similarity = self.calculate_similarity(original_answer, paraphrase)
                    
                    positives.append({
                        "text": paraphrase,
                        "similarity": similarity,
                        "type": "word_variation"
                    })
            
            # If we still need more, use simple sentence structure variations
            if len(positives) < n:
                # Try sentence structure variations
                if directions and landmarks:
                    direction = directions[0]
                    landmark = landmarks[0]
                    color_text = colors[0] if colors else ""
                    
                    variations = [
                        f"You should move {direction} to find the {color_text} {landmark}.",
                        f"If you go {direction}, you will see the {color_text} {landmark}.",
                        f"The {color_text} {landmark} can be found when you head {direction}.",
                        f"Continue {direction} until you reach the {color_text} {landmark}."
                    ]
                    
                    for variation in variations[:n - len(positives)]:
                        similarity = self.calculate_similarity(original_answer, variation)
                        positives.append({
                            "text": variation,
                            "similarity": similarity,
                            "type": "structure_variation"
                        })
        
        # If still not enough, just duplicate the original
        while len(positives) < n:
            positives.append({
                "text": original_answer,
                "similarity": 1.0,
                "type": "duplicate"
            })
        
        return positives[:n]
    
    def generate_positive_examples(self, original_answer, n=3):
        """Generate positive examples (paraphrases) for an answer.
        
        Uses a mixed approach with LM-based and template-based paraphrases.
        """
        # If n is less than 3, adjust the counts accordingly
        lm_count = 1 if n >= 1 else 0
        template_count = n - lm_count
        
        # Get LM-based paraphrases if available
        lm_paraphrases = []
        if lm_count > 0 and self.has_paraphraser:
            try:
                lm_paraphrases = self.generate_lm_paraphrases(original_answer, lm_count)
            except Exception as e:
                self.logger.warning(f"Error generating LM paraphrases: {str(e)}")
                # If LM generation fails, use templates for all
                template_count = n
        
        # Get template-based paraphrases for the remainder
        template_paraphrases = self.generate_template_paraphrases(original_answer, template_count)
        
        # Combine the results
        all_paraphrases = lm_paraphrases + template_paraphrases
        
        return all_paraphrases[:n]
    
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
        """Generate negative examples using a language model.
        
        Args:
            original_answer: Original answer to contradict
            n: Number of negative examples to generate
            
        Returns:
            List of negative examples with similarity scores
        """
        negatives = []
        
        # Skip if answer is too short or if paraphraser is not available
        if len(original_answer.split()) < 3 or not self.has_paraphraser:
            return negatives
        
        try:
            # Prepare prompt for contradiction generation
            nav_terms = self._extract_navigation_info(original_answer)
            directions = nav_terms["directions"]
            landmarks = nav_terms["landmarks"]
            
            # Create a prompt that encourages the model to generate contradictions
            prompt = f"Original navigation instruction: \"{original_answer}\"\n\nGenerate a navigation instruction that CONTRADICTS the original by using opposite directions, different landmarks, or incorrect spatial relationships. Make it sound natural and fluent, but ensure it would guide to a WRONG destination:"
            
            # Use the paraphrasing model to generate contradictions
            model_outputs = self.paraphraser(
                prompt,
                max_length=150,
                num_return_sequences=n+3,  # Generate extra in case some are filtered
                num_beams=10,
                temperature=1.2  # Higher temperature for diversity
            )
            
            # Process the generated contradictions
            for output in model_outputs:
                contradiction = output['generated_text']
                
                # Clean up the contradiction to extract just the instruction
                if ":" in contradiction:
                    contradiction = contradiction.split(":", 1)[1].strip()
                
                # Skip if it's empty or too similar to original
                if not contradiction or contradiction.strip() == original_answer.strip():
                    continue
                
                # Calculate similarity
                similarity = self.calculate_similarity(original_answer, contradiction)
                
                # Verify it's different enough but still on topic
                if similarity < 0.85 and similarity > 0.3:
                    negatives.append({
                        "text": contradiction,
                        "similarity": similarity,
                        "type": "lm_contradiction"
                    })
                    
                    if len(negatives) >= n:
                        break
            
            # If we didn't get enough good contradictions, try a more direct approach
            if len(negatives) < n and directions:
                # Make sure we have at least one direction before trying to access it
                if len(directions) > 0:
                    direction = directions[0]
                    opposite_dirs = self._get_opposite_direction(direction)
                    
                    # Make sure we have at least one opposite direction
                    if opposite_dirs and len(opposite_dirs) > 0:
                        opp_dir = opposite_dirs[0]
                        prompt = f"Original: \"{original_answer}\"\n\nRewrite this navigation instruction, but replace '{direction}' with '{opp_dir}':"
                        
                        try:
                            outputs = self.paraphraser(
                                prompt,
                                max_length=100,
                                num_return_sequences=2,
                                temperature=1.0
                            )
                            
                            for output in outputs:
                                contradiction = output['generated_text']
                                
                                # Clean up the contradiction
                                if ":" in contradiction:
                                    contradiction = contradiction.split(":", 1)[1].strip()
                                
                                similarity = self.calculate_similarity(original_answer, contradiction)
                                
                                if similarity < 0.9 and similarity > 0.3:
                                    negatives.append({
                                        "text": contradiction,
                                        "similarity": similarity,
                                        "type": "lm_direction_flip"
                                    })
                                    
                                    if len(negatives) >= n:
                                        break
                        except Exception as e:
                            self.logger.warning(f"Error with direction-specific prompt: {str(e)}")
        
        except Exception as e:
            self.logger.warning(f"Error generating LM negatives: {str(e)}")
            # Use more detailed logging to help with debugging
            import traceback
            self.logger.warning(f"Exception details: {traceback.format_exc()}")
        
        # If we couldn't generate any negative examples, return an empty list
        # The calling code will handle this by using only rule-based negatives
        return negatives[:n]
    
    def generate_rule_based_negatives(self, original_answer, n=2):
        """Generate negative examples using rule-based strategies."""
        negatives = []
        strategies = []
        
        # Extract key information from original answer
        extracted_info = self._extract_navigation_info(original_answer)
        directions = extracted_info["directions"]
        landmarks = extracted_info["landmarks"]
        colors = extracted_info["colors"] 
        shapes = extracted_info["shapes"]
        spatial_relations = extracted_info["spatial_relations"]
        sizes = extracted_info["sizes"]
        
        # Strategy 1: Semantic Frame Transformation
        if directions and (landmarks or shapes or colors):
            strategies.append(self._generate_semantic_frame_negatives(
                original_answer, directions, landmarks, colors, shapes, spatial_relations, sizes
            ))
        
        # Strategy 2: Landmark Substitution
        if landmarks:
            strategies.append(self._generate_landmark_substitution_negatives(
                original_answer, landmarks
            ))
        
        # Strategy 3: Direction Reversal
        if directions:
            strategies.append(self._generate_direction_reversal_negatives(
                original_answer, directions
            ))
        
        # Strategy 4: Spatial Relation Transformation
        if spatial_relations:
            strategies.append(self._generate_spatial_relation_negatives(
                original_answer, spatial_relations
            ))
        
        # If no strategies worked (likely because no key elements were found),
        # generate generic opposition negative
        if all(len(s) == 0 for s in strategies):
            strategies.append(self._generate_generic_opposition_negatives(original_answer))
        
        # Combine and sample from all strategies
        all_negatives = []
        for strat_negatives in strategies:
            all_negatives.extend(strat_negatives)
        
        # If we have enough negatives, sample randomly
        if len(all_negatives) > n:
            # Sort by similarity ascending - we want the most dissimilar examples
            all_negatives.sort(key=lambda x: x["similarity"])
            # Select n examples with preference for more dissimilar ones
            # Use exponential weighting to favor lower similarity
            weights = [np.exp(-3 * neg["similarity"]) for neg in all_negatives]
            weight_sum = sum(weights)
            if weight_sum > 0:  # Avoid division by zero
                weights = [w / weight_sum for w in weights]
                indices = np.random.choice(len(all_negatives), min(n, len(all_negatives)), replace=False, p=weights)
                negatives = [all_negatives[i] for i in indices]
            else:
                negatives = random.sample(all_negatives, min(n, len(all_negatives)))
        else:
            negatives = all_negatives
        
        # If still not enough negatives, add random negatives
        while len(negatives) < n:
            random_negative = self._generate_random_negative(original_answer)
            if random_negative:
                negatives.append(random_negative)
        
        return negatives[:n]  # Return at most n negatives
    
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