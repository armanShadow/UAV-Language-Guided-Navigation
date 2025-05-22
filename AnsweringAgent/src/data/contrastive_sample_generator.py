import torch
import torch.nn.functional as F
import json
import random
import numpy as np
import os
import tqdm
import logging
from transformers import AutoTokenizer, AutoModel, pipeline
import re

class ContrastiveSampleGenerator:
    """
    Generator for contrastive learning samples specific to UAV navigation tasks.
    
    This class handles the generation of positive examples (paraphrases) and
    negative examples (contradictions) for navigation instructions, to be used
    in contrastive learning for UAV navigation.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", 
                 paraphrase_model_name="prithivida/parrot_paraphraser_on_T5",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the contrastive sample generator.
        
        Args:
            model_name: Name of the sentence embedding model to use
            paraphrase_model_name: Name of the language model for paraphrasing
            device: Device to use for models ('cuda' or 'cpu')
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Load sentence embedding model
        self.logger.info(f"Loading sentence embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Initialize language models for paraphrasing and negation
        self._init_language_models(paraphrase_model_name)
        
        # Initialize navigation terminology datasets
        self._init_navigation_terminology()
        
        # Store alternative answers for negative examples
        self.alternative_answers = []
    
    def _init_language_models(self, paraphrase_model_name):
        """Initialize paraphrasing and negative generation models."""
        self.logger.info(f"Loading paraphrasing model: {paraphrase_model_name}")
        try:
            # Set up paraphrasing model with appropriate device configuration
            if torch.cuda.is_available() and self.device == "cuda":
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
                if torch.cuda.is_available() and self.device == "cuda":
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
    
    def _init_navigation_terminology(self):
        """Initialize UAV navigation-specific terminology lists based on AVDN dataset."""
        # Direction-related terms
        self.direction_terms = [
            "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest",
            "left", "right", "forward", "backward", "ahead", "behind", "clockwise",
            "counterclockwise", "turn", "rotate", "face", "head", "proceed", "continue", "go", "move",
            "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock", "5 o'clock", "6 o'clock", 
            "7 o'clock", "8 o'clock", "9 o'clock", "10 o'clock", "11 o'clock", "12 o'clock",
            "straight", "straight ahead", "opposite direction", "180 degrees", "90 degrees",
            "upposit", "opposite", "around"
        ]
        
        # Store clock directions separately for special handling
        self.clock_direction_terms = [
            "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock", "5 o'clock", "6 o'clock", 
            "7 o'clock", "8 o'clock", "9 o'clock", "10 o'clock", "11 o'clock", "12 o'clock"
        ]
        
        # Landmark-related terms
        self.landmark_terms = [
            "building", "house", "structure", "tower", "road", "bridge", "highway", 
            "hill", "mountain", "tree", "field", "river", "lake", "landfill",
            "rooftop", "parking lot", "intersection", "corner", "entrance", "exit",
            "area", "zone", "region", "block", "neighborhood", "complex", "facility",
            "stadium", "park", "garden", "campus", "fence", "wall", "path", "trail",
            "destination", "target", "location", "container", "containers", "cargo",
            "cargo containers", "island", "dirt", "island of dirt", "row", "white container",
            "blue containers", "small containers", "long row"
        ]
        
        # Visual attribute terms
        self.color_terms = [
            "red", "blue", "green", "yellow", "black", "white", "gray", "grey",
            "brown", "purple", "orange", "beige", "pink", "tan", "golden", "silver",
            "dark", "light", "bright", "dull", "vibrant", "colorful", "lighter", "darker"
        ]
        
        self.shape_terms = [
            "square", "rectangular", "round", "circular", "oval", "triangular", 
            "dome", "L-shaped", "U-shaped", "flat", "tall", "short", "wide", "narrow",
            "curved", "straight", "zigzag", "angled", "sloped", "long", "row"
        ]
        
        # Spatial relationship terms
        self.spatial_relation_terms = [
            "above", "below", "under", "over", "on top of", "beneath", "adjacent to",
            "next to", "beside", "alongside", "in front of", "behind", "between",
            "among", "surrounding", "inside", "outside", "within", "near", "far",
            "close to", "distant from", "across from", "opposite to", "parallel to",
            "perpendicular to", "diagonal from", "at the edge of", "in the center of",
            "in the middle of", "at the corner of", "at the intersection of",
            "bisecting", "few feet", "very close"
        ]
        
        # Size-related terms
        self.size_terms = [
            "large", "small", "big", "tiny", "huge", "massive", "enormous", "giant",
            "little", "medium", "medium-sized", "compact", "expansive", "long", "short",
            "wide", "narrow", "thick", "thin", "many"
        ]
        
        # Sequencing terms (useful for multi-step instructions)
        self.sequence_terms = [
            "first", "second", "third", "then", "next", "after", "before", "finally",
            "once", "when", "until", "while", "and"
        ]
        
        # Position-related terms
        self.position_terms = [
            "first", "last", "middle", "center", "edge", "corner", "end", "beginning", "top", "bottom"
        ]
        
        # UAV-specific instruction patterns from AVDN dataset
        self.uav_instruction_patterns = [
            "turn {direction} degrees",
            "go straight until",
            "fly to your {direction}",
            "turn around",
            "that is your destination",
            "you are {spatial_relation} your destination",
            "you will be right on top of your destination"
        ]
    
    def generate_embedding(self, text):
        """
        Generate an embedding vector for a text using the sentence embedding model.
        
        Args:
            text: Input text to embed
            
        Returns:
            Normalized embedding vector
        """
        # Tokenize and prepare inputs
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply mean pooling to get sentence embedding
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
        
        # Normalize the embedding vector
        return F.normalize(embeddings.squeeze(0), p=2, dim=0)
    
    def calculate_similarity(self, text1, text2):
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        emb1 = self.generate_embedding(text1)
        emb2 = self.generate_embedding(text2)
        return float(F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item())
    
    def generate_template_paraphrases(self, original_answer, n=1):
        """
        Generate positive examples (paraphrases) using templates specific to UAV navigation.
        
        Args:
            original_answer: Original answer to paraphrase
            n: Number of paraphrases to generate
            
        Returns:
            List of paraphrases with similarity scores
        """
        positives = []
        
        # Extract key navigation information
        extracted_info = self._extract_navigation_info(original_answer)
        directions = extracted_info["directions"]
        landmarks = extracted_info["landmarks"]
        colors = extracted_info["colors"] 
        shapes = extracted_info["shapes"]
        spatial_relations = extracted_info["spatial_relations"]
        sizes = extracted_info["sizes"]
        
        # Special handling for clock directions
        if "clock_directions" in extracted_info:
            clock_directions = extracted_info["clock_directions"]
        else:
            clock_directions = self._get_clock_directions()
            
        # Check if there are clock directions in the original text
        original_answer_lower = original_answer.lower()
        contains_clock = "o'clock" in original_answer_lower or "oclock" in original_answer_lower
        
        # Get appropriate templates
        templates = self._get_navigation_templates()
        action_verbs = self._get_navigation_action_verbs()
        features = self._get_landmark_features()
        
        # Select usable templates based on available information
        usable_templates = self._select_usable_templates(
            templates, 
            directions, 
            landmarks, 
            colors, 
            shapes, 
            spatial_relations, 
            sizes
        )
        
        # If original has clock directions, prioritize clock direction templates
        if contains_clock:
            clock_templates = [t for t in usable_templates if "{clock_direction}" in t]
            if clock_templates:
                # Move clock templates to the front of the list
                for t in clock_templates:
                    usable_templates.remove(t)
                usable_templates = clock_templates + usable_templates
        
        # Add generic templates that don't require specific extracted elements
        generic_templates = self._get_generic_templates()
        usable_templates.extend(generic_templates)
        
        # Generate paraphrases using templates
        paraphrase_count = 0
        max_attempts = min(30, len(usable_templates) * 3)  # Limit attempts to avoid infinite loops
        attempts = 0
        
        # Track similarity to ensure we get diverse paraphrases
        similarity_threshold = 0.6  # Minimum similarity to be considered a good paraphrase
        
        while attempts < max_attempts:
            attempts += 1
            
            if not usable_templates:
                break
                
            template = random.choice(usable_templates)
            
            try:
                # Fill the template with appropriate values
                paraphrase = self._fill_template(
                    template,
                    directions,
                    landmarks,
                    colors,
                    shapes,
                    spatial_relations,
                    sizes,
                    clock_directions,
                    features,
                    action_verbs
                )
                
                # Calculate similarity with original answer
                similarity = self.calculate_similarity(original_answer, paraphrase)
                
                # Only include if similar enough to original but not identical
                if similarity > similarity_threshold and paraphrase != original_answer:
                    # Check if this paraphrase is significantly different from previously generated ones
                    is_unique = True
                    for existing in positives:
                        existing_similarity = self.calculate_similarity(existing["text"], paraphrase)
                        if existing_similarity > 0.90:  # Very similar to existing paraphrase
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

        positives.sort(key=lambda x: x["similarity"], reverse=True)
        return positives[:n]
    
    def _get_navigation_templates(self):
        """Get templates for UAV navigation instructions based on AVDN dataset."""
        return [
            # Direction-focused templates (simple)
            "The destination is {direction}. It's {shape} and {color}.",
            "If you look {direction}, you'll see the destination which is {shape} with {color} color.",
            "Your destination is {direction} from your position. Look for a {color} {shape} structure.",
            "Head {direction} to reach your destination. It's {color} and {shape}.",
            "{direction} is where you'll find your destination. It's a {color} {shape} building.",
            "Fly {direction} to find your target. The {color} {shape} structure is your destination.",
            "Move your drone {direction} and look for a {color} {shape} landmark.",
            
            # AVDN-specific templates
            "Destination is a {shape} {landmark} {spatial_relation} at your {clock_direction} direction.",
            "Turn {direction} degrees. Go straight until you are over a {color} {landmark}. That is your destination.",
            "Yes, you are {spatial_relation} your destination. Turn to your {clock_direction} and go straight forward from there.",
            "{direction}, fly to your {direction} and turn around and go the {direction} direction just a few feet to your destination.",
            "You are very close to your destination. Turn to your {clock_direction} and you will be right on top of your destination.",
            
            # Clock direction templates (preserving exact clock direction)
            "Your destination is at {clock_direction}. It's the {color} {landmark}.",
            "If you go {clock_direction}, you'll see the {landmark} that is your destination.",
            "Head towards {clock_direction} to find your destination.", 
            "Go to {clock_direction} and you'll find your destination - {size} {landmark}.",
            "At {clock_direction} you'll see the {landmark} which is your destination.",
            "Your target is at {clock_direction} - it's the {color} {landmark} you'll see.",
            "If you look at {clock_direction}, that {color} {landmark} is your target.",
            "Please go to the {direction} direction at {clock_direction}. The destination is {color} {landmark}.",
            
            # Multi-step instructions
            "First, {action_verb} {direction}, then you'll see your destination - a {color} {landmark}.",
            "{action_verb} {direction} until you reach the {landmark}, that's your destination.",
            "Once you {action_verb} {direction}, look for the {color} {landmark} which is your target.",
            "{action_verb} {direction} and {sequence} look for the {landmark} as your destination.",
            "Turn {direction} degrees. Go straight until you are over a {color} {landmark}. That is your destination.",
            
            # Position-based templates
            "Your destination is the {position} {landmark} in the {direction}.",
            "The {position} {landmark} that you see {direction} is your target.",
            "Look for the {position} {color} {landmark} {direction}, that's your destination.",
            
            # Landmark-focused templates
            "Look for a {color} {landmark} {direction} from your current position.",
            "Your target is the {size} {color} {landmark} that you can see {direction}.",
            "The {landmark} with the {color} {feature} is your destination, located {direction}.",
            "Fly {direction} toward the {color} {landmark}.",
            "Navigate to the {size} {landmark} {spatial_relation} the area.",
            "The {color} {landmark} is your destination. It's {direction} from where you are now.",
            "Your destination is the {size} {landmark} with {color} features that you can see {direction}.",
            
            # Combined attribute templates
            "The {size} {landmark} with {color} {feature} {spatial_relation} is your target.",
            "Head to the {size} {color} {shape} structure {direction} of your position.",
            "Your goal is the {color} {landmark} with {shape} features {spatial_relation}.",
            "Navigate to the {color} {landmark} that appears {spatial_relation} your view area.",
            "Fly towards the {size} {landmark} with {color} exteriors located {direction}.",
            "Your destination is a {size} {color} structure with {shape} architecture {spatial_relation}.",
            "Move the drone to the {color} {landmark} that has {shape} features {direction} of your current position.",
            
            # AVDN dialog-specific templates
            "Yes, you are {spatial_relation} your destination.",
            "Turn {direction} degrees.",
            "Go straight until you are over a {color} {landmark}.",
            "That is your destination.",
            "You will be right on top of your destination.",
            "You are very close to your destination."
        ]
    
    def _get_navigation_action_verbs(self):
        """Get UAV-specific action verbs for navigation templates."""
        return [
            "fly", "navigate", "move", "head", "proceed", "travel", "hover near", 
            "go toward", "approach", "make your way to", "steer toward", "direct yourself to"
        ]
    
    def _get_clock_directions(self):
        """Get clock direction references for navigation templates."""
        return [
            "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock", "5 o'clock", "6 o'clock",
            "7 o'clock", "8 o'clock", "9 o'clock", "10 o'clock", "11 o'clock", "12 o'clock"
        ]
    
    def _get_landmark_features(self):
        """Get features that can describe landmarks."""
        return ["roof", "exterior", "walls", "edge", "perimeter", "facade", "structure", "design"]
    
    def _get_generic_templates(self):
        """Get generic templates that don't require specific extracted elements."""
        return [
            "Your destination is located {generic_direction}.",
            "Head {generic_direction} to find your target.",
            "The target is {generic_direction} from your current position.",
            "Navigate {generic_direction} to reach your destination.",
            "{generic_action} {generic_direction} to arrive at your destination.",
            "Your target can be found if you go {generic_direction}.",
            "To reach the destination, {generic_action} {generic_direction}.",
            
            # AVDN-specific generic templates
            "Turn {generic_direction} and go straight.",
            "You are very close to your destination.",
            "That is your destination.",
            "You will be right on top of your destination.",
            "Turn around completely.",
            "Turn 180 degrees.",
            "Go straight until you reach your destination.",
            "Yes, you are near the destination.",
            "You are on the right track."
        ]
    
    def _select_usable_templates(self, templates, directions, landmarks, colors, shapes, spatial_relations, sizes):
        """
        Select templates that can be used based on available information.
        
        Args:
            templates: List of all templates
            directions, landmarks, etc.: Extracted navigation information
            
        Returns:
            List of usable templates
        """
        usable_templates = []
        extracted_info = {
            "directions": directions,
            "landmarks": landmarks,
            "colors": colors,
            "shapes": shapes,
            "spatial_relations": spatial_relations,
            "sizes": sizes
        }
        
        # Calculate template complexity score
        template_complexity = {}
        for template in templates:
            # Count total placeholders as a rough complexity measure
            score = 0
            for key in extracted_info:
                placeholder = f"{{{key[:-1]}}}"  # Convert "directions" to "{direction}" etc.
                if placeholder in template:
                    score += 1
            template_complexity[template] = score
            
        for template in templates:
            can_use = True
            
            # Check required elements
            if "{direction}" in template and (not directions or len(directions) == 0):
                can_use = False
            if "{clock_direction}" in template and (not extracted_info.get("clock_directions") or len(extracted_info.get("clock_directions", [])) == 0):
                can_use = False
            if "{landmark}" in template and (not landmarks or len(landmarks) == 0):
                can_use = False
            if "{color}" in template and (not colors or len(colors) == 0):
                can_use = False
            if "{shape}" in template and (not shapes or len(shapes) == 0):
                can_use = False
            if "{spatial_relation}" in template and (not spatial_relations or len(spatial_relations) == 0):
                can_use = False
            if "{size}" in template and (not sizes or len(sizes) == 0):
                can_use = False
            if "{sequence}" in template and (not extracted_info.get("sequences") or len(extracted_info.get("sequences", [])) == 0):
                can_use = False
            if "{position}" in template and (not extracted_info.get("positions") or len(extracted_info.get("positions", [])) == 0):
                can_use = False
                
            if can_use:
                usable_templates.append(template)
        
        # Sort usable templates by complexity score (higher complexity first)
        # This makes us prefer more complex templates when available
        if usable_templates:
            usable_templates.sort(key=lambda t: template_complexity.get(t, 0), reverse=True)
        
        return usable_templates
    
    def _fill_template(self, template, directions, landmarks, colors, shapes, spatial_relations, sizes, 
                     clock_directions, features, action_verbs):
        """
        Fill a template with appropriate values, preserving original information when possible.
        
        Args:
            template: Template string with placeholders
            directions, landmarks, etc.: Navigation information and options
            
        Returns:
            Filled template as a paraphrase
        """
        paraphrase = template
        
        # Extract specific clock directions from the original text
        extracted_clock_directions = []
        for direction in directions:
            if any(clock in direction for clock in ["o'clock", "oclock"]):
                extracted_clock_directions.append(direction)
        
        # Replace clock direction placeholder with EXACT clock direction from original if available
        if "{clock_direction}" in paraphrase and extracted_clock_directions:
            # Use the exact clock direction from the original text
            paraphrase = paraphrase.replace("{clock_direction}", random.choice(extracted_clock_directions))
        elif "{clock_direction}" in paraphrase:
            # If no clock direction in original, use a random one
            paraphrase = paraphrase.replace("{clock_direction}", random.choice(clock_directions))
            
        # Replace direction placeholder
        if "{direction}" in paraphrase:
            if directions and len(directions) > 0:
                # Filter out clock directions which should be handled separately
                non_clock_directions = [d for d in directions if not any(clock in d for clock in ["o'clock", "oclock"])]
                if non_clock_directions:
                    paraphrase = paraphrase.replace("{direction}", random.choice(non_clock_directions))
                else:
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
                
        # Replace sequence placeholder
        if "{sequence}" in paraphrase:
            if hasattr(self, 'sequence_terms'):
                paraphrase = paraphrase.replace("{sequence}", random.choice(self.sequence_terms))
            else:
                paraphrase = paraphrase.replace("{sequence}", "then")
                
        # Replace position placeholder
        if "{position}" in paraphrase:
            if hasattr(self, 'position_terms'):
                paraphrase = paraphrase.replace("{position}", random.choice(self.position_terms))
            else:
                paraphrase = paraphrase.replace("{position}", "first")
        
        # Replace action verb placeholder
        if "{action_verb}" in paraphrase:
            paraphrase = paraphrase.replace("{action_verb}", random.choice(action_verbs))
            
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
            
        return paraphrase
    
    def generate_positive_examples(self, original_answer, n=3):
        """
        Generate positive examples (paraphrases) using a hybrid approach optimized for UAV navigation.
        
        Uses both language model-based and template-based paraphrases to create
        diverse positive examples for contrastive learning, with special handling
        for UAV navigation instructions from the AVDN dataset.
        
        Args:
            original_answer: The original navigation instruction to paraphrase
            n: Number of positive examples to generate (default: 3)
            
        Returns:
            List of positive examples with similarity scores and type labels
        """
        self.logger.info(f"Generating {n} positive examples using hybrid approach")
        positives = []
        
        # Handle short answers specially - they're difficult to paraphrase meaningfully
        if len(original_answer.split()) < 3:
            return self._generate_simple_variations(original_answer, n)
        
        # Check if this is a UAV navigation instruction with specific patterns
        is_uav_instruction = self._is_uav_navigation_instruction(original_answer)
        
        # Check if this is a multi-step instruction
        is_multi_step = self._is_multi_step_instruction(original_answer)
        
        # Step 1: For multi-step instructions, prioritize the language model approach
        # as it better preserves the sequential nature of instructions
        if is_multi_step and self.has_paraphraser:
            # Generate more LM paraphrases for multi-step instructions
            lm_paraphrases = self._generate_lm_paraphrase(
                original_answer, 
                max_paraphrases=min(n, 2),  # Try to get at least 2 LM paraphrases
                context="UAV multi-step navigation"
            )
            positives.extend(lm_paraphrases)
        # For non-multi-step instructions, use standard approach
        elif self.has_paraphraser:
            # For UAV instructions, add context to help the paraphraser
            if is_uav_instruction:
                lm_paraphrases = self._generate_lm_paraphrase(original_answer, context="UAV navigation")
            else:
                lm_paraphrases = self._generate_lm_paraphrase(original_answer)
            positives.extend(lm_paraphrases)
        
        # Step 2: Generate template-based paraphrases with priority for UAV instructions
        remaining = n - len(positives)
        if remaining > 0:
            self.logger.info(f"Generating {remaining} template-based paraphrases")
            # For multi-step instructions, use more templates to get better variety
            if is_multi_step:
                template_paraphrases = self.generate_template_paraphrases(original_answer, remaining + 2)
                # Sort by similarity and take the most appropriate ones
                template_paraphrases.sort(key=lambda x: x["similarity"], reverse=True)
                positives.extend(template_paraphrases[:remaining])
            # For UAV instructions, use more templates to get better variety
            elif is_uav_instruction:
                template_paraphrases = self.generate_template_paraphrases(original_answer, remaining + 2)
                # Sort by similarity and take the most appropriate ones
                template_paraphrases.sort(key=lambda x: x["similarity"], reverse=True)
                positives.extend(template_paraphrases[:remaining])
            else:
                template_paraphrases = self.generate_template_paraphrases(original_answer, remaining)
                positives.extend(template_paraphrases)
        
        # Step 3: Fill any remaining slots with simple variations if needed
        if len(positives) < n:
            self.logger.warning(f"Generated only {len(positives)}/{n} positive examples, adding simple variations")
            simple_variations = self._generate_simple_variations(original_answer, n - len(positives), positives)
            positives.extend(simple_variations)
        
        # Return the requested number of positives
        return positives[:n]
    
    def _is_uav_navigation_instruction(self, text):
        """
        Check if the text is a UAV navigation instruction based on patterns from AVDN dataset.
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if this is likely a UAV navigation instruction
        """
        text_lower = text.lower()
        
        # Check for UAV navigation patterns
        navigation_indicators = [
            "destination", "target", "direction", "o'clock", "turn", "degrees",
            "container", "straight", "fly", "navigate", "go", "move", "head",
            "left", "right", "north", "south", "east", "west"
        ]
        
        # Check for clock position references (common in AVDN)
        clock_references = any(f"{i} o'clock" in text_lower for i in range(1, 13))
        
        # Check for degree turn instructions (common in AVDN)
        degree_turns = any(f"{i} degree" in text_lower for i in [90, 180, 270, 360])
        
        # Count navigation indicators
        indicator_count = sum(1 for indicator in navigation_indicators if indicator in text_lower)
        
        # If it has clock references, degree turns, or multiple navigation indicators, it's likely a UAV instruction
        return clock_references or degree_turns or indicator_count >= 2
    
    def _generate_lm_paraphrase(self, original_answer, max_paraphrases=1, context=None):
        """
        Generate paraphrases using a language model.
        
        Args:
            original_answer: Original answer to paraphrase
            max_paraphrases: Maximum number of paraphrases to generate
            context: Optional context to add to the paraphrase input
            
        Returns:
            List of paraphrases with similarity scores
        """
        paraphrases = []
        
        if not self.has_paraphraser:
            return paraphrases
        
        self.logger.info("Generating LM-based paraphrase")
        
        try:
            # Format input for the T5 paraphraser with context if provided
            if context == "UAV navigation":
                paraphrase_input = f"In the context of Unmanned Aerial Vehicle navigation instructions, paraphrase: {original_answer}"
            else:
                paraphrase_input = f"Paraphrase: {original_answer}"
            
            # Generate paraphrases
            outputs = self.paraphraser(
                paraphrase_input,
                max_length=min(128, len(original_answer.split()) * 2),
                num_return_sequences=5,  # Try a couple to increase chances of success
                temperature=1.0,
                do_sample=True
            )
            
            # Process outputs
            for output in outputs:
                paraphrase = output['generated_text'].strip()
                
                # Skip if paraphrase is empty or too similar to original
                if not paraphrase or paraphrase == original_answer:
                    continue
                
                # Calculate similarity
                similarity = self.calculate_similarity(original_answer, paraphrase)
                
                # For UAV navigation, we want to ensure key directional information is preserved
                # so we use a higher similarity threshold
                min_similarity = 0.75 if context == "UAV navigation" else 0.7
                max_similarity = 0.95 if context == "UAV navigation" else 0.90
                
                # Only keep if reasonably similar to the original
                if min_similarity <= similarity <= max_similarity:
                    paraphrases.append({
                        "text": paraphrase,
                        "similarity": similarity,
                        "type": "lm_paraphrase"
                    })
            
            # Add best paraphrase to our collection
            if paraphrases:
                # Sort by descending similarity to find the best one
                paraphrases.sort(key=lambda x: x["similarity"], reverse=True)
                self.logger.info(f"Successfully generated LM paraphrase (similarity: {paraphrases[0]['similarity']:.3f})")
                # Return the top max_paraphrases, not nested in another list
                return paraphrases[:max_paraphrases]
            else:
                self.logger.warning("LM paraphraser didn't generate valid paraphrases")
        
        except Exception as e:
            self.logger.warning(f"Error generating LM paraphrase: {str(e)}")
            import traceback
            self.logger.debug(f"LM paraphrase exception details: {traceback.format_exc()}")
        
        return []
    
    def _generate_simple_variations(self, original_answer, n, existing_positives=None):
        """
        Generate simple variations of the original answer as a fallback method.
        
        Args:
            original_answer: Original answer text
            n: Number of variations to generate
            existing_positives: Existing positive examples to check for duplicates
            
        Returns:
            List of simple variations with similarity scores
        """
        # Check if this is likely a UAV navigation instruction
        is_uav_instruction = self._is_uav_navigation_instruction(original_answer)
        
        # Create variations based on the content type
        if is_uav_instruction:
            simple_variations = [
                f"I believe {original_answer}",
                f"{original_answer} That's your destination.",
                f"Based on what I can see, {original_answer.lower()}",
                f"From the drone's view, {original_answer.lower()}",
                f"Yes, {original_answer.lower()}",
                f"Looking at the aerial view, {original_answer.lower()}",
                f"From this position, {original_answer.lower()}",
                f"According to the UAV camera, {original_answer.lower()}",
                f"The drone should {original_answer.lower().replace('turn', '').replace('go', '').strip()}",
                f"To reach the destination, {original_answer.lower().replace('your destination is', '').replace('the destination is', '').strip()}"
            ]
        else:
            simple_variations = [
                f"I believe {original_answer}",
                f"{original_answer} That's your destination.",
                f"Based on what I can see, {original_answer.lower()}",
                f"From the drone's view, {original_answer.lower()}"
            ]
        
        positives = []
        existing_texts = set()
        
        # Get texts from existing positives to avoid duplicates
        if existing_positives:
            existing_texts = {pos["text"] for pos in existing_positives}
        
        # Add simple variations
        for variation in simple_variations:
            if len(positives) >= n:
                break
            
            # Skip if already in existing positives
            if variation in existing_texts:
                continue
                
            # Check if we already have something very similar
            is_unique = True
            if existing_positives:
                for existing in existing_positives:
                    if self.calculate_similarity(existing["text"], variation) > 0.9:
                        is_unique = False
                        break
            
            if is_unique:
                similarity = self.calculate_similarity(original_answer, variation)
                if similarity > 0.7 and variation != original_answer:
                    positives.append({
                        "text": variation,
                        "similarity": similarity,
                        "type": "simple_variation"
                    })
        
        return positives
    
    def generate_negative_examples(self, original_answer, n=3):
        """
        Generate diverse negative examples using multiple strategies.
        
        Uses exactly:
        - 1 template-based negative
        - 1 alternative answer from other dialog turns
        - 1 rule-based negative
        
        Args:
            original_answer: Original answer to generate negatives for
            n: Number of negatives to generate (defaults to 3 but will always return 3)
            
        Returns:
            List of negative examples with similarity scores
        """
        negatives = []
        similarity_threshold = 0.3  # Minimum similarity threshold for all negatives
        
        # Step 1: Try to get multiple template-based negatives and choose the best one
        template_candidates = self.generate_template_negatives(original_answer, 3)
        if template_candidates:
            # Sort by similarity to find the most appropriate template negative
            template_candidates.sort(key=lambda x: x["similarity"], reverse=True)
            # Take the template with highest similarity above threshold
            filtered_templates = [t for t in template_candidates if t["similarity"] >= similarity_threshold]
            if filtered_templates:
                negatives.append(filtered_templates[0])
                self.logger.info(f"Selected template-based negative with similarity: {filtered_templates[0]['similarity']:.3f}")
        
        # Step 2: Try to get an alternative answer from other dialog turns
        if self.alternative_answers:
            alternative_candidates = self.generate_alternative_answer_negatives(original_answer, 3)
            if alternative_candidates:
                # Sort by descending similarity to find the best alternative
                alternative_candidates.sort(key=lambda x: x["similarity"], reverse=True)
                # Take the alternative with highest similarity above threshold
                filtered_alternatives = [a for a in alternative_candidates if a["similarity"] >= similarity_threshold]
                if filtered_alternatives:
                    negatives.append(filtered_alternatives[0])
                    self.logger.info(f"Selected alternative-answer negative with similarity: {filtered_alternatives[0]['similarity']:.3f}")
        
        # Step 3: Try to get multiple rule-based negatives and choose the best one
        rule_candidates = self.generate_rule_based_negatives(original_answer, 3)
        if rule_candidates:
            # Sort by similarity to find the most appropriate rule-based negative
            rule_candidates.sort(key=lambda x: x["similarity"], reverse=True)
            # Take the rule-based with highest similarity above threshold
            filtered_rules = [r for r in rule_candidates if r["similarity"] >= similarity_threshold]
            if filtered_rules:
                negatives.append(filtered_rules[0])
                self.logger.info(f"Selected rule-based negative with similarity: {filtered_rules[0]['similarity']:.3f}")
        
        # If we don't have enough negatives (still < 3), add more from any category that worked
        remaining_slots = 3 - len(negatives)
        if remaining_slots > 0:
            self.logger.warning(f"Only generated {len(negatives)}/3 high-quality negatives, adding fallbacks")
            
            # Try to fill in with any remaining alternative answers
            if 'alternative_candidates' in locals() and len(alternative_candidates) > 1 and remaining_slots > 0:
                for candidate in alternative_candidates[1:]:  # Skip the first one we already used
                    if candidate not in negatives:
                        negatives.append(candidate)
                        remaining_slots -= 1
                        if remaining_slots == 0:
                            break


            # Try to fill in with any remaining template negatives
            if len(template_candidates) > 1 and remaining_slots > 0:
                for candidate in template_candidates[1:]:  # Skip the first one we already used
                    if candidate not in negatives:
                        negatives.append(candidate)
                        remaining_slots -= 1
                        if remaining_slots == 0:
                            break
            
            # Try to fill in with any remaining rule-based negatives
            if len(rule_candidates) > 1 and remaining_slots > 0:
                for candidate in rule_candidates[1:]:  # Skip the first one we already used
                    if candidate not in negatives:
                        negatives.append(candidate)
                        remaining_slots -= 1
                        if remaining_slots == 0:
                            break
            
            # Last resort: generate random negatives
            while remaining_slots > 0:
                random_negative = self._generate_random_negative(original_answer)[0]
                negatives.append({
                    "text": random_negative["text"],
                    "similarity": random_negative["similarity"],
                    "type": "random_negative"
                })
                remaining_slots -= 1
        
        return negatives[:3]  # Always return exactly 3 negatives
    
    def generate_template_negatives(self, original_answer, n=3):
        """
        Generate negative examples using templates that create contradictory navigation instructions.
        
        Args:
            original_answer: Original answer to generate negatives for
            n: Number of negatives to generate
            
        Returns:
            List of negative examples with similarity scores
        """
        negatives = []
        
        # Extract key navigation information
        extracted_info = self._extract_navigation_info(original_answer)
        directions = extracted_info["directions"]
        landmarks = extracted_info["landmarks"]
        colors = extracted_info["colors"] 
        shapes = extracted_info["shapes"]
        spatial_relations = extracted_info["spatial_relations"]
        sizes = extracted_info["sizes"]
        
        # Get templates for negative examples
        templates = self._get_negative_templates()
        
        # Select usable templates based on available information
        usable_templates = []
        for template in templates:
            can_use = True
            
            # Check required elements based on template placeholders
            if "{opposite_direction}" in template and (not directions or len(directions) == 0):
                can_use = False
            if "{different_landmark}" in template and (not landmarks or len(landmarks) == 0):
                can_use = False
            if "{different_color}" in template and (not colors or len(colors) == 0):
                can_use = False
            if "{different_shape}" in template and (not shapes or len(shapes) == 0):
                can_use = False
            
            if can_use:
                usable_templates.append(template)
        
        # Add generic negative templates that don't require specific extracted elements
        generic_templates = self._get_generic_negative_templates()
        usable_templates.extend(generic_templates)
        
        # Generate negatives using templates
        negative_count = 0
        max_attempts = min(30, len(usable_templates) * 3)  # Limit attempts
        attempts = 0
        generated_texts = set()  # Track generated texts to avoid duplicates
        
        while attempts < max_attempts:
            attempts += 1
            
            if not usable_templates:
                break
                
            template = random.choice(usable_templates)
            
            try:
                # Fill the template with appropriate values
                negative = self._fill_negative_template(
                    template,
                    directions,
                    landmarks,
                    colors,
                    shapes,
                    spatial_relations,
                    sizes
                )
                
                # Skip if duplicate or identical to original
                if negative in generated_texts or negative == original_answer:
                    continue
                    
                generated_texts.add(negative)
                    
                # Calculate similarity with original answer
                similarity = self.calculate_similarity(original_answer, negative)
                
                # Only include if different enough from original but not completely unrelated
                if 0.3 <= similarity <= 0.85:
                    negatives.append({
                        "text": negative,
                        "similarity": similarity,
                        "type": "template_negative"
                    })
                    negative_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Error generating template negative: {str(e)}")
                continue
        
        self.logger.info(f"Generated {len(negatives)} template-based negatives")
        negatives.sort(key=lambda x: x["similarity"], reverse=True)
        return negatives[:n]
    
    def _get_negative_templates(self):
        """Get templates for generating contradictory UAV navigation instructions."""
        return [
            # Direction reversal templates
            "Head {opposite_direction} instead of {direction}.",
            "Go {opposite_direction}, not {direction}.",
            "The destination is {opposite_direction}, not {direction} as previously indicated.",
            "You should fly {opposite_direction} to reach the destination.",
            
            # Landmark substitution templates
            "Look for a {different_landmark}, not a {landmark}.",
            "The destination is a {different_landmark}, not a {landmark}.",
            "Your target is the {different_landmark} {direction}, not the {landmark}.",
            
            # Color substitution templates
            "The landmark is {different_color}, not {color}.",
            "Look for a {different_color} building, not a {color} one.",
            "The {landmark} has {different_color} features, not {color}.",
            
            # Shape substitution templates
            "The structure is {different_shape}, not {shape}.",
            "The building has a {different_shape} architecture, not {shape}.",
            
            # Combined attribute templates
            "Go {opposite_direction} to find a {different_color} {different_landmark}, not {direction} to the {color} {landmark}.",
            "The target is a {different_color} {different_shape} structure {opposite_direction}, not a {color} {shape} one {direction}.",
            "Your destination is the {different_color} {different_landmark} with {different_shape} features, not the {color} {landmark} with {shape} elements."
        ]
    
    def _get_generic_negative_templates(self):
        """Get generic negative templates that don't require specific extracted elements."""
        return [
            "The destination is in the completely opposite direction.",
            "You're looking at the wrong area entirely.",
            "That's not the target at all. Look elsewhere.",
            "The target is not where you're currently looking.",
            "You need to search in a different area.",
            "The destination is nowhere near that location.",
            "You're facing the wrong way. Turn around completely.",
            "That's not your destination. Look in another direction.",
            "You need to look for a completely different landmark."
        ]
    
    def _fill_negative_template(self, template, directions, landmarks, colors, shapes, spatial_relations, sizes):
        """
        Fill a negative template with appropriate contradictory values.
        
        Args:
            template: Template string with placeholders
            directions, landmarks, etc.: Navigation information from original answer
            
        Returns:
            Filled template as a negative example
        """
        negative = template
        
        # Get opposite directions
        direction_opposites = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "northeast": "southwest",
            "southwest": "northeast",
            "northwest": "southeast",
            "southeast": "northwest",
            "left": "right",
            "right": "left",
            "forward": "backward",
            "backward": "forward",
            "ahead": "behind",
            "behind": "ahead",
            "clockwise": "counterclockwise",
            "counterclockwise": "clockwise"
        }
        
        # Replace direction placeholders
        if "{direction}" in negative and directions and len(directions) > 0:
            direction = random.choice(directions)
            negative = negative.replace("{direction}", direction)
            
            # Replace opposite direction placeholder
            if "{opposite_direction}" in negative:
                if direction in direction_opposites:
                    opposite = direction_opposites[direction]
                else:
                    # For directions without clear opposites, use a different direction
                    available_opposites = [d for d in self.direction_terms if d != direction]
                    opposite = random.choice(available_opposites)
                negative = negative.replace("{opposite_direction}", opposite)
        elif "{opposite_direction}" in negative:
            # Just pick a random direction if no specific directions in original
            negative = negative.replace("{opposite_direction}", random.choice(self.direction_terms))
            
        # Replace landmark placeholders
        if "{landmark}" in negative and landmarks and len(landmarks) > 0:
            landmark = random.choice(landmarks)
            negative = negative.replace("{landmark}", landmark)
            
            # Replace different landmark placeholder
            if "{different_landmark}" in negative:
                available_alternatives = [l for l in self.landmark_terms if l not in landmarks]
                if available_alternatives:
                    negative = negative.replace("{different_landmark}", random.choice(available_alternatives))
                else:
                    negative = negative.replace("{different_landmark}", "different structure")
        elif "{different_landmark}" in negative:
            negative = negative.replace("{different_landmark}", random.choice(self.landmark_terms))
            
        # Replace color placeholders
        if "{color}" in negative and colors and len(colors) > 0:
            color = random.choice(colors)
            negative = negative.replace("{color}", color)
            
            # Replace different color placeholder
            if "{different_color}" in negative:
                available_alternatives = [c for c in self.color_terms if c not in colors]
                if available_alternatives:
                    negative = negative.replace("{different_color}", random.choice(available_alternatives))
                else:
                    negative = negative.replace("{different_color}", "differently colored")
        elif "{different_color}" in negative:
            negative = negative.replace("{different_color}", random.choice(self.color_terms))
            
        # Replace shape placeholders
        if "{shape}" in negative and shapes and len(shapes) > 0:
            shape = random.choice(shapes)
            negative = negative.replace("{shape}", shape)
            
            # Replace different shape placeholder
            if "{different_shape}" in negative:
                available_alternatives = [s for s in self.shape_terms if s not in shapes]
                if available_alternatives:
                    negative = negative.replace("{different_shape}", random.choice(available_alternatives))
                else:
                    negative = negative.replace("{different_shape}", "differently shaped")
        elif "{different_shape}" in negative:
            negative = negative.replace("{different_shape}", random.choice(self.shape_terms))
            
        return negative
    
    def generate_alternative_answer_negatives(self, original_answer, n=3):
        """
        Generate negative examples using answers from other dialog turns.
        
        Args:
            original_answer: Original answer to generate negatives for
            n: Number of negatives to generate
            
        Returns:
            List of negative examples with similarity scores
        """
        negatives = []
        
        if not self.alternative_answers:
            return negatives
            
        # Filter out answers too similar to original
        filtered_alternatives = []
        for alt_answer in self.alternative_answers:
            if alt_answer != original_answer:
                similarity = self.calculate_similarity(original_answer, alt_answer)
                # Only use as negative if it's different enough but not completely unrelated
                if 0.3 <= similarity <= 0.7:
                    filtered_alternatives.append({
                        "text": alt_answer,
                        "similarity": similarity
                    })
        
        # Sort by ascending similarity (we want the more different ones first)
        filtered_alternatives.sort(key=lambda x: x["similarity"])
        
        # Take the most different alternatives up to n
        for i in range(min(n, len(filtered_alternatives))):
            negatives.append({
                "text": filtered_alternatives[i]["text"],
                "similarity": filtered_alternatives[i]["similarity"],
                "type": "alternative_answer"
            })
            
        self.logger.info(f"Generated {len(negatives)} alternative-answer negatives")
        return negatives
    
    def _extract_navigation_info(self, text):
        """
        Extract key navigation information from text.
        
        Identifies directions, landmarks, colors, shapes, spatial relations, and sizes
        present in the provided text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing lists of extracted terms by category
        """
        text_lower = text.lower()
        
        # Extract each type of navigation information
        directions = self._extract_terms(text_lower, self.direction_terms)
        
        # Special handling for clock directions - they need exact match
        clock_directions = []
        for clock in self.clock_direction_terms:
            if clock in text_lower:
                clock_directions.append(clock)
        
        landmarks = self._extract_terms(text_lower, self.landmark_terms)
        colors = self._extract_terms(text_lower, self.color_terms)
        shapes = self._extract_terms(text_lower, self.shape_terms)
        spatial_relations = self._extract_terms(text_lower, self.spatial_relation_terms)
        sizes = self._extract_terms(text_lower, self.size_terms)
        sequences = self._extract_terms(text_lower, self.sequence_terms)
        positions = self._extract_terms(text_lower, self.position_terms)
        
        return {
            "directions": directions,
            "clock_directions": clock_directions,
            "landmarks": landmarks,
            "colors": colors,
            "shapes": shapes,
            "spatial_relations": spatial_relations,
            "sizes": sizes,
            "sequences": sequences,
            "positions": positions
        }
    
    def _extract_terms(self, text, term_list):
        """
        Extract terms from text that match items in a list.
        
        Args:
            text: Text to search in
            term_list: List of terms to search for
            
        Returns:
            List of matching terms found in the text
        """
        return [term for term in term_list if term in text]
    
    def generate_rule_based_negatives(self, original_answer, n=2):
        """
        Generate negative examples using rule-based methods focused on navigation elements.
        
        This method applies various transformation strategies to create semantically
        opposite navigation instructions.
        
        Args:
            original_answer: Original answer to negate
            n: Number of negatives to generate
            
        Returns:
            List of negative examples with similarity scores
        """
        negatives = []
        
        # Extract navigation elements from the original answer
        extracted_info = self._extract_navigation_info(original_answer)
        
        # Create a list of applicable strategies based on available information
        strategies = self._get_applicable_negative_strategies(extracted_info)
        
        # Ensure we have enough strategies
        while len(strategies) < n:
            strategies.append(random.choice(strategies))
        
        # Randomize strategy order for diversity
        random.shuffle(strategies)
        
        # Generate negatives using different strategies
        generated_texts = set()  # Track generated texts to avoid duplicates
        strategy_idx = 0
        max_attempts = 10  # Avoid infinite loop if strategies keep failing
        
        while len(negatives) < n and strategy_idx < len(strategies) and max_attempts > 0:
            strategy = strategies[strategy_idx]
            try:
                # Call the strategy with appropriate parameters
                new_negatives = self._apply_negative_strategy(
                    strategy, original_answer, extracted_info)
                
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
    
    def _get_applicable_negative_strategies(self, extracted_info):
        """
        Get a list of applicable negative generation strategies based on available information.
        
        Args:
            extracted_info: Dictionary containing extracted navigation information
            
        Returns:
            List of applicable strategy methods
        """
        strategies = []
        
        # Add strategies based on available information
        if extracted_info["directions"]:
            strategies.append(self._generate_direction_reversal_negatives)
            
        if extracted_info["landmarks"]:
            strategies.append(self._generate_landmark_substitution_negatives)
            
        if extracted_info["colors"]:
            strategies.append(self._generate_color_substitution_negatives)
            
        if extracted_info["shapes"]:
            strategies.append(self._generate_shape_substitution_negatives)
            
        if extracted_info["spatial_relations"]:
            strategies.append(self._generate_spatial_relation_negatives)
            
        if extracted_info["sizes"]:
            strategies.append(self._generate_size_substitution_negatives)
            
        # Add semantic frame strategy if we have multiple elements to combine
        has_multiple_elements = sum(1 for elements in extracted_info.values() if elements) >= 2
        if has_multiple_elements:
            strategies.append(self._generate_semantic_frame_negatives)
            
        # Always add these fallback strategies
        strategies.append(self._generate_generic_opposition_negatives)
        strategies.append(self._generate_random_negative)
        
        return strategies
    
    def _apply_negative_strategy(self, strategy, original_answer, extracted_info):
        """
        Apply a negative generation strategy with appropriate parameters.
        
        Args:
            strategy: The strategy method to apply
            original_answer: The original answer text
            extracted_info: Dictionary of extracted navigation information
            
        Returns:
            List of generated negatives
        """
        # Call strategies with the appropriate arguments
        if strategy == self._generate_semantic_frame_negatives:
            return strategy(
                original_answer,
                extracted_info["directions"],
                extracted_info["landmarks"],
                extracted_info["colors"],
                extracted_info["shapes"],
                extracted_info["spatial_relations"],
                extracted_info["sizes"]
            )
        elif strategy == self._generate_landmark_substitution_negatives:
            return strategy(original_answer, extracted_info["landmarks"])
        elif strategy == self._generate_direction_reversal_negatives:
            return strategy(original_answer, extracted_info["directions"])
        elif strategy == self._generate_spatial_relation_negatives:
            return strategy(original_answer, extracted_info["spatial_relations"])
        elif strategy == self._generate_color_substitution_negatives:
            return strategy(original_answer, extracted_info["colors"])
        elif strategy == self._generate_shape_substitution_negatives:
            return strategy(original_answer, extracted_info["shapes"])
        elif strategy == self._generate_size_substitution_negatives:
            return strategy(original_answer, extracted_info["sizes"])
        else:
            return strategy(original_answer)
    
    def _generate_direction_reversal_negatives(self, original_answer, directions):
        """
        Generate negatives by reversing or changing directions mentioned in the answer.
        
        Args:
            original_answer: The original answer text
            directions: List of direction terms found in the original answer
            
        Returns:
            List of negatives with reversed directions
        """
        if not directions:
            return []
            
        negatives = []
        
        # Map of directions to their opposites
        direction_opposites = {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "northeast": "southwest",
            "southwest": "northeast",
            "northwest": "southeast",
            "southeast": "northwest",
            "left": "right",
            "right": "left",
            "forward": "backward",
            "backward": "forward",
            "ahead": "behind",
            "behind": "ahead",
            "clockwise": "counterclockwise",
            "counterclockwise": "clockwise"
        }
        
        # Get a direction from the original answer
        direction = random.choice(directions)
        
        # Generate a negative with the opposite direction
        if direction in direction_opposites:
            opposite = direction_opposites[direction]
            negative_text = original_answer.replace(direction, opposite)
            
            # Only use if it actually changed something
            if negative_text != original_answer:
                similarity = self.calculate_similarity(original_answer, negative_text)
                negatives.append({
                    "text": negative_text,
                    "similarity": similarity,
                    "type": "direction_reversal"
                })
        
        return negatives
    
    def _generate_landmark_substitution_negatives(self, original_answer, landmarks):
        """
        Generate negatives by substituting landmarks with different ones.
        
        Args:
            original_answer: The original answer text
            landmarks: List of landmark terms found in the original answer
            
        Returns:
            List of negatives with substituted landmarks
        """
        if not landmarks:
            return []
            
        negatives = []
        
        # Get a landmark from the original answer
        landmark = random.choice(landmarks)
        
        # Get a list of alternative landmarks not in the original answer
        alternative_landmarks = [l for l in self.landmark_terms if l not in landmarks]
        
        if alternative_landmarks:
            # Generate a negative with a different landmark
            alternative = random.choice(alternative_landmarks)
            negative_text = original_answer.replace(landmark, alternative)
            
            # Only use if it actually changed something
            if negative_text != original_answer:
                similarity = self.calculate_similarity(original_answer, negative_text)
                negatives.append({
                    "text": negative_text,
                    "similarity": similarity,
                    "type": "landmark_substitution"
                })
        
        return negatives
    
    def _generate_color_substitution_negatives(self, original_answer, colors):
        """
        Generate negatives by substituting colors with different ones.
        
        Args:
            original_answer: The original answer text
            colors: List of color terms found in the original answer
            
        Returns:
            List of negatives with substituted colors
        """
        if not colors:
            return []
            
        negatives = []
        
        # Get a color from the original answer
        color = random.choice(colors)
        
        # Get a list of alternative colors not in the original answer
        alternative_colors = [c for c in self.color_terms if c not in colors]
        
        if alternative_colors:
            # Generate a negative with a different color
            alternative = random.choice(alternative_colors)
            negative_text = original_answer.replace(color, alternative)
            
            # Only use if it actually changed something
            if negative_text != original_answer:
                similarity = self.calculate_similarity(original_answer, negative_text)
                negatives.append({
                    "text": negative_text,
                    "similarity": similarity,
                    "type": "color_substitution"
                })
        
        return negatives
    
    def _generate_shape_substitution_negatives(self, original_answer, shapes):
        """
        Generate negatives by substituting shapes with different ones.
        
        Args:
            original_answer: The original answer text
            shapes: List of shape terms found in the original answer
            
        Returns:
            List of negatives with substituted shapes
        """
        if not shapes:
            return []
            
        negatives = []
        
        # Get a shape from the original answer
        shape = random.choice(shapes)
        
        # Get a list of alternative shapes not in the original answer
        alternative_shapes = [s for s in self.shape_terms if s not in shapes]
        
        if alternative_shapes:
            # Generate a negative with a different shape
            alternative = random.choice(alternative_shapes)
            negative_text = original_answer.replace(shape, alternative)
            
            # Only use if it actually changed something
            if negative_text != original_answer:
                similarity = self.calculate_similarity(original_answer, negative_text)
                negatives.append({
                    "text": negative_text,
                    "similarity": similarity,
                    "type": "shape_substitution"
                })
        
        return negatives
    
    def _generate_spatial_relation_negatives(self, original_answer, spatial_relations):
        """
        Generate negatives by substituting spatial relations with different ones.
        
        Args:
            original_answer: The original answer text
            spatial_relations: List of spatial relation terms found in the original answer
            
        Returns:
            List of negatives with substituted spatial relations
        """
        if not spatial_relations:
            return []
            
        negatives = []
        
        # Get a spatial relation from the original answer
        relation = random.choice(spatial_relations)
        
        # Get a list of alternative spatial relations not in the original answer
        alternative_relations = [r for r in self.spatial_relation_terms if r not in spatial_relations]
        
        if alternative_relations:
            # Generate a negative with a different spatial relation
            alternative = random.choice(alternative_relations)
            negative_text = original_answer.replace(relation, alternative)
            
            # Only use if it actually changed something
            if negative_text != original_answer:
                similarity = self.calculate_similarity(original_answer, negative_text)
                negatives.append({
                    "text": negative_text,
                    "similarity": similarity,
                    "type": "spatial_relation_substitution"
                })
        
        return negatives
    
    def _generate_size_substitution_negatives(self, original_answer, sizes):
        """
        Generate negatives by substituting sizes with different ones.
        
        Args:
            original_answer: The original answer text
            sizes: List of size terms found in the original answer
            
        Returns:
            List of negatives with substituted sizes
        """
        if not sizes:
            return []
            
        negatives = []
        
        # Get a size from the original answer
        size = random.choice(sizes)
        
        # Get a list of alternative sizes not in the original answer
        alternative_sizes = [s for s in self.size_terms if s not in sizes]
        
        if alternative_sizes:
            # Generate a negative with a different size
            alternative = random.choice(alternative_sizes)
            negative_text = original_answer.replace(size, alternative)
            
            # Only use if it actually changed something
            if negative_text != original_answer:
                similarity = self.calculate_similarity(original_answer, negative_text)
                negatives.append({
                    "text": negative_text,
                    "similarity": similarity,
                    "type": "size_substitution"
                })
        
        return negatives
    
    def _generate_semantic_frame_negatives(self, original_answer, directions, landmarks, colors, shapes, spatial_relations, sizes):
        """
        Generate negatives by changing multiple elements of the navigation instruction.
        
        Args:
            original_answer: The original answer text
            directions, landmarks, etc.: Lists of terms found in the original answer
            
        Returns:
            List of negatives with multiple element changes
        """
        negatives = []
        
        # Need to have at least two types of elements to create meaningful changes
        element_types = []
        if directions:
            element_types.append("directions")
        if landmarks:
            element_types.append("landmarks")
        if colors:
            element_types.append("colors")
        if shapes:
            element_types.append("shapes")
        if spatial_relations:
            element_types.append("spatial_relations")
        if sizes:
            element_types.append("sizes")
        
        if len(element_types) < 2:
            return []
        
        # Select two random element types to change
        selected_types = random.sample(element_types, 2)
        
        # Create a working copy of the answer
        negative_text = original_answer
        
        # Apply changes for each selected element type
        for elem_type in selected_types:
            if elem_type == "directions" and directions:
                direction = random.choice(directions)
                opposite_map = {
                    "north": "south", "south": "north", "east": "west", "west": "east",
                    "left": "right", "right": "left", "forward": "backward", "backward": "forward"
                }
                if direction in opposite_map:
                    negative_text = negative_text.replace(direction, opposite_map[direction])
            
            elif elem_type == "landmarks" and landmarks:
                landmark = random.choice(landmarks)
                alternative_landmarks = [l for l in self.landmark_terms if l not in landmarks]
                if alternative_landmarks:
                    negative_text = negative_text.replace(landmark, random.choice(alternative_landmarks))
            
            elif elem_type == "colors" and colors:
                color = random.choice(colors)
                alternative_colors = [c for c in self.color_terms if c not in colors]
                if alternative_colors:
                    negative_text = negative_text.replace(color, random.choice(alternative_colors))
            
            elif elem_type == "shapes" and shapes:
                shape = random.choice(shapes)
                alternative_shapes = [s for s in self.shape_terms if s not in shapes]
                if alternative_shapes:
                    negative_text = negative_text.replace(shape, random.choice(alternative_shapes))
            
            elif elem_type == "spatial_relations" and spatial_relations:
                relation = random.choice(spatial_relations)
                alternative_relations = [r for r in self.spatial_relation_terms if r not in spatial_relations]
                if alternative_relations:
                    negative_text = negative_text.replace(relation, random.choice(alternative_relations))
            
            elif elem_type == "sizes" and sizes:
                size = random.choice(sizes)
                alternative_sizes = [s for s in self.size_terms if s not in sizes]
                if alternative_sizes:
                    negative_text = negative_text.replace(size, random.choice(alternative_sizes))
        
        # Only add if we managed to change the text
        if negative_text != original_answer:
            similarity = self.calculate_similarity(original_answer, negative_text)
            negatives.append({
                "text": negative_text,
                "similarity": similarity,
                "type": "semantic_frame"
            })
        
        return negatives
    
    def _generate_generic_opposition_negatives(self, original_answer):
        """
        Generate generic negatives that contradict the original answer.
        
        Args:
            original_answer: The original answer text
            
        Returns:
            List containing a generic contradictory negative
        """
        generic_oppositions = [
            "The destination is in the opposite direction.",
            "You need to look in a completely different location.",
            "That's not the right place. Your target is elsewhere.",
            "The destination is not there. You need to look in another direction.",
            "You're facing the wrong way. The target is in the opposite direction.",
            "That's incorrect. The landmark you should be looking for is completely different."
        ]
        
        negative_text = random.choice(generic_oppositions)
        similarity = self.calculate_similarity(original_answer, negative_text)
        
        return [{
            "text": negative_text,
            "similarity": similarity,
            "type": "generic_opposition"
        }]
    
    def _generate_random_negative(self, original_answer):
        """
        Generate a random negative example as a fallback strategy.
        
        Args:
            original_answer: Original answer to compare against
            
        Returns:
            List containing a single random negative example
        """
        # Generate random navigation instruction from templates
        templates = [
            "Head {direction} until you reach the {color} {landmark}.",
            "Make a {direction} turn and go towards the {size} {landmark}.",
            "Fly in the {direction} direction to find a {shape} building.",
            "Go {direction} about {distance} meters and look for a {color} {landmark}.",
            "Your destination is the {color} {landmark} to the {direction}.",
            "Turn to {time} o'clock direction and proceed to the {landmark}."
        ]
        
        # Select a template and fill in random values
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
        
        return [{
            "text": negative_text,
            "similarity": similarity,
            "type": "random_negative"
        }]
    
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

    def augment_dialog_turn(self, question, answer, num_positives=2, num_negatives=3, dialog_answers=None):
        """
        Generate contrastive samples for a dialog turn from AVDN dataset.
        
        Args:
            question: The question from the dialog turn
            answer: The answer to generate contrastive samples for
            num_positives: Number of positive examples to generate
            num_negatives: Number of negative examples to generate
            dialog_answers: List of answers from other dialog turns to use as potential negatives
            
        Returns:
            Dictionary with contrastive samples and complexity metadata
        """
        if not answer:
            return None
        
        # Extract instruction from question if it follows AVDN format
        # In AVDN, questions often contain "[QUE]" and answers contain "[INS]"
        extracted_instruction = None
        if question and "[INS]" in question:
            # Extract instruction part from the question
            ins_start = question.find("[INS]")
            if ins_start >= 0:
                extracted_instruction = question[ins_start+5:].strip()
        
        # Store alternative answers for negative generation
        if dialog_answers:
            self.alternative_answers = [a for a in dialog_answers if a != answer]
        else:
            self.alternative_answers = []
        
        # If we have an extracted instruction, use it to enhance the answer context
        enhanced_answer = answer
        if extracted_instruction:
            # Check if the instruction provides additional context not in the answer
            if self.calculate_similarity(extracted_instruction, answer) < 0.8:
                # Combine instruction with answer for better contrastive samples
                enhanced_answer = f"{extracted_instruction} {answer}"
        
        contrastive_samples = {
            "positive_examples": self.generate_positive_examples(answer, num_positives),
            "negative_examples": self.generate_negative_examples(enhanced_answer, num_negatives)
        }
        
        complexity_metadata = self.analyze_complexity(question, answer)
        
        return {
            "contrastive_samples": contrastive_samples,
            "complexity_metadata": complexity_metadata
        }
    
    def augment_dataset(self, input_json_path, output_json_path, num_positives=2, num_negatives=3):
        """
        Augment an entire dataset with contrastive samples.
        
        Args:
            input_json_path: Path to input JSON dataset
            output_json_path: Path to output augmented JSON dataset
            num_positives: Number of positive examples per dialog
            num_negatives: Number of negative examples per dialog
            
        Returns:
            Number of augmented dialog turns
        """
        # Load dataset
        self.logger.info(f"Loading dataset from {input_json_path}")
        with open(input_json_path, 'r') as f:
            data = json.load(f)
        
        self.logger.info(f"Processing {len(data)} episodes")
        
        augmented_count = 0
        total_dialogs = 0
        
        # Process each episode
        for episode in tqdm.tqdm(data, desc="Augmenting episodes"):
            # Collect all answers in this episode for use as potential negatives
            episode_answers = []
            
            # Check for AVDN dataset format
            is_avdn_format = False
            if "pre_dialogs" in episode or "instructions" in episode:
                is_avdn_format = True
            
            if is_avdn_format:
                # AVDN format - extract instructions from the episode
                if "instructions" in episode:
                    instruction = episode["instructions"]
                    if isinstance(instruction, str) and "[INS]" in instruction:
                        # Extract the instruction part
                        ins_start = instruction.find("[INS]")
                        if ins_start >= 0:
                            answer = instruction[ins_start+5:].strip()
                            episode_answers.append(answer)
                
                # Process pre_dialogs if available
                if "pre_dialogs" in episode and isinstance(episode["pre_dialogs"], list):
                    for dialog in episode["pre_dialogs"]:
                        if isinstance(dialog, str) and "[INS]" in dialog:
                            # Extract the instruction part
                            ins_start = dialog.find("[INS]")
                            if ins_start >= 0:
                                answer = dialog[ins_start+5:].strip()
                                episode_answers.append(answer)
                
                # For AVDN format, we augment the current instruction
                if "instructions" in episode and isinstance(episode["instructions"], str):
                    total_dialogs += 1
                    instruction = episode["instructions"]
                    
                    # Extract question and answer parts
                    question = None
                    answer = None
                    
                    if "[QUE]" in instruction and "[INS]" in instruction:
                        # Split into question and answer
                        que_start = instruction.find("[QUE]")
                        ins_start = instruction.find("[INS]")
                        
                        if que_start >= 0 and ins_start >= 0:
                            question = instruction[que_start+5:ins_start].strip()
                            answer = instruction[ins_start+5:].strip()
                    elif "[INS]" in instruction:
                        # Only instruction part
                        ins_start = instruction.find("[INS]")
                        if ins_start >= 0:
                            answer = instruction[ins_start+5:].strip()
                    
                    if answer:
                        # Augment this dialog turn
                        augmentation = self.augment_dialog_turn(
                            question=question,
                            answer=answer,
                            num_positives=num_positives,
                            num_negatives=num_negatives,
                            dialog_answers=episode_answers
                        )
                        
                        if augmentation:
                            # Add augmentation to the episode
                            episode["contrastive_augmentation"] = augmentation
                            augmented_count += 1
            else:
                # Standard format with dialogs list
                for dialog in episode.get("dialogs", []):
                    if dialog.get("answer"):
                        episode_answers.append(dialog["answer"])
                
                # Process each dialog
                for dialog in episode.get("dialogs", []):
                    total_dialogs += 1
                    if dialog.get("question") is not None and dialog.get("answer") is not None:
                        # Pass other answers from the episode as potential negatives
                        augmentation = self.augment_dialog_turn(
                            dialog["question"], 
                            dialog["answer"],
                            num_positives=num_positives,
                            num_negatives=num_negatives,
                            dialog_answers=episode_answers
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
    
    def process_avdn_episode(self, episode, num_positives=2, num_negatives=3):
        """
        Process an AVDN dataset episode to extract and augment instructions.
        
        Args:
            episode: AVDN dataset episode
            num_positives: Number of positive examples to generate
            num_negatives: Number of negative examples to generate
            
        Returns:
            Tuple of (augmented_count, total_dialogs, episode_answers)
        """
        augmented_count = 0
        total_dialogs = 0
        episode_answers = []
        
        # Extract instructions from the episode
        if "instructions" in episode:
            instruction = episode["instructions"]
            if isinstance(instruction, str) and "[INS]" in instruction:
                # Extract the instruction part
                ins_start = instruction.find("[INS]")
                if ins_start >= 0:
                    answer = instruction[ins_start+5:].strip()
                    episode_answers.append(answer)
        
        # Process pre_dialogs if available
        if "pre_dialogs" in episode and isinstance(episode["pre_dialogs"], list):
            for dialog in episode["pre_dialogs"]:
                if isinstance(dialog, str) and "[INS]" in dialog:
                    # Extract the instruction part
                    ins_start = dialog.find("[INS]")
                    if ins_start >= 0:
                        answer = dialog[ins_start+5:].strip()
                        episode_answers.append(answer)
        
        # For AVDN format, we augment the current instruction
        if "instructions" in episode and isinstance(episode["instructions"], str):
            total_dialogs += 1
            instruction = episode["instructions"]
            
            # Extract question and answer parts
            question = None
            answer = None
            
            if "[QUE]" in instruction and "[INS]" in instruction:
                # Split into question and answer
                que_start = instruction.find("[QUE]")
                ins_start = instruction.find("[INS]")
                
                if que_start >= 0 and ins_start >= 0:
                    question = instruction[que_start+5:ins_start].strip()
                    answer = instruction[ins_start+5:].strip()
            elif "[INS]" in instruction:
                # Only instruction part
                ins_start = instruction.find("[INS]")
                if ins_start >= 0:
                    answer = instruction[ins_start+5:].strip()
            
            if answer:
                # Augment this dialog turn
                augmentation = self.augment_dialog_turn(
                    question=question,
                    answer=answer,
                    num_positives=num_positives,
                    num_negatives=num_negatives,
                    dialog_answers=episode_answers
                )
                
                if augmentation:
                    # Add augmentation to the episode
                    episode["contrastive_augmentation"] = augmentation
                    augmented_count += 1
        
        return augmented_count, total_dialogs, episode_answers
    
    def extract_avdn_instructions(self, episode):
        """
        Extract instructions from an AVDN dataset episode.
        
        Args:
            episode: AVDN dataset episode
            
        Returns:
            List of extracted instructions
        """
        instructions = []
        
        # Extract main instruction
        if "instructions" in episode and isinstance(episode["instructions"], str):
            instruction = episode["instructions"]
            if "[INS]" in instruction:
                ins_start = instruction.find("[INS]")
                if ins_start >= 0:
                    answer = instruction[ins_start+5:].strip()
                    instructions.append(answer)
        
        # Extract from pre_dialogs
        if "pre_dialogs" in episode and isinstance(episode["pre_dialogs"], list):
            for dialog in episode["pre_dialogs"]:
                if isinstance(dialog, str) and "[INS]" in dialog:
                    ins_start = dialog.find("[INS]")
                    if ins_start >= 0:
                        answer = dialog[ins_start+5:].strip()
                        instructions.append(answer)
        
        return instructions
    
    def _preserve_multi_step_structure(self, original_text):
        """
        Analyze and preserve multi-step structure in navigation instructions.
        
        Args:
            original_text: Original navigation instruction text
            
        Returns:
            Dictionary with identified steps and their components
        """
        # Check if text contains multiple steps (indicated by periods, commas, or sequence terms)
        original_lower = original_text.lower()
        sentences = []
        
        # First try to split by periods
        if "." in original_text:
            # Split by periods but preserve them
            raw_sentences = original_text.split(".")
            sentences = [s.strip() + "." for s in raw_sentences if s.strip()]
            # Remove period from last sentence if it doesn't end with one
            if sentences and not original_text.endswith("."):
                sentences[-1] = sentences[-1][:-1]
        # If no periods, try commas
        elif "," in original_text:
            raw_sentences = original_text.split(",")
            sentences = [s.strip() + "," for s in raw_sentences if s.strip()]
            # Remove comma from last sentence if it doesn't end with one
            if sentences and not original_text.endswith(","):
                sentences[-1] = sentences[-1][:-1]
        # If no clear sentence breaks, check for sequence terms
        else:
            for term in ["then", "after", "next", "when", "until"]:
                if term in original_lower:
                    parts = re.split(f"({term})", original_text, flags=re.IGNORECASE)
                    if len(parts) > 1:
                        sentences = []
                        for i in range(0, len(parts)-1, 2):
                            if i+1 < len(parts):
                                sentences.append(f"{parts[i].strip()}")
                                sentences.append(f"{parts[i+1]} {parts[i+2].strip()}")
                        break
        
        # If still no clear structure, treat as single instruction
        if not sentences:
            sentences = [original_text]
            
        # Extract key components from each step
        steps = []
        for i, sentence in enumerate(sentences):
            step_info = {
                "text": sentence,
                "index": i,
                "has_direction": False,
                "has_landmark": False,
                "has_clock": False,
                "has_action": False,
                "is_destination": "destination" in sentence.lower() or "arrived" in sentence.lower()
            }
            
            # Check for directions
            for direction in self.direction_terms:
                if direction in sentence.lower():
                    step_info["has_direction"] = True
                    break
                    
            # Check for landmarks
            for landmark in self.landmark_terms:
                if landmark in sentence.lower():
                    step_info["has_landmark"] = True
                    break
                    
            # Check for clock references
            step_info["has_clock"] = any(clock in sentence.lower() for clock in ["o'clock", "oclock", "am", "pm"])
            
            # Check for action verbs
            for verb in self._get_navigation_action_verbs():
                if verb in sentence.lower():
                    step_info["has_action"] = True
                    break
                    
            steps.append(step_info)
            
        return {
            "original_text": original_text,
            "steps": steps,
            "is_multi_step": len(steps) > 1
        }
    
    def _is_multi_step_instruction(self, text):
        """
        Check if the text contains multiple navigation steps.
        
        Args:
            text: Text to analyze
            
        Returns:
            Boolean indicating if this is a multi-step instruction
        """
        text_lower = text.lower()
        
        # Check for multiple sentences (periods)
        if text.count('.') > 1:
            return True
            
        # Check for sequence terms that indicate multiple steps
        sequence_indicators = ["then", "after", "next", "first", "second", "finally", "when", "until"]
        for indicator in sequence_indicators:
            if indicator in text_lower:
                return True
                
        # Check for multiple landmarks or directions
        landmark_count = sum(1 for landmark in self.landmark_terms if landmark in text_lower)
        direction_count = sum(1 for direction in self.direction_terms if direction in text_lower)
        
        # If text has multiple landmarks and directions, likely multi-step
        if landmark_count > 1 and direction_count > 1:
            return True
            
        return False
    
    def _generate_multi_step_paraphrases(self, original_answer, n=1):
        """
        Generate paraphrases for multi-step navigation instructions.
        
        Args:
            original_answer: Original multi-step instruction
            n: Number of paraphrases to generate
            
        Returns:
            List of paraphrases with similarity scores
        """
        # For now, fall back to standard template paraphrasing
        # This is a simplified implementation that will be improved later
        return self.generate_template_paraphrases(original_answer, n)