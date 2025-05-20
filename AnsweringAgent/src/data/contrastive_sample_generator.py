import torch
import torch.nn.functional as F
import json
import random
import numpy as np
import os
import tqdm
import logging
from transformers import AutoTokenizer, AutoModel

class ContrastiveSampleGenerator:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the contrastive sample generator with a sentence embedding model."""
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loading sentence embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Load templates for negative examples
        self.direction_terms = ["north", "south", "east", "west", "left", "right", 
                               "forward", "backward", "ahead", "behind", "clockwise",
                               "counterclockwise", "o'clock"]
        self.landmark_terms = ["building", "house", "structure", "tower", "road", "bridge", 
                              "hill", "mountain", "tree", "field", "river", "lake"]
        self.color_terms = ["red", "blue", "green", "yellow", "black", "white", "gray", 
                           "brown", "purple", "orange", "beige", "pink"]
        self.shape_terms = ["square", "rectangular", "round", "circular", "oval", "triangular", 
                           "dome", "L-shaped", "U-shaped", "flat"]
        
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
    
    def generate_positive_examples(self, original_answer, n=2):
        """Generate positive examples (paraphrases) for an answer."""
        positives = []
        
        # Template-based paraphrasing
        templates = [
            "The destination can be found {direction}. It's {shape} and {color}.",
            "If you look {direction}, you'll see the destination which is {shape} with {color} color.",
            "Your destination is {direction} from your position. Look for a {color} {shape} structure.",
            "Head {direction} to reach your destination. It's {color} and {shape}.",
            "{direction} is where you'll find your destination. It's a {color} {shape} building."
        ]
        
        # Extract key information from original answer
        direction = None
        shape = None
        color = None
        
        for term in self.direction_terms:
            if term in original_answer.lower():
                direction = term
                break
                
        for term in self.shape_terms:
            if term in original_answer.lower():
                shape = term
                break
                
        for term in self.color_terms:
            if term in original_answer.lower():
                color = term
                break
        
        # Generate paraphrases if we have the key information
        if direction and (shape or color):
            available_templates = templates.copy()
            for i in range(min(n, len(available_templates))):
                template = random.choice(available_templates)
                available_templates.remove(template)  # Avoid duplicates
                
                paraphrase = template.format(
                    direction=direction,
                    shape=shape or "building",
                    color=color or "regular"
                )
                
                # Calculate similarity
                similarity = self.calculate_similarity(original_answer, paraphrase)
                
                positives.append({
                    "text": paraphrase,
                    "similarity": similarity,
                    "type": "paraphrase"
                })
        
        # If we couldn't generate enough paraphrases, add minor variations
        while len(positives) < n:
            # Minor word substitutions
            words = original_answer.split()
            if len(words) >= 3:
                idx = random.randint(0, len(words) - 1)
                if words[idx].lower() in ["the", "a", "is", "your"]:  # Safe words to modify
                    synonyms = {"the": "your", "a": "the", "is": "appears to be", "your": "the"}
                    if words[idx].lower() in synonyms:
                        words[idx] = synonyms[words[idx].lower()]
                    
                paraphrase = " ".join(words)
                similarity = self.calculate_similarity(original_answer, paraphrase)
                
                positives.append({
                    "text": paraphrase,
                    "similarity": similarity,
                    "type": "minor_variation"
                })
            else:
                # If we can't make a sensible paraphrase, just duplicate
                positives.append({
                    "text": original_answer,
                    "similarity": 1.0,
                    "type": "duplicate"
                })
        
        return positives
    
    def generate_negative_examples(self, original_answer, n=3):
        """Generate negative examples by modifying key aspects of the answer."""
        negatives = []
        
        # Extract key information from original answer
        direction = None
        shape = None
        color = None
        
        for term in self.direction_terms:
            if term in original_answer.lower():
                direction = term
                break
                
        for term in self.shape_terms:
            if term in original_answer.lower():
                shape = term
                break
                
        for term in self.color_terms:
            if term in original_answer.lower():
                color = term
                break
        
        # 1. Change direction
        if direction:
            wrong_directions = [d for d in self.direction_terms if d != direction]
            if wrong_directions:
                wrong_dir = random.choice(wrong_directions)
                negative = original_answer.replace(direction, wrong_dir)
                similarity = self.calculate_similarity(original_answer, negative)
                
                negatives.append({
                    "text": negative,
                    "similarity": similarity,
                    "type": "wrong_direction"
                })
        
        # 2. Change shape/structure
        if shape:
            wrong_shapes = [s for s in self.shape_terms if s != shape]
            if wrong_shapes:
                wrong_shape = random.choice(wrong_shapes)
                negative = original_answer.replace(shape, wrong_shape)
                similarity = self.calculate_similarity(original_answer, negative)
                
                negatives.append({
                    "text": negative,
                    "similarity": similarity,
                    "type": "wrong_shape"
                })
        
        # 3. Change color
        if color:
            wrong_colors = [c for c in self.color_terms if c != color]
            if wrong_colors:
                wrong_color = random.choice(wrong_colors)
                negative = original_answer.replace(color, wrong_color)
                similarity = self.calculate_similarity(original_answer, negative)
                
                negatives.append({
                    "text": negative,
                    "similarity": similarity,
                    "type": "wrong_color"
                })
        
        # 4. Completely unrelated answer (if needed to fill quota)
        unrelated_templates = [
            "I don't see any destination in that area.",
            "Continue flying straight ahead for now.",
            "There are no buildings matching that description.",
            "The destination is not visible from your current position.",
            "You need to change altitude to locate the destination."
        ]
        
        while len(negatives) < n:
            unrelated = random.choice(unrelated_templates)
            similarity = self.calculate_similarity(original_answer, unrelated)
            
            negatives.append({
                "text": unrelated,
                "similarity": similarity,
                "type": "unrelated"
            })
        
        return negatives[:n]  # Return at most n examples
    
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