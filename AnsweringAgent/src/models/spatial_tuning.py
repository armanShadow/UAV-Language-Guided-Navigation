#!/usr/bin/env python3
"""
Spatial Fine-tuning Helper
Additional utilities for improving spatial reasoning performance.
"""

import torch
import torch.nn as nn
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class SpatialTuningHelper:
    """Helper class for spatial reasoning fine-tuning adjustments."""
    
    @staticmethod
    def unfreeze_additional_decoder_layers(model, num_additional_layers: int = 2):
        """
        Unfreeze additional decoder layers for better spatial reasoning.
        
        Args:
            model: AnsweringAgent model
            num_additional_layers: Number of additional decoder layers to unfreeze
        """
        # Get the actual T5 model (handle DDP wrapping)
        t5_model = model.module.t5_model if hasattr(model, 'module') else model.t5_model
        
        # Currently last 3 encoder blocks are unfrozen
        # Now also unfreeze last N decoder blocks
        total_decoder_blocks = len(t5_model.decoder.block)
        
        logger.info(f"Unfreezing last {num_additional_layers} decoder blocks for spatial reasoning")
        logger.info(f"Total decoder blocks: {total_decoder_blocks}")
        
        for idx in range(total_decoder_blocks - num_additional_layers, total_decoder_blocks):
            for name, param in t5_model.decoder.block[idx].named_parameters():
                param.requires_grad = True
                logger.debug(f"Unfrozen decoder block {idx}: {name}")
        
        # Also unfreeze final layer norm
        if hasattr(t5_model.decoder, 'final_layer_norm'):
            t5_model.decoder.final_layer_norm.weight.requires_grad = True
            if t5_model.decoder.final_layer_norm.bias is not None:
                t5_model.decoder.final_layer_norm.bias.requires_grad = True
            logger.info("Unfrozen decoder final layer norm")
        
        # Count new trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"Updated trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable percentage: {trainable_params/total_params*100:.2f}%")
        
        return trainable_params
    
    @staticmethod
    def add_directional_auxiliary_head(model, num_directions: int = 12):
        """
        Add auxiliary directional classification head for explicit spatial learning.
        
        Args:
            model: AnsweringAgent model
            num_directions: Number of directional classes (12 for clock directions)
        """
        # Get the actual model (handle DDP wrapping)
        actual_model = model.module if hasattr(model, 'module') else model
        
        # Add directional classification head
        actual_model.directional_head = nn.Sequential(
            nn.Linear(actual_model.config.model.hidden_size, actual_model.config.model.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(actual_model.config.model.hidden_size // 2, num_directions)
        )
        
        # Move to same device as model
        device = next(actual_model.parameters()).device
        actual_model.directional_head.to(device)
        
        logger.info(f"Added directional auxiliary head with {num_directions} classes")
        return actual_model.directional_head
    
    @staticmethod
    def create_spatial_prompt_variants() -> List[str]:
        """Create different spatial reasoning prompt variants for testing."""
        prompts = [
            "Provide precise navigation with clock directions (1-12 o'clock), landmark descriptions, and clear spatial instructions. ",
            "Answer with exact directions using o'clock positions, building colors/shapes, and specific movement instructions. ",
            "Give detailed spatial guidance including clock directions, landmark features, and navigation steps. ",
            "Respond with precise clock-based directions, describe visible landmarks, and provide clear movement commands. ",
            "Use clock positions (1-12 o'clock), landmark characteristics, and specific directional guidance in your answer. "
        ]
        return prompts
    
    @staticmethod
    def extract_spatial_metrics(predicted_text: str, target_text: str) -> Dict[str, float]:
        """
        Extract spatial reasoning metrics from generated vs target text.
        
        Args:
            predicted_text: Generated text
            target_text: Ground truth text
            
        Returns:
            Dictionary with spatial accuracy metrics
        """
        import re
        
        def extract_clock_directions(text):
            """Extract clock directions from text."""
            pattern = r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*o\'?clock'
            matches = re.findall(pattern, text.lower())
            # Normalize to numbers
            word_to_num = {
                'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
                'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10', 'eleven': '11', 'twelve': '12'
            }
            normalized = [word_to_num.get(m, m) for m in matches]
            return set(normalized)
        
        def extract_directions(text):
            """Extract directional words."""
            directions = ['north', 'south', 'east', 'west', 'left', 'right', 'forward', 'backward', 'straight']
            found = []
            for direction in directions:
                if re.search(rf'\b{direction}\b', text.lower()):
                    found.append(direction)
            return set(found)
        
        def extract_landmarks(text):
            """Extract landmark words."""
            landmarks = ['building', 'structure', 'road', 'street', 'house', 'parking', 'lot']
            found = []
            for landmark in landmarks:
                if re.search(rf'\b{landmark}\b', text.lower()):
                    found.append(landmark)
            return set(found)
        
        # Extract features
        pred_clocks = extract_clock_directions(predicted_text)
        target_clocks = extract_clock_directions(target_text)
        
        pred_directions = extract_directions(predicted_text)
        target_directions = extract_directions(target_text)
        
        pred_landmarks = extract_landmarks(predicted_text)
        target_landmarks = extract_landmarks(target_text)
        
        # Calculate accuracy scores
        def jaccard_similarity(set1, set2):
            if not set1 and not set2:
                return 1.0
            union = set1 | set2
            intersection = set1 & set2
            return len(intersection) / len(union) if union else 0.0
        
        clock_accuracy = jaccard_similarity(pred_clocks, target_clocks)
        direction_accuracy = jaccard_similarity(pred_directions, target_directions)
        landmark_accuracy = jaccard_similarity(pred_landmarks, target_landmarks)
        
        # Overall spatial accuracy
        spatial_accuracy = (clock_accuracy + direction_accuracy + landmark_accuracy) / 3
        
        return {
            'clock_accuracy': clock_accuracy,
            'direction_accuracy': direction_accuracy,
            'landmark_accuracy': landmark_accuracy,
            'spatial_accuracy': spatial_accuracy,
            'pred_clocks': list(pred_clocks),
            'target_clocks': list(target_clocks),
            'pred_directions': list(pred_directions),
            'target_directions': list(target_directions),
            'pred_landmarks': list(pred_landmarks),
            'target_landmarks': list(target_landmarks)
        }

# Quick test function
def test_spatial_metrics():
    """Test spatial metrics extraction."""
    predicted = "Turn to your 3 o'clock direction and head towards the red building."
    target = "Go to 3 o'clock and move to the red building structure."
    
    metrics = SpatialTuningHelper.extract_spatial_metrics(predicted, target)
    print("Spatial metrics test:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_spatial_metrics() 