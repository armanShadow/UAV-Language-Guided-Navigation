import torch
import cv2
import numpy as np
import pandas as pd
import json
import os
import random
from typing import List, Tuple, Dict, Any, Union, Optional


class AnsweringAgentNormalizer:
    """A comprehensive normalization module for Aerial Vision and Dialog Navigation (AVDN) data."""
    
    # AVDN's RGB normalization values
    RGB_MEAN = np.array([60.134, 49.697, 40.746], dtype=np.float32).reshape((3, 1, 1))
    RGB_STD = np.array([29.99, 24.498, 22.046], dtype=np.float32).reshape((3, 1, 1))
    
    # GPS normalization ranges
    GPS_RANGES = {
        'lat': {'min': -90, 'max': 90},
        'lon': {'min': -180, 'max': 180}
    }

    def __init__(self, tokenizer=None, config=None):
        """Initialize the normalizer."""
        # Add image cache to avoid repeated disk reads
        self.image_cache = {}
        # Maximum cache size (adjust based on available memory)
        self.max_cache_size = 100
        self.tokenizer = tokenizer
        self.config = config

        if config is None:
            self.config = Config()

        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def load_image(self, file_path: str) -> np.ndarray:
        """Load an image from file and ensure RGB format.
        
        Args:
            file_path (str): Path to the image file
            
        Returns:
            np.ndarray: RGB image as float32 array
            
        Raises:
            FileNotFoundError: If image file cannot be loaded
        """
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"File not found: {file_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image.astype(np.float32)

    def normalize_pixel_values(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values using AVDN's mean-variance normalization.
        
        Args:
            image (np.ndarray): Input image in (H, W, C) format
            
        Returns:
            np.ndarray: Normalized image in (C, H, W) format
        """
        # Transpose image to match AVDN's format (C, H, W)
        image = image.transpose(2, 0, 1)
        
        # Apply normalization
        image = (image - self.RGB_MEAN) / self.RGB_STD
        
        return image

    def apply_visual_augmentation(self, image: np.ndarray, 
                                  augment_prob: float = 0.5,
                                  brightness_range: Tuple[float, float] = (0.8, 1.2),
                                  contrast_range: Tuple[float, float] = (0.8, 1.2),
                                  noise_level: float = 0.02) -> np.ndarray:
        """Apply visual augmentations to the image.
        
        Args:
            image: Input image (C, H, W) format after normalize_pixel_values
            augment_prob: Probability of applying each augmentation
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            noise_level: Standard deviation for Gaussian noise
            
        Returns:
            Augmented image
        """
        # Skip augmentation with some probability
        if random.random() > augment_prob:
            return image
            
        # Brightness adjustment
        if random.random() > 0.5:
            factor = random.uniform(brightness_range[0], brightness_range[1])
            # Apply to each channel
            image = image * factor
            
        # Contrast adjustment
        if random.random() > 0.5:
            factor = random.uniform(contrast_range[0], contrast_range[1])
            mean = np.mean(image, axis=(1, 2), keepdims=True)
            image = (image - mean) * factor + mean
            
        # Add Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
            image = image + noise
            
        return image

    def normalize_position(self, lat: float, lon: float) -> Tuple[float, float]:
        """Normalize GPS positions to a [0,1] scale.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            
        Returns:
            Tuple[float, float]: Normalized (latitude, longitude)
        """
        norm_lat = (lat - self.GPS_RANGES['lat']['min']) / \
                  (self.GPS_RANGES['lat']['max'] - self.GPS_RANGES['lat']['min'])
        norm_lon = (lon - self.GPS_RANGES['lon']['min']) / \
                  (self.GPS_RANGES['lon']['max'] - self.GPS_RANGES['lon']['min'])
        return norm_lat, norm_lon

    def gps_to_img_coords(self, gps: List[float], gps_botm_left: np.ndarray, 
                         gps_top_right: np.ndarray, lat_ratio: float, lng_ratio: float) -> np.ndarray:
        """
        Convert GPS coordinates to image coordinates using AVDN's approach.
        
        Args:
            gps: GPS coordinates [longitude, latitude] representing a corner
            gps_botm_left: Bottom left GPS coordinates of the image
            gps_top_right: Top right GPS coordinates of the image
            lat_ratio: Latitude ratio for scaling
            lng_ratio: Longitude ratio for scaling
            
        Returns:
            np.ndarray: Image coordinates (x, y)
        """
        # Convert inputs to numpy arrays if they aren't already
        gps = np.array(gps)
        gps_botm_left = np.array(gps_botm_left)
        gps_top_right = np.array(gps_top_right)
        
        x = int(round((gps[1] - gps_botm_left[1]) / lat_ratio))
        y = int(round((gps_top_right[0] - gps[0]) / lng_ratio))
        return np.array([x, y], dtype=np.float32)

    def normalize_coordinates(self, coords: np.ndarray, gps_botm_left: np.ndarray, 
                            gps_top_right: np.ndarray, lat_ratio: float, lng_ratio: float) -> np.ndarray:
        """
        Normalize coordinates using AVDN's approach.
        
        Args:
            coords: Coordinates to normalize
            gps_botm_left: Bottom left GPS coordinates
            gps_top_right: Top right GPS coordinates
            lat_ratio: Latitude ratio for scaling
            lng_ratio: Longitude ratio for scaling
            
        Returns:
            np.ndarray: Normalized coordinates in [0,1] range
        """
        # Convert to image coordinates
        img_coords = np.array([self.gps_to_img_coords(coord, gps_botm_left, gps_top_right, lat_ratio, lng_ratio) 
                              for coord in coords])
        
        # Calculate image dimensions in GPS coordinates
        img_width = (gps_top_right[0] - gps_botm_left[0]) / lng_ratio
        img_height = (gps_top_right[1] - gps_botm_left[1]) / lat_ratio
        
        # Normalize to [0, 1] range
        normalized = img_coords / np.array([img_width, img_height])
        return normalized

    def normalize_view_area(self, view_area: List[List[float]], gps_botm_left: np.ndarray, 
                          gps_top_right: np.ndarray, lat_ratio: float, lng_ratio: float,
                          image: np.ndarray, output_size: Tuple[int, int],
                          apply_augmentation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize a single view area by converting GPS coordinates to image coordinates and applying perspective transform.
        
        Args:
            view_area: List of 4 GPS coordinates representing view area corners
            gps_botm_left: Bottom left GPS coordinates of the image
            gps_top_right: Top right GPS coordinates of the image
            lat_ratio: Latitude ratio for scaling
            lng_ratio: Longitude ratio for scaling
            image: Original image to transform
            output_size: Output image size (width, height)
            apply_augmentation: Whether to apply visual augmentations
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Transformed image
                - View area corners in image coordinates
        """
        # Convert GPS coordinates to image coordinates for all corners (vectorized)
        img_coord_corners = np.array([
            self.gps_to_img_coords(corner, gps_botm_left, gps_top_right, lat_ratio, lng_ratio)
            for corner in view_area
        ], dtype=np.float32)
        
        # Apply perspective transformation
        width, height = output_size
        dst_pts = np.array([[0, 0], [width - 1, 0], 
                           [width - 1, height - 1], [0, height - 1]], 
                          dtype=np.float32)
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(img_coord_corners, dst_pts)
        
        # Apply perspective transform
        transformed_image = cv2.warpPerspective(
            image, M, (width, height), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        transformed_image = self.normalize_pixel_values(transformed_image)
        
        # Apply visual augmentation if requested
        if apply_augmentation:
            transformed_image = self.apply_visual_augmentation(transformed_image)
        
        return transformed_image

    def process_coordinates_to_image(self, coords: List[List[float]], 
                                  map_name: str, 
                                  image_dir: str,
                                  gps_botm_left: np.ndarray,
                                  gps_top_right: np.ndarray,
                                  lat_ratio: float, 
                                  lng_ratio: float,
                                  output_size: Tuple[int, int] = (224, 224),
                                  apply_augmentation: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Process coordinates to get the corresponding view area image.
        
        Args:
            coords: Coordinates defining the view area
            map_name: Name of the map
            image_dir: Directory containing map images
            gps_botm_left: Bottom left GPS coordinates
            gps_top_right: Top right GPS coordinates
            lat_ratio: Latitude ratio
            lng_ratio: Longitude ratio
            output_size: Output image size
            apply_augmentation: Whether to apply augmentation
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformed image and image coordinates
        """
        # Ensure map_name is properly formatted
        map_image_name = f"{map_name}.tif" if not map_name.endswith(".tif") else map_name
        
        # Load and cache map image
        if map_image_name not in self.image_cache:
            image_path = os.path.join(image_dir, map_image_name)
            img = self.load_image(image_path)

            # Manage cache size
            if len(self.image_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
                
            self.image_cache[map_image_name] = img
        else:
            img = self.image_cache[map_image_name]

        # Convert coordinates to view area image
        transformed_image = self.normalize_view_area(
            coords, gps_botm_left, gps_top_right, lat_ratio, lng_ratio,
            img, output_size, apply_augmentation=apply_augmentation
        )
        
        return transformed_image

    def process_contrastive_samples(self, dialog_turn: Dict[str, Any], max_length: int = 128) -> Dict[str, Any]:
        """Process contrastive samples in a dialog turn.
        
        Args:
            dialog_turn: Dialog turn data containing contrastive samples
            max_length: Maximum sequence length for tokenization
            
        Returns:
            Dict containing processed contrastive samples
        """
        contrastive_data = {}
        
        # Process positive examples
        if "contrastive_samples" in dialog_turn and "positive_examples" in dialog_turn["contrastive_samples"]:
            positive_examples = dialog_turn["contrastive_samples"]["positive_examples"]
            contrastive_data["positive_examples"] = []
            
            for example in positive_examples:
                # Tokenize the positive example
                tokenized = self.tokenizer(
                    example["text"],
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                contrastive_data["positive_examples"].append({
                    "text": example["text"],
                    "tokenized": {
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"]
                    },
                    "similarity": example.get("similarity", 1.0),
                    "type": example.get("type", "paraphrase")
                })
        
        # Process negative examples
        if "contrastive_samples" in dialog_turn and "negative_examples" in dialog_turn["contrastive_samples"]:
            negative_examples = dialog_turn["contrastive_samples"]["negative_examples"]
            contrastive_data["negative_examples"] = []
            
            for example in negative_examples:
                # Tokenize the negative example
                tokenized = self.tokenizer(
                    example["text"],
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                contrastive_data["negative_examples"].append({
                    "text": example["text"],
                    "tokenized": {
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"]
                    },
                    "similarity": example.get("similarity", 0.0),
                    "type": example.get("type", "unrelated")
                })
        
        # Add complexity metadata if present
        if "complexity_metadata" in dialog_turn:
            contrastive_data["complexity_metadata"] = dialog_turn["complexity_metadata"]
        
        return contrastive_data

    def process_dialog_turn(self, episode: Dict[str, Any], dialog_turn: Dict[str, Any], 
                          image_dir: str, output_size: Tuple[int, int] = (224, 224),
                          apply_augmentation: bool = False, max_history: int = 3) -> Dict[str, Any]:
        """Process a single dialog turn to create a training example.
        
        Args:
            episode: Episode data
            dialog_turn: Dialog turn data
            image_dir: Path to image directory
            output_size: Output image size
            apply_augmentation: Whether to apply data augmentation
            max_history: Maximum number of previous dialog turns to include
            
        Returns:
            Dict containing processed dialog turn data
        """
        result = {}
        
        # Get map name and GPS coordinates
        map_name = episode['map_name']
        gps_botm_left = np.array(episode['gps_botm_left'])
        gps_top_right = np.array(episode['gps_top_right'])
        lat_ratio = episode['lat_ratio']
        lng_ratio = episode['lng_ratio']
        
        # Process current observation
        if 'observation' in dialog_turn and 'view_area_coords' in dialog_turn['observation']:
            current_view, coords = self.process_coordinates_to_image(
                dialog_turn['observation']['view_area_coords'],
                map_name, image_dir, gps_botm_left, gps_top_right,
                lat_ratio, lng_ratio, output_size, apply_augmentation
            )
            result['current_view_image'] = torch.tensor(current_view)
            result['current_coords'] = coords
        else:
            # If no observation, create a blank image
            current_view = np.zeros((3, output_size[0], output_size[1]), dtype=np.float32)
            result['current_view_image'] = torch.tensor(current_view)
            result['current_coords'] = np.zeros((4, 2), dtype=np.float32)
        
        # Process previous observations
        if 'previous_observations' in dialog_turn and len(dialog_turn['previous_observations']) > 0:
            prev_views = []
            prev_coords = []
            
            for prev_obs in dialog_turn['previous_observations'][-max_history:]:
                prev_view, coords = self.process_coordinates_to_image(
                    prev_obs, map_name, image_dir, gps_botm_left, gps_top_right,
                    lat_ratio, lng_ratio, output_size, apply_augmentation
                )
                prev_views.append(torch.tensor(prev_view))
                prev_coords.append(coords)
                
            result['previous_views_image'] = prev_views
            result['previous_coords'] = prev_coords
        else:
            result['previous_views_image'] = []
            result['previous_coords'] = []
        
        # Process destination coordinates if available
        if 'destination' in episode:
            destination_view, dest_coords = self.process_coordinates_to_image(
                episode['destination'], map_name, image_dir, gps_botm_left, gps_top_right,
                lat_ratio, lng_ratio, output_size, False  # No augmentation for destination
                )
            result['destination_image'] = torch.tensor(destination_view)
            result['destination_coords'] = dest_coords
        
        # Process dialog history
        dialog_history = []
        if 'dialog_history' in dialog_turn and len(dialog_turn['dialog_history']) > 0:
            dialog_history = dialog_turn['dialog_history'][-max_history:]
        
        # Process first instruction
        first_instruction = episode.get('first_instruction', '')
        result['first_instruction'] = first_instruction
        
        # Process current question and answer
        question = dialog_turn.get('question', '')
        answer = dialog_turn.get('answer', '')
        
        # Create dialog context by combining history, first instruction, and current question
        dialog_ctx = []
        if first_instruction:
            dialog_ctx.append(f"First Instruction: {first_instruction}")
        
        for hist in dialog_history:
            dialog_ctx.append(hist)
            
        if question:
            dialog_ctx.append(f"Question: {question}")
            
        # Combine dialog context
        dialog_context = " ".join(dialog_ctx)
        
        # Tokenize input and answer
        if self.tokenizer:
            max_length = self.config.data.max_seq_length if self.config else 512
            
            result['tokenized_input'] = self.tokenizer(
                dialog_context, 
                max_length=max_length,
                padding="max_length",
            truncation=True,
                return_tensors="pt"
        )

            result['tokenized_answer'] = self.tokenizer(
                answer,
                max_length=max_length,
                padding="max_length",
            truncation=True,
                return_tensors="pt"
            )
        
        # Store raw text for reference
        result['dialog_context'] = dialog_context
        result['question'] = question
        result['answer'] = answer
        
        # Process contrastive samples if present
        if "contrastive_samples" in dialog_turn or "complexity_metadata" in dialog_turn:
            result['contrastive_data'] = self.process_contrastive_samples(
                dialog_turn,
                max_length=self.config.data.max_seq_length if self.config else 512
            )
        
        return result

    def preprocess_all_data(self, data_path: str, image_dir: str, 
                           output_size: Tuple[int, int] = (224, 224), 
                           apply_augmentation: bool = False):
        """
        Preprocess all data in the JSON file at once.
        
        Args:
            data_path: Path to the JSON dataset file
            image_dir: Directory containing satellite images
            output_size: Size of the output image (width, height)
            apply_augmentation: Whether to apply visual augmentations
            
        Returns:
            Dict: Dictionary where keys are indices and values are processed data items
        """
        print(f"Pre-processing data from {data_path}...")
        
        # Load the JSON file
        with open(data_path, 'r') as f:
            episodes = json.load(f)
        
        # Create flattened list of turns for processing
        flattened_turns = []
        for episode in episodes:
            for dialog in episode["dialogs"]:
                # Skip first turn with no Q&A for most purposes
                if dialog["turn_id"] > 0:
                    flattened_turns.append({
                        "episode_id": episode["episode_id"],
                        "map_name": episode["map_name"],
                        "turn_id": dialog["turn_id"],
                        "question": dialog["question"],
                        "answer": dialog["answer"],
                        "first_instruction": episode["first_instruction"],
                        "current_view_coords": dialog["observation"]["view_area_coords"],
                        "previous_observations": dialog["previous_observations"],
                        "dialog_history": dialog["dialog_history"],
                        "destination": episode.get("destination"),
                        "gps_data": {
                            "gps_botm_left": episode["gps_botm_left"],
                            "gps_top_right": episode["gps_top_right"],
                            "lng_ratio": episode["lng_ratio"],
                            "lat_ratio": episode["lat_ratio"]
                        }
                    })
        
        processed_dataset = {}
        total_items = len(flattened_turns)
        print(f"Processing {total_items} dialog turns...")
        
        # Process each turn
        for idx, turn in enumerate(flattened_turns):
            try:
                processed_data = self.process_dialog_turn(
                    turn,
                    image_dir,
                    output_size=output_size,
                    apply_augmentation=apply_augmentation
                )
                
                # Store with a unique ID based on episode and turn
                processed_dataset[idx] = processed_data
                
                # Log progress
                if (idx + 1) % 100 == 0 or idx == total_items - 1:
                    print(f"Progress: {idx + 1}/{total_items} turns processed ({(idx + 1)/total_items*100:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing turn {idx} (episode: {turn['episode_id']}, turn: {turn['turn_id']}): {str(e)}")
                raise e
        
        print(f"Pre-processing complete. {len(processed_dataset)} turns processed.")
        return processed_dataset


# Example usage
if __name__ == '__main__':
    # Import needed dependencies for standalone execution
    from transformers import T5Tokenizer
    from AnsweringAgent.src.config import Config
    
    # Initialize tokenizer and config
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=config.data.max_seq_length)
    
    # Initialize normalizer
    normalizer = AnsweringAgentNormalizer(tokenizer, config)
    
    # Example JSON file path and image directory
    json_file = "processed_data/train_data.json"
    image_dir = "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images"
    
    # Process data
    processed_data = normalizer.preprocess_all_data(json_file, image_dir, apply_augmentation=True)

    max_length = 0
    max_length_answer = 0
    for key, value in processed_data.items():
        input_ids = value['tokenized_input']['input_ids'][0]
        eos_index = np.where(input_ids == 1)[0][0]
        if eos_index > max_length:
            max_length = eos_index

        answer_ids = value['tokenized_answer']['input_ids'][0]
        eos_index_answer = np.where(answer_ids == 1)[0][0]
        if eos_index_answer > max_length_answer:
            max_length_answer = eos_index_answer

    print(f"Max length: {max_length}")
    print(f"Max length answer: {max_length_answer}")
    
    print(f"Processed {len(processed_data)} items.")
