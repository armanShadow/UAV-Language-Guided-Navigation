import torch
import cv2
import numpy as np
import pandas as pd
import json
import os
from typing import List, Tuple, Dict, Any
from transformers import BertTokenizerFast


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

    def __init__(self, tokenizer):
        """Initialize the normalizer with BERT tokenizer."""
        #TODO: #4 BertTokenizerFast vs BertTokenizer. is it confusing the model?
        self.tokenizer = tokenizer
        # Add image cache to avoid repeated disk reads
        self.image_cache = {}
        # Maximum cache size (adjust based on available memory)
        self.max_cache_size = 100

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
            np.ndarray: Normalized image in (H, W, C) format
        """
        # Transpose image to match AVDN's format (C, H, W)
        image = image.transpose(2, 0, 1)
        
        # Apply normalization
        image = (image - self.RGB_MEAN) / self.RGB_STD
        
        
        return image

    def normalize_text(self, data: Dict[str, Any], max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Normalize text using BERT tokenizer and return tokenized output.
        
        Args:
            data (Dict[str, Any]): Dictionary containing all text fields
            max_length (int): Maximum sequence length
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - input_ids: Tokenized input IDs
                - attention_mask: Attention mask for padding
                - token_type_ids: Token type IDs for BERT
        """
        # Get text inputs
        question = data['question'].strip()
        first_instruction = data['first_instruction'].strip()
        history = data['history'].strip()
        
        # Concatenate with special tokens for each input type and [SEP] between contexts
        concatenated_text = f"[ASKED QUE] {question} [SEP] [FIRST INS] {first_instruction} [SEP] [HIST] {history}"
        
        # Tokenize text
        tokenized_text = self.tokenizer(
            concatenated_text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Get label text
        label_text = data['answer'].strip()
        
        # Tokenize label
        tokenized_label = self.tokenizer(
            label_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Remove the batch dimension from label if it's 1
        if tokenized_label['input_ids'].size(0) == 1:
            tokenized_label = {k: v.squeeze(0) for k, v in tokenized_label.items()}
        
        return tokenized_text, tokenized_label

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
                          image: np.ndarray, output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize a single view area by converting GPS coordinates to image coordinates and applying perspective transform.
        
        Args:
            view_area: List of 4 GPS coordinates representing view area corners
            gps_botm_left: Bottom left GPS coordinates of the image
            gps_top_right: Top right GPS coordinates of the image
            lat_ratio: Latitude ratio for scaling
            lng_ratio: Longitude ratio for scaling
            image: Original image to transform
            output_size: Output image size (width, height)
            
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
        
        # Normalize image
        transformed_image = self.normalize_pixel_values(transformed_image)
        
        return transformed_image, img_coord_corners

    def process_data(self, data: Dict[str, Any], image_dir: str, output_size: Tuple[int, int] = (224, 224), max_seq_length: int = 512) -> Dict[str, Any]:
        """Process image, GPS coordinates, and normalize view areas using AVDN transformations.
        
        Args:
            data: Dictionary containing observation data
            image_dir: Directory containing satellite images
            output_size: Size of the output image (width, height)
            
        Returns:
            Dict[str, Any]: Processed data with normalized coordinates and transformed images
        """
        processed_data = data.copy()
        
        # Process text data
        tokenized_input_text, tokenized_label = self.normalize_text(data, max_length=max_seq_length)
        processed_data['text_input'] = tokenized_input_text
        processed_data['text_label'] = tokenized_label
        
        # Convert string coordinates to numpy arrays if they are strings
        gps_botm_left = np.array(json.loads(data['gps_botm_left']))
        gps_top_right = np.array(json.loads(data['gps_top_right']))

        lat_ratio = float(data['lat_ratio'])
        lng_ratio = float(data['lng_ratio'])
        
        # Cache and reuse loaded images
        map_name = str(data['map_name']) + '.tif'
        
        if map_name not in self.image_cache:
            # Load image only if not in cache
            image_path = os.path.join(image_dir, map_name)
            img = self.load_image(image_path)
            if len(self.image_cache) >= self.max_cache_size:
                # Clear oldest image if cache is full
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
            self.image_cache[map_name] = img
        else:
            img = self.image_cache[map_name]
        
        # Process current view coordinates
        if 'current_view_coord' in data:
            # Convert string coordinates to numpy array
            current_view_coord = np.array(json.loads(data['current_view_coord']))
            processed_data['current_view_coord_normalized'] = self.normalize_coordinates(
                current_view_coord, gps_botm_left, gps_top_right, lat_ratio, lng_ratio
            )
            
            # Transform current view to standard size using cached image
            transformed_image, img_coord_corners = self.normalize_view_area(
                current_view_coord.tolist(), gps_botm_left, gps_top_right, lat_ratio, lng_ratio,
                img, output_size
            )
            processed_data['current_view_image'] = torch.from_numpy(transformed_image)
            processed_data['current_view_coord_pixel'] = torch.from_numpy(img_coord_corners)
        
        # Process previous views coordinates
        if 'previous_views_coord' in data:
            # Convert string coordinates to numpy arrays
            previous_views_coord = [np.array(view_coords) for view_coords in json.loads(data['previous_views_coord'])]
            processed_data['previous_views_coord_normalized'] = [
                self.normalize_coordinates(view_coords, gps_botm_left, gps_top_right, lat_ratio, lng_ratio)
                for view_coords in previous_views_coord
            ]
            
            # Transform previous views to standard size
            processed_data['previous_views_image'] = []
            processed_data['previous_views_coord_pixel'] = []
            for view_coords in previous_views_coord:
                transformed_image, img_coord_corners = self.normalize_view_area(
                    view_coords.tolist(), gps_botm_left, gps_top_right, lat_ratio, lng_ratio,
                    img, output_size
                )
                processed_data['previous_views_image'].append(torch.from_numpy(transformed_image))
                processed_data['previous_views_coord_pixel'].append(torch.from_numpy(img_coord_corners))
        
        return processed_data

    def preprocess_all_data(self, data_df, image_dir, output_size=(224, 224), max_seq_length=512):
        """
        Preprocess all data in the dataframe at once to avoid repeated processing during training.
        Uses the existing process_data function for each item, which already handles image caching.
        
        Args:
            data_df: Pandas DataFrame containing the dataset
            image_dir: Directory containing satellite images
            output_size: Size of the output image (width, height)
            max_seq_length: Maximum sequence length for text
            
        Returns:
            Dict: Dictionary where keys are indices and values are processed data items
        """
        print(f"Pre-processing {len(data_df)} items...")
        processed_dataset = {}
        
        # Track progress
        total_items = len(data_df)
        
        # Simply iterate through all rows
        for idx, data in data_df.iterrows():
            try:
                # Process this item using the existing process_data function
                processed_data = self.process_data(
                    data, 
                    image_dir,
                    output_size=output_size,
                    max_seq_length=max_seq_length
                )
                # Store the processed data with original dataframe index
                processed_dataset[idx] = processed_data
                
                # Log progress
                if len(processed_dataset) % 100 == 0:
                    print(f"Progress: {len(processed_dataset)}/{total_items} items processed ({len(processed_dataset)/total_items*100:.1f}%)")
                    
            except Exception as e:
                print(f"Error processing item {idx}: {str(e)}")
        
        print(f"Pre-processing complete. {len(processed_dataset)} items processed.")
        return processed_dataset


# Example usage
if __name__ == '__main__':
    train_df = pd.read_csv('train_data.csv')
    result = train_df[train_df['map_name'] == 2128].iloc[1]
    image_dir = "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images"
    
    # Initialize normalizer
    normalizer = AnsweringAgentNormalizer()
    
    # Use the normalizer
    result = normalizer.process_data(
        result, image_dir
    )
    
    # Display result
    cv2.imshow("Normalized View Area", result['current_view_image'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
