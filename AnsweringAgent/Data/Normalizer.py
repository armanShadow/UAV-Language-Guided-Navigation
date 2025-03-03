import torch
import cv2
import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Union, Dict, Any
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

    def __init__(self):
        """Initialize the normalizer with BERT tokenizer."""
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

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
        
        # Transpose back to (H, W, C)
        return image.transpose(1, 2, 0)

    def normalize_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Normalize text using BERT tokenizer and return tokenized output.
        
        Args:
            text (str): Input text
            max_length (int): Maximum sequence length
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - input_ids: Tokenized input IDs
                - attention_mask: Attention mask for padding
                - token_type_ids: Token type IDs for BERT
        """
        # Convert to lowercase and strip whitespace
        text = text.lower().strip()
        
        # Tokenize text using BERT tokenizer
        tokenized = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return tokenized

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

    def gps_to_img_coords(self, gps: List[float], ob: Dict[str, Any]) -> Tuple[int, int]:
        """Convert GPS coordinates to pixel coordinates.
        
        Args:
            gps (List[float]): GPS coordinates [lon, lat]
            ob (Dict[str, Any]): Object containing GPS conversion parameters
            
        Returns:
            Tuple[int, int]: Pixel coordinates (x, y)
        """
        # Ensure gps_botm_left and gps_top_right are lists of floats
        gps_botm_left = ob['gps_botm_left']
        gps_top_right = ob['gps_top_right']

        if isinstance(gps_botm_left, str):
            gps_botm_left = json.loads(gps_botm_left)
        if isinstance(gps_top_right, str):
            gps_top_right = json.loads(gps_top_right)

        # Convert all values to float
        gps_botm_left = list(map(float, gps_botm_left))
        gps_top_right = list(map(float, gps_top_right))
        gps = list(map(float, gps))

        lng_ratio = float(ob['lng_ratio'])
        lat_ratio = float(ob['lat_ratio'])

        return int(round((gps[1] - gps_botm_left[1]) / lat_ratio)), \
               int(round((gps_top_right[0] - gps[0]) / lat_ratio))

    def get_direction(self, start: np.ndarray, end: np.ndarray) -> float:
        """Compute rotation angle for alignment based on AVDN method.
        
        Args:
            start (np.ndarray): Starting point coordinates
            end (np.ndarray): Ending point coordinates
            
        Returns:
            float: Rotation angle in degrees
        """
        vec = np.array(end) - np.array(start)
        if vec[1] > 0:
            _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90
        elif vec[1] < 0:
            _angle = np.arctan(vec[0] / vec[1]) / 1.57 * 90 + 180
        else:
            _angle = 90 if np.sign(vec[0]) == 1 else 270
        return (360 - _angle + 90) % 360

    def process_view_area(self, result: Dict[str, Any], image_dir: str, output_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Process image, GPS coordinates, and normalize view areas using AVDN transformations.
        
        Args:
            result (Dict[str, Any]): Dictionary containing view area information
            image_dir (str): Directory containing the image files
            output_size (Tuple[int, int]): Output image size (width, height)
            
        Returns:
            Tuple[np.ndarray, np.ndarray, Dict[str, Any]]: 
                - Normalized and transformed image
                - View area corners in image coordinates
                - Processed result dictionary
        """
        # Load and process image
        file_path = f"{image_dir}/{result['map_name']}.tif"
        image = self.load_image(file_path)

        # Get view area corners
        view_area = json.loads(result["current_view"])
        img_coord_view_area_corners = np.array(
            [self.gps_to_img_coords(coord, result) for coord in view_area], 
            dtype=np.float32
        )

        # Apply perspective transformation
        width, height = output_size
        dst_pts = np.array([[0, 0], [width - 1, 0], 
                           [width - 1, height - 1], [0, height - 1]], 
                          dtype=np.float32)
        M = cv2.getPerspectiveTransform(img_coord_view_area_corners, dst_pts)
        rotated_image = cv2.warpPerspective(image, M, (width, height))

        # Normalize image
        rotated_image = self.normalize_pixel_values(rotated_image)

        # Normalize text fields using BERT tokenizer
        result['question'] = self.normalize_text(result['question'])
        result['answer'] = self.normalize_text(result['answer'])
        result['first_instruction'] = self.normalize_text(result['first_instruction'])
        result['history'] = self.normalize_text(result['history'])
        
        # Normalize GPS coordinates
        result['gps_botm_left'] = self.normalize_position(
            *json.loads(result['gps_botm_left'])
        )
        result['gps_top_right'] = self.normalize_position(
            *json.loads(result['gps_top_right'])
        )

        return rotated_image, img_coord_view_area_corners, result


# Example usage
if __name__ == '__main__':
    train_df = pd.read_csv('train_data')
    result = train_df[train_df['map_name'] == 2128].iloc[-1]
    image_dir = "/Users/arman/Desktop/UTA/Thesis/DATA/train_images"
    
    # Initialize normalizer
    normalizer = AnsweringAgentNormalizer()
    
    # Use the normalizer
    processed_image, processed_view, processed_text = normalizer.process_view_area(
        result, image_dir
    )
    
    # Display result
    cv2.imshow("Normalized View Area", processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
