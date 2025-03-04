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
        # Get text inputs and convert to lowercase
        question = data['question'].lower().strip()
        first_instruction = data['first_instruction'].lower().strip()
        history = data['history'].lower().strip()
        
        # Concatenate with special tokens for each input type and [SEP] between contexts
        concatenated_text = f"[QUE] {question} [SEP] [FIRST INS] {first_instruction} [SEP] [HIST] {history}"
        
        # Tokenize text using BERT tokenizer
        tokenized_input_text = self.tokenizer(
            concatenated_text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize label with proper batch handling
        tokenized_label = self.tokenizer(
            data["answer"],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        # Remove the batch dimension from label if it's 1
        if tokenized_label['input_ids'].size(0) == 1:
            tokenized_label = {k: v.squeeze(0) for k, v in tokenized_label.items()}
        
        return tokenized_input_text, tokenized_label

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

    def normalize_view_area(self, view_area: List[List[float]], data: Dict[str, Any], image: np.ndarray, output_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize a single view area by converting GPS coordinates to image coordinates and applying perspective transform.
        
        Args:
            view_area (List[List[float]]): List of GPS coordinates representing view area corners
            data (Dict[str, Any]): Dictionary containing GPS conversion parameters
            image (np.ndarray): Original image to transform
            output_size (Tuple[int, int]): Output image size (width, height)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Transformed image
                - View area corners in image coordinates
        """
        # Convert GPS coordinates to image coordinates
        img_coord_corners = np.array(
            [self.gps_to_img_coords(coord, data) for coord in view_area], 
            dtype=np.float32
        )
        
        # Apply perspective transformation
        width, height = output_size
        dst_pts = np.array([[0, 0], [width - 1, 0], 
                           [width - 1, height - 1], [0, height - 1]], 
                          dtype=np.float32)
        M = cv2.getPerspectiveTransform(img_coord_corners, dst_pts)
        transformed_image = cv2.warpPerspective(image, M, (width, height))
        
         # Normalize image
        transformed_image = self.normalize_pixel_values(transformed_image)

        return transformed_image, img_coord_corners

    

    def process_data(self, data: Dict[str, Any], image_dir: str, output_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
        """Process image, GPS coordinates, and normalize view areas using AVDN transformations.
        
        Args:
            data (Dict[str, Any]): Dictionary containing view area information
            image_dir (str): Directory containing the image files
            output_size (Tuple[int, int]): Output image size (width, height)
            
        Returns:
            Dict[str, Any]: Processed data dictionary
        """
        # Load and process image
        file_path = f"{image_dir}/{data['map_name']}.tif"
        image = self.load_image(file_path)

        # Normalize current view area
        current_view = json.loads(data["current_view_coord"])
        rotated_image, img_coord_view_area_corners = self.normalize_view_area(
            current_view, data, image, output_size
        )

        data["current_view_coord"] = img_coord_view_area_corners
        data["current_view_image"] = rotated_image

        # Normalize previous view areas if they exist and are not empty
        if "previous_views_coord" in data and data["previous_views_coord"]:
            previous_views_coord = json.loads(data["previous_views_coord"])
            if previous_views_coord:  # Check if the list is not empty
                data["previous_views_coord"] = [
                    self.normalize_view_area(view, data, image, output_size)[1]
                    for view in previous_views_coord
                ]
                data["previous_views_image"] = [
                    self.normalize_view_area(view, data, image, output_size)[0]
                    for view in previous_views_coord
                ]
            else:
                # Remove empty previous views from data
                if isinstance(data, pd.Series):
                    data = data.drop("previous_views_coord")
                else:
                    data.pop("previous_views_coord", None)
        
        # Normalize concatenated text using BERT tokenizer
        tokenized_input_text, tokenized_label = self.normalize_text(data)
        data['text_input'] = tokenized_input_text
        data['text_label'] = tokenized_label
        
        # Normalize GPS coordinates
        data['gps_botm_left'] = self.normalize_position(
            *json.loads(data['gps_botm_left'])
        )
        data['gps_top_right'] = self.normalize_position(
            *json.loads(data['gps_top_right'])
        )

        return data


# Example usage
if __name__ == '__main__':
    train_df = pd.read_csv('train_data.csv')
    result = train_df[train_df['map_name'] == 2128].iloc[-1]
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
