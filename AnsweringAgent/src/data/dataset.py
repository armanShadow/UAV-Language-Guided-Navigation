import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from typing import Dict, Any, List, Union
from data.Normalizer import AnsweringAgentNormalizer
from config import Config
import numpy as np

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AnsweringDataset(Dataset):
    """Dataset class for the Answering Agent.
    
    Note on device handling:
    - This dataset returns all tensors on CPU
    - Device transfer is handled by the DataLoader with pin_memory=True
    - This is optimal for multi-GPU training as it allows efficient data loading
    """
    def __init__(self, config: Config, tokenizer):
        self.config = config
        self.csv_path = config.data.train_csv_path
        self.image_dir = config.data.avdn_image_dir
        
        # Use configuration for max previous views (default or from memory optimization)
        self.max_previous_views = config.data.max_previous_views
        
        # Initialize normalizer
        self.normalizer = AnsweringAgentNormalizer(tokenizer)
        
        # Load data from CSV
        self.data = pd.read_csv(self.csv_path)
        
        # Set image size from config 
        self.img_size = config.model.img_size
        
        # Get max sequence length for memory efficiency
        self.max_seq_length = config.data.max_seq_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data item with proper device handling.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - text_input: Tokenized text inputs
                - current_view_image: Current view image tensor
                - previous_views_image: Previous views image tensor (if available)
                - text_label: Tokenized answer label
        """
        # Get the data
        data = self.data.iloc[idx]
        
        # Process the data using normalizer with max sequence length limit
        processed_data = self.normalizer.process_data(
            data,
            self.image_dir,
            max_length=self.max_seq_length
        )
        
        # The image is already a tensor from the normalizer, convert with memory optimization
        current_view = processed_data['current_view_image'].to(torch.float32)
        current_view = current_view.permute(2, 0, 1)  # Convert to (C, H, W)
        
        # Process previous views if available
        if 'previous_views_image' in processed_data and processed_data['previous_views_image']:
            # Get only up to max_previous_views
            views_to_process = processed_data['previous_views_image'][:self.max_previous_views]
            
            # Calculate padding needed
            pad_length = max(0, self.max_previous_views - len(views_to_process))
            
            # Process available views
            previous_views = []
            for img in views_to_process:
                prev_view = img.to(torch.float32)  # More memory efficient than .float()
                prev_view = prev_view.permute(2, 0, 1)  # Convert to (C, H, W)
                previous_views.append(prev_view)
            
            # If we need padding
            if pad_length > 0:
                # Use a single zeros tensor and repeat it in the stack operation
                # More memory efficient than creating multiple zero tensors
                zero_tensor = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
                padding = [zero_tensor] * pad_length
                previous_views.extend(padding)
            
            previous_views = torch.stack(previous_views)
        else:
            # Create a tensor of zeros with shape (max_previous_views, C, H, W)
            # More efficient than creating multiple zero tensors
            previous_views = torch.zeros(
                (self.max_previous_views, 3, self.img_size, self.img_size), 
                dtype=torch.float32
            )
        
        # Extract only input_ids from text_label to save memory
        text_label = processed_data['text_label']['input_ids']
        
        # Return tensors in CPU, DataLoader will handle device transfer
        return {
            'text_input': processed_data['text_input'],
            'text_label': text_label,
            'current_view_image': current_view,
            'previous_views_image': previous_views
        } 