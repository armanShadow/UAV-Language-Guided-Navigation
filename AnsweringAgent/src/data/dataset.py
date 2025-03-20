import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from typing import Dict, Any
from data.Normalizer import AnsweringAgentNormalizer
from config import Config

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
        self.max_previous_views = config.data.max_previous_views
        self.max_seq_length = config.data.max_seq_length
        
        # Initialize normalizer
        self.normalizer = AnsweringAgentNormalizer(tokenizer)
        
        # Load data from CSV
        self.data = pd.read_csv(self.csv_path)
        
        # Pre-process all data if enabled (much faster training, higher memory usage)
        
        print("Pre-processing dataset...")
        self.processed_data = self.normalizer.preprocess_all_data(
            self.data, 
            self.image_dir, 
            output_size=(224, 224), 
            max_seq_length=self.max_seq_length
        )
        print("Pre-processing complete.")
    
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
        # Get pre-processed data directly
        print(f"Getting pre-processed data for index {idx}")
        print("Total pre-processed data length:", len(self.processed_data))
        processed_data = self.processed_data[idx]
        current_view = processed_data['current_view_image']
        
        
        # Process previous views if available
        if 'previous_views_image' in processed_data:
            # Pad or truncate to max_previous_views
            if len(previous_views) > self.max_previous_views:
                previous_views = previous_views[:self.max_previous_views]
            elif len(previous_views) < self.max_previous_views:
                # Pad with zero tensors
                padding = [torch.zeros((3, 224, 224), dtype=torch.float32) 
                            for _ in range(self.max_previous_views - len(previous_views))]
                previous_views.extend(padding)
            
            # Stack correctly along a new dimension - will be [batch, num_views, C, H, W]
            previous_views = torch.stack(previous_views, dim=0)
        else:
            # Create a tensor of zero tensors with shape (max_previous_views, C, H, W)
            previous_views = torch.zeros((self.max_previous_views, 3, 224, 224), dtype=torch.float32)
        
        # Return pre-processed tensors
        return {
            'text_input': processed_data['text_input'],
            'text_label': processed_data['text_label']['input_ids'],
            'current_view_image': current_view,
            'previous_views_image': previous_views
        }