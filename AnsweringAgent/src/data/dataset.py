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
        #TODO: #3 Redundant tokenizer. No Usage. AnsweringAgentNormalizer creates a tokenizer
        self.image_dir = config.data.avdn_image_dir
        self.max_previous_views = config.data.max_previous_views
        self.max_seq_length = config.data.max_seq_length
        
        # Initialize normalizer
        self.normalizer = AnsweringAgentNormalizer(tokenizer)
        
        # Load data from CSV
        self.data = pd.read_csv(self.csv_path)
    
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
        
        # Process the data using normalizer
        processed_data = self.normalizer.process_data(
            data,
            self.image_dir,
            max_seq_length=self.max_seq_length,
        )
        
        # The image is already a tensor from the normalizer, just ensure it's float
        current_view = processed_data['current_view_image'].float()
        current_view = current_view.permute(2, 0, 1)  # Convert to (C, H, W)
        
        # Process previous views if available
        if 'previous_views_image' in processed_data:
            previous_views = []
            for img in processed_data['previous_views_image']:
                prev_view = img.float()
                prev_view = prev_view.permute(2, 0, 1)  # Convert to (C, H, W)
                previous_views.append(prev_view)
            
            # Pad or truncate to max_previous_views
            if len(previous_views) > self.max_previous_views:
                previous_views = previous_views[:self.max_previous_views]
            elif len(previous_views) < self.max_previous_views:
                # Pad with zero tensors
                padding = [torch.zeros((3, 224, 224), dtype=torch.float32) 
                          for _ in range(self.max_previous_views - len(previous_views))]
                previous_views.extend(padding)
            
            previous_views = torch.stack(previous_views, dim=1)
        else:
            # Create a tensor of zero tensors with shape (max_previous_views, C, H, W)
            previous_views = torch.zeros((self.max_previous_views, 3, 224, 224), dtype=torch.float32)
        
        # Return tensors in CPU, DataLoader will handle device transfer
        return {
            'text_input': processed_data['text_input'],
            'text_label': processed_data['text_label']['input_ids'],
            'current_view_image': current_view,
            'previous_views_image': previous_views
        } 