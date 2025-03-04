import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import os
import json
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from data.Normalizer import AnsweringAgentNormalizer

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AnsweringDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 avdn_image_dir: str,
                 tokenizer: BertTokenizerFast,
                 max_length: int = 512,
                 max_previous_views: int = 4):
        """
        Dataset for the answering agent.
        
        Args:
            csv_path (str): Path to the CSV file containing the data
            avdn_image_dir (str): Path to AVDN's train_images directory
            tokenizer (BertTokenizerFast): BERT tokenizer
            max_length (int): Maximum sequence length for text
            max_previous_views (int): Maximum number of previous views to include
        """
        self.avdn_image_dir = avdn_image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_previous_views = max_previous_views
        
        # Initialize normalizer
        self.normalizer = AnsweringAgentNormalizer()
        
        # Load data from CSV
        self.data = pd.read_csv(csv_path)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data item.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - text_input: Tokenized text inputs
                - current_view_image: Current view image tensor
                - previous_views_image: Previous views image tensor (if available)
                - text_label: Tokenized answer label
        """
        item = self.data.iloc[idx]
        
        # Convert pandas Series to dict for normalizer
        item_dict = item.to_dict()
        
        # Process data using normalizer
        processed_data = self.normalizer.process_data(
            item_dict,
            self.avdn_image_dir
        )
        
        # Convert processed images to tensors
        current_view = torch.from_numpy(processed_data['current_view_image']).float()
        current_view = current_view.permute(2, 0, 1)  # Convert to (C, H, W)
        
        # Process previous views if available
        if 'previous_views_image' in processed_data:
            previous_views = []
            for img in processed_data['previous_views_image']:
                prev_view = torch.from_numpy(img).float()
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
            
            previous_views = torch.stack(previous_views)
        else:
            # Create a tensor of zero tensors with shape (max_previous_views, C, H, W)
            previous_views = torch.zeros((self.max_previous_views, 3, 224, 224), dtype=torch.float32)
        
        # Get text inputs and labels and convert to resizable tensors
        text_inputs = {
            'input_ids': processed_data['text_input']['input_ids'].squeeze(0).clone(),
            'attention_mask': processed_data['text_input']['attention_mask'].squeeze(0).clone(),
            'token_type_ids': processed_data['text_input']['token_type_ids'].squeeze(0).clone()
        }
        text_label = processed_data['text_label']['input_ids'].squeeze(0).clone()
        
        return {
            'text_input': text_inputs,
            'current_view_image': current_view,
            'previous_views_image': previous_views,
            'text_label': text_label
        } 