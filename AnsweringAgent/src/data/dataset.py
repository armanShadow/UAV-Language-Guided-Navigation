import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from typing import Dict, Any
from data.Normalizer import AnsweringAgentNormalizer
from config import Config
import pickle

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class AnsweringDataset(Dataset):
    """Dataset class for the Answering Agent.
    
    Note on device handling:
    - This dataset returns all tensors on CPU
    - Device transfer is handled by the DataLoader with pin_memory=True
    - This is optimal for multi-GPU training as it allows efficient data loading
    """
    @staticmethod
    def preprocess_and_save(config: Config, tokenizer, logger=None):
        """
        Static method to preprocess the dataset once and save to disk.
        This should be called only on rank 0 before initializing datasets.
        
        Args:
            config: Configuration object
            tokenizer: BERT tokenizer
            logger: Logger for output messages
            
        Returns:
            str: Path to the preprocessed data file
        """
        if logger:
            logger.info("Starting dataset preprocessing (this will run only once)...")
        
        # Check if preprocessed data already exists
        processed_data_path = config.data.train_data_path
        
        if os.path.exists(processed_data_path):
            if logger:
                logger.info(f"Preprocessed data already exists at {processed_data_path}. Skipping preprocessing.")
            return processed_data_path
        
        # Load data from CSV
        data_df = pd.read_csv(config.data.train_csv_path)
        
        # Initialize normalizer
        normalizer = AnsweringAgentNormalizer(tokenizer)
        
        # Use preprocess_all_data method to process all items efficiently
        if logger:
            logger.info(f"Processing {len(data_df)} items with preprocess_all_data...")
        
        processed_items = normalizer.preprocess_all_data(
            data_df,
            config.data.avdn_image_dir,
            output_size=(224, 224),
            max_seq_length=config.data.max_seq_length
        )
        
        # Save the processed data to disk
        with open(processed_data_path, 'wb') as f:
            logger.info(f"Saving preprocessed data to {processed_data_path}")
            pickle.dump(processed_items, f)
            logger.info(f"Preprocessed data saved to {processed_data_path}")
        
        if logger:
            logger.info(f"Preprocessing complete. {len(processed_items)} items saved to {processed_data_path}")
        
        return processed_data_path
    
    def __init__(self, config: Config):
        """
        Initialize the dataset - only loads data, no preprocessing.
        
        Args:
            config: Configuration object
            tokenizer: BERT tokenizer
        """
        self.config = config
        self.max_previous_views = config.data.max_previous_views
        self.max_seq_length = config.data.max_seq_length        

        # Check if preprocessed data exists and load it
        preprocessed_path = config.data.train_data_path
        
        try:
            with open(preprocessed_path, 'rb') as f:
                print(f"Loading preprocessed data from {preprocessed_path}")
                self.preprocessed_data = pickle.load(f)
            print(f"Loaded {len(self.preprocessed_data)} preprocessed items")
        except Exception as e:
            print(f"Error loading preprocessed data: {str(e)}")
    
    def __len__(self) -> int:
        return len(self.preprocessed_data)
    
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
        
        processed_data = self.preprocessed_data[idx]
        
        # The image is already normalized from Normalizer.py and already in (C,H,W) format
        current_view = processed_data['current_view_image'].float()
        
        # Process previous views
        previous_views = []
        for img in processed_data['previous_views_image']:
            # Images are already in (C,H,W) format from normalize_pixel_values
            previous_views.append(img.float())
           
        # Pad or truncate to max_previous_views
        if len(previous_views) > self.max_previous_views:
            previous_views = previous_views[:self.max_previous_views]
        elif len(previous_views) < self.max_previous_views:
            # Pad with zero tensors
            padding = [torch.zeros_like(previous_views[0]) 
                      for _ in range(self.max_previous_views - len(previous_views))]
            previous_views.extend(padding)
        
        # Stack correctly along a new dimension - will be [num_views, C, H, W]
        previous_views = torch.stack(previous_views, dim=0)
        
        # Return tensors in CPU, DataLoader will handle device transfer
        return {
            'text_input': processed_data['text_input'],
            'text_label': processed_data['text_label']['input_ids'],
            'current_view_image': current_view,
            'previous_views_image': previous_views
        }