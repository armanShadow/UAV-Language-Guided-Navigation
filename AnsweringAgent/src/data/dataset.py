import torch
from torch.utils.data import Dataset
import os
import json
from typing import Dict, Any, List, Optional
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
    def preprocess_and_save(config: Config, tokenizer, split='train', logger=None):
        """
        Static method to preprocess the dataset once and save to disk.
        This should be called only on rank 0 before initializing datasets.
        
        Args:
            config: Configuration object
            split: Dataset split ('train', 'val_seen', 'val_unseen')
            logger: Logger for output messages
            
        Returns:
            str: Path to the preprocessed data file
        """
        if logger:
            logger.info(f"Starting {split} dataset preprocessing...")
        
        # Determine data paths based on split
        if split == 'train':
            json_path = config.data.train_json_path
            processed_data_path = config.data.train_processed_path
        elif split == 'val_seen':
            json_path = config.data.val_seen_json_path
            processed_data_path = config.data.val_seen_processed_path
        elif split == 'val_unseen':
            json_path = config.data.val_unseen_json_path
            processed_data_path = config.data.val_unseen_processed_path
        else:
            raise ValueError(f"Unknown split: {split}. Must be one of ['train', 'val_seen', 'val_unseen']")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        
        # Check if preprocessed data already exists
        if os.path.exists(processed_data_path):
            if logger:
                logger.info(f"Preprocessed {split} data already exists at {processed_data_path}. Skipping preprocessing.")
            return processed_data_path
        
        # Check if JSON data exists
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"{split} JSON data file not found at {json_path}")
            
        if logger:
            logger.info(f"Loading {split} JSON data from {json_path}...")
        
        # Initialize normalizer
        normalizer = AnsweringAgentNormalizer(tokenizer)
        
        # Use preprocess_all_data method to process all items efficiently
        if logger:
            logger.info(f"Processing {split} data from {json_path}...")
        
        # Apply augmentation only to training data
        apply_augmentation = config.training.use_augmentation and split == 'train'
        
        processed_items = normalizer.preprocess_all_data(
            json_path,
            config.data.avdn_image_dir,
            output_size=(config.model.img_size, config.model.img_size),
            apply_augmentation=apply_augmentation
        )
        
        # Save the processed data to disk
        with open(processed_data_path, 'wb') as f:
            if logger:
                logger.info(f"Saving preprocessed {split} data to {processed_data_path}")
            pickle.dump(processed_items, f)
        
        if logger:
            logger.info(f"{split} preprocessing complete. {len(processed_items)} items saved to {processed_data_path}")
        
        return processed_data_path
    
    def __init__(self, config: Config, split='train'):
        """
        Initialize the dataset - loads preprocessed data.
        
        Args:
            config: Configuration object
            tokenizer: Text tokenizer to use (T5, BERT, etc.)
            split: Dataset split ('train', 'val_seen', 'val_unseen')
        """
        self.config = config
        self.max_prev_views = config.data.max_previous_views
        self.split = split
        
        # Determine which processed file to load based on split
        if split == 'train':
            preprocessed_path = config.data.train_processed_path
        elif split == 'val_seen':
            preprocessed_path = config.data.val_seen_processed_path
        elif split == 'val_unseen':
            preprocessed_path = config.data.val_unseen_processed_path
        else:
            raise ValueError(f"Unknown split: {split}. Must be one of ['train', 'val_seen', 'val_unseen']")

        # Load the preprocessed data
        try:
            with open(preprocessed_path, 'rb') as f:
                print(f"Loading {split} data from {preprocessed_path}")
                self.preprocessed_data = pickle.load(f)
            print(f"Loaded {len(self.preprocessed_data)} preprocessed items for {split}")
            
            # Convert dict to list for easier indexing
            self.data_items = list(self.preprocessed_data.values())
        except Exception as e:
            print(f"Error loading preprocessed data: {str(e)}")
            raise
    
    def __len__(self) -> int:
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data item
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - text_input: Tokenized text inputs
                - current_view_image: Current view image tensor
                - previous_views_image: Previous views image tensor (if available)
                - text_label: Tokenized answer label
                - destination_image: Destination image tensor (if available)
                - raw_question: Raw question string
                - raw_answer: Raw answer string
                - first_instruction: First instruction string
                - dialog_history: Dialog history list
        """
        # Get the preprocessed data item
        item = self.data_items[idx]
        
        # Get pre-tokenized text data
        tokenized_text = item['tokenized_input']
        tokenized_answer = item['tokenized_answer']
        
        # Get image data
        current_view = item['current_view_image']
        
        # Process previous views
        previous_views = []
        if 'previous_views_image' in item and len(item['previous_views_image']) > 0:
            for img in item['previous_views_image']:
                previous_views.append(img)
        
        # Handle case where previous_views is empty
        if len(previous_views) == 0:
            # Instead of zeros, replicate the current view for all previous views
            default_views = torch.stack([current_view] * self.max_prev_views, dim=0)
            return {
                'text_input': tokenized_text,
                'text_label': tokenized_answer,
                'current_view_image': current_view,
                'previous_views_image': default_views,
                'raw_question': item['question'],
                'raw_answer': item['answer'],
                'first_instruction': item['first_instruction'],
                'dialog_history': item['dialog_history']
            }
            
        # Pad or truncate to max_previous_views
        if len(previous_views) > self.max_prev_views:
            previous_views = previous_views[:self.max_prev_views]
        elif len(previous_views) < self.max_prev_views:
            # Instead of zero padding, replicate the current view for padding
            padding = [current_view.clone() 
                      for _ in range(self.max_prev_views - len(previous_views))]
            previous_views.extend(padding)
        
        # Stack correctly along a new dimension - will be [num_views, C, H, W]
        previous_views = torch.stack(previous_views, dim=0)
        
        # Include destination if available (for curriculum learning)
        result = {
            'text_input': tokenized_text,
            'text_label': tokenized_answer,
            'current_view_image': current_view,
            'previous_views_image': previous_views,
            'raw_question': item['question'],
            'raw_answer': item['answer'],
            'first_instruction': item['first_instruction'],
            'dialog_history': item['dialog_history']
        }
        
        # Add destination if available (important for curriculum learning)
        if 'destination_image' in item:
            result['destination_image'] = item['destination_image']
            
        return result
    
    @staticmethod
    def create_datasets(config: Config, logger=None, splits=['train', 'val_seen', 'val_unseen']):
        """
        Create all three datasets (train, val_seen, val_unseen) at once.
        
        Args:
            config: Configuration object
            logger: Logger for output messages
            splits: List of splits to create
        Returns:
            Dict[str, Dataset]: Dictionary of datasets
        """
        # Preprocess all splits
        datasets = {}
        for split in splits:
            datasets[split] = AnsweringDataset(config, split=split)

        return datasets