import torch
from torch.utils.data import Dataset
import os
import json
from typing import Dict, Any, List, Optional
from data.Normalizer import AnsweringAgentNormalizer
from config import Config
import pickle
import math
import traceback
import random

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
    def save_in_chunks(data, chunk_size, output_dir):
        """
        Save data in chunks to disk.
        """
        items = list(data.items())
        for i in range(0, len(items), chunk_size):
            chunk = dict(items[i:i+chunk_size])
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"chunk_{i//chunk_size}.pkl"), "wb") as f:
                print(f"Saving chunk {i//chunk_size} to {os.path.join(output_dir, f'chunk_{i//chunk_size}.pkl')}")
                pickle.dump(chunk, f)

    @staticmethod
    def load_train_chunks(preprocessed_path):
        """
        Load train data from multiple chunks.
        """
        data = {}
        for file in os.listdir(preprocessed_path):
            if file.endswith('.pkl'):
                with open(os.path.join(preprocessed_path, file), 'rb') as f:
                    print(f"Loading chunk {file} from {os.path.join(preprocessed_path, file)}")
                    data.update(pickle.load(f))
        return data 

    def __init__(self, config: Config, split='train', tokenizer=None):
        """
        Initialize the dataset - loads preprocessed data.
        
        Args:
            config: Configuration object
            split: Dataset split ('train', 'val_seen', 'val_unseen')
            tokenizer: Tokenizer for potential on-the-fly processing
        """
        self.config = config
        self.max_prev_views = config.data.max_previous_views
        self.split = split
        self.tokenizer = tokenizer  # Store tokenizer for context-aware processing
        
        # Determine which processed file to load based on split
        if split == 'train':
            preprocessed_path = config.data.train_processed_path_dir
        elif split == 'val_seen':
            preprocessed_path = config.data.val_seen_processed_path
        elif split == 'val_unseen':
            preprocessed_path = config.data.val_unseen_processed_path
        else:
            raise ValueError(f"Unknown split: {split}. Must be one of ['train', 'val_seen', 'val_unseen']")

        # Load the preprocessed data
        try:
            if split == 'train':
                self.preprocessed_data = AnsweringDataset.load_train_chunks(preprocessed_path)
            else:
                with open(preprocessed_path, 'rb') as f:
                    print(f"Loading {split} data from {preprocessed_path}")
                    self.preprocessed_data = pickle.load(f)
                print(f"Loaded {len(self.preprocessed_data)} preprocessed items for {split}")
        
            # Convert dict to list for easier indexing
            self.data_items = list(self.preprocessed_data.values())
        except Exception as e:
            print(f"Error loading preprocessed data: {str(e)}")
            traceback.print_exc()
            raise
    
    def __len__(self) -> int:
        return len(self.data_items)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a data item using pre-tokenized data from the normalizer.
        
        Args:
            idx (int): Index of the item to get
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - text_input: Tokenized text inputs
                - current_view_image: Current view image tensor
                - previous_views_image: Previous views image tensor (if available)
                - text_label: Tokenized answer label
                - positive_example: Tokenized positive contrastive example (if available)
                - negative_example: Tokenized negative contrastive example (if available)
        """
        # Get the preprocessed data item
        item = self.data_items[idx]
        
        # Get pre-tokenized text data
        tokenized_text = {
            'input_ids': item['tokenized_input']['input_ids'].squeeze(0),
            'attention_mask': item['tokenized_input']['attention_mask'].squeeze(0)
        }
        tokenized_answer = {
            'input_ids': item['tokenized_answer']['input_ids'].squeeze(0),
            'attention_mask': item['tokenized_answer']['attention_mask'].squeeze(0)
        }
        
        # Get separate tokenized components for hierarchical processing
        tokenized_first_instruction = {
            'input_ids': item['tokenized_first_instruction']['input_ids'].squeeze(0),
            'attention_mask': item['tokenized_first_instruction']['attention_mask'].squeeze(0)
        }
        tokenized_current_question = {
            'input_ids': item['tokenized_current_question']['input_ids'].squeeze(0), 
            'attention_mask': item['tokenized_current_question']['attention_mask'].squeeze(0)
        }

        

        # Get image data - already tensors from normalizer
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

            result = {
                'text_input': tokenized_text,
                'text_label': tokenized_answer,
                'first_instruction_input': tokenized_first_instruction,
                'current_question_input': tokenized_current_question,
                'current_view_image': current_view,
                'previous_views_image': default_views,
            }
            
            if 'destination_image' in item:
                result['destination_image'] = item['destination_image']

            # Add contrastive examples if available
            if 'contrastive_data' in item:
                self._add_contrastive_examples(item['contrastive_data'], result)

            return result   
        
        # Pad or truncate to max_previous_views
        if len(previous_views) > self.max_prev_views:
            previous_views = previous_views[:self.max_prev_views]
        elif len(previous_views) < self.max_prev_views:
            # Instead of zero padding, replicate the current view for padding
            padding = [current_view.clone() for _ in range(self.max_prev_views - len(previous_views))]
            previous_views.extend(padding)
        
        # Stack the views into a single tensor
        previous_views_tensor = torch.stack(previous_views, dim=0)
        
        # Build result dictionary
        result = {
            'text_input': tokenized_text,
            'text_label': tokenized_answer,
            'first_instruction_input': tokenized_first_instruction,
            'current_question_input': tokenized_current_question,
            'current_view_image': current_view,
            'previous_views_image': previous_views_tensor,
        }
        
        # Add destination image if available
        if 'destination_image' in item:
            result['destination_image'] = item['destination_image']
            
        # Add contrastive examples if available
        if 'contrastive_data' in item:
            self._add_contrastive_examples(item['contrastive_data'], result)
            
        return result
        
    def _add_contrastive_examples(self, contrastive_data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Add contrastive examples to the result dictionary.
        Retrieve tokenized data from normalizer.
        
        Args:
            contrastive_data: Dictionary containing contrastive samples from normalizer
            result: Result dictionary to update with contrastive examples
        """
        # Add tokenized positive examples
        if 'tokenized_positive' in contrastive_data:
            result['positive_input'] = {
                'input_ids': contrastive_data['tokenized_positive']['input_ids'].squeeze(0),
                'attention_mask': contrastive_data['tokenized_positive']['attention_mask'].squeeze(0)
            }
            
        if 'tokenized_positive_2' in contrastive_data:
            result['positive_input_2'] = {
                'input_ids': contrastive_data['tokenized_positive_2']['input_ids'].squeeze(0),
                'attention_mask': contrastive_data['tokenized_positive_2']['attention_mask'].squeeze(0)
            }
            
        if 'tokenized_negative' in contrastive_data:
            result['negative_input'] = {
                'input_ids': contrastive_data['tokenized_negative']['input_ids'].squeeze(0),
                'attention_mask': contrastive_data['tokenized_negative']['attention_mask'].squeeze(0)
            }
        
        # Also include raw text for separate encoding approach
        if 'positive_text' in contrastive_data:
            result['positive_text'] = contrastive_data['positive_text']
            
        if 'positive_text_2' in contrastive_data:
            result['positive_text_2'] = contrastive_data['positive_text_2']
            
        if 'negative_text' in contrastive_data:
            result['negative_text'] = contrastive_data['negative_text']
    
    @staticmethod
    def create_datasets(config: Config, logger=None, splits=['train', 'val_seen', 'val_unseen'], tokenizer=None):
        """
        Create all three datasets (train, val_seen, val_unseen) at once.
        
        Args:
            config: Configuration object
            logger: Logger for output messages
            splits: List of splits to create
        Returns:
            Dict[str, Dataset]: Dictionary of datasets
        """
        # Load all splits
        datasets = {}
        for split in splits:
            datasets[split] = AnsweringDataset(config, split=split, tokenizer=tokenizer)

        return datasets
    
    @staticmethod
    def preprocess_and_save(config: Config, tokenizer, split='train', logger=None):
        """
        Preprocess and save a dataset split.
        
        Args:
            config: Configuration object
            tokenizer: Tokenizer for text processing
            split: Dataset split ('train', 'val_seen', 'val_unseen')
            logger: Logger for output messages
            
        Returns:
            str: Path to the saved preprocessed data
        """
        if logger is None:
            import logging
            logger = logging.getLogger(__name__)
        
        logger.info(f"📊 Preprocessing {split} dataset...")
        
        # Get JSON file path
        json_path = config.data.get_json_path(split)
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        # Get image directory
        if split == 'train':
            image_dir = config.data.avdn_image_dir
        elif split == 'val_seen':
            image_dir = config.data.avdn_image_dir
        else:  # val_unseen
            image_dir = config.data.avdn_image_dir
        
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Initialize normalizer
        normalizer = AnsweringAgentNormalizer(tokenizer, config)
        
        # Load and filter JSON data (skip turn 0)
        logger.info(f"Loading JSON data from {json_path}")
        with open(json_path, 'r') as f:
            episodes = json.load(f)
        
        # Filter out turn 0 (which has no Q&A and no paraphrases)
        filtered_episodes = []
        total_turns = 0
        filtered_turns = 0
        
        for episode in episodes:
            episode_turns = []
            for dialog in episode["dialogs"]:
                total_turns += 1
                # Skip turn 0 (no Q&A, no paraphrases)
                if dialog["turn_id"] > 0:
                    episode_turns.append(dialog)
                    filtered_turns += 1
            
            if episode_turns:  # Only keep episodes with valid turns
                episode_copy = episode.copy()
                episode_copy["dialogs"] = episode_turns
                filtered_episodes.append(episode_copy)
        
        logger.info(f"Filtered {total_turns} total turns to {filtered_turns} valid turns (excluding turn 0)")
        
        # Preprocess data using normalizer
        logger.info("Processing data with normalizer...")
        processed_data = normalizer.preprocess_all_data(
            filtered_episodes,  # Pass filtered episodes directly
            image_dir,
            output_size=(224, 224),
            apply_augmentation=config.data.use_augmentation
        )
        
        # Save processed data
        if split == 'train':
            output_dir = config.data.train_processed_path_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Save in chunks for train data
            chunk_size = 1000
            AnsweringDataset.save_in_chunks(processed_data, chunk_size, output_dir)
            logger.info(f"✅ Train data saved in chunks to {output_dir}")
            return output_dir
        else:
            # Save as single file for validation data
            if split == 'val_seen':
                output_path = config.data.val_seen_processed_path
            else:  # val_unseen
                output_path = config.data.val_unseen_processed_path
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(processed_data, f)
            
            logger.info(f"✅ {split} data saved to {output_path}")
            return output_path