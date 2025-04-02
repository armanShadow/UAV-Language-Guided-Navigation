import torch
import os
import pickle
import json
from torch.utils.data import DataLoader
from data.dataset import AnsweringDataset
from config import Config
from transformers import T5Tokenizer
import traceback

def check_tensor_shapes(data_item, key_prefix=""):
    """Check tensor shapes in a data item recursively"""
    issues = []
    for key, value in data_item.items():
        full_key = f"{key_prefix}.{key}" if key_prefix else key
        
        if isinstance(value, dict):
            # Recursive check for nested dictionaries
            nested_issues = check_tensor_shapes(value, full_key)
            issues.extend(nested_issues)
        elif isinstance(value, torch.Tensor):
            # Record tensor shape
            issues.append(f"{full_key}: {value.shape}, dtype={value.dtype}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            # Handle list of tensors
            shapes = [tensor.shape for tensor in value]
            if len(set(str(shape) for shape in shapes)) > 1:
                # Different shapes found in list
                issues.append(f"{full_key}: List contains tensors with inconsistent shapes: {shapes}")
            else:
                issues.append(f"{full_key}: List of {len(value)} tensors with shape {shapes[0]}")
    
    return issues

def check_train_dataset():
    """Check the training dataset for inconsistencies"""
    config = Config()
    
    try:
        # Load a few chunks from the train dataset
        train_path = config.data.train_processed_path_dir
        print(f"Checking train dataset chunks in {train_path}")
        
        all_files = [f for f in os.listdir(train_path) if f.endswith('.pkl')]
        sample_files = all_files[:min(3, len(all_files))]  # Take up to 3 files
        
        # Check each file
        for filename in sample_files:
            filepath = os.path.join(train_path, filename)
            print(f"\nChecking file: {filepath}")
            
            with open(filepath, 'rb') as f:
                chunk_data = pickle.load(f)
            
            # Get a few sample items
            sample_keys = list(chunk_data.keys())[:5]
            
            for key in sample_keys:
                item = chunk_data[key]
                print(f"\nItem {key} keys: {list(item.keys())}")
                
                # Check tensor shapes
                issues = check_tensor_shapes(item)
                for issue in issues:
                    print(f"  {issue}")
                
                # Check if tokenized_input is consistent
                if 'tokenized_input' in item:
                    print(f"  tokenized_input keys: {list(item['tokenized_input'].keys())}")
                
                # Check text_label shape
                if 'tokenized_answer' in item:
                    print(f"  tokenized_answer keys: {list(item['tokenized_answer'].keys())}")
                    if 'input_ids' in item['tokenized_answer']:
                        print(f"  tokenized_answer.input_ids shape: {item['tokenized_answer']['input_ids'].shape}")
        
        # Now try to create a DataLoader and iterate over it
        print("\nTrying to create DataLoader...")
        dataset = AnsweringDataset(config, split='train')
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        # Try to get the first batch
        try:
            first_batch = next(iter(loader))
            print("Successfully got first batch!")
            print(f"Batch keys: {list(first_batch.keys())}")
            
            for key, value in first_batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key} (dict):")
                    for subkey, subval in value.items():
                        print(f"    {subkey}: {subval.shape if isinstance(subval, torch.Tensor) else type(subval)}")
                else:
                    print(f"  {key}: {type(value)}")
            
        except Exception as e:
            print(f"Error getting first batch: {str(e)}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error checking dataset: {str(e)}")
        traceback.print_exc()

def check_val_dataset():
    """Check the validation dataset for inconsistencies"""
    config = Config()
    
    try:
        # Load the validation dataset
        val_path = config.data.val_seen_processed_path
        print(f"\nChecking val_seen dataset at {val_path}")
        
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
        
        # Get a few sample items
        sample_keys = list(val_data.keys())[:5]
        
        for key in sample_keys:
            item = val_data[key]
            print(f"\nItem {key} keys: {list(item.keys())}")
            
            # Check tensor shapes
            issues = check_tensor_shapes(item)
            for issue in issues:
                print(f"  {issue}")
        
        # Now try to create a DataLoader and iterate over it
        print("\nTrying to create DataLoader for val_seen...")
        dataset = AnsweringDataset(config, split='val_seen')
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        
        # Try to get the first batch
        try:
            first_batch = next(iter(loader))
            print("Successfully got first batch!")
            
            for key, value in first_batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  {key} (dict):")
                    for subkey, subval in value.items():
                        print(f"    {subkey}: {subval.shape if isinstance(subval, torch.Tensor) else type(subval)}")
        
        except Exception as e:
            print(f"Error getting first batch: {str(e)}")
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error checking dataset: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting dataset debug...")
    check_train_dataset()
    check_val_dataset()
    print("Debug complete!") 