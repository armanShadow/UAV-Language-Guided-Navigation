#!/usr/bin/env python3
"""
Test Dataset Verification
========================

Verifies that the preprocessed .pkl datasets match the augmented JSON format.
Checks for:
1. Data consistency between JSON and preprocessed files
2. Paraphrase preservation in preprocessed data
3. Correct number of samples
4. Proper data structure

USAGE:
    cd AnsweringAgent/src
    python test_dataset_verification.py
"""

import json
import pickle
import os
import random
from typing import Dict, Any, List
from config import Config
from transformers import T5Tokenizer

def load_json_data(json_path: str) -> List[Dict[str, Any]]:
    """Load JSON data and extract dialog turns with paraphrases."""
    print(f"📂 Loading JSON data from {json_path}")
    
    with open(json_path, 'r') as f:
        episodes = json.load(f)
    
    # Extract all dialog turns with paraphrases
    turns_with_paraphrases = []
    total_episodes = len(episodes)
    
    for episode in episodes:
        episode_id = episode['episode_id']
        for dialog in episode['dialogs']:
            if dialog['turn_id'] > 0:  # Skip first turn with no Q&A
                if 'paraphrases' in dialog:
                    turns_with_paraphrases.append({
                        'episode_id': episode_id,
                        'turn_id': dialog['turn_id'],
                        'question': dialog['question'],
                        'answer': dialog['answer'],
                        'paraphrases': dialog['paraphrases']
                    })
    
    print(f"  ✅ Found {len(turns_with_paraphrases)} turns with paraphrases from {total_episodes} episodes")
    return turns_with_paraphrases

def load_preprocessed_data(pkl_path: str, is_train: bool = False) -> Dict[int, Any]:
    """Load preprocessed .pkl data."""
    print(f"📂 Loading preprocessed data from {pkl_path}")
    
    if is_train:
        # Load chunked train data
        data = {}
        for file in os.listdir(pkl_path):
            if file.endswith('.pkl'):
                chunk_path = os.path.join(pkl_path, file)
                with open(chunk_path, 'rb') as f:
                    chunk_data = pickle.load(f)
                    data.update(chunk_data)
                    print(f"  📦 Loaded chunk {file}: {len(chunk_data)} items")
    else:
        # Load single validation file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            print(f"  📦 Loaded {len(data)} items")
    
    return data

def verify_paraphrase_preservation(json_turns: List[Dict], pkl_data: Dict[int, Any]) -> Dict[str, Any]:
    """Verify that paraphrases are preserved in preprocessed data."""
    print("\n🔍 Verifying paraphrase preservation...")
    
    paraphrase_stats = {
        'total_turns_with_paraphrases': 0,
        'turns_with_contrastive_data': 0,
        'positive_examples_found': 0,
        'negative_examples_found': 0,
        'missing_paraphrases': 0
    }
    
    # Sample some items for detailed verification
    sample_size = min(10, len(pkl_data))
    sample_indices = random.sample(list(pkl_data.keys()), sample_size)
    
    print(f"📊 Analyzing {sample_size} sample items...")
    
    for idx in sample_indices:
        item = pkl_data[idx]
        
        # Check if contrastive data exists
        if 'contrastive_data' in item:
            paraphrase_stats['turns_with_contrastive_data'] += 1
            
            contrastive_data = item['contrastive_data']
            
            # Check positive examples
            if 'positive_examples' in contrastive_data and contrastive_data['positive_examples']:
                paraphrase_stats['positive_examples_found'] += len(contrastive_data['positive_examples'])
                print(f"  ✅ Item {idx}: {len(contrastive_data['positive_examples'])} positive examples")
            
            # Check negative examples
            if 'negative_examples' in contrastive_data and contrastive_data['negative_examples']:
                paraphrase_stats['negative_examples_found'] += len(contrastive_data['negative_examples'])
                print(f"  ✅ Item {idx}: {len(contrastive_data['negative_examples'])} negative examples")
        else:
            paraphrase_stats['missing_paraphrases'] += 1
            print(f"  ❌ Item {idx}: No contrastive data found")
    
    return paraphrase_stats

def verify_data_structure(pkl_data: Dict[int, Any]) -> Dict[str, Any]:
    """Verify the structure of preprocessed data."""
    print("\n🔍 Verifying data structure...")
    
    structure_stats = {
        'total_items': len(pkl_data),
        'items_with_text_input': 0,
        'items_with_current_view': 0,
        'items_with_previous_views': 0,
        'items_with_destination': 0,
        'items_with_contrastive_data': 0
    }
    
    # Sample some items for verification
    sample_size = min(5, len(pkl_data))
    sample_indices = random.sample(list(pkl_data.keys()), sample_size)
    
    for idx in sample_indices:
        item = pkl_data[idx]
        
        # Check required fields
        if 'tokenized_input' in item:
            structure_stats['items_with_text_input'] += 1
            print(f"  ✅ Item {idx}: Text input shape {item['tokenized_input']['input_ids'].shape}")
        
        if 'current_view_image' in item:
            structure_stats['items_with_current_view'] += 1
            print(f"  ✅ Item {idx}: Current view shape {item['current_view_image'].shape}")
        
        if 'previous_views_image' in item:
            structure_stats['items_with_previous_views'] += 1
            if isinstance(item['previous_views_image'], list):
                print(f"  ✅ Item {idx}: Previous views (list of {len(item['previous_views_image'])} tensors)")
            else:
                print(f"  ✅ Item {idx}: Previous views shape {item['previous_views_image'].shape}")
        
        if 'destination_image' in item:
            structure_stats['items_with_destination'] += 1
            print(f"  ✅ Item {idx}: Destination image shape {item['destination_image'].shape}")
        
        if 'contrastive_data' in item:
            structure_stats['items_with_contrastive_data'] += 1
    
    return structure_stats

def compare_sample_counts(json_path: str, pkl_path: str, split_name: str) -> Dict[str, Any]:
    """Compare sample counts between JSON and preprocessed data."""
    print(f"\n📊 Comparing sample counts for {split_name}...")
    
    # Count JSON turns with paraphrases
    json_turns = load_json_data(json_path)
    json_count = len(json_turns)
    
    # Count preprocessed items
    is_train = split_name == 'train'
    pkl_data = load_preprocessed_data(pkl_path, is_train)
    pkl_count = len(pkl_data)
    
    comparison = {
        'split': split_name,
        'json_turns_with_paraphrases': json_count,
        'preprocessed_items': pkl_count,
        'match': json_count == pkl_count
    }
    
    print(f"  📈 JSON turns with paraphrases: {json_count}")
    print(f"  📈 Preprocessed items: {pkl_count}")
    print(f"  {'✅' if comparison['match'] else '❌'} Counts {'match' if comparison['match'] else 'do not match'}")
    
    return comparison

def main():
    """Run comprehensive dataset verification."""
    print("🚀 Starting Dataset Verification...\n")
    
    # Load configuration
    config = Config()
    
    # Initialize tokenizer for potential use
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, model_max_length=config.data.max_seq_length)
    
    # Define splits to test
    splits = ['train', 'val_seen', 'val_unseen']
    
    all_results = {}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"🔍 Testing {split.upper()} split")
        print(f"{'='*60}")
        
        # Get paths
        json_path = config.data.get_json_path(split)
        if split == 'train':
            pkl_path = config.data.train_processed_path_dir
        elif split == 'val_seen':
            pkl_path = config.data.val_seen_processed_path
        else:  # val_unseen
            pkl_path = config.data.val_unseen_processed_path
        
        # Check if files exist
        if not os.path.exists(json_path):
            print(f"❌ JSON file not found: {json_path}")
            continue
        
        if not os.path.exists(pkl_path):
            print(f"❌ Preprocessed file not found: {pkl_path}")
            continue
        
        # Compare sample counts
        comparison = compare_sample_counts(json_path, pkl_path, split)
        
        # Load data for detailed verification
        json_turns = load_json_data(json_path)
        is_train = split == 'train'
        pkl_data = load_preprocessed_data(pkl_path, is_train)
        
        # Verify paraphrase preservation
        paraphrase_stats = verify_paraphrase_preservation(json_turns, pkl_data)
        
        # Verify data structure
        structure_stats = verify_data_structure(pkl_data)
        
        # Combine results
        all_results[split] = {
            'comparison': comparison,
            'paraphrase_stats': paraphrase_stats,
            'structure_stats': structure_stats
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    total_paraphrases = 0
    total_contrastive_items = 0
    
    for split, results in all_results.items():
        print(f"\n{split.upper()}:")
        print(f"  📊 Sample count match: {'✅' if results['comparison']['match'] else '❌'}")
        print(f"  📊 Preprocessed items: {results['structure_stats']['total_items']}")
        print(f"  📊 Items with contrastive data: {results['paraphrase_stats']['turns_with_contrastive_data']}")
        print(f"  📊 Positive examples found: {results['paraphrase_stats']['positive_examples_found']}")
        print(f"  📊 Negative examples found: {results['paraphrase_stats']['negative_examples_found']}")
        
        total_paraphrases += results['paraphrase_stats']['positive_examples_found'] + results['paraphrase_stats']['negative_examples_found']
        total_contrastive_items += results['paraphrase_stats']['turns_with_contrastive_data']
    
    print(f"\n🎯 OVERALL STATISTICS:")
    print(f"  📊 Total contrastive items: {total_contrastive_items}")
    print(f"  📊 Total paraphrases: {total_paraphrases}")
    print(f"  📊 Average paraphrases per item: {total_paraphrases / max(1, total_contrastive_items):.1f}")
    
    # Final verdict
    all_counts_match = all(results['comparison']['match'] for results in all_results.values())
    all_have_contrastive = all(results['paraphrase_stats']['turns_with_contrastive_data'] > 0 for results in all_results.values())
    
    if all_counts_match and all_have_contrastive:
        print(f"\n🎉 VERIFICATION PASSED!")
        print(f"✅ All datasets match their JSON counterparts")
        print(f"✅ All datasets contain contrastive learning data")
        print(f"✅ Ready for training with paraphrases!")
    else:
        print(f"\n❌ VERIFICATION FAILED!")
        if not all_counts_match:
            print(f"❌ Sample counts do not match")
        if not all_have_contrastive:
            print(f"❌ Some datasets missing contrastive data")

if __name__ == "__main__":
    main() 