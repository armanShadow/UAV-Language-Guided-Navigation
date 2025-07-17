#!/usr/bin/env python3
"""
Test Dataset Verification
========================

Verifies that the preprocessed .pkl datasets match the augmented JSON format.
Checks for:
1. Episode count matching between JSON and PKL
2. Dialog turn count matching (excluding turn 0)
3. Exact content matching for each turn
4. Paraphrase preservation in preprocessed data

USAGE:
    cd AnsweringAgent/src
    python test_dataset_verification.py
"""

import json
import pickle
import os
import random
from typing import Dict, Any, List, Tuple
from config import Config
from transformers import T5Tokenizer

def load_json_data(json_path: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """Load JSON data and extract episodes and dialog turns with paraphrases."""
    print(f"📂 Loading JSON data from {json_path}")
    
    with open(json_path, 'r') as f:
        episodes = json.load(f)
    
    # Extract all dialog turns with paraphrases (excluding turn 0)
    turns_with_paraphrases = []
    total_episodes = len(episodes)
    total_turns = 0
    valid_turns = 0
    turns_with_paraphrases_count = 0
    
    for episode in episodes:
        episode_id = episode['episode_id']
        
        for dialog in episode['dialogs']:
            total_turns += 1
            if dialog['turn_id'] > 0:  # Skip turn 0
                valid_turns += 1
                turn_data = {
                    'episode_id': episode_id,
                    'turn_id': dialog['turn_id'],
                    'question': dialog['question'],
                    'answer': dialog['answer'],
                    'dialog_history': dialog['dialog_history'],
                    'first_instruction': episode['first_instruction'],
                    'observation': dialog['observation'],
                    'previous_observations': dialog['previous_observations']
                }
                
                if 'paraphrases' in dialog:
                    turn_data['paraphrases'] = dialog['paraphrases']
                    turns_with_paraphrases_count += 1
                
                turns_with_paraphrases.append(turn_data)
    
    stats = {
        'total_episodes': total_episodes,
        'total_turns': total_turns,
        'valid_turns': valid_turns,
        'turns_with_paraphrases': turns_with_paraphrases_count
    }
    
    print(f"  ✅ Found {total_episodes} episodes")
    print(f"  ✅ Total turns: {total_turns}, Valid turns (excluding turn 0): {valid_turns}")
    print(f"  ✅ Turns with paraphrases: {stats['turns_with_paraphrases']}")
    
    return episodes, stats

def load_preprocessed_data(pkl_path: str, is_train: bool = False) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    """Load preprocessed .pkl data."""
    print(f"📂 Loading preprocessed data from {pkl_path}")
    
    if is_train:
        # Load chunked train data
        data = {}
        total_chunks = 0
        for file in os.listdir(pkl_path):
            if file.endswith('.pkl'):
                chunk_path = os.path.join(pkl_path, file)
                with open(chunk_path, 'rb') as f:
                    chunk_data = pickle.load(f)
                    data.update(chunk_data)
                    total_chunks += 1
                    print(f"  📦 Loaded chunk {file}: {len(chunk_data)} items")
    else:
        # Load single validation file
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            print(f"  📦 Loaded {len(data)} items")
    
    # Analyze data structure
    stats = {
        'total_items': len(data),
        'items_with_contrastive_data': 0,
        'items_with_paraphrases': 0,
        'total_positive_examples': 0,
        'total_negative_examples': 0
    }
    
    # Sample some items to check structure
    sample_size = min(10, len(data))
    sample_indices = random.sample(list(data.keys()), sample_size)
    
    for idx in sample_indices:
        item = data[idx]
        if 'contrastive_data' in item:
            stats['items_with_contrastive_data'] += 1
            contrastive_data = item['contrastive_data']
            if 'positive_examples' in contrastive_data and contrastive_data['positive_examples']:
                stats['items_with_paraphrases'] += len(contrastive_data['positive_examples'])
                stats['total_positive_examples'] += len(contrastive_data['positive_examples'])
            if 'negative_examples' in contrastive_data and contrastive_data['negative_examples']:
                stats['total_negative_examples'] += len(contrastive_data['negative_examples'])
    
    # Scale up sample statistics
    if sample_size > 0:
        scale_factor = len(data) / sample_size
        stats['items_with_contrastive_data'] = int(stats['items_with_contrastive_data'] * scale_factor)
        stats['total_positive_examples'] = int(stats['total_positive_examples'] * scale_factor)
        stats['total_negative_examples'] = int(stats['total_negative_examples'] * scale_factor)
    
    return data, stats

def verify_episode_and_turn_counts(json_episodes: List[Dict], pkl_data: Dict[int, Any], split_name: str) -> Dict[str, Any]:
    """Verify episode and turn counts match between JSON and PKL."""
    print(f"\n📊 Verifying episode and turn counts for {split_name}...")
    
    # Count JSON episodes and valid turns
    json_episode_count = len(json_episodes)
    json_valid_turn_count = 0
    json_turns_with_paraphrases = 0
    
    for episode in json_episodes:
        for dialog in episode['dialogs']:
            if dialog['turn_id'] > 0:  # Valid turns only
                json_valid_turn_count += 1
                if 'paraphrases' in dialog:
                    json_turns_with_paraphrases += 1
    
    # Count PKL items
    pkl_item_count = len(pkl_data)
    
    verification = {
        'split': split_name,
        'json_episodes': json_episode_count,
        'json_valid_turns': json_valid_turn_count,
        'json_turns_with_paraphrases': json_turns_with_paraphrases,
        'pkl_items': pkl_item_count,
        'turn_count_match': json_valid_turn_count == pkl_item_count,
        'paraphrase_coverage': json_turns_with_paraphrases / max(1, json_valid_turn_count)
    }
    
    print(f"  📈 JSON episodes: {json_episode_count}")
    print(f"  📈 JSON valid turns (excluding turn 0): {json_valid_turn_count}")
    print(f"  📈 JSON turns with paraphrases: {json_turns_with_paraphrases}")
    print(f"  📈 PKL items: {pkl_item_count}")
    print(f"  📈 Paraphrase coverage: {verification['paraphrase_coverage']:.2%}")
    print(f"  {'✅' if verification['turn_count_match'] else '❌'} Turn counts {'match' if verification['turn_count_match'] else 'do not match'}")
    
    return verification

def verify_content_structure(pkl_data: Dict[int, Any]) -> Dict[str, Any]:
    """Verify the structure and content of PKL data."""
    print(f"\n🔍 Verifying PKL data structure...")
    
    content_stats = {
        'total_checks': 0,
        'items_with_required_fields': 0,
        'items_with_tokenized_input': 0,
        'items_with_tokenized_answer': 0,
        'items_with_current_view': 0,
        'items_with_previous_views': 0,
        'items_with_dialog_context': 0,
        'items_with_question': 0,
        'items_with_answer': 0,
        'items_with_contrastive_data': 0,
        'items_with_positive_examples': 0,
        'items_with_negative_examples': 0,
        'errors': []
    }
    
    # Check a sample of PKL items for structure
    sample_size = min(20, len(pkl_data))
    sample_indices = random.sample(list(pkl_data.keys()), sample_size)
    
    print(f"📊 Checking {sample_size} sample items for structure verification...")
    
    required_fields = ['tokenized_input', 'tokenized_answer', 'current_view_image', 'dialog_context', 'question', 'answer']
    
    for idx in sample_indices:
        item = pkl_data[idx]
        content_stats['total_checks'] += 1
        
        try:
            # Check required fields
            has_all_required = all(field in item for field in required_fields)
            if has_all_required:
                content_stats['items_with_required_fields'] += 1
                print(f"  ✅ Item {idx}: All required fields present")
            else:
                missing = [field for field in required_fields if field not in item]
                print(f"  ❌ Item {idx}: Missing fields: {missing}")
            
            # Check individual fields
            if 'tokenized_input' in item:
                content_stats['items_with_tokenized_input'] += 1
            if 'tokenized_answer' in item:
                content_stats['items_with_tokenized_answer'] += 1
            if 'current_view_image' in item:
                content_stats['items_with_current_view'] += 1
            if 'previous_views_image' in item:
                content_stats['items_with_previous_views'] += 1
            if 'dialog_context' in item:
                content_stats['items_with_dialog_context'] += 1
            if 'question' in item:
                content_stats['items_with_question'] += 1
            if 'answer' in item:
                content_stats['items_with_answer'] += 1
            
            # Check contrastive data (paraphrases)
            if 'contrastive_data' in item:
                content_stats['items_with_contrastive_data'] += 1
                contrastive_data = item['contrastive_data']
                
                if 'positive_examples' in contrastive_data and contrastive_data['positive_examples']:
                    content_stats['items_with_positive_examples'] += 1
                    print(f"  ✅ Item {idx}: {len(contrastive_data['positive_examples'])} positive examples")
                
                if 'negative_examples' in contrastive_data and contrastive_data['negative_examples']:
                    content_stats['items_with_negative_examples'] += 1
                    print(f"  ✅ Item {idx}: {len(contrastive_data['negative_examples'])} negative examples")
            else:
                print(f"  ⚠️ Item {idx}: No contrastive data found")
                
        except Exception as e:
            content_stats['errors'].append(f"Item {idx}: {str(e)}")
            print(f"  ❌ Item {idx}: Error during verification - {str(e)}")
    
    # Scale up sample statistics
    if sample_size > 0:
        scale_factor = len(pkl_data) / sample_size
        for key in content_stats:
            if key not in ['total_checks', 'errors'] and isinstance(content_stats[key], int):
                content_stats[key] = int(content_stats[key] * scale_factor)
    
    return content_stats

def verify_paraphrase_quality(pkl_data: Dict[int, Any]) -> Dict[str, Any]:
    """Verify the quality and structure of paraphrases in preprocessed data."""
    print(f"\n🔍 Verifying paraphrase quality...")
    
    paraphrase_stats = {
        'items_checked': 0,
        'items_with_contrastive_data': 0,
        'items_with_2_positives': 0,
        'items_with_1_negative': 0,
        'items_with_correct_structure': 0,
        'total_positive_examples': 0,
        'total_negative_examples': 0,
        'avg_positive_length': 0.0,
        'avg_negative_length': 0.0
    }
    
    # Sample items for detailed verification
    sample_size = min(15, len(pkl_data))
    sample_indices = random.sample(list(pkl_data.keys()), sample_size)
    
    print(f"📊 Analyzing {sample_size} sample items for paraphrase quality...")
    
    positive_lengths = []
    negative_lengths = []
    
    for idx in sample_indices:
        item = pkl_data[idx]
        paraphrase_stats['items_checked'] += 1
        
        # Check if contrastive data exists
        if 'contrastive_data' in item:
            paraphrase_stats['items_with_contrastive_data'] += 1
            contrastive_data = item['contrastive_data']
            
            # Check positive examples
            positive_examples = contrastive_data.get('positive_examples', [])
            if positive_examples:
                paraphrase_stats['total_positive_examples'] += len(positive_examples)
                if len(positive_examples) == 2:
                    paraphrase_stats['items_with_2_positives'] += 1
                    print(f"  ✅ Item {idx}: Correct positive count (2)")
                else:
                    print(f"  ❌ Item {idx}: Incorrect positive count ({len(positive_examples)})")
                
                # Calculate lengths
                for pos in positive_examples:
                    if hasattr(pos, 'get') and 'text' in pos:
                        positive_lengths.append(len(pos['text']))
                    elif isinstance(pos, str):
                        positive_lengths.append(len(pos))
            
            # Check negative examples
            negative_examples = contrastive_data.get('negative_examples', [])
            if negative_examples:
                paraphrase_stats['total_negative_examples'] += len(negative_examples)
                if len(negative_examples) == 1:
                    paraphrase_stats['items_with_1_negative'] += 1
                    print(f"  ✅ Item {idx}: Correct negative count (1)")
                else:
                    print(f"  ❌ Item {idx}: Incorrect negative count ({len(negative_examples)})")
                
                # Calculate lengths
                for neg in negative_examples:
                    if hasattr(neg, 'get') and 'text' in neg:
                        negative_lengths.append(len(neg['text']))
                    elif isinstance(neg, str):
                        negative_lengths.append(len(neg))
            
            # Check structure correctness (2P + 1N)
            if len(positive_examples) == 2 and len(negative_examples) == 1:
                paraphrase_stats['items_with_correct_structure'] += 1
        else:
            print(f"  ❌ Item {idx}: No contrastive data found")
    
    # Calculate averages
    if positive_lengths:
        paraphrase_stats['avg_positive_length'] = sum(positive_lengths) / len(positive_lengths)
    if negative_lengths:
        paraphrase_stats['avg_negative_length'] = sum(negative_lengths) / len(negative_lengths)
    
    # Scale up sample statistics
    if sample_size > 0:
        scale_factor = len(pkl_data) / sample_size
        for key in ['items_with_contrastive_data', 'items_with_2_positives', 'items_with_1_negative', 
                   'items_with_correct_structure', 'total_positive_examples', 'total_negative_examples']:
            paraphrase_stats[key] = int(paraphrase_stats[key] * scale_factor)
    
    return paraphrase_stats

def verify_tokenization_quality(pkl_data: Dict[int, Any]) -> Dict[str, Any]:
    """Verify tokenization quality in preprocessed data."""
    print(f"\n🔍 Verifying tokenization quality...")
    
    token_stats = {
        'items_checked': 0,
        'items_with_valid_input_tokens': 0,
        'items_with_valid_answer_tokens': 0,
        'avg_input_length': 0.0,
        'avg_answer_length': 0.0,
        'input_lengths': [],
        'answer_lengths': []
    }
    
    # Sample items for verification
    sample_size = min(10, len(pkl_data))
    sample_indices = random.sample(list(pkl_data.keys()), sample_size)
    
    for idx in sample_indices:
        item = pkl_data[idx]
        token_stats['items_checked'] += 1
        
        # Check input tokenization
        if 'tokenized_input' in item:
            tokenized_input = item['tokenized_input']
            if 'input_ids' in tokenized_input:
                input_ids = tokenized_input['input_ids']
                if hasattr(input_ids, 'shape'):
                    input_length = input_ids.shape[-1]
                    token_stats['input_lengths'].append(input_length)
                    token_stats['items_with_valid_input_tokens'] += 1
                    print(f"  ✅ Item {idx}: Input tokens shape {input_ids.shape}")
        
        # Check answer tokenization
        if 'tokenized_answer' in item:
            tokenized_answer = item['tokenized_answer']
            if 'input_ids' in tokenized_answer:
                answer_ids = tokenized_answer['input_ids']
                if hasattr(answer_ids, 'shape'):
                    answer_length = answer_ids.shape[-1]
                    token_stats['answer_lengths'].append(answer_length)
                    token_stats['items_with_valid_answer_tokens'] += 1
                    print(f"  ✅ Item {idx}: Answer tokens shape {answer_ids.shape}")
    
    # Calculate averages
    if token_stats['input_lengths']:
        token_stats['avg_input_length'] = sum(token_stats['input_lengths']) / len(token_stats['input_lengths'])
    if token_stats['answer_lengths']:
        token_stats['avg_answer_length'] = sum(token_stats['answer_lengths']) / len(token_stats['answer_lengths'])
    
    return token_stats

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
        
        # Load data
        json_episodes, json_stats = load_json_data(json_path)
        is_train = split == 'train'
        pkl_data, pkl_stats = load_preprocessed_data(pkl_path, is_train)
        
        # Verify episode and turn counts
        count_verification = verify_episode_and_turn_counts(json_episodes, pkl_data, split)
        
        # Verify content structure
        content_verification = verify_content_structure(pkl_data)
        
        # Verify paraphrase quality
        paraphrase_stats = verify_paraphrase_quality(pkl_data)
        
        # Verify tokenization quality
        tokenization_stats = verify_tokenization_quality(pkl_data)
        
        # Combine results
        all_results[split] = {
            'json_stats': json_stats,
            'pkl_stats': pkl_stats,
            'count_verification': count_verification,
            'content_verification': content_verification,
            'paraphrase_stats': paraphrase_stats,
            'tokenization_stats': tokenization_stats
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    total_json_turns = 0
    total_pkl_items = 0
    total_paraphrases = 0
    total_contrastive_items = 0
    
    for split, results in all_results.items():
        print(f"\n{split.upper()}:")
        print(f"  📊 JSON episodes: {results['json_stats']['total_episodes']}")
        print(f"  📊 JSON valid turns: {results['json_stats']['valid_turns']}")
        print(f"  📊 JSON turns with paraphrases: {results['json_stats']['turns_with_paraphrases']}")
        print(f"  📊 PKL items: {results['pkl_stats']['total_items']}")
        print(f"  📊 Turn count match: {'✅' if results['count_verification']['turn_count_match'] else '❌'}")
        print(f"  📊 Paraphrase coverage: {results['count_verification']['paraphrase_coverage']:.2%}")
        print(f"  📊 PKL items with contrastive data: {results['paraphrase_stats']['items_with_contrastive_data']}")
        print(f"  📊 PKL items with correct structure (2P+1N): {results['paraphrase_stats']['items_with_correct_structure']}")
        
        total_json_turns += results['json_stats']['valid_turns']
        total_pkl_items += results['pkl_stats']['total_items']
        total_paraphrases += results['paraphrase_stats']['total_positive_examples'] + results['paraphrase_stats']['total_negative_examples']
        total_contrastive_items += results['paraphrase_stats']['items_with_contrastive_data']
    
    print(f"\n🎯 OVERALL STATISTICS:")
    print(f"  📊 Total JSON valid turns: {total_json_turns}")
    print(f"  📊 Total PKL items: {total_pkl_items}")
    print(f"  📊 Total contrastive items: {total_contrastive_items}")
    print(f"  📊 Total paraphrases: {total_paraphrases}")
    print(f"  📊 Average paraphrases per contrastive item: {total_paraphrases / max(1, total_contrastive_items):.1f}")
    
    # Final verdict
    all_counts_match = all(results['count_verification']['turn_count_match'] for results in all_results.values())
    all_have_contrastive = all(results['paraphrase_stats']['items_with_contrastive_data'] > 0 for results in all_results.values())
    all_have_correct_structure = all(results['paraphrase_stats']['items_with_correct_structure'] > 0 for results in all_results.values())
    
    if all_counts_match and all_have_contrastive and all_have_correct_structure:
        print(f"\n🎉 VERIFICATION PASSED!")
        print(f"✅ All datasets have matching turn counts")
        print(f"✅ All datasets contain contrastive learning data")
        print(f"✅ All datasets have correct paraphrase structure (2P+1N)")
        print(f"✅ Ready for training with paraphrases!")
    else:
        print(f"\n❌ VERIFICATION FAILED!")
        if not all_counts_match:
            print(f"❌ Turn counts do not match between JSON and PKL")
        if not all_have_contrastive:
            print(f"❌ Some datasets missing contrastive data")
        if not all_have_correct_structure:
            print(f"❌ Some datasets have incorrect paraphrase structure")

if __name__ == "__main__":
    main() 