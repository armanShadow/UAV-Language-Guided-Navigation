#!/usr/bin/env python3
"""
Debug Episode Counts
===================

Compare episode counts between original and augmented datasets to understand the discrepancy.
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def analyze_dataset(json_path: str, dataset_name: str):
    """Analyze a dataset and print statistics."""
    print(f"\n📊 Analyzing {dataset_name}: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        episodes = json.load(f)
    
    total_episodes = len(episodes)
    total_turns = 0
    valid_turns = 0
    episodes_with_paraphrases = 0
    
    for episode in episodes:
        episode_turns = 0
        episode_has_paraphrases = False
        
        for dialog in episode['dialogs']:
            total_turns += 1
            if dialog['turn_id'] > 0:
                valid_turns += 1
                episode_turns += 1
                
                if 'paraphrases' in dialog:
                    episode_has_paraphrases = True
        
        if episode_has_paraphrases:
            episodes_with_paraphrases += 1
    
    stats = {
        'total_episodes': total_episodes,
        'total_turns': total_turns,
        'valid_turns': valid_turns,
        'episodes_with_paraphrases': episodes_with_paraphrases,
        'avg_turns_per_episode': total_turns / total_episodes if total_episodes > 0 else 0
    }
    
    print(f"  📈 Total episodes: {total_episodes}")
    print(f"  📈 Total turns: {total_turns}")
    print(f"  📈 Valid turns (excluding turn 0): {valid_turns}")
    print(f"  📈 Episodes with paraphrases: {episodes_with_paraphrases}")
    print(f"  📈 Average turns per episode: {stats['avg_turns_per_episode']:.2f}")
    
    return stats

def main():
    """Compare original vs augmented datasets."""
    print("🔍 Debugging Episode Counts...")
    
    config = Config()
    
    # Analyze original datasets
    print("\n" + "="*60)
    print("📋 ORIGINAL DATASETS")
    print("="*60)
    
    original_train_stats = analyze_dataset(config.data.train_json_path, "Original Train")
    original_val_seen_stats = analyze_dataset(config.data.val_seen_json_path, "Original Val Seen")
    original_val_unseen_stats = analyze_dataset(config.data.val_unseen_json_path, "Original Val Unseen")
    
    # Analyze augmented datasets
    print("\n" + "="*60)
    print("📋 AUGMENTED DATASETS")
    print("="*60)
    
    augmented_train_stats = analyze_dataset(config.data.train_augmented_json_path, "Augmented Train")
    augmented_val_seen_stats = analyze_dataset(config.data.val_seen_augmented_json_path, "Augmented Val Seen")
    augmented_val_unseen_stats = analyze_dataset(config.data.val_unseen_augmented_json_path, "Augmented Val Unseen")
    
    # Compare results
    print("\n" + "="*60)
    print("📊 COMPARISON SUMMARY")
    print("="*60)
    
    if original_train_stats and augmented_train_stats:
        print(f"\nTRAIN DATASET:")
        print(f"  📈 Original episodes: {original_train_stats['total_episodes']}")
        print(f"  📈 Augmented episodes: {augmented_train_stats['total_episodes']}")
        print(f"  📈 Difference: {augmented_train_stats['total_episodes'] - original_train_stats['total_episodes']}")
        print(f"  📈 Original valid turns: {original_train_stats['valid_turns']}")
        print(f"  📈 Augmented valid turns: {augmented_train_stats['valid_turns']}")
        print(f"  📈 Episodes with paraphrases: {augmented_train_stats['episodes_with_paraphrases']}")
    
    if original_val_seen_stats and augmented_val_seen_stats:
        print(f"\nVAL SEEN DATASET:")
        print(f"  📈 Original episodes: {original_val_seen_stats['total_episodes']}")
        print(f"  📈 Augmented episodes: {augmented_val_seen_stats['total_episodes']}")
        print(f"  📈 Difference: {augmented_val_seen_stats['total_episodes'] - original_val_seen_stats['total_episodes']}")
    
    if original_val_unseen_stats and augmented_val_unseen_stats:
        print(f"\nVAL UNSEEN DATASET:")
        print(f"  📈 Original episodes: {original_val_unseen_stats['total_episodes']}")
        print(f"  📈 Augmented episodes: {augmented_val_unseen_stats['total_episodes']}")
        print(f"  📈 Difference: {augmented_val_unseen_stats['total_episodes'] - original_val_unseen_stats['total_episodes']}")

if __name__ == "__main__":
    main() 