#!/usr/bin/env python3
"""
Count Missing Turns for All Datasets
====================================

Precisely count missing turns across all three datasets (train, val_seen, val_unseen).
"""

import json
import os
import sys
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def analyze_missing_turns_precisely(json_path: str, dataset_name: str):
    """Precisely analyze missing turns vs episodes with missing turns."""
    print(f"🔍 Analyzing {dataset_name} missing turns: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        episodes = json.load(f)
    
    total_episodes = len(episodes)
    total_turns = 0
    valid_turns = 0
    turns_with_paraphrases = 0
    
    # Track missing turns
    missing_turns = []
    episodes_with_missing_turns = []
    episodes_with_all_paraphrases = 0
    
    for episode in episodes:
        episode_id = episode['episode_id']
        episode_missing_turns = []
        episode_valid_turns = 0
        episode_turns_with_paraphrases = 0
        
        for dialog in episode['dialogs']:
            total_turns += 1
            if dialog['turn_id'] > 0:
                valid_turns += 1
                episode_valid_turns += 1
                
                if 'paraphrases' in dialog:
                    turns_with_paraphrases += 1
                    episode_turns_with_paraphrases += 1
                    
                    # Check if it has correct 2P + 1N structure
                    paraphrases = dialog['paraphrases']
                    positives = paraphrases.get('positives', [])
                    negatives = paraphrases.get('negatives', [])
                    
                    if len(positives) != 2 or len(negatives) != 1:
                        episode_missing_turns.append(dialog['turn_id'])
                        missing_turns.append({
                            'episode_id': episode_id,
                            'turn_id': dialog['turn_id'],
                            'reason': f'Wrong structure: {len(positives)}P+{len(negatives)}N',
                            'question': dialog['question'],
                            'answer': dialog['answer']
                        })
                        print(f"  ⚠️ Episode {episode_id}, Turn {dialog['turn_id']}: Wrong structure ({len(positives)}P+{len(negatives)}N)")
                else:
                    episode_missing_turns.append(dialog['turn_id'])
                    missing_turns.append({
                        'episode_id': episode_id,
                        'turn_id': dialog['turn_id'],
                        'reason': 'No paraphrases',
                        'question': dialog['question'],
                        'answer': dialog['answer']
                    })
                    print(f"  ❌ Episode {episode_id}, Turn {dialog['turn_id']}: No paraphrases - '{dialog['answer']}'")
        
        # Track episodes with missing turns
        if episode_missing_turns:
            episodes_with_missing_turns.append({
                'episode_id': episode_id,
                'missing_turns': episode_missing_turns,
                'total_valid_turns': episode_valid_turns,
                'turns_with_paraphrases': episode_turns_with_paraphrases
            })
        else:
            episodes_with_all_paraphrases += 1
    
    stats = {
        'dataset_name': dataset_name,
        'total_episodes': total_episodes,
        'total_turns': total_turns,
        'valid_turns': valid_turns,
        'turns_with_paraphrases': turns_with_paraphrases,
        'missing_turns': missing_turns,
        'episodes_with_missing_turns': episodes_with_missing_turns,
        'episodes_with_all_paraphrases': episodes_with_all_paraphrases
    }
    
    print(f"  📈 Total episodes: {total_episodes}")
    print(f"  📈 Valid turns (excluding turn 0): {valid_turns}")
    print(f"  📈 Turns with paraphrases: {turns_with_paraphrases}")
    print(f"  📈 Missing turns (exact count): {len(missing_turns)}")
    print(f"  📈 Coverage: {(turns_with_paraphrases/valid_turns*100):.2f}%")
    
    # Breakdown of missing turns
    no_paraphrases = sum(1 for t in missing_turns if t['reason'] == 'No paraphrases')
    wrong_structure = sum(1 for t in missing_turns if 'Wrong structure' in t['reason'])
    
    print(f"  ❌ Turns with no paraphrases: {no_paraphrases}")
    print(f"  ⚠️ Turns with wrong structure: {wrong_structure}")
    
    return stats

def main():
    """Count missing turns across all datasets."""
    print("🔢 Counting Missing Turns Across All Datasets...")
    
    config = Config()
    
    # Analyze all three datasets
    datasets = ['train', 'val_seen', 'val_unseen']
    all_stats = {}
    total_missing = 0
    total_valid = 0
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"📋 {dataset_name.upper()} DATASET ANALYSIS")
        print(f"{'='*60}")
        
        json_path = config.data.get_json_path(dataset_name)
        stats = analyze_missing_turns_precisely(json_path, dataset_name)
        
        if stats:
            all_stats[dataset_name] = stats
            total_missing += len(stats['missing_turns'])
            total_valid += stats['valid_turns']
    
    # Overall summary
    print(f"\n{'='*60}")
    print(f"📊 OVERALL SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, stats in all_stats.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  📈 Valid turns: {stats['valid_turns']}")
        print(f"  ✅ With paraphrases: {stats['turns_with_paraphrases']}")
        print(f"  ❌ Missing: {len(stats['missing_turns'])}")
        print(f"  📊 Coverage: {(stats['turns_with_paraphrases']/stats['valid_turns']*100):.2f}%")
    
    overall_coverage = ((total_valid - total_missing) / total_valid * 100) if total_valid > 0 else 0
    
    print(f"\n🎯 TOTAL ACROSS ALL DATASETS:")
    print(f"  📈 Total valid turns: {total_valid}")
    print(f"  ❌ Total missing: {total_missing}")
    print(f"  📊 Overall coverage: {overall_coverage:.2f}%")
    
    if total_missing == 0:
        print(f"\n🎉 PERFECT! All datasets have 100% paraphrase coverage!")
        print(f"✅ Ready for preprocessing and training!")
    else:
        print(f"\n⚠️ Still have {total_missing} missing turns across all datasets")
        print(f"💡 Run fix_short_answers.py to fix remaining issues")

if __name__ == "__main__":
    main() 