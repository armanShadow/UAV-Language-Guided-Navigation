#!/usr/bin/env python3
"""
Count Missing Turns Precisely
=============================

Precisely count the exact number of missing turns vs episodes with missing turns.
"""

import json
import os
import sys
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def analyze_missing_turns_precisely(json_path: str):
    """Precisely analyze missing turns vs episodes with missing turns."""
    print(f"🔍 Analyzing missing turns precisely: {json_path}")
    
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
    
    print(f"\n📊 Detailed Analysis:")
    print(f"{'Episode ID':<30} {'Turn':<6} {'Status':<20} {'Question':<50} {'Answer':<30}")
    print(f"{'-'*30} {'-'*6} {'-'*20} {'-'*50} {'-'*30}")
    
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
                
                question = dialog['question'][:47] + "..." if len(dialog['question']) > 50 else dialog['question']
                answer = dialog['answer'][:27] + "..." if len(dialog['answer']) > 30 else dialog['answer']
                
                if 'paraphrases' in dialog:
                    turns_with_paraphrases += 1
                    episode_turns_with_paraphrases += 1
                    
                    # Check if it has correct 2P + 1N structure
                    paraphrases = dialog['paraphrases']
                    positives = paraphrases.get('positives', [])
                    negatives = paraphrases.get('negatives', [])
                    
                    if len(positives) == 2 and len(negatives) == 1:
                        status = "✅ Complete (2P+1N)"
                    else:
                        status = f"⚠️ Wrong structure ({len(positives)}P+{len(negatives)}N)"
                        episode_missing_turns.append(dialog['turn_id'])
                        missing_turns.append({
                            'episode_id': episode_id,
                            'turn_id': dialog['turn_id'],
                            'reason': f'Wrong structure: {len(positives)}P+{len(negatives)}N',
                            'question': dialog['question'],
                            'answer': dialog['answer']
                        })
                    
                    print(f"{episode_id:<30} {dialog['turn_id']:<6} {status:<20} {question:<50} {answer:<30}")
                else:
                    status = "❌ No paraphrases"
                    episode_missing_turns.append(dialog['turn_id'])
                    missing_turns.append({
                        'episode_id': episode_id,
                        'turn_id': dialog['turn_id'],
                        'reason': 'No paraphrases',
                        'question': dialog['question'],
                        'answer': dialog['answer']
                    })
                    print(f"{episode_id:<30} {dialog['turn_id']:<6} {status:<20} {question:<50} {answer:<30}")
        
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
    
    print(f"\n📊 PRECISE SUMMARY:")
    print(f"  📈 Total episodes: {total_episodes}")
    print(f"  📈 Total turns: {total_turns}")
    print(f"  📈 Valid turns (excluding turn 0): {valid_turns}")
    print(f"  📈 Turns with paraphrases: {turns_with_paraphrases}")
    print(f"  📈 Missing turns (exact count): {len(missing_turns)}")
    print(f"  📈 Episodes with all paraphrases: {episodes_with_all_paraphrases}")
    print(f"  📈 Episodes with missing turns: {len(episodes_with_missing_turns)}")
    
    print(f"\n🔍 MISSING TURNS BREAKDOWN:")
    no_paraphrases = sum(1 for t in missing_turns if t['reason'] == 'No paraphrases')
    wrong_structure = sum(1 for t in missing_turns if 'Wrong structure' in t['reason'])
    
    print(f"  ❌ Turns with no paraphrases: {no_paraphrases}")
    print(f"  ⚠️ Turns with wrong structure: {wrong_structure}")
    print(f"  📈 Total missing: {no_paraphrases + wrong_structure}")
    
    if missing_turns:
        print(f"\n📋 FIRST 10 MISSING TURNS:")
        for i, turn in enumerate(missing_turns[:10]):
            print(f"  {i+1}. Episode {turn['episode_id']}, Turn {turn['turn_id']}: {turn['reason']}")
            print(f"     Q: {turn['question'][:80]}...")
            print(f"     A: {turn['answer'][:80]}...")
    
    return {
        'total_episodes': total_episodes,
        'valid_turns': valid_turns,
        'turns_with_paraphrases': turns_with_paraphrases,
        'missing_turns': missing_turns,
        'episodes_with_missing_turns': episodes_with_missing_turns
    }

def main():
    """Main function to count missing turns precisely."""
    print("🔢 Counting Missing Turns Precisely...")
    
    config = Config()
    
    # Analyze train dataset
    print(f"\n{'='*60}")
    print(f"📋 TRAIN DATASET ANALYSIS")
    print(f"{'='*60}")
    
    train_stats = analyze_missing_turns_precisely(config.data.train_augmented_json_path)
    
    print(f"\n🎯 CONCLUSION:")
    expected_missing = train_stats['valid_turns'] - train_stats['turns_with_paraphrases']
    actual_missing = len(train_stats['missing_turns'])
    
    print(f"  📈 Expected missing turns: {expected_missing}")
    print(f"  📈 Actual missing turns found: {actual_missing}")
    
    if expected_missing == actual_missing:
        print(f"  ✅ Counts match! Analysis is correct.")
    else:
        print(f"  ❌ Counts don't match. Something is wrong with the analysis.")

if __name__ == "__main__":
    main() 