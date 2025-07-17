#!/usr/bin/env python3
"""
Investigate Paraphrase Coverage
===============================

Investigate why not all episodes have paraphrases when comprehensive pipeline 
was supposed to generate 100% coverage for all 4,326 dialog turns.
"""

import json
import os
import sys
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

def analyze_paraphrase_coverage(json_path: str, dataset_name: str):
    """Analyze paraphrase coverage in detail."""
    print(f"\n🔍 Analyzing {dataset_name}: {json_path}")
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return None
    
    with open(json_path, 'r') as f:
        episodes = json.load(f)
    
    total_episodes = len(episodes)
    total_turns = 0
    valid_turns = 0
    turns_with_paraphrases = 0
    episodes_with_paraphrases = 0
    episodes_without_paraphrases = []
    
    # Detailed analysis
    paraphrase_stats = defaultdict(int)
    
    for episode in episodes:
        episode_id = episode['episode_id']
        episode_has_paraphrases = False
        episode_turns_with_paraphrases = 0
        episode_total_turns = 0
        
        for dialog in episode['dialogs']:
            total_turns += 1
            if dialog['turn_id'] > 0:
                valid_turns += 1
                episode_total_turns += 1
                
                if 'paraphrases' in dialog:
                    turns_with_paraphrases += 1
                    episode_turns_with_paraphrases += 1
                    episode_has_paraphrases = True
                    
                    # Analyze paraphrase structure
                    paraphrases = dialog['paraphrases']
                    if 'positives' in paraphrases:
                        paraphrase_stats['positives'] += len(paraphrases['positives'])
                    if 'negatives' in paraphrases:
                        paraphrase_stats['negatives'] += len(paraphrases['negatives'])
                    if 'validation_analysis' in paraphrases:
                        paraphrase_stats['validation_analysis'] += 1
                else:
                    print(f"  ❌ Episode {episode_id}, Turn {dialog['turn_id']}: Missing paraphrases")
                    print(f"      Question: {dialog['question'][:50]}...")
                    print(f"      Answer: {dialog['answer'][:50]}...")
        
        if episode_has_paraphrases:
            episodes_with_paraphrases += 1
        else:
            episodes_without_paraphrases.append(episode_id)
    
    stats = {
        'total_episodes': total_episodes,
        'total_turns': total_turns,
        'valid_turns': valid_turns,
        'turns_with_paraphrases': turns_with_paraphrases,
        'episodes_with_paraphrases': episodes_with_paraphrases,
        'episodes_without_paraphrases': episodes_without_paraphrases,
        'paraphrase_stats': dict(paraphrase_stats),
        'coverage_rate': (turns_with_paraphrases / valid_turns * 100) if valid_turns > 0 else 0
    }
    
    print(f"  📈 Total episodes: {total_episodes}")
    print(f"  📈 Total turns: {total_turns}")
    print(f"  📈 Valid turns (excluding turn 0): {valid_turns}")
    print(f"  📈 Turns with paraphrases: {turns_with_paraphrases}")
    print(f"  📈 Episodes with paraphrases: {episodes_with_paraphrases}")
    print(f"  📈 Coverage rate: {stats['coverage_rate']:.2f}%")
    print(f"  📈 Positive paraphrases: {paraphrase_stats['positives']}")
    print(f"  📈 Negative paraphrases: {paraphrase_stats['negatives']}")
    print(f"  📈 With validation analysis: {paraphrase_stats['validation_analysis']}")
    
    if episodes_without_paraphrases:
        print(f"  ❌ Episodes without paraphrases: {len(episodes_without_paraphrases)}")
        print(f"      First 10: {episodes_without_paraphrases[:10]}")
    
    return stats

def check_comprehensive_pipeline_output():
    """Check if comprehensive pipeline output exists and analyze it."""
    print("\n🔍 Checking Comprehensive Pipeline Output...")
    
    # Check if comprehensive pipeline output exists
    comprehensive_output_path = "comprehensive_avdn_pipeline_output.json"
    if os.path.exists(comprehensive_output_path):
        print(f"✅ Found comprehensive pipeline output: {comprehensive_output_path}")
        with open(comprehensive_output_path, 'r') as f:
            output_data = json.load(f)
        print(f"  📈 Pipeline processed: {output_data.get('total_episodes', 'N/A')} episodes")
        print(f"  📈 Pipeline generated: {output_data.get('total_paraphrases', 'N/A')} paraphrases")
    else:
        print(f"❌ Comprehensive pipeline output not found: {comprehensive_output_path}")
    
    # Check for any log files
    log_files = [f for f in os.listdir('.') if f.endswith('.log') and 'comprehensive' in f.lower()]
    if log_files:
        print(f"📋 Found log files: {log_files}")
        for log_file in log_files[:3]:  # Show first 3
            with open(log_file, 'r') as f:
                lines = f.readlines()
                print(f"  📄 {log_file}: {len(lines)} lines")
                # Show last few lines
                for line in lines[-5:]:
                    print(f"      {line.strip()}")
    else:
        print("❌ No comprehensive pipeline log files found")

def main():
    """Investigate paraphrase coverage issue."""
    print("🔍 Investigating Paraphrase Coverage Issue...")
    
    config = Config()
    
    # Check comprehensive pipeline output
    check_comprehensive_pipeline_output()
    
    # Analyze augmented datasets
    print("\n" + "="*60)
    print("📋 AUGMENTED DATASET ANALYSIS")
    print("="*60)
    
    train_stats = analyze_paraphrase_coverage(config.data.train_augmented_json_path, "Augmented Train")
    val_seen_stats = analyze_paraphrase_coverage(config.data.val_seen_augmented_json_path, "Augmented Val Seen")
    val_unseen_stats = analyze_paraphrase_coverage(config.data.val_unseen_augmented_json_path, "Augmented Val Unseen")
    
    # Summary
    print("\n" + "="*60)
    print("📊 COVERAGE SUMMARY")
    print("="*60)
    
    if train_stats:
        print(f"\nTRAIN DATASET:")
        print(f"  📈 Coverage: {train_stats['coverage_rate']:.2f}%")
        print(f"  📈 Expected: 100% (from comprehensive pipeline)")
        print(f"  📈 Missing: {train_stats['valid_turns'] - train_stats['turns_with_paraphrases']} turns")
        print(f"  📈 Episodes without paraphrases: {len(train_stats['episodes_without_paraphrases'])}")
    
    # Check if this matches the comprehensive pipeline output
    print(f"\n🎯 EXPECTED FROM COMPREHENSIVE PIPELINE:")
    print(f"  📈 Total episodes: 3,883")
    print(f"  📈 Dialog turns with answers: 4,326")
    print(f"  📈 Paraphrases generated: 12,978 (8,652 positives + 4,326 negatives)")
    print(f"  📈 Coverage: 100%")
    
    print(f"\n🔍 POSSIBLE ISSUES:")
    print(f"  1. Comprehensive pipeline didn't complete successfully")
    print(f"  2. Data was filtered incorrectly during processing")
    print(f"  3. Some episodes failed paraphrase generation")
    print(f"  4. File corruption or incomplete processing")

if __name__ == "__main__":
    main() 