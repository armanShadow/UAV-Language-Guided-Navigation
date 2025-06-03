#!/usr/bin/env python3
"""
Focused AVDN Dataset Pattern Analysis
Extract core patterns from 300 random samples for Strategy 1 expansion
"""

import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

def load_300_random_samples():
    """Load 300 random samples from the AVDN dataset."""
    dataset_paths = [
        "processed_data/train_data.json",
        "../processed_data/train_data.json", 
        "../../processed_data/train_data.json"
    ]
    
    for path in dataset_paths:
        if Path(path).exists():
            print(f"📂 Loading dataset from: {path}")
            with open(path, 'r') as f:
                episodes = json.load(f)
            
            # Extract all answers
            all_samples = []
            for episode in episodes:
                dialogs = episode.get('dialogs', [])
                for dialog in dialogs:
                    if dialog and dialog.get('answer'):
                        answer = dialog['answer'].strip()
                        if answer and len(answer.split()) >= 3:
                            all_samples.append(answer)
            
            # Get 300 random samples
            random.seed(42)  # Consistent results
            if len(all_samples) >= 300:
                random_samples = random.sample(all_samples, 300)
                print(f"📊 Selected 300 random samples from {len(all_samples)} total")
                return random_samples
            else:
                return all_samples
    
    print("❌ Could not find dataset.")
    return []

def analyze_core_patterns(samples):
    """Analyze the most frequent and important patterns in 300 samples."""
    print("\n🔍 ANALYZING CORE PATTERNS (300 samples)")
    print("="*60)
    
    # Focus on the most important patterns based on what we saw
    core_patterns = {
        # Clock patterns - very important for UAV
        r"(\d+)\s*o'?clock": "X o'clock",
        r"(\d+):(\d+)": "X:Y time format",
        r"turn\s+to\s+your\s+(\d+)\s*o'?clock": "turn to your X o'clock",
        r"move\s+towards\s+your\s+(\d+)\s*o'?clock": "move towards your X o'clock",
        r"turn\s+on\s+(\d+)\s*o'?clock": "turn on X o'clock",
        
        # Movement patterns
        r"head\s+(\w+)": "head X",
        r"turn\s+(\w+)": "turn X",
        r"move\s+(\w+)": "move X", 
        r"go\s+(\w+)": "go X",
        
        # Destination patterns
        r"destination\s+is\s+(.+)": "destination is X",
        r"your\s+destination\s+is\s+(.+)": "your destination is X",
        r"that'?s\s+your\s+destination": "that's your destination",
        
        # Positional descriptions
        r"you\s+will\s+see\s+(.+)": "you will see X",
        r"looks?\s+like\s+(.+)": "looks like X",
        r"it\s+is\s+(.+)": "it is X",
        
        # Spatial relations
        r"pass\s+(.+)": "pass X",
        r"cross\s+(.+)": "cross X",
        r"over\s+(.+)": "over X",
        r"next\s+to\s+(.+)": "next to X",
        r"in\s+front\s+of\s+(.+)": "in front of X",
        
        # Building descriptions
        r"(\w+)\s+building": "X building",
        r"building\s+with\s+(.+)": "building with X",
        r"structure\s+(.+)": "structure X",
        
        # Distance/proximity
        r"you\s+are\s+(near|close)": "you are near/close",
        r"very\s+close": "very close",
        r"nearly\s+there": "nearly there",
    }
    
    pattern_counts = defaultdict(list)
    
    for sample in samples:
        sample_lower = sample.lower()
        for pattern, description in core_patterns.items():
            matches = re.findall(pattern, sample_lower)
            if matches:
                pattern_counts[description].append((sample, matches))
    
    print("\n📊 CORE PATTERN FREQUENCY:")
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: len(x[1]), reverse=True)
    
    for description, matches in sorted_patterns:
        if len(matches) >= 2:  # Only show patterns that appear at least twice
            print(f"  {description}: {len(matches)} matches")
            # Show top 2 examples
            for i, (sample, match_groups) in enumerate(matches[:2]):
                if i == 0:
                    print(f"    Examples:")
                print(f"      → \"{sample[:70]}{'...' if len(sample) > 70 else ''}\"")

def analyze_key_vocabulary(samples):
    """Analyze key vocabulary for synonyms."""
    print("\n🔍 ANALYZING KEY VOCABULARY")
    print("="*60)
    
    # Count key spatial terms
    key_terms = {
        'movement_verbs': ['turn', 'move', 'go', 'head', 'fly', 'proceed'],
        'direction_words': ['north', 'south', 'east', 'west', 'left', 'right', 'forward', 'straight'],
        'landmark_types': ['building', 'structure', 'house', 'parking', 'road', 'street'],
        'spatial_relations': ['next', 'front', 'behind', 'over', 'under', 'near', 'close'],
        'descriptors': ['large', 'small', 'long', 'rectangular', 'gray', 'grey']
    }
    
    term_counts = defaultdict(Counter)
    
    for sample in samples:
        words = sample.lower().split()
        for category, terms in key_terms.items():
            for word in words:
                if word in terms:
                    term_counts[category][word] += 1
    
    for category, counts in term_counts.items():
        if counts:
            print(f"\n  {category.upper()}:")
            for term, count in counts.most_common(10):
                print(f"    {term}: {count}")

def find_missing_patterns(samples):
    """Find patterns we might have missed in Strategy 1."""
    print("\n🔍 POTENTIAL MISSING PATTERNS")
    print("="*60)
    
    # Look for common sentence starters and structures we might have missed
    sentence_starters = Counter()
    
    for sample in samples:
        # Get first few words
        words = sample.lower().split()
        if len(words) >= 2:
            starter = ' '.join(words[:2])
            sentence_starters[starter] += 1
        if len(words) >= 3:
            starter = ' '.join(words[:3])
            sentence_starters[starter] += 1
    
    print("\n📊 COMMON SENTENCE STARTERS:")
    for starter, count in sentence_starters.most_common(15):
        if count >= 3:  # Only show if appears 3+ times
            print(f"  '{starter}': {count} times")

def generate_strategy1_recommendations(samples):
    """Generate specific recommendations for Strategy 1 based on analysis."""
    print("\n🚀 STRATEGY 1 RECOMMENDATIONS")
    print("="*60)
    
    print("📋 HIGH-PRIORITY PATTERNS TO ADD:")
    
    # Analyze most common patterns not yet covered
    high_priority = []
    
    # Check for clock direction variations
    clock_samples = [s for s in samples if re.search(r'\d+\s*o\'?clock', s.lower())]
    if clock_samples:
        high_priority.append(f"Clock directions: {len(clock_samples)} samples")
    
    # Check for informal language
    informal_samples = [s for s in samples if any(word in s.lower() for word in ['you are', 'you will', 'you should'])]
    if informal_samples:
        high_priority.append(f"Informal language: {len(informal_samples)} samples")
    
    # Check for building descriptions  
    building_samples = [s for s in samples if 'building' in s.lower()]
    if building_samples:
        high_priority.append(f"Building descriptions: {len(building_samples)} samples")
    
    for priority in high_priority:
        print(f"  • {priority}")
    
    print("\n📋 SUGGESTED NEW PATTERNS:")
    suggested_patterns = [
        ("you will see X", "you'll observe X / you can spot X"),
        ("head X", "go X / move X"),
        ("turn on X o'clock", "rotate to X o'clock"),
        ("looks like X", "appears to be X / resembles X"), 
        ("your destination", "your target / your goal"),
        ("pass X", "go past X / move beyond X"),
        ("cross X", "go across X / traverse X"),
    ]
    
    for original, suggestion in suggested_patterns:
        print(f"  {original} → {suggestion}")

if __name__ == "__main__":
    print("🔍 FOCUSED AVDN PATTERN ANALYSIS (300 samples)")
    print("="*60)
    
    # Load 300 random samples
    samples = load_300_random_samples()
    if not samples:
        exit(1)
    
    # Run focused analyses
    analyze_core_patterns(samples)
    analyze_key_vocabulary(samples)
    find_missing_patterns(samples)
    generate_strategy1_recommendations(samples)
    
    print(f"\n{'='*60}")
    print("🎯 FOCUSED ANALYSIS COMPLETE")
    print(f"Based on {len(samples)} random samples")
    print('='*60) 