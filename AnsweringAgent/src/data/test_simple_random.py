#!/usr/bin/env python3
"""
Simple Random Dataset Test for Strategy 1
Always uses random samples from the actual AVDN dataset
"""

import json
import random
import logging
from pathlib import Path
from contrastive_sample_generator import ContrastiveSampleGenerator

# Setup simple logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

def load_random_samples(n_samples=3):
    """Load random samples from the dataset, prioritizing ones with spatial elements."""
    
    # Try to find the dataset
    dataset_paths = [
        "processed_data/train_data.json",
        "../processed_data/train_data.json", 
        "../../processed_data/train_data.json"
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if Path(path).exists():
            dataset_path = path
            break
    
    if not dataset_path:
        print("❌ Could not find dataset. Please check paths.")
        return []
    
    print(f"📂 Loading dataset from: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        episodes = json.load(f)
    
    # Extract all dialog samples
    all_samples = []
    spatial_samples = []  # Samples with spatial elements
    
    for episode in episodes:
        dialogs = episode.get('dialogs', [])
        if not dialogs:
            continue
            
        for dialog in dialogs:
            if not dialog:  # Skip None dialogs
                continue
                
            answer = dialog.get('answer')
            if not answer:  # Skip None/empty answers
                continue
                
            answer = answer.strip()
            if answer and len(answer.split()) >= 5:  # Only meaningful answers
                sample = {
                    'answer': answer,
                    'episode_id': episode.get('episode_id', 'unknown'),
                    'question': dialog.get('question', ''),
                    'instruction': episode.get('instruction', '')
                }
                
                all_samples.append(sample)
                
                # Check if this sample has spatial elements we can work with
                answer_lower = answer.lower()
                has_spatial = (
                    'o\'clock' in answer_lower or
                    'oclock' in answer_lower or
                    ':' in answer and any(f'{i}:' in answer for i in range(1, 13)) or
                    any(word in answer_lower for word in ['turn', 'move', 'head', 'go']) or
                    any(word in answer_lower for word in ['building', 'destination', 'target']) or
                    any(word in answer_lower for word in ['north', 'south', 'east', 'west', 'left', 'right'])
                )
                
                if has_spatial:
                    spatial_samples.append(sample)
    
    print(f"📊 Found {len(all_samples)} total samples, {len(spatial_samples)} with spatial elements")
    
    # Prefer spatial samples, but include regular ones if needed
    if len(spatial_samples) >= n_samples:
        return random.sample(spatial_samples, n_samples)
    else:
        selected = spatial_samples.copy()
        remaining = n_samples - len(spatial_samples)
        non_spatial = [s for s in all_samples if s not in spatial_samples]
        if non_spatial and remaining > 0:
            selected.extend(random.sample(non_spatial, min(remaining, len(non_spatial))))
        return selected

def test_strategy1_on_samples():
    """Test Strategy 1 on random dataset samples."""
    
    print("🚀 SIMPLE RANDOM STRATEGY 1 TEST")
    print("="*60)
    
    # Load random samples
    samples = load_random_samples(n_samples=3)
    
    if not samples:
        print("❌ No samples loaded. Exiting.")
        return
    
    # Initialize generator
    generator = ContrastiveSampleGenerator()
    
    # Test each sample
    for i, sample in enumerate(samples, 1):
        positives = generator.generate_positive_examples(sample["answer"])
        print(f"Sample {i}: {sample['answer']}")
        print(positives)
        print("-"*60)

def test_specific_sample():
    """Test with a specific challenging sample."""
    
    print(f"\n{'='*60}")
    print("SPECIFIC TEST: Complex Navigation Sample")
    print('='*60)
    
    # A complex sample to test all strategies
    test_sample = "Turn right and move towards 3 o'clock. You'll see a large red building near the intersection. That's your destination."
    
    print(f"Original: {test_sample}")
    
    generator = ContrastiveSampleGenerator()
    
    # Test Strategy 1
    print(f"\n🔧 Strategy 1 Results:")
    strategy1_results = generator.generate_positive_examples(test_sample)
    print(f"  Total: {len(strategy1_results)} transformations")
    
    print(strategy1_results)

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run tests
    test_strategy1_on_samples()
    test_specific_sample()
    
    print(f"\n{'='*60}")
    print("🎯 TEST COMPLETE")
    print('='*60) 