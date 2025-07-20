#!/usr/bin/env python3
"""
Quick test script to verify hard negative mining improvements.
"""

import os
import sys
import torch

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from transformers import T5Tokenizer
from data.add_hard_negatives import HardNegativeMiner

def test_phrase_diversity():
    """Test the improved phrase diversity logic."""
    print("🧪 Testing Phrase Diversity...")
    
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    miner = HardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        image_dir="dummy",
        k_nn=100,  # Increased from 15
        cosine_threshold=0.2,  # Lowered from 0.3
        use_diverse_negatives=True,
        diverse_ratio=0.3,
        min_answer_length=20
    )
    
    miner.debug_mode = True
    
    # Test phrases
    test_phrases = [
        "your goal is the big building right in front of you",  # Should be allowed once
        "your goal is the big building right in front of you",  # Should be rejected (duplicate)
        "Navigate to the large building with the red roof and continue straight",  # Should be allowed
        "Go south to black lot",  # Should be allowed once
        "Go south to black lot",  # Should be rejected (duplicate)
        "You should turn left at the intersection and continue for about 100 meters",  # Should be allowed
    ]
    
    print("📊 Testing phrase diversity:")
    for i, phrase in enumerate(test_phrases, 1):
        is_diverse = miner._is_phrase_diverse(phrase)
        if is_diverse:
            miner._track_phrase_usage(phrase)  # Track if accepted
        
        status = "✅ ACCEPTED" if is_diverse else "❌ REJECTED"
        print(f"  {i}. {status}: '{phrase[:50]}{'...' if len(phrase) > 50 else ''}'")
    
    print(f"\n📈 Final phrase usage: {dict(miner.used_phrases)}")

def test_clustering():
    """Test the improved clustering logic."""
    print("\n🧪 Testing Clustering...")
    
    # Simulate small dataset clustering
    import numpy as np
    
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    miner = HardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        image_dir="dummy",
        k_nn=100,  # Increased from 15
        cosine_threshold=0.2,  # Lowered from 0.3
        use_diverse_negatives=True,
        diverse_ratio=0.3,
        min_answer_length=20
    )
    
    # Simulate visual features for 13 samples
    np.random.seed(42)
    miner.visual_indices = list(range(13))
    miner.visual_features = {i: np.random.rand(192).astype(np.float32) for i in range(13)}
    
    # Create dummy dataset
    dummy_dataset = {}
    for i in range(13):
        dummy_dataset[i] = {
            'first_instruction': f'instruction_{i % 3}',  # 3 different instructions
            'answer': f'Answer for sample {i} with sufficient length to pass filtering',
            'dialog_context': f'Context for sample {i}',
        }
    
    # Test clustering
    miner.build_visual_clusters(dummy_dataset)
    
    if miner.cluster_labels is not None:
        unique_labels = np.unique(miner.cluster_labels)
        print(f"✅ Created {len(unique_labels)} clusters for 13 samples")
        print(f"📊 Cluster labels: {miner.cluster_labels}")
        
        # Test diverse negative finding
        result = miner.find_diverse_negative(0, dummy_dataset)
        if result:
            negative_idx, anchor_cluster, negative_cluster, visual_similarity = result
            print(f"✅ Found diverse negative: sample {negative_idx}")
            print(f"📊 Cluster transition: {anchor_cluster} → {negative_cluster}")
        else:
            print("❌ No diverse negative found")
    else:
        print("❌ Clustering failed")

def test_answer_similarity():
    """Test the improved answer-level similarity logic."""
    print("\n🧪 Testing Answer-Level Similarity...")
    
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    miner = HardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        image_dir="dummy",
        k_nn=100,
        cosine_threshold=0.2,
        use_diverse_negatives=True,
        diverse_ratio=0.3,
        min_answer_length=15  # Lenient for testing
    )
    
    # Test different answer pairs
    test_pairs = [
        # Different answers (should have low similarity)
        ("Turn left at the intersection and continue straight for 100 meters", 
         "Go south to the black parking lot near the trees"),
        
        # Similar answers (should have high similarity)
        ("Turn left at the intersection", 
         "Turn right at the intersection"),
        
        # Very different answers
        ("The destination is a large red building with windows",
         "Navigate to coordinates 45.2, -122.3 using GPS"),
    ]
    
    print("📊 Testing answer similarity pairs:")
    for i, (answer1, answer2) in enumerate(test_pairs, 1):
        features1 = miner.extract_text_features(answer1)
        features2 = miner.extract_text_features(answer2)
        similarity = np.dot(features1, features2)
        
        print(f"  {i}. Similarity: {similarity:.3f}")
        print(f"     Answer 1: '{answer1[:40]}{'...' if len(answer1) > 40 else ''}'")
        print(f"     Answer 2: '{answer2[:40]}{'...' if len(answer2) > 40 else ''}'")
        print()
    
    print("✅ Answer-level similarity test completed!")

if __name__ == '__main__':
    print("🚀 Running Quick Tests...")
    
    if torch.cuda.is_available():
        print(f"🔧 GPU available: {torch.cuda.get_device_name(0)}")
    
    test_phrase_diversity()
    test_clustering()
    test_answer_similarity()
    
    print("\n✅ Quick tests completed!") 