#!/usr/bin/env python3
"""
Test script for hard negative mining functionality.
"""

import os
import sys
import pickle
import torch
import numpy as np
from transformers import T5Tokenizer

# Add the parent directory to the path to access config and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.add_hard_negatives import HardNegativeMiner, load_dataset, save_dataset

def test_hard_negative_mining():
    """Test the negative mining functionality with a small subset."""
    
    print("🧪 Testing Negative Mining...")
    
    # Load configuration
    config = Config()
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    # Test with a small subset of the dataset
    print("📊 Loading small subset of train dataset...")
    
    # Load a small subset for testing
    try:
        from data.dataset import AnsweringDataset
        dataset = AnsweringDataset.load_train_chunks(config.data.train_processed_path_dir)
        
        # Take only first 100 samples for testing
        test_dataset = {k: v for k, v in list(dataset.items())[:100]}
        print(f"✅ Loaded {len(test_dataset)} samples for testing")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Initialize hard negative miner
    image_dir = "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images"
    miner = HardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        image_dir=image_dir,
        k_nn=10,  # Smaller K for testing
        cosine_threshold=0.3,
        use_diverse_negatives=True,
        diverse_ratio=0.5,
        min_answer_length=20
    )
    
    # Test mining negatives (hard + diverse)
    print("⛏️ Testing negative mining...")
    negatives = miner.mine_hard_negatives(test_dataset, max_samples=50, debug_mode=True)
    
    print(f"✅ Mined {len(negatives)} negatives total")
    
    # Count types
    hard_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'hard')
    diverse_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'diverse')
    print(f"📊 Hard negatives: {hard_count}, Diverse negatives: {diverse_count}")
    
    # Test adding negatives to dataset
    print("➕ Testing negative addition...")
    updated_dataset = miner.add_hard_negatives_to_dataset(test_dataset, negatives)
    
    # Verify negatives were added
    negative_count = 0
    for idx, item in updated_dataset.items():
        if 'negative_text_2' in item:
            negative_count += 1
    
    print(f"✅ Added negatives to {negative_count} samples")
    
    # Test a few samples to verify structure
    print("🔍 Verifying sample structure...")
    for idx, item in list(updated_dataset.items())[:3]:
        if 'negative_text_2' in item:
            print(f"  Sample {idx}:")
            print(f"    Original answer: {item.get('answer', '')[:50]}...")
            print(f"    Negative_2: {item['negative_text_2'][:50]}...")
            print(f"    Negative type: {item.get('negative_type_2', 'unknown')}")
            print(f"    Has tokenized negative_2: {'tokenized_negative_2' in item}")
            
            # Check validation metadata for negative_2
            if 'contrastive_data' in item and 'validation_metadata_2' in item['contrastive_data']:
                metadata = item['contrastive_data']['validation_metadata_2']
                print(f"    Validation metadata for negative_2:")
                print(f"      Type: {metadata.get('negative_type_2', 'unknown')}")
                if 'text_similarity' in metadata:
                    print(f"      Text similarity: {metadata['text_similarity']:.3f}")
                if 'visual_similarity' in metadata:
                    print(f"      Visual similarity: {metadata['visual_similarity']:.3f}")
                if 'anchor_cluster' in metadata:
                    print(f"      Clusters: {metadata['anchor_cluster']} -> {metadata['negative_cluster']}")
            break
    
    # Test answer quality filtering
    print("🔍 Testing answer quality filtering...")
    test_answers = [
        "turn left",  # Should be filtered
        "go straight ahead",  # Should be filtered
        "i don't know",  # Should be filtered
        "maybe turn right",  # Should be filtered
        "You should turn left at the intersection and continue for about 100 meters",  # Should pass
        "Navigate to the building on your right and follow the path around it",  # Should pass
        "Take the second right turn and proceed towards the landmark",  # Should pass
    ]
    
    for answer in test_answers:
        is_good = miner.is_good_answer(answer)
        status = "✅ PASS" if is_good else "❌ FILTERED"
        print(f"  {status}: '{answer[:50]}{'...' if len(answer) > 50 else ''}'")
    
    print("✅ Negative mining test completed successfully!")
    return True

if __name__ == '__main__':
    success = test_hard_negative_mining()
    if success:
        print("🎉 All tests passed!")
    else:
        print("❌ Tests failed!")
        sys.exit(1) 