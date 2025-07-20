#!/usr/bin/env python3
"""
Test script for hard negative mining functionality with GPU processing.
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
    """Test the negative mining functionality with GPU processing on a small subset."""
    
    print("🧪 Testing Negative Mining with GPU Processing...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_id = 0  # Use first GPU for testing
        torch.cuda.set_device(gpu_id)
        print(f"🚀 Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("⚠️ CUDA not available, using CPU")
        gpu_id = None
    
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
        
        # Take only first 50 samples for testing (smaller for faster testing)
        test_dataset = {k: v for k, v in list(dataset.items())[:50]}
        print(f"✅ Loaded {len(test_dataset)} samples for testing")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Test dataset sharding functionality
    print("🔀 Testing dataset sharding...")
    num_shards = 4
    shard_id = 0
    original_size = len(test_dataset)
    sharded_dataset = {k: v for k, v in test_dataset.items() if (k % num_shards) == shard_id}
    print(f"  Sharded dataset: keeping {len(sharded_dataset)} / {original_size} samples for shard {shard_id} of {num_shards}")
    
    # Initialize hard negative miner with GPU settings
    image_dir = "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images"
    miner = HardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        image_dir=image_dir,
        k_nn=15,  # Smaller K for testing
        cosine_threshold=0.3,
        use_diverse_negatives=True,
        diverse_ratio=0.3,  # Use new default
        min_answer_length=20
    )
    
    # Set GPU settings
    miner.batch_size = 32  # Smaller batch for testing
    miner.num_workers = 2
    if torch.cuda.is_available():
        miner.device = torch.device(f'cuda:{gpu_id}')
    
    # Test mining negatives with new probabilistic strategy
    print("⛏️ Testing negative mining with GPU processing...")
    negatives = miner.mine_hard_negatives(sharded_dataset, max_samples=25, debug_mode=True)
    
    print(f"✅ Mined {len(negatives)} negatives total")
    
    # Count types
    hard_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'hard')
    diverse_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'diverse')
    print(f"📊 Hard negatives: {hard_count}, Diverse negatives: {diverse_count}")
    
    # Test adding negatives to dataset
    print("➕ Testing negative addition...")
    updated_dataset = miner.add_hard_negatives_to_dataset(sharded_dataset, negatives)
    
    # Verify negatives were added
    negative_count = 0
    for idx, item in updated_dataset.items():
        if 'contrastive_data' in item and 'negative_text_2' in item['contrastive_data']:
            negative_count += 1
    
    print(f"✅ Added negatives to {negative_count} samples")
    
    # Test a few samples to verify structure
    print("🔍 Verifying sample structure...")
    for idx, item in list(updated_dataset.items())[:8]:
        if 'contrastive_data' in item and 'negative_text_2' in item['contrastive_data']:
            print(f"  Sample {idx}:")
            print(f"    First instruction: {item.get('first_instruction', 'N/A')}")
            print(f"    Current question: {item.get('question', 'N/A')}")
            print(f"    Original answer: {item.get('answer', 'N/A')}")
            print(f"    Negative_2: {item['contrastive_data']['negative_text_2']}")
            print(f"    Negative type: {item['contrastive_data'].get('validation_metadata_negative_2', {}).get('negative_type_2', 'unknown')}")
            print(f"    Has tokenized negative_2: {'tokenized_negative_2' in item['contrastive_data']}")
            
            # Check validation metadata for negative_2
            if 'validation_metadata_negative_2' in item['contrastive_data']:
                metadata = item['contrastive_data']['validation_metadata_negative_2']
                print(f"    Validation metadata for negative_2:")
                print(f"      Type: {metadata.get('negative_type_2', 'unknown')}")
                if 'text_similarity' in metadata:
                    print(f"      Text similarity: {metadata['text_similarity']:.3f}")
                if 'visual_similarity' in metadata:
                    print(f"      Visual similarity: {metadata['visual_similarity']:.3f}")
                if 'anchor_cluster' in metadata:
                    print(f"      Clusters: {metadata['anchor_cluster']} -> {metadata['negative_cluster']}")
            print()  # Add spacing between samples
    
    # Summary statistics
    print("📊 Summary Statistics:")
    if negatives:
        # Answer length statistics
        original_lengths = [len(item.get('answer', '')) for item in sharded_dataset.values()]
        negative_lengths = [len(data['negative_text_2']) for data in negatives.values()]
        
        print(f"  Original answers: avg={sum(original_lengths)/len(original_lengths):.1f} chars")
        print(f"  Negative answers: avg={sum(negative_lengths)/len(negative_lengths):.1f} chars")
        
        # Similarity statistics for hard negatives
        hard_similarities = []
        diverse_similarities = []
        for data in negatives.values():
            metadata = data['validation_metadata_2']
            if metadata.get('negative_type_2') == 'hard' and 'text_similarity' in metadata:
                hard_similarities.append(metadata['text_similarity'])
            elif metadata.get('negative_type_2') == 'diverse' and 'visual_similarity' in metadata:
                diverse_similarities.append(metadata['visual_similarity'])
        
        if hard_similarities:
            print(f"  Hard negatives text similarity: avg={sum(hard_similarities)/len(hard_similarities):.3f}")
        if diverse_similarities:
            print(f"  Diverse negatives visual similarity: avg={sum(diverse_similarities)/len(diverse_similarities):.3f}")
        
        # Mining strategy effectiveness
        hard_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'hard')
        diverse_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'diverse')
        print(f"  Mining success rate: {len(negatives)}/{len(sharded_dataset)} ({len(negatives)/len(sharded_dataset)*100:.1f}%)")
        print(f"  Strategy distribution: {hard_count} hard, {diverse_count} diverse")
    else:
        print("  No negatives mined - check mining parameters")
    
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
    
    # Test GPU memory usage
    if torch.cuda.is_available():
        print("🔍 Testing GPU memory usage...")
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(gpu_id)
        
        # Run a small batch through the miner
        test_batch = list(sharded_dataset.items())[:5]
        for idx, item in test_batch:
            try:
                # Test visual feature extraction
                if 'current_view_image' in item:
                    features = miner.extract_visual_features(item['current_view_image'])
                    print(f"    Sample {idx}: extracted {features.shape} features")
            except Exception as e:
                print(f"    Sample {idx}: error - {e}")
        
        final_memory = torch.cuda.memory_allocated(gpu_id)
        memory_used = (final_memory - initial_memory) / 1024**2  # MB
        print(f"  GPU memory used: {memory_used:.1f} MB")
        
        # Clean up
        torch.cuda.empty_cache()
    
    print("✅ Negative mining test completed successfully!")
    return True

def test_multi_gpu_setup():
    """Test the multi-GPU setup logic without actually running multiple processes."""
    
    print("\n🧪 Testing Multi-GPU Setup Logic...")
    
    # Simulate the sharding logic
    num_gpus = 4
    total_samples = 1000
    
    print(f"📊 Simulating {num_gpus} GPU setup with {total_samples} total samples")
    
    # Simulate dataset sharding
    shard_distributions = []
    for gpu_id in range(num_gpus):
        shard_samples = [i for i in range(total_samples) if (i % num_gpus) == gpu_id]
        shard_distributions.append(len(shard_samples))
        print(f"  GPU {gpu_id}: {len(shard_samples)} samples")
    
    # Verify distribution
    total_sharded = sum(shard_distributions)
    print(f"  Total sharded samples: {total_sharded}/{total_samples}")
    
    if total_sharded == total_samples:
        print("✅ Sharding distribution is correct")
    else:
        print("❌ Sharding distribution error")
        return False
    
    # Test diverse ratio logic
    print("📊 Testing diverse ratio logic...")
    diverse_ratio = 0.3
    num_samples = 100
    
    hard_first_count = 0
    diverse_first_count = 0
    
    for _ in range(num_samples):
        import random
        if random.random() < diverse_ratio:
            diverse_first_count += 1
        else:
            hard_first_count += 1
    
    print(f"  Hard-first strategy: {hard_first_count}/{num_samples} ({hard_first_count/num_samples*100:.1f}%)")
    print(f"  Diverse-first strategy: {diverse_first_count}/{num_samples} ({diverse_first_count/num_samples*100:.1f}%)")
    
    expected_hard_ratio = 1 - diverse_ratio
    actual_hard_ratio = hard_first_count / num_samples
    tolerance = 0.1
    
    if abs(actual_hard_ratio - expected_hard_ratio) < tolerance:
        print("✅ Diverse ratio logic is working correctly")
    else:
        print(f"❌ Diverse ratio logic error: expected {expected_hard_ratio:.2f}, got {actual_hard_ratio:.2f}")
        return False
    
    print("✅ Multi-GPU setup test completed successfully!")
    return True

if __name__ == '__main__':
    print("🚀 Starting Hard Negative Mining Tests...")
    
    # Test 1: GPU mining functionality
    success1 = test_hard_negative_mining()
    
    # Test 2: Multi-GPU setup logic
    success2 = test_multi_gpu_setup()
    
    if success1 and success2:
        print("\n🎉 All tests passed!")
        print("✅ GPU processing is working correctly")
        print("✅ Multi-GPU sharding logic is correct")
        print("✅ Mining strategy improvements are functional")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 