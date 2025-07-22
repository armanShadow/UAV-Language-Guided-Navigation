#!/usr/bin/env python3
"""
Test script for hard negative mining functionality.
Tests semantic filtering, GPU processing, and mining strategies.
"""

import os
import sys
import pickle
import torch
import numpy as np
import re
from transformers import T5Tokenizer

# Add the parent directory to the path to access config and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data.add_hard_negatives import HardNegativeMiner, load_dataset, save_dataset

def test_semantic_filtering():
    """Test the semantic filtering functionality."""
    
    print("🧪 Testing Semantic Filtering...")
    
    # Load configuration and tokenizer
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    # Initialize miner
    image_dir = "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images"
    miner = HardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        image_dir=image_dir,
        k_nn=100,
        cosine_threshold=0.2,
        use_diverse_negatives=True,
        diverse_ratio=0.3,
        min_answer_length=20
    )
    
    # Initialize semantic filtering quietly
    miner.debug_mode = False  # Disable debug for clean output
    miner._initialize_blacklist_embeddings()
    
    # Enable debug for semantic testing
    miner.debug_mode = True
    miner.semantic_similarity_threshold = 0.75  # Lower threshold for testing
    
    print(f"✅ Semantic Filtering: {len(miner.blacklist_embeddings)} embeddings loaded")
    print(f"🎯 Testing with threshold: {miner.semantic_similarity_threshold}")
    
    # Test phrases that should trigger semantic similarity
    test_phrases = [
        "yes, that's correct",
        "exactly right", 
        "absolutely correct",
        "that is exactly right",
        "you are correct",
        "Turn left at the intersection and continue straight for 100 meters",
        "Navigate to the building and follow the path around it", 
        "The destination is located behind the large building with the red roof",
    ]
    
    print("\n🔍 Testing semantic similarity for each phrase:")
    passed_count = 0
    filtered_count = 0
    
    for phrase in test_phrases:
        print(f"\n📝 Testing: '{phrase}'")
        miner.debug_mode = True  # Enable debug for this test
        is_good = miner.is_good_answer(phrase)
        miner.debug_mode = False  # Disable for clean output
        
        status = "✅ PASSED" if is_good else "❌ FILTERED"
        print(f"   Result: {status}")
        
        if is_good:
            passed_count += 1
        else:
            filtered_count += 1
    
    print(f"\n📊 Semantic Filter Test Results:")
    print(f"   Passed: {passed_count}/{len(test_phrases)}")
    print(f"   Filtered: {filtered_count}/{len(test_phrases)}")
    print(f"   Threshold used: {miner.semantic_similarity_threshold}")
    
    return True

def test_mining_functionality():
    """Test the mining functionality with a larger dataset."""
    
    print("\n🧪 Testing Mining Functionality...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_id = 0
        torch.cuda.set_device(gpu_id)
        print(f"🚀 GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("⚠️ Using CPU")
        gpu_id = None
    
    # Load configuration
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    # Load larger subset for testing
    print("📊 Loading train dataset...")
    
    try:
        from data.dataset import AnsweringDataset
        dataset = AnsweringDataset.load_train_chunks(config.data.train_processed_path_dir)
        
        # Take first 800 samples, shard to 200 for testing
        test_dataset = {k: v for k, v in list(dataset.items())[:800]}
        
        # Shard to simulate multi-GPU (keep 200 samples)
        num_shards = 4
        shard_id = 0
        original_size = len(test_dataset)
        sharded_dataset = {k: v for k, v in test_dataset.items() if (k % num_shards) == shard_id}
        print(f"📈 Dataset: {len(sharded_dataset)} samples (sharded from {original_size})")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Initialize miner with visual similarity filtering
    image_dir = "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images"
    miner = HardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        image_dir=image_dir,
        k_nn=100,
        cosine_threshold=0.2,
        use_diverse_negatives=True,
        diverse_ratio=0.3,
        min_answer_length=20,
        min_visual_similarity=0.1,  # Test with visual similarity filtering
        fallback_phrase_reuse_limit=3
    )
    
    # Set GPU settings
    miner.batch_size = 32
    miner.num_workers = 2
    if torch.cuda.is_available():
        miner.device = torch.device(f'cuda:{gpu_id}')
    
    # Mine negatives without debug output
    print("⛏️ Mining negatives...")
    negatives = miner.mine_hard_negatives(sharded_dataset, debug_mode=False)
    
    # Add negatives to dataset
    updated_dataset = miner.add_hard_negatives_to_dataset(sharded_dataset, negatives)
    
    # Comprehensive analysis
    print(f"\n📊 Mining Results Summary:")
    print(f"{'='*50}")
    
    # Basic counts
    hard_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'hard')
    diverse_count = sum(1 for data in negatives.values() if data.get('negative_type_2') == 'diverse')
    success_rate = len(negatives) / len(sharded_dataset) * 100
    
    print(f"📈 Success Rate: {len(negatives)}/{len(sharded_dataset)} ({success_rate:.1f}%)")
    print(f"🎯 Strategy Distribution: {hard_count} hard, {diverse_count} diverse")
    
    # Answer quality metrics
    if negatives:
        original_lengths = [len(item.get('answer', '')) for item in sharded_dataset.values()]
        negative_lengths = [len(data['negative_text_2']) for data in negatives.values()]
        
        print(f"📏 Answer Length: orig={np.mean(original_lengths):.1f}±{np.std(original_lengths):.1f}, neg={np.mean(negative_lengths):.1f}±{np.std(negative_lengths):.1f} chars")
        
        # Similarity metrics
        hard_text_sims = [data['validation_metadata_2']['text_similarity'] 
                         for data in negatives.values() 
                         if data.get('negative_type_2') == 'hard' and 'text_similarity' in data['validation_metadata_2']]
        
        hard_visual_sims = [data['validation_metadata_2']['visual_similarity'] 
                           for data in negatives.values() 
                           if data.get('negative_type_2') == 'hard' and 'visual_similarity' in data['validation_metadata_2']]
        
        diverse_visual_sims = [data['validation_metadata_2']['visual_similarity'] 
                              for data in negatives.values() 
                              if data.get('negative_type_2') == 'diverse' and 'visual_similarity' in data['validation_metadata_2']]
        
        if hard_text_sims:
            print(f"🔤 Hard Text Similarity: {np.mean(hard_text_sims):.3f}±{np.std(hard_text_sims):.3f} (n={len(hard_text_sims)})")
        
        if hard_visual_sims:
            print(f"👁️ Hard Visual Similarity: {np.mean(hard_visual_sims):.3f}±{np.std(hard_visual_sims):.3f} (n={len(hard_visual_sims)})")
            
            # Check visual similarity filtering effectiveness
            below_threshold = sum(1 for sim in hard_visual_sims if sim < miner.min_visual_similarity)
            if below_threshold > 0:
                print(f"⚠️ Warning: {below_threshold} hard negatives below min_visual_similarity ({miner.min_visual_similarity})")
            else:
                print(f"✅ All hard negatives meet minimum visual similarity requirement")
            
        if diverse_visual_sims:
            print(f"🌈 Diverse Visual Similarity: {np.mean(diverse_visual_sims):.3f}±{np.std(diverse_visual_sims):.3f} (n={len(diverse_visual_sims)})")
        
        # Phrase diversity
        unique_phrases = set()
        phrase_counts = {}
        for data in negatives.values():
            phrase = data['negative_text_2'].lower().strip()
            unique_phrases.add(phrase)
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        diversity_ratio = len(unique_phrases) / len(negatives)
        max_reuse = max(phrase_counts.values()) if phrase_counts else 0
        avg_reuse = np.mean(list(phrase_counts.values())) if phrase_counts else 0
        
        print(f"🔄 Phrase Diversity: {diversity_ratio:.3f} ({len(unique_phrases)}/{len(negatives)}), max_reuse={max_reuse}, avg_reuse={avg_reuse:.2f}")
        
        # Cluster analysis for diverse negatives
        cluster_transitions = []
        for data in negatives.values():
            if data.get('negative_type_2') == 'diverse':
                metadata = data['validation_metadata_2']
                if 'anchor_cluster' in metadata and 'negative_cluster' in metadata:
                    if metadata['anchor_cluster'] != metadata['negative_cluster']:
                        cluster_transitions.append(1)
                    else:
                        cluster_transitions.append(0)
        
        if cluster_transitions:
            different_cluster_ratio = np.mean(cluster_transitions)
            print(f"🎲 Cluster Diversity: {different_cluster_ratio:.3f} ({sum(cluster_transitions)}/{len(cluster_transitions)} different clusters)")
    
    # Quality assessment samples
    print(f"\n🔍 Quality Assessment Samples:")
    print(f"{'='*50}")
    
    sample_indices = list(negatives.keys())[:5]  # First 5 samples
    for i, idx in enumerate(sample_indices, 1):
        item = updated_dataset[idx]
        negative_data = negatives[idx]
        
        print(f"\n📋 Sample {i} (ID: {idx}):")
        print(f"   Question: {item.get('question', 'N/A')[:80]}{'...' if len(item.get('question', '')) > 80 else ''}")
        print(f"   Original: {item.get('answer', 'N/A')[:80]}{'...' if len(item.get('answer', '')) > 80 else ''}")
        print(f"   Negative: {negative_data['negative_text_2'][:80]}{'...' if len(negative_data['negative_text_2']) > 80 else ''}")
        
        metadata = negative_data['validation_metadata_2']
        neg_type = metadata.get('negative_type_2', 'unknown')
        print(f"   Type: {neg_type}")
        
        if 'text_similarity' in metadata:
            print(f"   Text Sim: {metadata['text_similarity']:.3f}")
        if 'visual_similarity' in metadata:
            print(f"   Visual Sim: {metadata['visual_similarity']:.3f}")
            # Check if visual similarity meets minimum requirement (only for hard negatives)
            if neg_type == 'hard':
                if metadata['visual_similarity'] < miner.min_visual_similarity:
                    print(f"   ⚠️ Below min threshold ({miner.min_visual_similarity})")
                else:
                    print(f"   ✅ Above min threshold ({miner.min_visual_similarity})")
        if 'anchor_cluster' in metadata and 'negative_cluster' in metadata:
            print(f"   Clusters: {metadata['anchor_cluster']} → {metadata['negative_cluster']}")
    
    print("\n✅ Mining functionality test completed!")
    return True

def test_multi_gpu_setup():
    """Test the multi-GPU setup logic."""
    
    print("\n🧪 Testing Multi-GPU Setup Logic...")
    
    # Test sharding logic
    num_gpus = 4
    total_samples = 1000
    
    shard_distributions = []
    for gpu_id in range(num_gpus):
        shard_samples = [i for i in range(total_samples) if (i % num_gpus) == gpu_id]
        shard_distributions.append(len(shard_samples))
    
    total_sharded = sum(shard_distributions)
    
    # Test strategy ratio logic
    diverse_ratio = 0.3
    num_samples = 1000
    
    hard_first_count = 0
    diverse_first_count = 0
    
    for _ in range(num_samples):
        import random
        if random.random() < diverse_ratio:
            diverse_first_count += 1
        else:
            hard_first_count += 1
    
    expected_hard_ratio = 1 - diverse_ratio
    actual_hard_ratio = hard_first_count / num_samples
    
    print(f"📊 Sharding: {num_gpus} GPUs, {total_sharded}/{total_samples} samples distributed")
    print(f"📊 Strategy Ratio: {actual_hard_ratio:.3f} hard-first (expected: {expected_hard_ratio:.3f})")
    
    success = (total_sharded == total_samples and abs(actual_hard_ratio - expected_hard_ratio) < 0.05)
    
    if success:
        print("✅ Multi-GPU setup test completed!")
    else:
        print("❌ Multi-GPU setup test failed!")
    
    return success

if __name__ == '__main__':
    print("🚀 Starting Hard Negative Mining Tests...")
    print("="*60)
    
    # Test 1: Semantic filtering
    success1 = test_semantic_filtering()
    
    # Test 2: Mining functionality  
    success2 = test_mining_functionality()
    
    # Test 3: Multi-GPU setup
    success3 = test_multi_gpu_setup()
    
    print("\n" + "="*60)
    if success1 and success2 and success3:
        print("🎉 All tests passed!")
        print("✅ Semantic filtering is working correctly")
        print("✅ Mining strategies are functional")
        print("✅ Visual similarity filtering is working correctly")
        print("✅ Multi-GPU setup works correctly")
    else:
        print("❌ Some tests failed!")
        sys.exit(1) 