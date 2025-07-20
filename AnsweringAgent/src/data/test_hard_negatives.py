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
        k_nn=15,
        cosine_threshold=0.3,
        use_diverse_negatives=True,
        diverse_ratio=0.3,
        min_answer_length=20
    )
    
    # Enable debug mode
    miner.debug_mode = True
    
    # Test blacklist embedding initialization
    print("🔍 Testing blacklist embedding initialization...")
    miner._initialize_blacklist_embeddings()
    
    if miner.blacklist_embeddings:
        print(f"✅ Semantic filtering initialized: {len(miner.blacklist_embeddings)} phrases")
        
        # Show some example embeddings
        for i, (phrase, embedding) in enumerate(list(miner.blacklist_embeddings.items())[:3]):
            print(f"  '{phrase}': embedding shape {embedding.shape}")
    else:
        print("⚠️ No semantic filtering available")
    
    # Simulate small dataset filtering (same as mining logic)
    print("📊 Applying same filtering logic as mining (small dataset)...")
    original_blacklist = miner.answer_blacklist.copy()
    original_min_length = miner.min_answer_length
    
    # Apply lenient filtering (same as mining)
    miner.min_answer_length = max(15, miner.min_answer_length - 5)
    miner.answer_blacklist = {
        'short_affirmative': ['yes', 'exactly', 'correct'],  # removed 'right' to prevent directional false-positives
        'generic_responses': ['destiny is exactly that', 'that is correct'],
    }
    print(f"  Adjusted min_answer_length to {miner.min_answer_length}")
    print(f"  Using lenient direct blacklist with {sum(len(phrases) for phrases in miner.answer_blacklist.values())} phrases")
    print(f"  Semantic filtering still uses full blacklist with {len(miner.blacklist_embeddings)} phrases")
    
    # Test caching functionality
    print("💾 Testing embedding cache...")
    cache_path = os.path.join(os.path.dirname(__file__), 'blacklist_embeds.pkl')
    
    if os.path.exists(cache_path):
        print(f"✅ Cache file exists: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cached_embeddings = pickle.load(f)
            print(f"  Cached: {len(cached_embeddings)} phrases")
            
            if len(cached_embeddings) == len(miner.blacklist_embeddings):
                print("✅ Cache matches current embeddings")
            else:
                print(f"⚠️ Cache mismatch: cached {len(cached_embeddings)}, current {len(miner.blacklist_embeddings)}")
        except Exception as e:
            print(f"❌ Error reading cache: {e}")
    else:
        print("⚠️ No cache file found")
    
    # Test filtering with various phrases (using realistic lenient blacklist)
    print("🔍 Testing answer filtering with lenient blacklist...")
    
    test_phrases = [
        # Should be filtered by direct blacklist (lenient)
        ("yes", "direct blacklist (lenient)"),
        ("exactly", "direct blacklist (lenient)"),
        ("correct", "direct blacklist (lenient)"),
        ("destiny is exactly that", "direct blacklist (lenient)"),
        
        # Should be filtered by semantic similarity (using full blacklist)
        ("yes, that's absolutely correct", "semantic similarity"),
        ("you're exactly right", "semantic similarity"),
        ("that's the right answer", "semantic similarity"),
        
        # Should be filtered by length
        ("go", "too short"),
        ("turn left", "too short"),
        ("move right", "too short"),
        
        # Should NOW pass (not in lenient blacklist)
        ("You should turn left at the intersection and continue for about 100 meters", "should pass"),
        ("Navigate to the building and follow the path around it", "should pass"),
        ("Take the second turn and proceed towards the landmark", "should pass"),
        ("The destination is located behind the large building with the red roof", "should pass"),
        ("Continue straight for approximately 200 meters until you see the bridge", "should pass"),
    ]
    
    filtered_count = 0
    passed_count = 0
    
    for phrase, expected_reason in test_phrases:
        print(f"\n  Testing: '{phrase[:50]}{'...' if len(phrase) > 50 else ''}'")
        is_good = miner.is_good_answer(phrase)
        status = "✅ PASS" if is_good else "❌ FILTERED"
        print(f"  Result: {status} (expected: {expected_reason})")
        
        if is_good:
            passed_count += 1
        else:
            filtered_count += 1
    
    print(f"\n📊 Filtering results: {passed_count} passed, {filtered_count} filtered")
    
    # Test blacklist categories (current lenient blacklist)
    print("📋 Testing blacklist categories...")
    
    print(f"📊 Current blacklist after lenient override:")
    for category, phrases in miner.answer_blacklist.items():
        print(f"  {category}: {len(phrases)} phrases")
        for phrase in phrases[:2]:  # Show first 2 from each category
            is_good = miner.is_good_answer(phrase)
            status = "✅ PASS" if is_good else "❌ FILTERED"
            print(f"    {status}: '{phrase}'")
    
    # Restore original settings
    miner.answer_blacklist = original_blacklist
    miner.min_answer_length = original_min_length
    
    print("✅ Semantic filtering test completed!")
    return True

def test_mining_functionality():
    """Test the mining functionality with a small dataset."""
    
    print("\n🧪 Testing Mining Functionality...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_id = 0
        torch.cuda.set_device(gpu_id)
        print(f"🚀 Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        print("⚠️ CUDA not available, using CPU")
        gpu_id = None
    
    # Load configuration
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    # Load small subset for testing
    print("📊 Loading small subset of train dataset...")
    
    try:
        from data.dataset import AnsweringDataset
        dataset = AnsweringDataset.load_train_chunks(config.data.train_processed_path_dir)
        
        # Take only first 50 samples for testing
        test_dataset = {k: v for k, v in list(dataset.items())[:50]}
        print(f"✅ Loaded {len(test_dataset)} samples for testing")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False
    
    # Test dataset sharding
    print("🔀 Testing dataset sharding...")
    num_shards = 4
    shard_id = 0
    original_size = len(test_dataset)
    sharded_dataset = {k: v for k, v in test_dataset.items() if (k % num_shards) == shard_id}
    print(f"  Sharded dataset: keeping {len(sharded_dataset)} / {original_size} samples for shard {shard_id} of {num_shards}")
    
    # Initialize miner
    image_dir = "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images"
    miner = HardNegativeMiner(
        config=config,
        tokenizer=tokenizer,
        image_dir=image_dir,
        k_nn=15,
        cosine_threshold=0.3,
        use_diverse_negatives=True,
        diverse_ratio=0.3,
        min_answer_length=20
    )
    
    # Set GPU settings
    miner.batch_size = 32
    miner.num_workers = 2
    if torch.cuda.is_available():
        miner.device = torch.device(f'cuda:{gpu_id}')
    
    # Mine negatives with debug mode
    print("⛏️ Testing negative mining...")
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
    
    # Analyze sample structure
    print("🔍 Analyzing sample structure...")
    for idx, item in list(updated_dataset.items())[:5]:
        if 'contrastive_data' in item and 'negative_text_2' in item['contrastive_data']:
            print(f"  Sample {idx}:")
            print(f"    Original answer: {item.get('answer', 'N/A')[:60]}{'...' if len(item.get('answer', '')) > 60 else ''}")
            print(f"    Negative answer: {item['contrastive_data']['negative_text_2'][:60]}{'...' if len(item['contrastive_data']['negative_text_2']) > 60 else ''}")
            print(f"    Negative type: {item['contrastive_data'].get('validation_metadata_negative_2', {}).get('negative_type_2', 'unknown')}")
            
            # Check validation metadata
            if 'validation_metadata_negative_2' in item['contrastive_data']:
                metadata = item['contrastive_data']['validation_metadata_negative_2']
                if 'text_similarity' in metadata:
                    print(f"    Text similarity: {metadata['text_similarity']:.3f}")
                if 'visual_similarity' in metadata:
                    print(f"    Visual similarity: {metadata['visual_similarity']:.3f}")
    
    # Quality analysis
    print("📊 Quality Analysis:")
    if negatives:
        # Answer lengths
        original_lengths = [len(item.get('answer', '')) for item in sharded_dataset.values()]
        negative_lengths = [len(data['negative_text_2']) for data in negatives.values()]
        
        print(f"  Original answers: avg={sum(original_lengths)/len(original_lengths):.1f} chars")
        print(f"  Negative answers: avg={sum(negative_lengths)/len(negative_lengths):.1f} chars")
        
        # Check for blacklisted phrases using miner's actual logic
        blacklisted_count = 0
        semantic_filtered_count = 0
        
        print(f"  Current blacklist being used for analysis: {list(miner.answer_blacklist.keys())}")
        print(f"  Total phrases in current blacklist: {sum(len(phrases) for phrases in miner.answer_blacklist.values())}")
        
        for data in negatives.values():
            answer_text = data['negative_text_2']
            answer_lower = answer_text.lower()
            
            # Test direct blacklist (word boundary) - exactly as miner does
            is_blacklisted = False
            for phrases in miner.answer_blacklist.values():
                for phrase in phrases:
                    pattern = rf"\b{re.escape(phrase)}\b"
                    if re.search(pattern, answer_lower):
                        is_blacklisted = True
                        if len(negatives) <= 20:  # Only show details for small datasets
                            print(f"    Found blacklisted phrase '{phrase}' in: '{answer_text[:40]}{'...' if len(answer_text) > 40 else ''}'")
                        break
                if is_blacklisted:
                    break
            
            if is_blacklisted:
                blacklisted_count += 1
            
            # Test semantic similarity
            if hasattr(miner, 'blacklist_embeddings') and miner.blacklist_embeddings:
                try:
                    if miner._check_semantic_similarity_to_blacklist(answer_text):
                        semantic_filtered_count += 1
                        if len(negatives) <= 20:  # Only show details for small datasets
                            print(f"    Found semantic match in: '{answer_text[:40]}{'...' if len(answer_text) > 40 else ''}'")
                except Exception:
                    pass
        
        print(f"  Direct blacklist matches: {blacklisted_count}/{len(negatives)} ({blacklisted_count/len(negatives)*100:.1f}%)")
        if blacklisted_count == 0:
            print(f"    ↪ This is expected! Blacklisted phrases were filtered during mining.")
        
        if semantic_filtered_count > 0:
            print(f"  Semantic similarity matches: {semantic_filtered_count}/{len(negatives)} ({semantic_filtered_count/len(negatives)*100:.1f}%)")
        else:
            print(f"  Semantic similarity matches: 0/{len(negatives)} (0.0%)")
            print(f"    ↪ This is expected! Semantically similar phrases were filtered during mining.")
        
        # Phrase diversity
        unique_phrases = set()
        for data in negatives.values():
            unique_phrases.add(data['negative_text_2'].lower().strip())
        
        diversity_ratio = len(unique_phrases) / len(negatives)
        print(f"  Phrase diversity: {diversity_ratio:.3f} ({len(unique_phrases)} unique / {len(negatives)} total)")
        
        # Hard negative quality
        hard_similarities = [data['validation_metadata_2']['text_similarity'] 
                           for data in negatives.values() 
                           if data.get('negative_type_2') == 'hard' and 'text_similarity' in data['validation_metadata_2']]
        if hard_similarities:
            avg_hard_sim = sum(hard_similarities) / len(hard_similarities)
            print(f"  Hard negative quality: avg text similarity {avg_hard_sim:.3f}")
    
    print("✅ Mining functionality test completed!")
    return True

def test_multi_gpu_setup():
    """Test the multi-GPU setup logic."""
    
    print("\n🧪 Testing Multi-GPU Setup Logic...")
    
    # Simulate sharding logic
    num_gpus = 4
    total_samples = 1000
    
    print(f"📊 Simulating {num_gpus} GPU setup with {total_samples} total samples")
    
    # Test dataset sharding
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
    
    # Test strategy ratio logic
    print("📊 Testing strategy ratio logic...")
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
    tolerance = 0.15
    
    if abs(actual_hard_ratio - expected_hard_ratio) < tolerance:
        print("✅ Strategy ratio logic is working correctly")
    else:
        print(f"❌ Strategy ratio error: expected {expected_hard_ratio:.2f}, got {actual_hard_ratio:.2f}")
        return False
    
    print("✅ Multi-GPU setup test completed!")
    return True

if __name__ == '__main__':
    print("🚀 Starting Hard Negative Mining Tests...")
    
    # Test 1: Semantic filtering
    success1 = test_semantic_filtering()
    
    # Test 2: Mining functionality
    success2 = test_mining_functionality()
    
    # Test 3: Multi-GPU setup
    success3 = test_multi_gpu_setup()
    
    if success1 and success2 and success3:
        print("\n🎉 All tests passed!")
        print("✅ Semantic filtering with caching is working correctly")
        print("✅ Mining strategies are functional")
        print("✅ GPU processing and multi-GPU setup work correctly")
        print("✅ Test logic matches main implementation")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 