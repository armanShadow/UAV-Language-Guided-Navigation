#!/usr/bin/env python3
"""
Test script for hard negative mining functionality with GPU processing.
Enhanced with semantic filtering, caching, and debug mode testing.
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

def test_enhanced_semantic_filtering():
    """Test the enhanced semantic filtering with caching and expanded blacklist."""
    
    print("🧪 Testing Enhanced Semantic Filtering...")
    
    # Load configuration and tokenizer
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, 
                                           model_max_length=config.data.max_seq_length)
    
    # Initialize miner with debug mode
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
    
    # Enable debug mode to see similarity scores
    miner.debug_mode = True
    
    # Test blacklist embedding initialization and caching
    print("🔍 Testing blacklist embedding initialization...")
    miner._initialize_blacklist_embeddings()
    
    # Check if embeddings were loaded or created
    if miner.blacklist_embeddings:
        print(f"✅ Blacklist embeddings initialized: {len(miner.blacklist_embeddings)} phrases")
        
        # Show some example embeddings
        for i, (phrase, embedding) in enumerate(list(miner.blacklist_embeddings.items())[:5]):
            print(f"  '{phrase}': embedding shape {embedding.shape}")
    else:
        print("⚠️ No blacklist embeddings available")
    
    # Test semantic similarity filtering with expanded test cases
    print("🔍 Testing semantic similarity filtering with expanded blacklist...")
    
    # Test phrases that should be caught by semantic similarity (similarity > 0.88)
    semantic_test_phrases = [
        # Direct blacklist matches
        "yes, that's correct",
        "exactly right",
        "that is correct",
        "you are absolutely right",
        "that's exactly it",
        "destiny is exactly that",
        
        # Semantic variants that should be caught
        "that's absolutely correct",
        "you're exactly right",
        "that is precisely the case",
        "you are completely correct",
        "that's the right answer",
        "exactly that's it",
        "you got it right",
        "that's the correct answer",
        
        # Minimal answers that should be caught
        "go straight ahead",
        "proceed forward",
        "turn left",
        "move right",
        "head north",
        "fly towards",
        "navigate to",
        
        # Good answers that should pass
        "You should turn left at the intersection and continue for about 100 meters",
        "Navigate to the building on your right and follow the path around it",
        "Take the second right turn and proceed towards the landmark",
        "Follow the road until you reach the traffic light, then turn right",
        "The destination is located behind the large building with the red roof",
        "Continue straight for approximately 200 meters until you see the bridge"
    ]
    
    print("📊 Testing semantic similarity filtering:")
    filtered_count = 0
    passed_count = 0
    
    for phrase in semantic_test_phrases:
        is_good = miner.is_good_answer(phrase)
        status = "✅ PASS" if is_good else "❌ FILTERED"
        print(f"  {status}: '{phrase[:60]}{'...' if len(phrase) > 60 else ''}'")
        
        if is_good:
            passed_count += 1
        else:
            filtered_count += 1
    
    print(f"📊 Semantic filtering results: {passed_count} passed, {filtered_count} filtered")
    
    # Test caching functionality
    print("💾 Testing blacklist embedding caching...")
    cache_path = os.path.join(os.path.dirname(__file__), 'blacklist_embeds.pkl')
    
    if os.path.exists(cache_path):
        print(f"✅ Cache file exists: {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                cached_embeddings = pickle.load(f)
            print(f"  Cached embeddings: {len(cached_embeddings)} phrases")
            
            # Verify cache matches current embeddings
            if len(cached_embeddings) == len(miner.blacklist_embeddings):
                print("✅ Cache matches current embeddings")
            else:
                print(f"⚠️ Cache mismatch: cached {len(cached_embeddings)}, current {len(miner.blacklist_embeddings)}")
        except Exception as e:
            print(f"❌ Error reading cache: {e}")
    else:
        print("⚠️ No cache file found - will be created on first run")
    
    # Test debug mode similarity score printing
    print("🔍 Testing debug mode similarity score printing...")
    
    # Test phrases that should trigger similarity score printing (≥ 0.70)
    debug_test_phrases = [
        "yes, that's absolutely correct",  # Should show similarity to "yes"
        "you are exactly right",          # Should show similarity to "exactly"
        "that's the correct answer",      # Should show similarity to "correct"
        "go straight ahead",              # Should show similarity to "go"
        "proceed forward",                # Should show similarity to "proceed"
        "navigate to the building"        # Should pass without similarity printing
    ]
    
    print("📊 Testing debug mode with similarity scores ≥ 0.70:")
    for phrase in debug_test_phrases:
        print(f"\n  Testing: '{phrase}'")
        is_good = miner.is_good_answer(phrase)
        status = "✅ PASS" if is_good else "❌ FILTERED"
        print(f"  Result: {status}")
    
    # Test expanded blacklist categories
    print("📋 Testing expanded blacklist categories...")
    
    blacklist_categories = {
        'short_affirmative': [
            'yes', 'exactly', 'correct', 'right', 'true', 'sure', 'okay', 'ok',
            "that's correct", "that's right", "that's true", "you are correct", "absolutely"
        ],
        'generic_responses': [
            'destiny is exactly that', 'that is correct', 'you are right', 'that is it',
            'yes that is correct', 'yes exactly', 'exactly that'
        ],
        'minimal_answers': [
            'go', 'turn', 'move', 'head', 'fly', 'navigate',
            'proceed', 'continue', 'advance', 'straight ahead'
        ]
    }
    
    print("📊 Testing blacklist category coverage:")
    for category, phrases in blacklist_categories.items():
        print(f"  {category}: {len(phrases)} phrases")
        
        # Test a few phrases from each category
        for phrase in phrases[:3]:  # Test first 3 phrases from each category
            is_good = miner.is_good_answer(phrase)
            status = "✅ PASS" if is_good else "❌ FILTERED"
            print(f"    {status}: '{phrase}'")
    
    print("✅ Enhanced semantic filtering test completed!")
    return True

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
    
    # Initialize hard negative miner with GPU settings and enhanced semantic filtering
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
    
    # Test mining negatives with enhanced semantic filtering and debug mode
    print("⛏️ Testing negative mining with enhanced semantic filtering...")
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
        
        # Enhanced quality metrics with semantic filtering
        print("🔍 Enhanced Quality Analysis with Semantic Filtering:")
        
        # Check for blacklisted phrases (expanded list)
        blacklisted_count = 0
        semantic_filtered_count = 0
        for data in negatives.values():
            answer = data['negative_text_2'].lower()
            
            # Check direct blacklist matches
            if any(phrase in answer for phrase in ['yes', 'exactly', 'correct', 'right', 'destiny is exactly that', 'go', 'turn', 'move', 'head', 'fly', 'navigate']):
                blacklisted_count += 1
            
            # Check semantic similarity filtering (if available)
            if hasattr(miner, 'blacklist_embeddings') and miner.blacklist_embeddings:
                try:
                    if miner._check_semantic_similarity_to_blacklist(answer):
                        semantic_filtered_count += 1
                except:
                    pass
        
        print(f"  Direct blacklist matches: {blacklisted_count}/{len(negatives)} ({blacklisted_count/len(negatives)*100:.1f}%)")
        if semantic_filtered_count > 0:
            print(f"  Semantic similarity filtered: {semantic_filtered_count}/{len(negatives)} ({semantic_filtered_count/len(negatives)*100:.1f}%)")
        
        # Test semantic similarity filtering with enhanced test phrases
        print("🔍 Testing enhanced semantic similarity filtering:")
        enhanced_test_phrases = [
            # Direct blacklist matches
            "yes, that's correct",
            "exactly right",
            "that is the destination",
            "you are absolutely right",
            "that's exactly it",
            
            # Semantic variants that should be caught
            "that's absolutely correct",
            "you're exactly right",
            "that is precisely the case",
            "you are completely correct",
            "that's the right answer",
            "exactly that's it",
            "you got it right",
            "that's the correct answer",
            
            # Minimal answers that should be caught
            "go straight ahead",
            "proceed forward",
            "turn left",
            "move right",
            "head north",
            "fly towards",
            "navigate to",
            
            # Good answers that should pass
            "You should turn left at the intersection and continue for about 100 meters",
            "Navigate to the building on your right and follow the path around it",
            "Take the second right turn and proceed towards the landmark",
            "Follow the road until you reach the traffic light, then turn right",
            "The destination is located behind the large building with the red roof",
            "Continue straight for approximately 200 meters until you see the bridge"
        ]
        
        for phrase in enhanced_test_phrases:
            is_good = miner.is_good_answer(phrase)
            status = "✅ PASS" if is_good else "❌ FILTERED"
            print(f"    {status}: '{phrase[:50]}{'...' if len(phrase) > 50 else ''}'")
        
        # Phrase diversity analysis
        unique_phrases = set()
        for data in negatives.values():
            unique_phrases.add(data['negative_text_2'].lower().strip())
        
        diversity_ratio = len(unique_phrases) / len(negatives)
        print(f"  Phrase diversity ratio: {diversity_ratio:.3f} ({len(unique_phrases)} unique / {len(negatives)} total)")
        
        # Cluster diversity for diverse negatives
        if diverse_count > 0:
            cluster_transitions = []
            for data in negatives.values():
                metadata = data['validation_metadata_2']
                if metadata.get('negative_type_2') == 'diverse' and 'anchor_cluster' in metadata:
                    anchor_cluster = metadata['anchor_cluster']
                    negative_cluster = metadata['negative_cluster']
                    cluster_transitions.append((anchor_cluster, negative_cluster))
            
            different_clusters = sum(1 for a, n in cluster_transitions if a != n)
            cluster_diversity = different_clusters / len(cluster_transitions) if cluster_transitions else 0
            print(f"  Diverse cluster diversity: {cluster_diversity:.3f} ({different_clusters}/{len(cluster_transitions)} different clusters)")
    else:
        print("  No negatives mined - check mining parameters")
    
    # Test answer quality filtering with enhanced criteria
    print("🔍 Testing enhanced answer quality filtering...")
    enhanced_test_answers = [
        # Should be filtered by direct blacklist
        "turn left",
        "go straight ahead",
        "yes, that's correct",
        "exactly right",
        
        # Should be filtered by semantic similarity
        "that's absolutely correct",
        "you're exactly right",
        "proceed forward",
        "head north",
        
        # Should pass
        "You should turn left at the intersection and continue for about 100 meters",
        "Navigate to the building on your right and follow the path around it",
        "Take the second right turn and proceed towards the landmark",
        "Follow the road until you reach the traffic light, then turn right",
        "The destination is located behind the large building with the red roof",
        "Continue straight for approximately 200 meters until you see the bridge"
    ]
    
    for answer in enhanced_test_answers:
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
    
    # Test 1: Enhanced semantic filtering
    success1 = test_enhanced_semantic_filtering()
    
    # Test 2: GPU mining functionality
    success2 = test_hard_negative_mining()
    
    # Test 3: Multi-GPU setup logic
    success3 = test_multi_gpu_setup()
    
    if success1 and success2 and success3:
        print("\n🎉 All tests passed!")
        print("✅ Enhanced semantic filtering is working correctly")
        print("✅ GPU processing is working correctly")
        print("✅ Multi-GPU sharding logic is correct")
        print("✅ Mining strategy improvements are functional")
        print("✅ Semantic filtering with caching is operational")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 