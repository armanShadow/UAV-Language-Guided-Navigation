#!/usr/bin/env python3
"""
Test script for the complete two-pipeline architecture.
Tests paraphrase generation, validation, and iterative refinement.
"""

import sys
import time
import json
import random
import logging
from pathlib import Path
from typing import List

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
from validation_pipeline import ValidationPipeline
from iterative_contrastive_pipeline import IterativeContrastivePipeline

# Configure logging
logger = logging.getLogger(__name__)

def load_random_avdn_examples(num_examples: int = 4) -> List[str]:
    """
    Load random examples from the processed AVDN dataset for testing.
    Returns a list of navigation instructions.
    """
    # Possible dataset paths (prioritize processed data)
    dataset_paths = [
        "processed_data/train_data.json",
        "src/data/processed_data/train_data.json",
        "AnsweringAgent/src/data/processed_data/train_data.json",
        "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json",
        "../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json",
        "../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json",
        "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json"
    ]
    
    for path in dataset_paths:
        if Path(path).exists():
            logger.info(f"📂 Loading dataset from: {path}")
            try:
                with open(path, 'r') as f:
                    episodes = json.load(f)
                
                # Extract all answers from dialogs
                all_instructions = []
                for episode in episodes:
                    dialogs = episode.get('dialogs', [])
                    for dialog in dialogs:
                        if dialog and dialog.get('answer'):
                            answer = dialog['answer'].strip()
                            if answer and len(answer.split()) >= 3:
                                all_instructions.append(answer)
                
                # Get random examples
                if len(all_instructions) >= num_examples:
                    # Use current time for true randomness
                    random.seed()
                    random_examples = random.sample(all_instructions, num_examples)
                    logger.info(f"📊 Selected {num_examples} random examples from {len(all_instructions)} total")
                    return random_examples
                else:
                    logger.warning(f"Only {len(all_instructions)} examples available, returning all")
                    return all_instructions
                    
            except Exception as e:
                logger.error(f"Error loading dataset from {path}: {e}")
                continue
    
    logger.error("❌ Could not find AVDN dataset. Using fallback examples.")
    # Fallback to original hardcoded examples
    return [
        "Turn right and fly over the white building at 3 o'clock",
        "Go straight ahead towards the gray road near the parking area",
        "Navigate to the brown house at 6 o'clock position",
        "Fly north over the highway and turn left at the intersection"
    ]

def test_generation_pipeline():
    """Test the paraphrase generation pipeline independently."""
    print("🔧 Testing Paraphrase Generation Pipeline")
    print("="*60)
    
    pipeline = ParaphraseGenerationPipeline()
    
    # Test model loading
    print("Loading Mixtral model...")
    if not pipeline.load_model():
        print("❌ Failed to load generation model")
        return False
    print("✅ Generation model loaded successfully")
    
    # Test spatial term extraction with real AVDN example
    avdn_examples = load_random_avdn_examples(num_examples=1)
    test_instruction = avdn_examples[0]
    spatial_terms = pipeline.extract_spatial_terms(test_instruction)
    print(f"\nReal AVDN example: {test_instruction}")
    print(f"Spatial terms extracted: {spatial_terms}")
    
    # Test paraphrase generation
    print(f"\nTesting paraphrase generation...")
    results = pipeline.generate_paraphrases(test_instruction, strategy="combined")
    
    print(f"Generated {len(results['positives'])} positives:")
    for i, pos in enumerate(results['positives'], 1):
        print(f"  {i}. {pos}")
    
    print(f"Generated {len(results['negatives'])} negatives:")
    for i, neg in enumerate(results['negatives'], 1):
        print(f"  {i}. {neg}")
    
    return True

def test_validation_pipeline():
    """Test the validation pipeline independently."""
    print("\n🔍 Testing Validation Pipeline")
    print("="*60)
    
    pipeline = ValidationPipeline()
    
    # Test model loading
    print("Loading embedding model...")
    if not pipeline.load_embedding_model():
        print("❌ Failed to load validation model")
        return False
    print("✅ Validation model loaded successfully")
    
    # Test feature extraction with real AVDN example
    avdn_examples = load_random_avdn_examples(num_examples=1)
    test_instruction = avdn_examples[0]
    features = pipeline.extract_spatial_features(test_instruction)
    print(f"\nReal AVDN example: {test_instruction}")
    print(f"Spatial features extracted: {features}")
    
    # Test positive validation (create a simple positive paraphrase)
    # For demo purposes, create a basic paraphrase by replacing some words
    positive_paraphrase = test_instruction.replace("turn", "rotate").replace("go", "move").replace("fly", "navigate")
    if positive_paraphrase == test_instruction:
        positive_paraphrase = test_instruction.replace("the", "a")  # Fallback change
    pos_result = pipeline.validate_positive_paraphrase(test_instruction, positive_paraphrase)
    
    print(f"\nPositive validation:")
    print(f"  Original: {test_instruction}")
    print(f"  Paraphrase: {positive_paraphrase}")
    print(f"  Valid: {pos_result['is_valid']}")
    print(f"  Embedding similarity: {pos_result['embedding_similarity']:.3f}")
    print(f"  Direction similarity: {pos_result['direction_similarity']:.3f}")
    print(f"  Landmark similarity: {pos_result['landmark_similarity']:.3f}")
    
    # Test negative validation (create a simple negative by changing spatial elements)
    # For demo purposes, create a negative by changing directions and landmarks
    negative_paraphrase = test_instruction.replace("right", "left").replace("white", "gray").replace("3", "9")
    if negative_paraphrase == test_instruction:
        negative_paraphrase = test_instruction.replace("over", "under")  # Fallback change
    neg_result = pipeline.validate_negative_paraphrase(test_instruction, negative_paraphrase)
    
    print(f"\nNegative validation:")
    print(f"  Original: {test_instruction}")
    print(f"  Paraphrase: {negative_paraphrase}")
    print(f"  Valid: {neg_result['is_valid']}")
    print(f"  Embedding similarity: {neg_result['embedding_similarity']:.3f}")
    print(f"  Direction changed: {neg_result['direction_changed']}")
    print(f"  Landmark changed: {neg_result['landmark_changed']}")
    
    return True

def test_iterative_pipeline():
    """Test the complete iterative pipeline."""
    print("\n🔄 Testing Iterative Contrastive Pipeline")
    print("="*60)
    
    pipeline = IterativeContrastivePipeline(max_iterations=2)
    
    # Test initialization
    print("Initializing both pipelines...")
    if not pipeline.initialize():
        print("❌ Failed to initialize pipelines")
        return False
    print("✅ Both pipelines initialized successfully")
    
    # Test single instruction processing with real AVDN example
    avdn_examples = load_random_avdn_examples(num_examples=1)
    test_instruction = avdn_examples[0]
    print(f"\nProcessing real AVDN example: {test_instruction}")
    
    start_time = time.time()
    result = pipeline.generate_contrastive_samples(test_instruction)
    processing_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Success: {result['success']}")
    print(f"  Iterations used: {result['iterations_used']}")
    print(f"  Processing time: {processing_time:.1f}s")
    
    if result['positives']:
        print(f"  Valid positives ({len(result['positives'])}):")
        for i, pos in enumerate(result['positives'], 1):
            print(f"    {i}. {pos}")
    
    if result['negatives']:
        print(f"  Valid negatives ({len(result['negatives'])}):")
        for i, neg in enumerate(result['negatives'], 1):
            print(f"    {i}. {neg}")
    
    # Test batch processing with real AVDN examples
    print(f"\nTesting batch processing...")
    test_instructions = load_random_avdn_examples(num_examples=2)
    print(f"Using real AVDN examples for batch:")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"  {i}. {instruction}")
    
    batch_start = time.time()
    batch_results = pipeline.process_instruction_batch(test_instructions)
    batch_time = time.time() - batch_start
    
    successful = sum(1 for r in batch_results if r['success'])
    print(f"\nBatch results:")
    print(f"  Instructions processed: {len(batch_results)}")
    print(f"  Successful: {successful}")
    print(f"  Success rate: {successful/len(batch_results)*100:.1f}%")
    print(f"  Total time: {batch_time:.1f}s")
    print(f"  Average time per instruction: {batch_time/len(batch_results):.1f}s")
    
    # Show statistics
    stats = pipeline.get_statistics()
    print(f"\nPipeline statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    return True

def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    print("\n🚀 Testing End-to-End Workflow")
    print("="*60)
    
    # Load actual AVDN dataset samples
    print("Loading real AVDN dataset samples...")
    test_instructions = load_random_avdn_examples(num_examples=4)
    print(f"Loaded {len(test_instructions)} real AVDN examples:")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"  {i}. {instruction}")
    
    # Initialize pipeline
    pipeline = IterativeContrastivePipeline(
        max_iterations=2,
        target_positives=2,
        target_negatives=1
    )
    
    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return False
    
    # Process all instructions
    print(f"Processing {len(test_instructions)} test instructions...")
    
    all_results = []
    total_time = 0
    
    for i, instruction in enumerate(test_instructions, 1):
        print(f"\n--- Instruction {i}/{len(test_instructions)} ---")
        print(f"Original: {instruction}")
        
        start_time = time.time()
        result = pipeline.generate_contrastive_samples(instruction)
        processing_time = time.time() - start_time
        total_time += processing_time
        
        all_results.append(result)
        
        if result['success']:
            print(f"✅ Success in {result['iterations_used']} iterations ({processing_time:.1f}s)")
            print(f"   Positives: {result['positives']}")
            print(f"   Negatives: {result['negatives']}")
        else:
            print(f"❌ Failed after {result['iterations_used']} iterations ({processing_time:.1f}s)")
    
    # Summary
    successful = sum(1 for r in all_results if r['success'])
    success_rate = successful / len(all_results) * 100
    avg_time = total_time / len(all_results)
    
    print(f"\n{'='*60}")
    print(f"END-TO-END WORKFLOW SUMMARY")
    print(f"{'='*60}")
    print(f"Instructions processed: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average time per instruction: {avg_time:.1f}s")
    
    # Show final statistics
    final_stats = pipeline.get_statistics()
    print(f"\nFinal pipeline statistics:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    return success_rate > 50  # Consider test passed if >50% success rate

def main():
    """Run all tests for the two-pipeline architecture."""
    print("🧪 Testing Two-Pipeline Architecture")
    print("="*80)
    print("Testing separate generation and validation pipelines with iterative refinement")
    print("="*80)
    
    try:
        # Test individual components
        if not test_generation_pipeline():
            print("\n❌ Generation pipeline test failed")
            return False
        
        if not test_validation_pipeline():
            print("\n❌ Validation pipeline test failed")
            return False
        
        if not test_iterative_pipeline():
            print("\n❌ Iterative pipeline test failed")
            return False
        
        # Test complete workflow
        if not test_end_to_end_workflow():
            print("\n❌ End-to-end workflow test failed")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("Two-pipeline architecture is working correctly")
        return True
        
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 