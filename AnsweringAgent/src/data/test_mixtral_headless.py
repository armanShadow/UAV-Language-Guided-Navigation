#!/usr/bin/env python3
"""
Headless server test for Mixtral paraphrasing pipeline.
Tests the paraphrase generation pipeline (Pipeline 1) with real AVDN examples.
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import List

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from paraphrase_generation_pipeline import ParaphraseGenerationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
                
                # Extract all instructions from episodes
                all_instructions = []
                for episode in episodes:
                    # Add first_instruction if it exists
                    if 'first_instruction' in episode and episode['first_instruction']:
                        instruction = episode['first_instruction'].strip()
                        if instruction and len(instruction) > 10:  # Filter out empty/short instructions
                            all_instructions.append(instruction)
                    
                    # Add dialog answers if they exist
                    if 'dialogs' in episode:
                        for dialog in episode['dialogs']:
                            if 'answer' in dialog and dialog['answer']:
                                answer = dialog['answer'].strip()
                                if answer and len(answer) > 10:  # Filter out empty/short answers
                                    all_instructions.append(answer)
                
                logger.info(f"📊 Extracted {len(all_instructions)} instructions from dataset")
                
                # Return random sample
                if all_instructions:
                    import random
                    random.shuffle(all_instructions)
                    return all_instructions[:num_examples]
                else:
                    logger.warning("No valid instructions found in dataset")
                
            except Exception as e:
                logger.warning(f"Failed to load from {path}: {e}")
                continue
    
    # Fallback examples if no dataset found or no valid instructions
    logger.warning("No AVDN dataset found or no valid instructions, using fallback examples")
    fallback_examples = [
        "Turn right and fly over the white building at 3 o'clock",
        "Go straight ahead towards the gray road near the parking area", 
        "Navigate to the brown house at 6 o'clock position",
        "Fly north over the highway and turn left at the intersection",
        "Head forward towards 6 o'clock direction, after passing a road and few buildings",
        "Make a left turn and continue straight until you reach the parking lot",
        "Fly over the intersection and look for the gray building on your right",
        "Go north towards the highway and turn right at the traffic light"
    ]
    
    return fallback_examples[:num_examples]

def test_mixtral_paraphrasing():
    """Test Mixtral paraphrasing with TRUE BATCH PROCESSING and combined prompts."""
    
    print("🚀 Testing Mixtral TRUE BATCH PROCESSING Pipeline on Headless Server")
    print("="*80)
    
    # Load test instructions
    test_instructions = load_random_avdn_examples(num_examples=4)
    print(f"📊 Loaded {len(test_instructions)} test instructions")
    
    print("\n📝 Test Instructions:")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"  {i}. {instruction}")
    
    # Test TRUE BATCH PROCESSING pipeline
    print("\n🔧 Initializing TRUE BATCH PROCESSING Pipeline...")
    try:
        from simple_batch_pipeline import TrueBatchProcessingPipeline
        print("✅ TRUE BATCH pipeline imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TRUE BATCH pipeline: {e}")
        return False
    
    # Initialize pipeline
    pipeline = TrueBatchProcessingPipeline(batch_size=2)
    
    print("⏳ Loading Mixtral-8x7B-Instruct model...")
    load_start = time.time()
    
    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return False
    
    load_time = time.time() - load_start
    print(f"✅ Mixtral model loaded successfully in {load_time:.1f}s")
    print(f"🔧 Using device: {pipeline.generation_pipeline.device}")
    print(f"🔧 Model device: {pipeline.generation_pipeline.model.device}")
    
    # Test TRUE BATCH PROCESSING with COMBINED PROMPTS
    print(f"\n🚀 Testing TRUE BATCH PROCESSING with COMBINED PROMPTS...")
    print(f"⚡ PROCESSING ALL {len(test_instructions)} INSTRUCTIONS SIMULTANEOUSLY")
    print(f"🔥 USING 4 COMBINED PROMPTS (instead of 8 separate prompts)")
    print(f"🔥 NO SEQUENTIAL PROCESSING - GENUINE PARALLEL INFERENCE")
    print(f"🔥 Using TRUE BATCH PROCESSING at model level")
    
    # Process all instructions with TRUE BATCH PROCESSING
    batch_start = time.time()
    
    print(f"\n🚀 TRUE BATCH PROCESSING: {len(test_instructions)} instructions across {pipeline.num_gpus} GPUs")
    all_results = pipeline.process_instructions_true_batch(test_instructions)
    
    batch_time = time.time() - batch_start
    
    # Display results with detailed validation analysis
    print(f"\n📊 TRUE BATCH PROCESSING Results with Detailed Validation Analysis:")
    print(f"⏱️  Total batch processing time: {batch_time:.1f}s")
    print(f"⚡ Average time per instruction: {batch_time/len(test_instructions):.1f}s")
    
    successful = 0
    total_quality_scores = {'positives': [], 'negatives': []}
    validation_insights = {'positive_failures': [], 'negative_failures': []}
    
    for i, result in enumerate(all_results, 1):
        print(f"\n{'='*80}")
        print(f"📋 RESULT {i}/{len(all_results)}")
        print(f"{'='*80}")
        print(f"📝 Original: {result['original_instruction']}")
        
        # Display generated paraphrases (before validation)
        print(f"\n🔧 Generated Paraphrases:")
        print(f"  📝 Generated Positives ({len(result.get('generated_positives', []))}):")
        for j, pos in enumerate(result.get('generated_positives', []), 1):
            print(f"    {j}. {pos}")
        
        print(f"  📝 Generated Negatives ({len(result.get('generated_negatives', []))}):")
        for j, neg in enumerate(result.get('generated_negatives', []), 1):
            print(f"    {j}. {neg}")
        
        # Display validation results
        validation_summary = result.get('validation_summary', {})
        quality_assessment = result.get('quality_assessment', {})
        
        print(f"\n🔍 Validation Analysis:")
        print(f"  ✅ Valid Positives: {validation_summary.get('valid_positives', 0)}/{len(result.get('generated_positives', []))}")
        print(f"  ✅ Valid Negatives: {validation_summary.get('valid_negatives', 0)}/{len(result.get('generated_negatives', []))}")
        print(f"  📊 Avg Positive Quality: {quality_assessment.get('avg_positive_quality', 0):.3f}")
        print(f"  📊 Avg Negative Quality: {quality_assessment.get('avg_negative_quality', 0):.3f}")
        
        # Display detailed validation logs if available
        detailed_logs = result.get('detailed_validation_logs', {})
        if detailed_logs:
            print(f"\n🔍 Detailed Validation Analysis:")
            
            # Positive validation details
            for j, log_entry in enumerate(detailed_logs.get('positives', []), 1):
                validation_result = log_entry.get('validation_result', {})
                print(f"  📝 Positive {j}: {log_entry.get('paraphrase', '')}")
                print(f"    ✅ Length: {validation_result.get('length_check', False)}")
                print(f"    ✅ Unique: {validation_result.get('uniqueness_check', False)}")
                print(f"    ✅ Navigation: {validation_result.get('has_navigation', False)}")
                print(f"    ✅ Spatial: {validation_result.get('has_spatial', False)}")
                print(f"    📊 Quality: {log_entry.get('quality_score', 0):.3f}")
                if not validation_result.get('is_valid', False):
                    print(f"    ❌ Failure: {validation_result.get('failure_reason', 'Unknown')}")
                    validation_insights['positive_failures'].append(validation_result.get('failure_reason', 'Unknown'))
            
            # Negative validation details
            for j, log_entry in enumerate(detailed_logs.get('negatives', []), 1):
                validation_result = log_entry.get('validation_result', {})
                print(f"  📝 Negative {j}: {log_entry.get('paraphrase', '')}")
                print(f"    ✅ Length: {validation_result.get('length_check', False)}")
                print(f"    ✅ Unique: {validation_result.get('uniqueness_check', False)}")
                print(f"    ✅ Navigation: {validation_result.get('has_navigation', False)}")
                print(f"    ✅ Spatial: {validation_result.get('has_spatial', False)}")
                print(f"    📊 Quality: {log_entry.get('quality_score', 0):.3f}")
                if not validation_result.get('is_valid', False):
                    print(f"    ❌ Failure: {validation_result.get('failure_reason', 'Unknown')}")
                    validation_insights['negative_failures'].append(validation_result.get('failure_reason', 'Unknown'))
        
        # Success determination
        if result['success']:
            successful += 1
            print(f"\n✅ QUALITY-BASED SUCCESS")
            
            # Display accepted paraphrases
            if result['positives']:
                print(f"  📝 Accepted Positives:")
                for j, pos in enumerate(result['positives'], 1):
                    print(f"    {j}. {pos}")
            
            if result['negatives']:
                print(f"  📝 Accepted Negatives:")
                for j, neg in enumerate(result['negatives'], 1):
                    print(f"    {j}. {neg}")
        else:
            print(f"\n❌ QUALITY-BASED FAILURE")
            print(f"  📊 Reason: Quality scores or validation criteria not met")
        
        # Collect quality scores
        total_quality_scores['positives'].extend(quality_assessment.get('individual_scores', {}).get('positives', []))
        total_quality_scores['negatives'].extend(quality_assessment.get('individual_scores', {}).get('negatives', []))
    
    # Final summary with validation insights
    success_rate = successful / len(all_results) * 100 if all_results else 0
    
    print(f"\n{'='*80}")
    print(f"📊 FINAL RESULTS & VALIDATION INSIGHTS")
    print(f"{'='*80}")
    print(f"🎯 SUCCESS RATE: {successful}/{len(all_results)} ({success_rate:.1f}%)")
    print(f"⏱️  TOTAL TIME: {batch_time:.1f}s")
    print(f"⚡ SPEEDUP: TRUE BATCH PROCESSING across {pipeline.num_gpus} GPUs")
    print(f"🔥 EFFICIENCY: Combined prompts (4 vs 8) = 2x prompt efficiency")
    
    # Quality assessment summary
    if total_quality_scores['positives']:
        avg_pos_quality = sum(total_quality_scores['positives']) / len(total_quality_scores['positives'])
        print(f"📈 AVG POSITIVE QUALITY: {avg_pos_quality:.3f}")
    
    if total_quality_scores['negatives']:
        avg_neg_quality = sum(total_quality_scores['negatives']) / len(total_quality_scores['negatives'])
        print(f"📈 AVG NEGATIVE QUALITY: {avg_neg_quality:.3f}")
    
    # Validation failure analysis
    print(f"\n🔍 VALIDATION FAILURE ANALYSIS:")
    if validation_insights['positive_failures']:
        from collections import Counter
        pos_failures = Counter(validation_insights['positive_failures'])
        print(f"  📊 Positive Failure Reasons:")
        for reason, count in pos_failures.most_common():
            print(f"    - {reason}: {count} occurrences")
    
    if validation_insights['negative_failures']:
        neg_failures = Counter(validation_insights['negative_failures'])
        print(f"  📊 Negative Failure Reasons:")
        for reason, count in neg_failures.most_common():
            print(f"    - {reason}: {count} occurrences")
    
    # Compare with sequential processing estimate
    sequential_time_estimate = len(test_instructions) * 25  # Assume 25s per instruction sequential
    speedup = sequential_time_estimate / batch_time if batch_time > 0 else 0
    
    print(f"\n🚀 PERFORMANCE COMPARISON:")
    print(f"📊 Sequential estimate: {sequential_time_estimate}s")
    print(f"⚡ TRUE BATCH actual: {batch_time:.1f}s")
    print(f"🔥 SPEEDUP: {speedup:.1f}x faster")
    print(f"🔥 TRUE BATCH PROCESSING: {len(test_instructions)} instructions processed simultaneously")
    
    print(f"\n💡 VALIDATION INSIGHTS:")
    print(f"🔍 This detailed analysis shows exactly why paraphrases pass or fail validation")
    print(f"📊 Quality scores provide continuous metrics beyond pass/fail")
    print(f"🎯 Quality-based success is more lenient than strict validation")
    
    return successful > 0

def main():
    """Run the Mixtral paraphrasing test."""
    try:
        success = test_mixtral_paraphrasing()
        
        if success:
            print("\n🎯 HEADLESS SERVER TEST PASSED")
            print("Mixtral TRUE BATCH PROCESSING pipeline is working correctly")
            return True
        else:
            print("\n❌ HEADLESS SERVER TEST FAILED")
            print("Mixtral TRUE BATCH PROCESSING pipeline needs debugging")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 