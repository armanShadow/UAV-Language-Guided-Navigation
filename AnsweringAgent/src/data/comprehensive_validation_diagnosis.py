#!/usr/bin/env python3
"""
Comprehensive validation diagnosis script for both positive and negative paraphrases.
Analyzes validation pipeline behavior in detail to identify edge cases and threshold issues.
"""

import logging
from validation_pipeline import ValidationPipeline
from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def analyze_validation_detailed(val_pipeline, original, paraphrase, paraphrase_type):
    """Detailed analysis of validation for a single paraphrase."""
    print(f"\n🔬 DETAILED {paraphrase_type.upper()} VALIDATION ANALYSIS")
    print("=" * 60)
    print(f"Original: {original}")
    print(f"{paraphrase_type.capitalize()}: {paraphrase}")
    print()
    
    # Extract features
    orig_features = val_pipeline.extract_spatial_features(original)
    para_features = val_pipeline.extract_spatial_features(paraphrase)
    
    print("📊 SPATIAL FEATURES:")
    print(f"Original features: {orig_features}")
    print(f"Paraphrase features: {para_features}")
    print()
    
    # Compute similarities
    embedding_sim = val_pipeline.compute_embedding_similarity(original, paraphrase)
    
    orig_dirs = orig_features.get('directions', [])
    para_dirs = para_features.get('directions', [])
    direction_sim = val_pipeline._compute_direction_similarity(orig_dirs, para_dirs)
    
    orig_landmarks = orig_features.get('landmarks', [])
    para_landmarks = para_features.get('landmarks', [])
    landmark_sim = val_pipeline._compute_landmark_similarity(orig_landmarks, para_landmarks)
    
    print("📈 SIMILARITY SCORES:")
    print(f"  Embedding similarity: {embedding_sim:.3f}")
    print(f"  Direction similarity: {direction_sim:.3f}")
    print(f"  Landmark similarity: {landmark_sim:.3f}")
    print()
    
    # Analyze direction changes in detail
    if orig_dirs or para_dirs:
        print("🧭 DIRECTION ANALYSIS:")
        print(f"  Original directions: {orig_dirs}")
        print(f"  Paraphrase directions: {para_dirs}")
        
        # Extract clock hours
        orig_clock_hours = set()
        para_clock_hours = set()
        
        for direction in orig_dirs:
            clock_hour = val_pipeline._extract_clock_hour(direction)
            if clock_hour:
                orig_clock_hours.add(clock_hour)
                print(f"    Original '{direction}' → clock hour: {clock_hour}")
        
        for direction in para_dirs:
            clock_hour = val_pipeline._extract_clock_hour(direction)
            if clock_hour:
                para_clock_hours.add(clock_hour)
                print(f"    Paraphrase '{direction}' → clock hour: {clock_hour}")
        
        print(f"  Clock hours comparison: {orig_clock_hours} vs {para_clock_hours}")
        clock_similarity = 1.0 if orig_clock_hours == para_clock_hours else 0.0
        print(f"  Clock similarity: {clock_similarity}")
        print()
    
    # Run appropriate validation
    if paraphrase_type == "positive":
        result = val_pipeline.validate_positive_paraphrase(original, paraphrase)
        print("✅ POSITIVE VALIDATION LOGIC:")
        print(f"  Required: embedding > 0.75 AND (direction > 0.8 OR landmark > 0.7 OR combined > 0.75)")
        print(f"  Embedding check: {embedding_sim:.3f} > 0.75 = {embedding_sim > 0.75}")
        print(f"  Direction check: {direction_sim:.3f} > 0.8 = {direction_sim > 0.8}")
        print(f"  Landmark check: {landmark_sim:.3f} > 0.7 = {landmark_sim > 0.7}")
        combined_score = (direction_sim + landmark_sim) / 2
        print(f"  Combined check: {combined_score:.3f} > 0.75 = {combined_score > 0.75}")
        spatial_ok = direction_sim > 0.8 or landmark_sim > 0.7 or combined_score > 0.75
        print(f"  Spatial OK: {spatial_ok}")
        overall_valid = embedding_sim > 0.75 and spatial_ok
        print(f"  Overall valid: {overall_valid}")
        
    else:  # negative
        result = val_pipeline.validate_negative_paraphrase(original, paraphrase)
        print("❌ NEGATIVE VALIDATION LOGIC:")
        print(f"  Required: 0.4 < embedding < 0.9 AND spatial_changed")
        print(f"  Embedding check: 0.4 < {embedding_sim:.3f} < 0.9 = {0.4 < embedding_sim < 0.9}")
        
        direction_changed = direction_sim < 0.6
        landmark_changed = landmark_sim < 0.6
        spatial_changed = direction_changed or landmark_changed
        
        print(f"  Direction changed: {direction_sim:.3f} < 0.6 = {direction_changed}")
        print(f"  Landmark changed: {landmark_sim:.3f} < 0.6 = {landmark_changed}")
        print(f"  Spatial changed: {spatial_changed}")
        
        overall_valid = (0.4 < embedding_sim < 0.9) and spatial_changed
        print(f"  Overall valid: {overall_valid}")
    
    print(f"\n🎯 FINAL RESULT: {'✅ VALID' if result['is_valid'] else '❌ INVALID'}")
    
    if not result['is_valid']:
        print("❌ FAILURE REASONS:")
        if paraphrase_type == "positive":
            if embedding_sim <= 0.75:
                print(f"   - Embedding too low: {embedding_sim:.3f} ≤ 0.75")
            if not spatial_ok:
                print(f"   - Spatial preservation failed: direction={direction_sim:.3f}≤0.8, landmark={landmark_sim:.3f}≤0.7, combined={combined_score:.3f}≤0.75")
        else:
            if not (0.4 < embedding_sim < 0.9):
                if embedding_sim <= 0.4:
                    print(f"   - Embedding too low: {embedding_sim:.3f} ≤ 0.4")
                if embedding_sim >= 0.9:
                    print(f"   - Embedding too high: {embedding_sim:.3f} ≥ 0.9")
            if not spatial_changed:
                print(f"   - No spatial changes: direction={direction_sim:.3f}≥0.6, landmark={landmark_sim:.3f}≥0.6")
    
    return result

def test_comprehensive_validation():
    """Test comprehensive validation with various examples."""
    print("🧪 COMPREHENSIVE VALIDATION DIAGNOSIS")
    print("=" * 60)
    
    # Initialize validation pipeline only (no generation for faster testing)
    val_pipeline = ValidationPipeline()
    
    # Test cases covering different scenarios
    test_cases = [
        {
            'original': "Go in the 3 o'clock direction from your current position.",
            'positive': "Head towards the 3 o'clock direction from where you are now.",
            'negative': "From your current location, proceed in the 9 o'clock direction."
        },
        {
            'original': "Turn right and fly over the white building.",
            'positive': "Make a right turn and navigate above the white structure.",
            'negative': "Turn left and fly over the gray building."
        },
        {
            'original': "Navigate to the brown house at 6 o'clock position.",
            'positive': "Head to the brown house at the 6 o'clock location.",
            'negative': "Navigate to the white house at 12 o'clock position."
        },
        {
            'original': "Move forward towards the red building on your left.",
            'positive': "Proceed ahead toward the red structure on your left side.",
            'negative': "Move backward towards the blue building on your right."
        }
    ]
    
    overall_results = {
        'positive_valid': 0,
        'positive_total': 0,
        'negative_valid': 0,
        'negative_total': 0,
        'failures': []
    }
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🎯 TEST CASE {i}")
        print("=" * 40)
        
        # Test positive validation
        pos_result = analyze_validation_detailed(
            val_pipeline, 
            test_case['original'], 
            test_case['positive'], 
            "positive"
        )
        
        # Test negative validation
        neg_result = analyze_validation_detailed(
            val_pipeline, 
            test_case['original'], 
            test_case['negative'], 
            "negative"
        )
        
        # Update results
        overall_results['positive_total'] += 1
        overall_results['negative_total'] += 1
        
        if pos_result['is_valid']:
            overall_results['positive_valid'] += 1
        else:
            overall_results['failures'].append(f"Test {i} positive: {test_case['positive']}")
        
        if neg_result['is_valid']:
            overall_results['negative_valid'] += 1
        else:
            overall_results['failures'].append(f"Test {i} negative: {test_case['negative']}")
    
    # Overall summary
    print(f"\n📊 OVERALL VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Positive validation: {overall_results['positive_valid']}/{overall_results['positive_total']} ({overall_results['positive_valid']/overall_results['positive_total']*100:.1f}%)")
    print(f"Negative validation: {overall_results['negative_valid']}/{overall_results['negative_total']} ({overall_results['negative_valid']/overall_results['negative_total']*100:.1f}%)")
    
    if overall_results['failures']:
        print(f"\n❌ FAILURES ({len(overall_results['failures'])}):")
        for failure in overall_results['failures']:
            print(f"  - {failure}")
    
    # Threshold analysis
    print(f"\n🎯 THRESHOLD ANALYSIS")
    print("=" * 30)
    print("Current thresholds:")
    print("  Positive validation:")
    print("    - Embedding similarity: > 0.75")
    print("    - Direction similarity: > 0.8 OR landmark > 0.7 OR combined > 0.75")
    print("  Negative validation:")
    print("    - Embedding similarity: 0.4 < embedding < 0.9")
    print("    - Direction change: < 0.6")
    print("    - Landmark change: < 0.6")
    
    # Suggestions based on results
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 25)
    
    pos_rate = overall_results['positive_valid'] / overall_results['positive_total']
    neg_rate = overall_results['negative_valid'] / overall_results['negative_total']
    
    if pos_rate < 0.8:
        print("🔧 Positive validation issues:")
        print("  - Consider lowering embedding threshold from 0.75 to 0.65")
        print("  - Consider lowering direction threshold from 0.8 to 0.7")
        print("  - Consider lowering landmark threshold from 0.7 to 0.6")
    
    if neg_rate < 0.8:
        print("🔧 Negative validation issues:")
        print("  - Consider raising direction change threshold from 0.6 to 0.65")
        print("  - Consider raising landmark change threshold from 0.6 to 0.65")
        print("  - Consider lowering embedding threshold from 0.4 to 0.35")
        print("  - Consider raising embedding threshold from 0.9 to 0.95")
    
    if pos_rate >= 0.8 and neg_rate >= 0.8:
        print("✅ Validation thresholds appear well-calibrated!")
    
    return overall_results

def test_with_generation():
    """Test validation with actual generation pipeline."""
    print(f"\n🚀 TESTING WITH ACTUAL GENERATION")
    print("=" * 40)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("⚠️  Running on CPU - generation may be slow")
    
    try:
        # Initialize both pipelines
        gen_pipeline = ParaphraseGenerationPipeline()
        val_pipeline = ValidationPipeline()
        
        test_instruction = "Go in the 3 o'clock direction from your current position."
        
        print(f"🔄 Generating paraphrases for: {test_instruction}")
        
        # Generate paraphrases
        generation_result = gen_pipeline.generate_paraphrases(test_instruction)
        
        if generation_result['success']:
            positives = generation_result['positives']
            negatives = generation_result['negatives']
            
            print(f"✅ Generated {len(positives)} positives, {len(negatives)} negatives")
            
            # Validate each
            for i, positive in enumerate(positives):
                print(f"\n🔍 Testing generated positive {i+1}:")
                analyze_validation_detailed(val_pipeline, test_instruction, positive, "positive")
            
            for i, negative in enumerate(negatives):
                print(f"\n🔍 Testing generated negative {i+1}:")
                analyze_validation_detailed(val_pipeline, test_instruction, negative, "negative")
        
        else:
            print(f"❌ Generation failed: {generation_result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"❌ Error during generation testing: {e}")
        print("💡 This is expected if running on CPU without proper model setup")

if __name__ == "__main__":
    # Run comprehensive validation testing
    results = test_comprehensive_validation()
    
    # Optionally test with actual generation (may fail on CPU)
    print(f"\n" + "="*60)
    user_input = input("Test with actual generation pipeline? (y/n): ").lower().strip()
    if user_input == 'y':
        test_with_generation()
    else:
        print("Skipping generation testing.") 