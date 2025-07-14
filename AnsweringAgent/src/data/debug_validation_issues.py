#!/usr/bin/env python3
"""
Debug Validation Issues
Investigate why negative validation is consistently failing at 0% success rate.
"""

import logging
from test_two_pipeline_architecture import load_random_avdn_examples

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_negative_validation():
    """Debug why negative validation is failing."""
    print("🔍 DEBUGGING NEGATIVE VALIDATION ISSUES")
    print("="*60)
    
    # Load pipelines
    from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
    from validation_pipeline import ValidationPipeline
    
    # Initialize
    gen_pipeline = ParaphraseGenerationPipeline()
    val_pipeline = ValidationPipeline()
    
    print("🔧 Loading models...")
    if not gen_pipeline.load_model():
        print("❌ Failed to load generation model")
        return
    if not val_pipeline.load_embedding_model():
        print("❌ Failed to load validation model")
        return
    print("✅ Models loaded")
    
    # Get a test instruction
    examples = load_random_avdn_examples(num_examples=1)
    instruction = examples[0]
    
    print(f"\n📝 Test instruction: {instruction}")
    
    # Extract spatial features from original
    orig_features = val_pipeline.extract_spatial_features(instruction)
    print(f"\n🔍 Original spatial features: {orig_features}")
    
    # Generate paraphrases
    print(f"\n🚀 Generating paraphrases...")
    generation_result = gen_pipeline.generate_paraphrases(instruction, strategy="combined")
    
    if not generation_result:
        print("❌ No paraphrases generated")
        return
    
    positives = generation_result.get('positives', [])
    negatives = generation_result.get('negatives', [])
    
    print(f"\n📊 Generated:")
    print(f"  Positives: {len(positives)}")
    print(f"  Negatives: {len(negatives)}")
    
    # Debug each negative in detail
    for i, negative in enumerate(negatives):
        print(f"\n🔍 NEGATIVE {i+1} DEBUG:")
        print(f"  Text: {negative}")
        
        # Extract features
        neg_features = val_pipeline.extract_spatial_features(negative)
        print(f"  Features: {neg_features}")
        
        # Compute similarities
        embedding_sim = val_pipeline.compute_embedding_similarity(instruction, negative)
        direction_sim = val_pipeline._compute_direction_similarity(
            orig_features.get('directions', []), 
            neg_features.get('directions', [])
        )
        landmark_sim = val_pipeline._compute_landmark_similarity(
            orig_features.get('landmarks', []), 
            neg_features.get('landmarks', [])
        )
        
        print(f"  Embedding similarity: {embedding_sim:.3f}")
        print(f"  Direction similarity: {direction_sim:.3f}")
        print(f"  Landmark similarity: {landmark_sim:.3f}")
        
        # Check thresholds
        direction_changed = direction_sim < 0.7
        landmark_changed = landmark_sim < 0.7
        spatial_changed = direction_changed or landmark_changed
        embedding_ok = embedding_sim > 0.5
        
        print(f"  Direction changed: {direction_changed} (sim < 0.7: {direction_sim:.3f} < 0.7)")
        print(f"  Landmark changed: {landmark_changed} (sim < 0.7: {landmark_sim:.3f} < 0.7)")
        print(f"  Spatial changed: {spatial_changed}")
        print(f"  Embedding OK: {embedding_ok} (sim > 0.5: {embedding_sim:.3f} > 0.5)")
        
        # Final validation
        is_valid = embedding_ok and spatial_changed
        print(f"  VALID: {is_valid} = {embedding_ok} AND {spatial_changed}")
        
        if not is_valid:
            print(f"  ❌ FAILED because:")
            if not embedding_ok:
                print(f"     - Embedding similarity too low: {embedding_sim:.3f} <= 0.5")
            if not spatial_changed:
                print(f"     - No spatial changes detected:")
                print(f"       - Direction unchanged: {direction_sim:.3f} >= 0.7")
                print(f"       - Landmark unchanged: {landmark_sim:.3f} >= 0.7")
        else:
            print(f"  ✅ PASSED validation")
    
    # Debug each positive for comparison
    for i, positive in enumerate(positives):
        print(f"\n🔍 POSITIVE {i+1} DEBUG:")
        print(f"  Text: {positive}")
        
        # Validate using positive logic
        pos_result = val_pipeline.validate_positive_paraphrase(instruction, positive)
        print(f"  Valid: {pos_result['is_valid']}")
        print(f"  Embedding: {pos_result['embedding_similarity']:.3f}")
        print(f"  Direction: {pos_result['direction_similarity']:.3f}")
        print(f"  Landmark: {pos_result['landmark_similarity']:.3f}")

def suggest_validation_fixes():
    """Suggest fixes based on debug results."""
    print("\n💡 VALIDATION FIX SUGGESTIONS:")
    print("="*60)
    
    print("If negatives are failing because:")
    print("1. Embedding similarity too low → Reduce threshold from 0.5 to 0.3")
    print("2. No spatial changes detected → Check generation prompts are creating real changes")
    print("3. Direction/landmark similarity too high → Improve change detection logic")
    print("4. Feature extraction missing changes → Improve spatial feature patterns")
    
    print("\nQuick fixes to try:")
    print("- Lower embedding threshold: 0.5 → 0.3")
    print("- More lenient spatial change detection: 0.7 → 0.8")
    print("- Better negative generation prompts")
    print("- Expand spatial feature vocabulary")

if __name__ == "__main__":
    debug_negative_validation()
    suggest_validation_fixes() 