#!/usr/bin/env python3
"""
Diagnostic Negative Generation
See exactly what negatives are being generated and why they fail validation.
"""

def diagnose_negative_generation():
    """Diagnose what's happening with negative generation and validation."""
    print("🔬 DIAGNOSTIC: NEGATIVE GENERATION ANALYSIS")
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
    
    # Test with a simple, clear instruction
    test_instruction = "Go in the 3 o'clock direction from your current position."
    
    print(f"\n📝 Test instruction: {test_instruction}")
    
    # Extract original features
    orig_features = val_pipeline.extract_spatial_features(test_instruction)
    print(f"🔍 Original features: {orig_features}")
    
    # Generate paraphrases
    print(f"\n🚀 Generating paraphrases...")
    generation_result = gen_pipeline.generate_paraphrases(test_instruction, strategy="combined")
    
    if not generation_result:
        print("❌ No paraphrases generated")
        return
    
    positives = generation_result.get('positives', [])
    negatives = generation_result.get('negatives', [])
    
    print(f"\n📊 Generated paraphrases:")
    print(f"  Positives: {len(positives)}")
    for i, pos in enumerate(positives):
        print(f"    {i+1}. {pos}")
    
    print(f"  Negatives: {len(negatives)}")
    for i, neg in enumerate(negatives):
        print(f"    {i+1}. {neg}")
    
    # Analyze each negative in detail
    for i, negative in enumerate(negatives):
        print(f"\n🔬 DETAILED ANALYSIS - NEGATIVE {i+1}")
        print(f"Original: {test_instruction}")
        print(f"Negative: {negative}")
        
        # Check if they're actually different
        if negative.strip().lower() == test_instruction.strip().lower():
            print("❌ IDENTICAL: Negative is exactly the same as original!")
            continue
        
        # Extract features
        neg_features = val_pipeline.extract_spatial_features(negative)
        print(f"Original features: {orig_features}")
        print(f"Negative features: {neg_features}")
        
        # Check what changed
        direction_changes = []
        landmark_changes = []
        
        orig_dirs = orig_features.get('directions', [])
        neg_dirs = neg_features.get('directions', [])
        if set(orig_dirs) != set(neg_dirs):
            direction_changes.append(f"{orig_dirs} → {neg_dirs}")
        
        orig_landmarks = orig_features.get('landmarks', [])
        neg_landmarks = neg_features.get('landmarks', [])
        if set(orig_landmarks) != set(neg_landmarks):
            landmark_changes.append(f"{orig_landmarks} → {neg_landmarks}")
        
        print(f"Direction changes: {direction_changes if direction_changes else 'NONE'}")
        print(f"Landmark changes: {landmark_changes if landmark_changes else 'NONE'}")
        
        # Compute similarities
        embedding_sim = val_pipeline.compute_embedding_similarity(test_instruction, negative)
        direction_sim = val_pipeline._compute_direction_similarity(orig_dirs, neg_dirs)
        landmark_sim = val_pipeline._compute_landmark_similarity(orig_landmarks, neg_landmarks)
        
        print(f"Similarities:")
        print(f"  Embedding: {embedding_sim:.3f}")
        print(f"  Direction: {direction_sim:.3f}")
        print(f"  Landmark: {landmark_sim:.3f}")
        
        # Check current validation logic
        direction_changed = direction_sim < 0.9
        landmark_changed = landmark_sim < 0.9
        spatial_changed = direction_changed or landmark_changed
        embedding_ok = embedding_sim > 0.1
        
        print(f"Validation checks:")
        print(f"  Direction changed: {direction_changed} (similarity {direction_sim:.3f} < 0.9)")
        print(f"  Landmark changed: {landmark_changed} (similarity {landmark_sim:.3f} < 0.9)")
        print(f"  Spatial changed: {spatial_changed}")
        print(f"  Embedding OK: {embedding_ok} (similarity {embedding_sim:.3f} > 0.1)")
        
        is_valid = embedding_ok and spatial_changed
        print(f"FINAL RESULT: {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        if not is_valid:
            print("❌ Failure reasons:")
            if not embedding_ok:
                print(f"   - Embedding too low: {embedding_sim:.3f} ≤ 0.1")
            if not spatial_changed:
                print(f"   - No spatial changes: direction={direction_sim:.3f}≥0.9, landmark={landmark_sim:.3f}≥0.9")

def suggest_fixes():
    """Suggest specific fixes based on diagnostic results."""
    print(f"\n💡 DIAGNOSTIC-BASED FIXES:")
    print("="*60)
    
    print("Common issues and fixes:")
    print("1. Negatives identical to original → Improve generation prompts")
    print("2. No spatial feature changes detected → Expand feature vocabulary")
    print("3. Similarities too high → Check feature extraction logic")
    print("4. Embedding similarity too low → Lower threshold or improve generation")
    
    print("\nImmediate fixes to try:")
    print("- Make validation accept ANY text difference (bypass spatial requirements)")
    print("- Improve negative generation with more explicit change requirements")
    print("- Add debug logging to validation pipeline")

if __name__ == "__main__":
    diagnose_negative_generation()
    suggest_fixes() 