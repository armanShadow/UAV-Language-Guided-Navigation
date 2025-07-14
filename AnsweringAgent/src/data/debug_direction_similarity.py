#!/usr/bin/env python3
"""
Debug script to trace the direction similarity computation bug.
"""

from validation_pipeline import ValidationPipeline
import re

def debug_direction_similarity():
    """Debug the direction similarity computation for clock directions."""
    
    val_pipeline = ValidationPipeline()
    
    # Test case from diagnostic
    original = "Go in the 3 o'clock direction from your current position."
    negative = "From your current location, proceed in the 9 o'clock direction."
    
    print("🔍 DEBUGGING DIRECTION SIMILARITY")
    print("=" * 50)
    print(f"Original: {original}")
    print(f"Negative: {negative}")
    print()
    
    # Debug the spatial feature extraction step by step
    print("🔧 DEBUGGING SPATIAL FEATURE EXTRACTION:")
    print("Direction regex patterns:", val_pipeline.spatial_features['directions']['regex_patterns'])
    print("Direction string patterns:", val_pipeline.spatial_features['directions']['string_patterns'])
    print()
    
    # Test each pattern manually
    orig_lower = original.lower()
    neg_lower = negative.lower()
    
    print("Testing REGEX patterns against original text:")
    for i, pattern in enumerate(val_pipeline.spatial_features['directions']['regex_patterns']):
        matches = re.findall(pattern, orig_lower)
        print(f"  Pattern {i}: '{pattern}' → {matches}")
    
    print("\nTesting STRING patterns against original text:")
    for i, pattern in enumerate(val_pipeline.spatial_features['directions']['string_patterns']):
        match = re.search(r'\b' + re.escape(pattern) + r'\b', orig_lower)
        print(f"  Pattern {i}: '{pattern}' → {bool(match)}")
    
    print("\nTesting REGEX patterns against negative text:")
    for i, pattern in enumerate(val_pipeline.spatial_features['directions']['regex_patterns']):
        matches = re.findall(pattern, neg_lower)
        print(f"  Pattern {i}: '{pattern}' → {matches}")
    
    print("\nTesting STRING patterns against negative text:")
    for i, pattern in enumerate(val_pipeline.spatial_features['directions']['string_patterns']):
        match = re.search(r'\b' + re.escape(pattern) + r'\b', neg_lower)
        print(f"  Pattern {i}: '{pattern}' → {bool(match)}")
    
    print()
    
    # Extract spatial features
    orig_features = val_pipeline.extract_spatial_features(original)
    neg_features = val_pipeline.extract_spatial_features(negative)
    
    print("📊 EXTRACTED FEATURES:")
    print(f"Original directions: {orig_features.get('directions', [])}")
    print(f"Negative directions: {neg_features.get('directions', [])}")
    print()
    
    # Test direction similarity step by step
    orig_dirs = orig_features.get('directions', [])
    neg_dirs = neg_features.get('directions', [])
    
    print("🔧 STEP-BY-STEP SIMILARITY COMPUTATION:")
    
    # Extract clock hours
    orig_clock_hours = set()
    neg_clock_hours = set()
    
    for direction in orig_dirs:
        clock_hour = val_pipeline._extract_clock_hour(direction)
        print(f"  Original '{direction}' → clock hour: {clock_hour}")
        if clock_hour:
            orig_clock_hours.add(clock_hour)
    
    for direction in neg_dirs:
        clock_hour = val_pipeline._extract_clock_hour(direction)
        print(f"  Negative '{direction}' → clock hour: {clock_hour}")
        if clock_hour:
            neg_clock_hours.add(clock_hour)
    
    print(f"Original clock hours: {orig_clock_hours}")
    print(f"Negative clock hours: {neg_clock_hours}")
    
    # Clock similarity
    clock_similarity = 1.0 if orig_clock_hours == neg_clock_hours else 0.0
    print(f"Clock similarity: {clock_similarity}")
    
    # Synonym matching
    synonym_matches = 0
    total_directions = len(orig_dirs)
    
    for orig_dir in orig_dirs:
        synonym_match = val_pipeline._find_direction_synonym_match(orig_dir, neg_dirs)
        print(f"  '{orig_dir}' has synonym match in {neg_dirs}: {synonym_match}")
        if synonym_match:
            synonym_matches += 1
    
    synonym_similarity = synonym_matches / total_directions if total_directions > 0 else 0.0
    print(f"Synonym similarity: {synonym_similarity}")
    
    # Final similarity (max of clock and synonym)
    final_similarity = max(clock_similarity, synonym_similarity)
    print(f"Final similarity: max({clock_similarity}, {synonym_similarity}) = {final_similarity}")
    
    # Compare with actual method
    actual_similarity = val_pipeline._compute_direction_similarity(orig_dirs, neg_dirs)
    print(f"Actual method result: {actual_similarity}")
    
    if final_similarity != actual_similarity:
        print("⚠️  BUG DETECTED: Manual computation differs from method!")
    else:
        print("✅ Manual computation matches method")

if __name__ == "__main__":
    debug_direction_similarity() 