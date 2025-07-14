#!/usr/bin/env python3
"""
Debug script to test direction synonym matching specifically.
"""

from validation_pipeline import ValidationPipeline

def test_direction_synonym_bug():
    """Test the direction synonym matching bug."""
    
    val_pipeline = ValidationPipeline()
    
    # Test case that's failing
    original = "Turn right and fly over the white building."
    negative = "Turn left and fly over the gray building."
    
    print("🔍 DEBUGGING DIRECTION SYNONYM MATCHING")
    print("=" * 50)
    print(f"Original: {original}")
    print(f"Negative: {negative}")
    print()
    
    # Extract features
    orig_features = val_pipeline.extract_spatial_features(original)
    neg_features = val_pipeline.extract_spatial_features(negative)
    
    orig_dirs = orig_features.get('directions', [])
    neg_dirs = neg_features.get('directions', [])
    
    print(f"Original directions: {orig_dirs}")
    print(f"Negative directions: {neg_dirs}")
    print()
    
    # Test synonym matching step by step
    print("🔧 TESTING SYNONYM MATCHING:")
    synonyms = val_pipeline.spatial_features['directions']['synonyms']
    print(f"Available synonyms: {synonyms}")
    print()
    
    # Test each original direction
    for orig_dir in orig_dirs:
        print(f"Testing '{orig_dir}':")
        
        # Find which synonym group it belongs to
        found_group = None
        for base_dir, synonym_list in synonyms.items():
            if orig_dir.lower() in synonym_list:
                found_group = base_dir
                print(f"  Found in group '{base_dir}': {synonym_list}")
                break
        
        if not found_group:
            print(f"  Not found in any synonym group")
            continue
        
        # Test against negative directions
        for neg_dir in neg_dirs:
            print(f"  Testing against '{neg_dir}':")
            
            # Check if neg_dir matches any synonym in the same group
            synonym_list = synonyms[found_group]
            matches = [syn for syn in synonym_list if syn in neg_dir.lower()]
            print(f"    Matches in group: {matches}")
            
            # This is the actual logic from _find_direction_synonym_match
            has_match = any(syn in neg_dir.lower() for syn in synonym_list)
            print(f"    Has match: {has_match}")
    
    print()
    
    # Test the actual method
    print("🎯 ACTUAL METHOD RESULTS:")
    for orig_dir in orig_dirs:
        match_result = val_pipeline._find_direction_synonym_match(orig_dir, neg_dirs)
        print(f"  '{orig_dir}' matches {neg_dirs}: {match_result}")
    
    # Test overall similarity
    direction_sim = val_pipeline._compute_direction_similarity(orig_dirs, neg_dirs)
    print(f"\nOverall direction similarity: {direction_sim}")
    
    # Manual calculation
    synonym_matches = 0
    total_directions = len(orig_dirs)
    
    for orig_dir in orig_dirs:
        if val_pipeline._find_direction_synonym_match(orig_dir, neg_dirs):
            synonym_matches += 1
    
    manual_synonym_similarity = synonym_matches / total_directions if total_directions > 0 else 0.0
    print(f"Manual synonym similarity: {manual_synonym_similarity}")
    
    # Clock similarity should be 0 for this case
    orig_clock_hours = set()
    neg_clock_hours = set()
    
    for direction in orig_dirs:
        clock_hour = val_pipeline._extract_clock_hour(direction)
        if clock_hour:
            orig_clock_hours.add(clock_hour)
    
    for direction in neg_dirs:
        clock_hour = val_pipeline._extract_clock_hour(direction)
        if clock_hour:
            neg_clock_hours.add(clock_hour)
    
    # Use the corrected clock similarity logic
    if orig_clock_hours and neg_clock_hours:
        # Both have clock directions - compare them
        clock_similarity = 1.0 if orig_clock_hours == neg_clock_hours else 0.0
    elif not orig_clock_hours and not neg_clock_hours:
        # Neither has clock directions - no clock information to compare
        clock_similarity = 0.0
    else:
        # One has clock directions, other doesn't - different
        clock_similarity = 0.0
    
    print(f"Clock similarity: {clock_similarity}")
    
    manual_overall = max(clock_similarity, manual_synonym_similarity)
    print(f"Manual overall: max({clock_similarity}, {manual_synonym_similarity}) = {manual_overall}")
    
    if manual_overall != direction_sim:
        print("⚠️  BUG DETECTED: Manual calculation differs from method!")
    else:
        print("✅ Manual calculation matches method")

if __name__ == "__main__":
    test_direction_synonym_bug() 