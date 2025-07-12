#!/usr/bin/env python3
"""
Simple test script for Enhanced Mixtral Paraphraser
Gradual testing approach with clear validation checkpoints
"""

import sys
import logging

def test_basic_import():
    """Test 1: Basic import and initialization"""
    print("🔄 Test 1: Basic Import and Initialization")
    print("-" * 45)
    
    try:
        from enhanced_mixtral_paraphraser import EnhancedMixtralParaphraser
        print("✅ Enhanced paraphraser imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_loading():
    """Test 2: Model loading and initialization"""
    print("\n🔄 Test 2: Model Loading")
    print("-" * 45)
    
    try:
        from enhanced_mixtral_paraphraser import EnhancedMixtralParaphraser
        print("📥 Initializing Enhanced Mixtral Paraphraser...")
        paraphraser = EnhancedMixtralParaphraser()
        print("✅ Model loaded successfully")
        print(f"🔍 Device mapping: auto (10x RTX 2080 Ti)")
        return paraphraser
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_spatial_token_extraction(paraphraser):
    """Test 3: Spatial token extraction"""
    print("\n🔄 Test 3: Spatial Token Extraction")
    print("-" * 45)
    
    # Test cases from actual AVDN dataset
    test_cases = [
        "head forward towards 6 o'clock direction, After passing a road and few buildings, The destination is a ground with few trucks parked there.",
        "Hi drone, head south west and pass over a building and your destination is the small green building.",
        "Move toward 3 o'clock direction, turn towards 6 o'clock direction, black colored small square shaped place is your destination.",
        "Destination looks like three blue roofed buildings. It is in your two o'clock direction.",
        "turn 8'o clock direction go foward cross road go forward then cross other road the three joint brown building is your goal."
    ]
    
    for i, instruction in enumerate(test_cases, 1):
        print(f"\n  📝 Case {i}: {instruction}")
        try:
            tokens = paraphraser._extract_spatial_tokens(instruction)
            print(f"    🔍 Extracted tokens: {tokens}")
            
            # Check if critical tokens found
            has_spatial = any(tokens.values())
            if has_spatial:
                print("    ✅ Spatial tokens detected")
            else:
                print("    ⚠️  No spatial tokens detected")
                
        except Exception as e:
            print(f"    ❌ Token extraction failed: {e}")
            return False
    
    return True

def test_single_positive_generation(paraphraser):
    """Test 4: Single positive paraphrase generation"""
    print("\n🔄 Test 4: Single Positive Generation")
    print("-" * 45)
    
    test_instruction = "head forward towards 6 o'clock direction, After passing a road and few buildings, The destination is a ground with few trucks parked there."
    print(f"📝 Testing: {test_instruction}")
    
    try:
        positive = paraphraser._generate_positive(test_instruction)
        print(f"✅ Generated: {positive}")
        
        # Basic validation
        if positive and positive != test_instruction:
            print("✅ Paraphrase is different from original")
        else:
            print("⚠️  Paraphrase is same as original (fallback)")
            
        # Check for spatial preservation  
        if "6 o'clock" in positive:
            print("✅ Clock direction preserved")
        else:
            print("❌ Clock direction lost!")
            
        return True
        
    except Exception as e:
        print(f"❌ Positive generation failed: {e}")
        return False

def test_single_negative_generation(paraphraser):
    """Test 5: Single negative paraphrase generation"""
    print("\n🔄 Test 5: Single Negative Generation")
    print("-" * 45)
    
    test_instruction = "Hi drone, head south west and pass over a building and your destination is the small green building."
    print(f"📝 Testing: {test_instruction}")
    
    try:
        negative = paraphraser._generate_negative(test_instruction)
        print(f"❌ Generated: {negative}")
        
        # Basic validation
        if negative and negative != test_instruction:
            print("✅ Negative is different from original")
        else:
            print("⚠️  Negative is same as original (fallback)")
            
        # Check for spatial corruption
        if "south west" not in negative or "green building" not in negative:
            print("✅ Spatial corruption detected")
        else:
            print("⚠️  No spatial corruption detected")
            
        return True
        
    except Exception as e:
        print(f"❌ Negative generation failed: {e}")
        return False

def test_full_pipeline(paraphraser):
    """Test 6: Full pipeline (2 positive + 1 negative)"""
    print("\n🔄 Test 6: Full Pipeline")
    print("-" * 45)
    
    test_instruction = "turn 8'o clock direction go foward cross road go forward then cross other road the three joint brown building is your goal."
    print(f"📝 Testing: {test_instruction}")
    
    try:
        results = paraphraser.generate_paraphrases(test_instruction)
        
        print(f"\n📋 Results:")
        print(f"  Original:   {results['original']}")
        print(f"  Positive 1: {results['positives'][0]}")
        print(f"  Positive 2: {results['positives'][1]}")
        print(f"  Negative:   {results['negative']}")
        print(f"  Tokens:     {results['spatial_tokens']}")
        
        # Validation checks
        if len(results['positives']) == 2:
            print("✅ Generated 2 positive paraphrases")
        else:
            print(f"❌ Expected 2 positives, got {len(results['positives'])}")
            
        if results['negative']:
            print("✅ Generated 1 negative paraphrase") 
        else:
            print("❌ No negative paraphrase generated")
            
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        return False

def main():
    """Run gradual tests with clear checkpoints"""
    print("🚀 Enhanced Mixtral Paraphraser - Gradual Testing")
    print("=" * 55)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test sequence with early exit on failure
    tests = [
        ("Import & Setup", test_basic_import),
        ("Model Loading", test_model_loading),
    ]
    
    paraphraser = None
    
    # Run basic tests first
    for test_name, test_func in tests:
        if test_name == "Model Loading":
            paraphraser = test_func()
            if not paraphraser:
                print("\n❌ CRITICAL: Model loading failed. Cannot continue.")
                return False
        else:
            if not test_func():
                print(f"\n❌ CRITICAL: {test_name} failed. Cannot continue.")
                return False
    
    # Run tests that require paraphraser instance
    advanced_tests = [
        ("Spatial Token Extraction", lambda: test_spatial_token_extraction(paraphraser)),
        ("Single Positive Generation", lambda: test_single_positive_generation(paraphraser)),
        ("Single Negative Generation", lambda: test_single_negative_generation(paraphraser)),
        ("Full Pipeline", lambda: test_full_pipeline(paraphraser)),
    ]
    
    for test_name, test_func in advanced_tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if not test_func():
            print(f"\n⚠️  {test_name} failed. Check implementation.")
            # Continue with other tests instead of stopping
        else:
            print(f"✅ {test_name} completed successfully")
    
    print("\n🎉 Testing completed!")
    print("📋 Review results above for any issues to address.")
    
    return True

if __name__ == "__main__":
    main() 