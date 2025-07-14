#!/usr/bin/env python3
"""
Test Fixed Validation
Quick test to verify that the validation threshold fixes improve negative validation success.
"""

def test_fixed_validation():
    """Test the fixed validation thresholds."""
    print("🧪 TESTING FIXED VALIDATION THRESHOLDS")
    print("="*60)
    
    # Test the simple sequential pipeline with fixed validation
    try:
        from simple_sequential_pipeline import SimpleSequentialPipeline
        
        # Use a smaller test set for quick validation
        test_instructions = [
            "Go in the 3 o'clock direction from your current position.",
            "Just move forward straight. You will see few very long structures."
        ]
        
        print(f"📝 Testing with {len(test_instructions)} instructions")
        
        pipeline = SimpleSequentialPipeline()
        
        print("\n🔧 Initializing pipeline...")
        if not pipeline.initialize():
            print("❌ Failed to initialize pipeline")
            return False
        print("✅ Pipeline initialized")
        
        print("\n🚀 Testing with fixed validation thresholds...")
        results = pipeline.process_instructions_sequential(test_instructions)
        
        if not results:
            print("❌ No results returned")
            return False
        
        # Analyze results
        successful = sum(1 for r in results if r.get('success', False))
        total_positives = sum(len(r.get('positives', [])) for r in results if r.get('success'))
        total_negatives = sum(len(r.get('negatives', [])) for r in results if r.get('success'))
        
        print(f"\n📊 FIXED VALIDATION RESULTS:")
        print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        print(f"  Total valid positives: {total_positives}")
        print(f"  Total valid negatives: {total_negatives}")
        
        # Show improvement
        if total_negatives > 0:
            print(f"  ✅ IMPROVEMENT: Negative validation now working!")
            print(f"     - Previous: 0 negatives consistently")
            print(f"     - Current: {total_negatives} negatives")
        else:
            print(f"  ⚠️  Still no negatives - may need further adjustments")
        
        # Show sample results
        for i, result in enumerate(results):
            if result.get('success'):
                print(f"\n📝 Successful instruction {i+1}:")
                print(f"  Positives: {len(result.get('positives', []))}")
                print(f"  Negatives: {len(result.get('negatives', []))}")
                if result.get('negatives'):
                    print(f"  Sample negative: {result['negatives'][0][:50]}...")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_true_batch_fixed():
    """Test the true batch pipeline with fixed tokenizer."""
    print("\n🧪 TESTING TRUE BATCH WITH FIXES")
    print("="*60)
    
    try:
        from true_batch_processing_pipeline import TrueBatchProcessingPipeline
        
        # Use just 2 instructions for quick test
        test_instructions = [
            "Go in the 3 o'clock direction from your current position.",
            "Just move forward straight. You will see few very long structures."
        ]
        
        pipeline = TrueBatchProcessingPipeline(batch_size=2)
        
        print("🔧 Initializing true batch pipeline...")
        if not pipeline.initialize():
            print("❌ Failed to initialize true batch pipeline")
            return False
        print("✅ True batch pipeline initialized")
        
        print("\n⚡ Testing TRUE BATCH processing with fixes...")
        results = pipeline.process_instructions_true_batch(test_instructions)
        
        if not results:
            print("❌ True batch processing failed")
            return False
        
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n📊 TRUE BATCH FIXED RESULTS:")
        print(f"  Success rate: {successful}/{len(results)} ({successful/len(results)*100:.1f}%)")
        
        if successful > 0:
            print("  ✅ TRUE BATCH NOW WORKING!")
        else:
            print("  ❌ True batch still has issues")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ True batch test failed: {e}")
        return False

def main():
    """Run both tests to validate fixes."""
    print("🚀 TESTING VALIDATION FIXES")
    print("Testing fixes for:")
    print("1. Negative validation thresholds (0.5→0.3 embedding, 0.7→0.8 spatial)")
    print("2. True batch tokenizer padding issue")
    print("3. Improved negative generation prompts")
    print("="*80)
    
    # Test 1: Sequential with fixed validation
    sequential_success = test_fixed_validation()
    
    # Test 2: True batch with fixes
    batch_success = test_true_batch_fixed()
    
    # Summary
    print(f"\n" + "="*80)
    print("🎯 VALIDATION FIX RESULTS:")
    print(f"  Sequential pipeline: {'✅ IMPROVED' if sequential_success else '❌ Still issues'}")
    print(f"  True batch pipeline: {'✅ FIXED' if batch_success else '❌ Still issues'}")
    
    if sequential_success or batch_success:
        print("\n🎉 SUCCESS: At least one pipeline is now working better!")
        print("💡 RECOMMENDATIONS:")
        if sequential_success:
            print("   - Use SimpleSequentialPipeline for reliable processing")
        if batch_success:
            print("   - Use TrueBatchProcessingPipeline for faster parallel processing")
    else:
        print("\n⚠️  STILL ISSUES: May need further debugging")
        print("💡 NEXT STEPS:")
        print("   - Check model loading on headless server")
        print("   - Investigate negative generation quality")
        print("   - Consider even more lenient thresholds")

if __name__ == "__main__":
    main() 