#!/usr/bin/env python3
"""
Test New Pipeline Approaches
Compare different pipeline implementations to understand parallelization issues.
"""

import time
import sys
from typing import List, Dict

def load_test_instructions() -> List[str]:
    """Load test instructions for pipeline testing."""
    try:
        from test_two_pipeline_architecture import load_random_avdn_examples
        return load_random_avdn_examples(num_examples=4)
    except ImportError:
        # Fallback test instructions if AVDN not available
        return [
            "Move towards your 3:30 direction. You will see a traffic circle and a small rectangular structure.",
            "Turn left at the building and proceed straight until you reach the parking lot.",
            "Head north for 100 meters, then turn right at the white house.",
            "Follow the road until you see a red building on your right side."
        ]

def test_sequential_pipeline():
    """Test the simple sequential pipeline (baseline)."""
    print("\n" + "="*80)
    print("🔄 TESTING SEQUENTIAL PIPELINE (BASELINE)")
    print("="*80)
    
    try:
        from simple_sequential_pipeline import SimpleSequentialPipeline
        
        instructions = load_test_instructions()
        print(f"📝 Testing with {len(instructions)} instructions")
        
        pipeline = SimpleSequentialPipeline()
        
        print("\n🔧 Initializing sequential pipeline...")
        init_start = time.time()
        if not pipeline.initialize():
            print("❌ Failed to initialize sequential pipeline")
            return None
        init_time = time.time() - init_start
        print(f"✅ Sequential pipeline initialized in {init_time:.1f}s")
        
        print("\n🚀 Starting sequential processing...")
        start_time = time.time()
        results = pipeline.process_instructions_sequential(instructions)
        total_time = time.time() - start_time
        
        if not results:
            print("❌ Sequential processing failed")
            return None
        
        successful = sum(1 for r in results if r.get('success', False))
        success_rate = successful / len(results) * 100
        avg_time = total_time / len(results)
        
        print(f"\n📊 SEQUENTIAL RESULTS:")
        print(f"  Success rate: {successful}/{len(results)} ({success_rate:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average per instruction: {avg_time:.1f}s")
        print(f"  Initialization time: {init_time:.1f}s")
        
        return {
            "name": "Sequential",
            "success_rate": success_rate,
            "total_time": total_time,
            "avg_time": avg_time,
            "init_time": init_time,
            "successful": successful,
            "total": len(results)
        }
        
    except Exception as e:
        print(f"❌ Sequential pipeline test failed: {e}")
        return None

def test_true_batch_pipeline():
    """Test the true batch processing pipeline."""
    print("\n" + "="*80)
    print("⚡ TESTING TRUE BATCH PIPELINE")
    print("="*80)
    
    try:
        from true_batch_processing_pipeline import TrueBatchProcessingPipeline
        
        instructions = load_test_instructions()
        print(f"📝 Testing with {len(instructions)} instructions")
        
        pipeline = TrueBatchProcessingPipeline(batch_size=4)
        
        print("\n🔧 Initializing true batch pipeline...")
        init_start = time.time()
        if not pipeline.initialize():
            print("❌ Failed to initialize true batch pipeline")
            return None
        init_time = time.time() - init_start
        print(f"✅ True batch pipeline initialized in {init_time:.1f}s")
        
        print("\n⚡ Starting TRUE BATCH processing...")
        start_time = time.time()
        results = pipeline.process_instructions_true_batch(instructions)
        total_time = time.time() - start_time
        
        if not results:
            print("❌ True batch processing failed")
            return None
        
        successful = sum(1 for r in results if r.get('success', False))
        success_rate = successful / len(results) * 100
        avg_time = total_time / len(results)
        
        print(f"\n📊 TRUE BATCH RESULTS:")
        print(f"  Success rate: {successful}/{len(results)} ({success_rate:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average per instruction: {avg_time:.1f}s")
        print(f"  Initialization time: {init_time:.1f}s")
        
        return {
            "name": "True Batch",
            "success_rate": success_rate,
            "total_time": total_time,
            "avg_time": avg_time,
            "init_time": init_time,
            "successful": successful,
            "total": len(results)
        }
        
    except Exception as e:
        print(f"❌ True batch pipeline test failed: {e}")
        return None

def test_fake_batch_pipeline():
    """Test the fake batch processing pipeline (current implementation)."""
    print("\n" + "="*80)
    print("🔄 TESTING FAKE BATCH PIPELINE (CURRENT)")
    print("="*80)
    
    try:
        from batch_processing_pipeline import BatchProcessingPipeline
        
        instructions = load_test_instructions()
        print(f"📝 Testing with {len(instructions)} instructions")
        
        pipeline = BatchProcessingPipeline(batch_size=4)
        
        print("\n🔧 Initializing fake batch pipeline...")
        init_start = time.time()
        if not pipeline.initialize():
            print("❌ Failed to initialize fake batch pipeline")
            return None
        init_time = time.time() - init_start
        print(f"✅ Fake batch pipeline initialized in {init_time:.1f}s")
        
        print("\n🔄 Starting FAKE BATCH processing (actually sequential)...")
        start_time = time.time()
        results = pipeline.process_instructions_batch(instructions)
        total_time = time.time() - start_time
        
        if not results:
            print("❌ Fake batch processing failed")
            return None
        
        successful = sum(1 for r in results if r.get('success', False))
        success_rate = successful / len(results) * 100
        avg_time = total_time / len(results)
        
        print(f"\n📊 FAKE BATCH RESULTS:")
        print(f"  Success rate: {successful}/{len(results)} ({success_rate:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average per instruction: {avg_time:.1f}s")
        print(f"  Initialization time: {init_time:.1f}s")
        print(f"  ⚠️  Note: This is actually sequential processing, not true batching")
        
        return {
            "name": "Fake Batch",
            "success_rate": success_rate,
            "total_time": total_time,
            "avg_time": avg_time,
            "init_time": init_time,
            "successful": successful,
            "total": len(results)
        }
        
    except Exception as e:
        print(f"❌ Fake batch pipeline test failed: {e}")
        return None

def compare_pipeline_results(results: List[Dict]):
    """Compare results from different pipeline approaches."""
    print("\n" + "="*80)
    print("📊 PIPELINE COMPARISON")
    print("="*80)
    
    if not results:
        print("❌ No results to compare")
        return
    
    print(f"{'Pipeline':<15} {'Success Rate':<12} {'Total Time':<12} {'Avg Time':<12} {'Init Time':<12} {'Speedup':<10}")
    print("-" * 80)
    
    baseline_time = None
    for result in results:
        if result["name"] == "Sequential":
            baseline_time = result["total_time"]
            break
    
    for result in results:
        speedup = "1.0x" if not baseline_time else f"{baseline_time/result['total_time']:.1f}x"
        print(f"{result['name']:<15} {result['success_rate']:<12.1f}% {result['total_time']:<12.1f}s "
              f"{result['avg_time']:<12.1f}s {result['init_time']:<12.1f}s {speedup:<10}")
    
    print("\n🎯 Analysis:")
    
    # Find best success rate
    best_success = max(results, key=lambda x: x['success_rate'])
    print(f"  🏆 Best success rate: {best_success['name']} ({best_success['success_rate']:.1f}%)")
    
    # Find fastest
    fastest = min(results, key=lambda x: x['total_time'])
    print(f"  ⚡ Fastest processing: {fastest['name']} ({fastest['total_time']:.1f}s)")
    
    # Find most efficient (best avg time)
    most_efficient = min(results, key=lambda x: x['avg_time'])
    print(f"  📈 Most efficient: {most_efficient['name']} ({most_efficient['avg_time']:.1f}s per instruction)")
    
    # True parallelization analysis
    print(f"\n💡 Parallelization Analysis:")
    for result in results:
        if "Batch" in result['name']:
            if result['avg_time'] < baseline_time / 2:  # Assuming 2x speedup indicates some parallelization
                print(f"  ✅ {result['name']}: Shows genuine parallelization benefits")
            else:
                print(f"  ❌ {result['name']}: No real parallelization (similar to sequential)")

def main():
    """Run comprehensive pipeline comparison."""
    print("🚀 COMPREHENSIVE PIPELINE TESTING")
    print("Testing different approaches to understand parallelization issues")
    print("="*80)
    
    results = []
    
    # Test 1: Sequential (baseline)
    sequential_result = test_sequential_pipeline()
    if sequential_result:
        results.append(sequential_result)
    
    # Test 2: True batch processing
    batch_result = test_true_batch_pipeline()
    if batch_result:
        results.append(batch_result)
    
    # Test 3: Fake batch processing (current implementation)
    fake_batch_result = test_fake_batch_pipeline()
    if fake_batch_result:
        results.append(fake_batch_result)
    
    # Compare all results
    compare_pipeline_results(results)
    
    # Final recommendations
    print(f"\n" + "="*80)
    print("🎯 RECOMMENDATIONS")
    print("="*80)
    
    if len(results) >= 2:
        print("Based on testing results:")
        print("1. If you see similar times across all approaches → GPU memory/model size is the bottleneck")
        print("2. If true batch is faster → Model supports batch inference effectively") 
        print("3. If sequential is most reliable → Focus on optimizing single-instruction processing")
        print("4. If all fail → Check GPU memory usage and model loading issues")
    else:
        print("❌ Not enough successful tests to make recommendations")
        print("🔧 Debugging needed for pipeline initialization or model loading")
    
    print(f"\n⚡ Next Steps:")
    print("- Use the most reliable approach for production")
    print("- Focus on optimizing model inference rather than parallelization")
    print("- Consider reducing model size if memory is the constraint")

if __name__ == "__main__":
    main() 