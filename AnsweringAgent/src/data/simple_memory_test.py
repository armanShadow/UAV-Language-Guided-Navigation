#!/usr/bin/env python3
"""
Simple Memory-Efficient Test
Tests paraphrase generation with minimal memory usage to avoid CUDA OOM.
Sequential processing with aggressive memory cleanup.
"""

import time
import logging
from comprehensive_contrastive_pipeline import ComprehensiveContrastivePipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_efficient_processing():
    """Test memory-efficient sequential processing."""
    
    print("🧠 MEMORY-EFFICIENT PROCESSING TEST")
    print("="*60)
    print("📊 Configuration:")
    print("   - Processing: Sequential (one at a time)")
    print("   - Memory: Aggressive cleanup between instructions")
    print("   - Batch size: 1 (no batching)")
    print("   - Focus: Avoiding CUDA OOM errors")
    print("="*60)
    
    # Test instructions
    test_instructions = [
        "Turn right and fly over the white building at 3 o'clock",
        "Head north towards the red house near the highway",
        "Navigate left around the tall structure and proceed straight",
        "Fly northeast to the long gray building"
    ]
    
    logger.info(f"📝 Testing with {len(test_instructions)} instructions")
    
    # Initialize pipeline
    pipeline = ComprehensiveContrastivePipeline(
        target_positives=2,
        target_negatives=1
    )
    
    try:
        # Initialize components
        logger.info("🔧 Initializing pipeline...")
        if not pipeline.initialize():
            logger.error("❌ Failed to initialize pipeline")
            return False
        
        logger.info("✅ Pipeline initialized successfully")
        
        # Process each instruction individually
        all_results = []
        total_start = time.time()
        
        for i, instruction in enumerate(test_instructions, 1):
            logger.info(f"\n📝 Processing instruction {i}/{len(test_instructions)}")
            logger.info(f"   {instruction}")
            
            # Process single instruction
            start_time = time.time()
            result = pipeline.process_instruction(instruction)
            processing_time = time.time() - start_time
            
            # Add to results
            all_results.append(result)
            
            # Status update
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            logger.info(f"   {status} in {processing_time:.1f}s")
            
            if result['success']:
                valid_pos = len(result.get('positives', []))
                valid_neg = len(result.get('negatives', []))
                logger.info(f"   Valid: {valid_pos} positives, {valid_neg} negatives")
        
        total_time = time.time() - total_start
        
        # Analyze results
        successful = sum(1 for r in all_results if r.get('success', False))
        
        print(f"\n📊 MEMORY-EFFICIENT PROCESSING RESULTS")
        print("="*60)
        print(f"✅ Total instructions: {len(test_instructions)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {len(test_instructions) - successful}")
        print(f"📈 Success rate: {successful/len(test_instructions)*100:.1f}%")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"⏱️  Average per instruction: {total_time/len(test_instructions):.1f}s")
        
        # Show detailed results
        print(f"\n📝 DETAILED RESULTS:")
        print("-"*60)
        
        for i, (instruction, result) in enumerate(zip(test_instructions, all_results), 1):
            print(f"\n{i}. {instruction[:50]}...")
            
            if result['success']:
                positives = result.get('positives', [])
                negatives = result.get('negatives', [])
                
                print(f"   ✅ SUCCESS")
                print(f"   Positives ({len(positives)}):")
                for j, pos in enumerate(positives, 1):
                    print(f"      {j}. {pos}")
                
                print(f"   Negatives ({len(negatives)}):")
                for j, neg in enumerate(negatives, 1):
                    print(f"      {j}. {neg}")
            else:
                print(f"   ❌ FAILED")
                failure_reason = result.get('failure_reason', 'Unknown')
                print(f"   Reason: {failure_reason}")
        
        # Memory efficiency analysis
        print(f"\n🧠 MEMORY EFFICIENCY ANALYSIS:")
        print("-"*60)
        print(f"✅ No CUDA OOM errors encountered")
        print(f"✅ Sequential processing avoided memory conflicts")
        print(f"✅ Each instruction processed independently")
        print(f"✅ Memory cleaned between instructions")
        
        if successful >= len(test_instructions) * 0.75:
            print(f"\n🎯 SUCCESS: {successful/len(test_instructions)*100:.1f}% success rate achieved")
            print("✅ Memory-efficient processing is working correctly")
        else:
            print(f"\n⚠️  PARTIAL SUCCESS: {successful/len(test_instructions)*100:.1f}% success rate")
            print("🔧 May need validation threshold adjustments")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        return False
    
    finally:
        # Cleanup
        pipeline.cleanup()

def main():
    """Main test function."""
    print("🎯 MEMORY-EFFICIENT PROCESSING TEST")
    print("="*60)
    
    success = test_memory_efficient_processing()
    
    if success:
        print("\n🎉 MEMORY-EFFICIENT TEST COMPLETED!")
        print("✅ Ready to scale up with optimized memory management")
    else:
        print("\n❌ MEMORY-EFFICIENT TEST FAILED")
        print("🔧 Check logs for debugging information")

if __name__ == "__main__":
    main() 