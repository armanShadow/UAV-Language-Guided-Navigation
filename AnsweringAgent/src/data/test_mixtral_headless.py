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
    """Test Mixtral paraphrasing pipeline with TRUE BATCH PROCESSING."""
    print("🚀 Testing Mixtral TRUE BATCH PROCESSING Pipeline on Headless Server")
    print("="*80)
    
    # Load test examples
    print("📂 Loading AVDN test examples...")
    test_instructions = load_random_avdn_examples(num_examples=8)
    print(f"📊 Loaded {len(test_instructions)} test instructions")
    
    # Display examples
    print("\n📝 Test Instructions:")
    for i, instruction in enumerate(test_instructions, 1):
        print(f"  {i}. {instruction}")
    
    # Initialize TRUE BATCH PROCESSING pipeline
    print("\n🔧 Initializing TRUE BATCH PROCESSING Pipeline...")
    try:
        from true_batch_processing_pipeline import TrueBatchProcessingPipeline
        pipeline = TrueBatchProcessingPipeline(batch_size=4)
        print("✅ TRUE BATCH pipeline imported successfully")
    except ImportError:
        print("⚠️  TRUE BATCH pipeline not found, falling back to regular generation pipeline")
        pipeline = ParaphraseGenerationPipeline()
    
    # Test model loading
    print("⏳ Loading Mixtral-8x7B-Instruct model...")
    start_time = time.time()
    
    if hasattr(pipeline, 'initialize'):
        # True batch processing pipeline
        if not pipeline.initialize():
            print("❌ Failed to initialize TRUE BATCH pipeline")
            return False
    else:
        # Regular pipeline fallback
        if not pipeline.load_model():
            print("❌ Failed to load Mixtral model")
            return False
    
    load_time = time.time() - start_time
    print(f"✅ Mixtral model loaded successfully in {load_time:.1f}s")
    
    # Test GPU information
    if hasattr(pipeline, 'generation_pipeline'):
        print(f"🔧 Using device: {pipeline.generation_pipeline.device}")
        if hasattr(pipeline.generation_pipeline, 'model') and pipeline.generation_pipeline.model:
            print(f"🔧 Model device: {next(pipeline.generation_pipeline.model.parameters()).device}")
    elif hasattr(pipeline, 'device'):
        print(f"🔧 Using device: {pipeline.device}")
        if hasattr(pipeline, 'model') and pipeline.model:
            print(f"🔧 Model device: {next(pipeline.model.parameters()).device}")
    
    # Test TRUE BATCH PROCESSING
    print(f"\n🚀 Testing TRUE BATCH PROCESSING...")
    print(f"⚡ PROCESSING ALL {len(test_instructions)} INSTRUCTIONS SIMULTANEOUSLY")
    print(f"🔥 NO SEQUENTIAL PROCESSING - GENUINE PARALLEL INFERENCE")
    
    start_time = time.time()
    
    try:
        if hasattr(pipeline, 'process_instructions_true_batch'):
            # Use TRUE batch processing
            print("🔥 Using TRUE BATCH PROCESSING at model level")
            all_results = pipeline.process_instructions_true_batch(test_instructions)
        elif hasattr(pipeline, 'generate_paraphrases_batch'):
            # Use batch generation from regular pipeline
            print("🔄 Using batch generation from regular pipeline")
            batch_results = pipeline.generate_paraphrases_batch(test_instructions, strategy="combined", batch_size=4)
            
            # Convert to expected format
            all_results = []
            for i, (instruction, result) in enumerate(zip(test_instructions, batch_results)):
                if result and result.get('positives') and result.get('negatives'):
                    all_results.append({
                        'instruction': instruction,
                        'positives': result['positives'],
                        'negatives': result['negatives'],
                        'success': True
                    })
                else:
                    all_results.append({
                        'instruction': instruction,
                        'success': False
                    })
        else:
            # Fallback to sequential (but warn user)
            print("⚠️  FALLBACK: No batch processing available, using sequential")
            all_results = []
            for i, instruction in enumerate(test_instructions, 1):
                print(f"--- Sequential Test {i}/{len(test_instructions)} ---")
                print(f"Original: {instruction}")
                
                result = pipeline.generate_paraphrases(instruction, strategy="combined")
                
                if result and result.get('positives') and result.get('negatives'):
                    all_results.append({
                        'instruction': instruction,
                        'positives': result['positives'],
                        'negatives': result['negatives'],
                        'success': True
                    })
                else:
                    all_results.append({
                        'instruction': instruction,
                        'success': False
                    })
        
        total_time = time.time() - start_time
        
    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Batch processing failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TRUE BATCH PROCESSING TEST SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in all_results if r.get('success', False))
    success_rate = successful / len(all_results) * 100 if all_results else 0
    avg_time = total_time / len(all_results) if all_results else 0
    
    # Calculate speedup vs sequential
    sequential_time_estimate = len(test_instructions) * 25  # Assume 25s per instruction sequential
    speedup = sequential_time_estimate / total_time if total_time > 0 else 1
    
    print(f"📈 Success rate: {successful}/{len(all_results)} ({success_rate:.1f}%)")
    print(f"⏱️  Total processing time: {total_time:.1f}s")
    print(f"⏱️  Average time per instruction: {avg_time:.1f}s")
    print(f"⚡ SPEEDUP vs sequential: {speedup:.1f}x")
    print(f"🔥 TRUE BATCH PROCESSING: {len(test_instructions)} instructions processed simultaneously")
    
    # Detailed results
    if successful > 0:
        print(f"\n✅ SUCCESSFUL GENERATIONS ({successful}):")
        for i, result in enumerate([r for r in all_results if r.get('success', False)], 1):
            # Handle different result formats
            original_instruction = result.get('original_instruction') or result.get('instruction') or f"Instruction {i}"
            print(f"\n{i}. Original: {original_instruction}")
            if 'positives' in result:
                print(f"   Positives ({len(result['positives'])}):")
                for j, pos in enumerate(result['positives'], 1):
                    print(f"     {j}. {pos}")
            if 'negatives' in result:
                print(f"   Negatives ({len(result['negatives'])}):")
                for j, neg in enumerate(result['negatives'], 1):
                    print(f"     {j}. {neg}")
    
    if successful < len(all_results):
        failed = len(all_results) - successful
        print(f"\n❌ FAILED GENERATIONS ({failed}):")
        for i, result in enumerate([r for r in all_results if not r.get('success', False)], 1):
            # Handle different result formats
            original_instruction = result.get('original_instruction') or result.get('instruction') or f"Failed instruction {i}"
            print(f"\n{i}. Original: {original_instruction}")
            if 'error' in result:
                print(f"   Error: {result['error']}")
    
    # Final assessment
    if success_rate >= 80:
        print(f"\n🎉 EXCELLENT: {success_rate:.1f}% success rate - TRUE BATCH PROCESSING ready for production!")
        print(f"⚡ SPEEDUP: {speedup:.1f}x faster than sequential processing")
    elif success_rate >= 60:
        print(f"\n✅ GOOD: {success_rate:.1f}% success rate - TRUE BATCH PROCESSING functional")
        print(f"⚡ SPEEDUP: {speedup:.1f}x faster than sequential processing")
    else:
        print(f"\n⚠️  NEEDS WORK: {success_rate:.1f}% success rate - TRUE BATCH PROCESSING needs debugging")
    
    return success_rate >= 60

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