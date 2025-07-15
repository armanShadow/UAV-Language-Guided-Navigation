#!/usr/bin/env python3
"""
Comprehensive AVDN Samples Test
Tests 4 real samples from AVDN dataset through the complete paraphrasing and validation pipeline.
"""

import logging
import json
import time
from pathlib import Path
from comprehensive_contrastive_pipeline import ComprehensiveContrastivePipeline

# Configure logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_avdn_samples(num_samples: int = 4) -> list:
    """Load real samples from AVDN dataset."""
    dataset_paths = [
        "processed_data/train_data.json",
        "src/data/processed_data/train_data.json",
        "AnsweringAgent/src/data/processed_data/train_data.json",
        "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json",
        "../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json"
    ]
    
    for path in dataset_paths:
        if Path(path).exists():
            logger.info(f"📂 Loading AVDN dataset from: {path}")
            try:
                with open(path, 'r') as f:
                    episodes = json.load(f)
                
                # Extract navigation instructions from dialogs
                instructions = []
                for episode in episodes[:20]:  # Look at first 20 episodes
                    dialogs = episode.get('dialogs', [])
                    for dialog in dialogs:
                        if dialog and dialog.get('answer'):
                            answer = dialog['answer'].strip()
                            # Filter for good navigation instructions
                            if (answer and 
                                len(answer.split()) >= 5 and  # At least 5 words
                                len(answer.split()) <= 20 and  # Not too long
                                any(word in answer.lower() for word in ['turn', 'go', 'fly', 'head', 'move', 'navigate']) and
                                any(word in answer.lower() for word in ['right', 'left', 'north', 'south', 'building', 'house', 'road', 'o\'clock'])):
                                instructions.append(answer)
                                if len(instructions) >= num_samples:
                                    break
                    if len(instructions) >= num_samples:
                        break
                
                if len(instructions) >= num_samples:
                    logger.info(f"📊 Selected {num_samples} AVDN samples from {len(instructions)} candidates")
                    return instructions[:num_samples]
                else:
                    logger.warning(f"Only found {len(instructions)} suitable instructions")
                    return instructions
                    
            except Exception as e:
                logger.error(f"Error loading dataset from {path}: {e}")
                continue
    
    # Fallback to high-quality synthetic examples if no dataset found
    logger.warning("❌ Could not find AVDN dataset. Using high-quality synthetic examples.")
    return [
        "Turn right and fly over the white building at 3 o'clock",
        "Head north towards the red house near the highway",
        "Navigate left around the tall structure and proceed straight",
        "Fly backward to the parking lot at 9 o'clock"
    ]

def test_sample_detailed(pipeline, instruction: str, sample_num: int) -> dict:
    """Test a single sample with detailed analysis."""
    logger.info(f"\n{'='*60}")
    logger.info(f"📝 TESTING SAMPLE {sample_num}: '{instruction}'")
    logger.info(f"{'='*60}")
    
    start_time = time.time()
    result = pipeline.process_instruction(instruction)
    processing_time = time.time() - start_time
    
    # Detailed analysis
    print(f"\n📊 SAMPLE {sample_num} RESULTS:")
    print(f"Original: {instruction}")
    print(f"Success: {'✅' if result['success'] else '❌'}")
    print(f"Processing time: {processing_time:.2f}s")
    
    if result['success']:
        print(f"\n✅ VALID PARAPHRASES:")
        print(f"Positives ({len(result['positives'])}):")
        for i, pos in enumerate(result['positives'], 1):
            print(f"  {i}. {pos}")
        
        print(f"Negatives ({len(result['negatives'])}):")
        for i, neg in enumerate(result['negatives'], 1):
            print(f"  {i}. {neg}")
    else:
        print(f"\n❌ FAILED: {result.get('failure_reason', 'Unknown')}")
        
        # Show what was generated vs what was valid
        if 'validation_report' in result:
            report = result['validation_report']
            total_gen = report.get('total_generated', {})
            print(f"Generated: {total_gen.get('positives', 0)} positives, {total_gen.get('negatives', 0)} negatives")
            print(f"Valid: {len(result['positives'])} positives, {len(result['negatives'])} negatives")
            
            # Show failed validations
            positive_results = report.get('validation_details', {}).get('positive_results', [])
            for i, pos_result in enumerate(positive_results, 1):
                if not pos_result['is_valid']:
                    print(f"\n❌ Positive {i} failed: '{pos_result['paraphrase'][:60]}...'")
                    print(f"   Scores: emb={pos_result.get('embedding_similarity', 0):.3f}, "
                          f"dir={pos_result.get('direction_similarity', 0):.3f}, "
                          f"landmark={pos_result.get('landmark_similarity', 0):.3f}")
    
    # Validation summary
    if 'validation_report' in result:
        summary = result['validation_report']['summary']
        print(f"\n📈 VALIDATION SUMMARY:")
        print(f"Positive success rate: {summary['positive_success_rate']:.1%}")
        print(f"Negative success rate: {summary['negative_success_rate']:.1%}")
        print(f"Overall quality score: {summary['overall_quality_score']:.3f}")
    
    return result

def main():
    """Run comprehensive test on 4 AVDN samples."""
    logger.info("🚀 Starting Comprehensive AVDN Samples Test")
    logger.info("Testing 4 real AVDN samples through complete pipeline")
    
    start_time = time.time()
    
    try:
        # Load AVDN samples
        logger.info("\n📂 Loading AVDN samples...")
        samples = load_avdn_samples(num_samples=4)
        
        if not samples:
            logger.error("❌ No samples loaded")
            return
        
        logger.info(f"✅ Loaded {len(samples)} samples:")
        for i, sample in enumerate(samples, 1):
            logger.info(f"  {i}. {sample}")
        
        # Initialize pipeline
        logger.info("\n🔧 Initializing pipeline...")
        pipeline = ComprehensiveContrastivePipeline(
            target_positives=2,
            target_negatives=1
        )
        
        if not pipeline.initialize():
            logger.error("❌ Pipeline initialization failed")
            return
        
        logger.info("✅ Pipeline initialized successfully")
        
        # Test all samples
        results = []
        successful_samples = 0
        
        for i, sample in enumerate(samples, 1):
            result = test_sample_detailed(pipeline, sample, i)
            results.append(result)
            
            if result['success']:
                successful_samples += 1
            
            # Brief pause between samples
            if i < len(samples):
                time.sleep(1)
        
        # Overall summary
        total_time = time.time() - start_time
        logger.info(f"\n{'='*60}")
        logger.info(f"🏁 COMPREHENSIVE TEST COMPLETE")
        logger.info(f"{'='*60}")
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"Total samples tested: {len(samples)}")
        print(f"Successful samples: {successful_samples}")
        print(f"Success rate: {successful_samples/len(samples)*100:.1f}%")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per sample: {total_time/len(samples):.2f}s")
        
        # Detailed breakdown
        print(f"\n📋 DETAILED BREAKDOWN:")
        for i, (sample, result) in enumerate(zip(samples, results), 1):
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"Sample {i}: {status}")
            print(f"  '{sample[:50]}{'...' if len(sample) > 50 else ''}'")
            if result['success']:
                print(f"  → {len(result['positives'])} valid positives, {len(result['negatives'])} valid negatives")
            else:
                print(f"  → {result.get('failure_reason', 'Validation failed')}")
        
        # Pipeline statistics
        stats = pipeline.get_statistics()
        processing_stats = stats.get('processing_stats', {})
        print(f"\n📈 PIPELINE STATISTICS:")
        print(f"Total instructions processed: {processing_stats.get('total_instructions_processed', 0)}")
        print(f"Successful generations: {processing_stats.get('successful_generations', 0)}")
        print(f"Failed generations: {processing_stats.get('failed_generations', 0)}")
        print(f"Validation reports collected: {len(processing_stats.get('validation_reports', []))}")
        
        # Cleanup
        pipeline.cleanup()
        
        if successful_samples == len(samples):
            logger.info("🎉 All samples processed successfully!")
        else:
            logger.warning(f"⚠️ {len(samples) - successful_samples} samples failed. Check validation thresholds if needed.")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in test: {e}")

if __name__ == "__main__":
    main() 