#!/usr/bin/env python3
"""
Test script for the Comprehensive Contrastive Pipeline (Simplified)
Verifies that ParaphraseGenerationPipeline and ValidationPipeline work together properly.
"""

import sys
import time
import logging
from pathlib import Path

# Import the comprehensive pipeline
from comprehensive_contrastive_pipeline import ComprehensiveContrastivePipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline_initialization():
    """Test that the pipeline can be initialized properly."""
    logger.info("=== Testing Pipeline Initialization ===")
    
    pipeline = ComprehensiveContrastivePipeline(
        target_positives=2,
        target_negatives=1
    )
    
    try:
        success = pipeline.initialize()
        if success:
            logger.info("✅ Pipeline initialization successful")
            
            # Check that all components are available
            assert pipeline.generation_pipeline is not None, "ParaphraseGenerationPipeline not initialized"
            assert pipeline.validation_pipeline is not None, "ValidationPipeline not available"
            
            logger.info("✅ All pipeline components verified")
            return pipeline
        else:
            logger.error("❌ Pipeline initialization failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ Pipeline initialization error: {e}")
        return None

def test_single_instruction_processing(pipeline):
    """Test processing a single instruction with full validation reporting."""
    logger.info("\n=== Testing Single Instruction Processing ===")
    
    test_instruction = "Turn right and fly over the white building at 3 o'clock"
    logger.info(f"Testing instruction: '{test_instruction}'")
    
    try:
        result = pipeline.process_instruction(test_instruction)
        
        logger.info(f"Result success: {result['success']}")
        logger.info(f"Processing time: {result.get('processing_time', 'N/A'):.2f}s")
        
        if result['success']:
            logger.info(f"Valid positives: {len(result['positives'])}")
            logger.info(f"Valid negatives: {len(result['negatives'])}")
            
            # Show actual paraphrases
            print(f"\nValid Positive paraphrases:")
            for i, positive in enumerate(result['positives'], 1):
                print(f"  {i}. {positive}")
            
            print(f"\nValid Negative paraphrases:")
            for i, negative in enumerate(result['negatives'], 1):
                print(f"  {i}. {negative}")
            
            # Show validation report summary
            report_summary = result['validation_report']['summary']
            print(f"\nValidation Report Summary:")
            print(f"  Positive success rate: {report_summary['positive_success_rate']:.2%}")
            print(f"  Negative success rate: {report_summary['negative_success_rate']:.2%}")
            print(f"  Overall quality score: {report_summary['overall_quality_score']:.3f}")
            
            logger.info("✅ Single instruction processing successful")
            return True
        else:
            logger.warning(f"❌ Processing failed: {result.get('failure_reason', 'Unknown')}")
            
            # Show what was generated vs what was valid
            if 'validation_report' in result:
                total_gen = result['validation_report'].get('total_generated', {})
                print(f"\nGeneration vs Validation:")
                print(f"  Generated: {total_gen.get('positives', 0)} positives, {total_gen.get('negatives', 0)} negatives")
                print(f"  Valid: {len(result['positives'])} positives, {len(result['negatives'])} negatives")
            
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in single instruction processing: {e}")
        return False

def test_generation_only_mode(pipeline):
    """Test generation-only mode (without validation)."""
    logger.info("\n=== Testing Generation-Only Mode ===")
    
    test_instruction = "Head north towards the red house near the highway"
    
    try:
        result = pipeline.generate_paraphrases_only(test_instruction, strategy="combined")
        
        if result.get('success', False):
            logger.info("✅ Generation-only mode successful")
            logger.info(f"Generated {len(result.get('positives', []))} positives, {len(result.get('negatives', []))} negatives")
            
            # Show generated paraphrases
            print(f"\nGenerated Positives:")
            for i, positive in enumerate(result.get('positives', []), 1):
                print(f"  {i}. {positive}")
            
            print(f"\nGenerated Negatives:")
            for i, negative in enumerate(result.get('negatives', []), 1):
                print(f"  {i}. {negative}")
            
            return True
        else:
            logger.warning(f"❌ Generation-only mode failed: {result.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in generation-only mode: {e}")
        return False

def test_validation_only_mode(pipeline):
    """Test validation-only mode (without generation)."""
    logger.info("\n=== Testing Validation-Only Mode ===")
    
    original = "Turn left at the intersection"
    positives = [
        "Make a left turn at the intersection",
        "Go left when you reach the intersection"
    ]
    negatives = [
        "Turn right at the intersection"
    ]
    
    try:
        result = pipeline.validate_paraphrases_only(original, positives, negatives)
        
        if result.get('success', False):
            logger.info("✅ Validation-only mode successful")
            
            report = result['validation_report']
            valid_pos = len(report['valid_positives'])
            valid_neg = len(report['valid_negatives'])
            
            logger.info(f"Valid positives: {valid_pos}/{len(positives)}")
            logger.info(f"Valid negatives: {valid_neg}/{len(negatives)}")
            
            # Show detailed validation results
            print(f"\nValidation Details:")
            print(f"  Positive validation results:")
            for i, pos_result in enumerate(report['validation_details']['positive_results'], 1):
                status = "✅" if pos_result['is_valid'] else "❌"
                print(f"    {i}. {status} '{pos_result['paraphrase'][:50]}...' (score: {pos_result['score']:.3f})")
            
            print(f"  Negative validation results:")
            for i, neg_result in enumerate(report['validation_details']['negative_results'], 1):
                status = "✅" if neg_result['is_valid'] else "❌"
                print(f"    {i}. {status} '{neg_result['paraphrase'][:50]}...' (score: {neg_result['score']:.3f})")
            
            return True
        else:
            logger.warning(f"❌ Validation-only mode failed: {result.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in validation-only mode: {e}")
        return False

def test_multiple_instructions(pipeline):
    """Test processing multiple instructions sequentially."""
    logger.info("\n=== Testing Multiple Instructions Processing ===")
    
    test_instructions = [
        "Navigate left around the tall structure",
        "Fly backward to the parking lot at 9 o'clock"
    ]
    
    try:
        results = pipeline.process_instructions(test_instructions)
        
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"Processed {len(results)} instructions, {success_count} successful")
        
        # Show detailed results
        print(f"\nDetailed Results:")
        for i, result in enumerate(results, 1):
            status = "✅" if result['success'] else "❌"
            print(f"  Instruction {i}: {status}")
            if result['success']:
                print(f"    Valid: {len(result['positives'])} positives, {len(result['negatives'])} negatives")
            else:
                print(f"    Reason: {result.get('failure_reason', 'Unknown')}")
        
        if success_count > 0:
            logger.info("✅ Multiple instruction processing successful")
            return True
        else:
            logger.warning("❌ No successful instructions in batch")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in multiple instruction processing: {e}")
        return False

def test_validation_reporting(pipeline):
    """Test the validation reporting functionality."""
    logger.info("\n=== Testing Validation Reporting ===")
    
    test_instruction = "Turn right and head to the building"
    
    try:
        # Process instruction to get validation report
        result = pipeline.process_instruction(test_instruction)
        
        if 'validation_report' in result:
            report = result['validation_report']
            
            logger.info("✅ Validation report generated")
            
            # Check report structure
            required_keys = ['original_instruction', 'total_generated', 'valid_positives', 'valid_negatives', 'validation_details', 'summary']
            for key in required_keys:
                assert key in report, f"Missing key in validation report: {key}"
            
            logger.info("✅ Validation report structure verified")
            
            # Show report summary
            print(f"\nValidation Report Summary:")
            print(f"  Original: '{report['original_instruction']}'")
            print(f"  Generated: {report['total_generated']['positives']} pos, {report['total_generated']['negatives']} neg")
            print(f"  Valid: {len(report['valid_positives'])} pos, {len(report['valid_negatives'])} neg")
            print(f"  Success rates: {report['summary']['positive_success_rate']:.2%} pos, {report['summary']['negative_success_rate']:.2%} neg")
            print(f"  Quality score: {report['summary']['overall_quality_score']:.3f}")
            
            return True
        else:
            logger.warning("❌ No validation report in result")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error in validation reporting test: {e}")
        return False

def test_statistics_reporting(pipeline):
    """Test statistics reporting functionality."""
    logger.info("\n=== Testing Statistics Reporting ===")
    
    try:
        stats = pipeline.get_statistics()
        
        logger.info("Statistics categories:")
        for category in stats.keys():
            logger.info(f"  - {category}")
        
        processing_stats = stats.get('processing_stats', {})
        logger.info(f"Total processed: {processing_stats.get('total_instructions_processed', 0)}")
        logger.info(f"Success rate: {processing_stats.get('successful_generations', 0)}/{processing_stats.get('total_instructions_processed', 0)}")
        
        # Check for validation reports
        validation_reports = processing_stats.get('validation_reports', [])
        logger.info(f"Validation reports collected: {len(validation_reports)}")
        
        logger.info("✅ Statistics reporting working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in statistics reporting: {e}")
        return False

def main():
    """Run comprehensive tests on the simplified pipeline."""
    logger.info("🚀 Starting Comprehensive Pipeline Tests (Simplified Architecture)")
    start_time = time.time()
    
    test_results = []
    
    try:
        # Test 1: Initialize pipeline
        pipeline = test_pipeline_initialization()
        test_results.append(("Initialization", pipeline is not None))
        
        if pipeline is None:
            logger.error("❌ Cannot continue tests - initialization failed")
            return
        
        # Test 2: Single instruction processing
        single_success = test_single_instruction_processing(pipeline)
        test_results.append(("Single Instruction", single_success))
        
        # Test 3: Generation-only mode
        gen_success = test_generation_only_mode(pipeline)
        test_results.append(("Generation Only", gen_success))
        
        # Test 4: Validation-only mode
        val_success = test_validation_only_mode(pipeline)
        test_results.append(("Validation Only", val_success))
        
        # Test 5: Multiple instructions
        multi_success = test_multiple_instructions(pipeline)
        test_results.append(("Multiple Instructions", multi_success))
        
        # Test 6: Validation reporting
        report_success = test_validation_reporting(pipeline)
        test_results.append(("Validation Reporting", report_success))
        
        # Test 7: Statistics reporting
        stats_success = test_statistics_reporting(pipeline)
        test_results.append(("Statistics", stats_success))
        
        # Cleanup
        pipeline.cleanup()
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in tests: {e}")
    
    # Final results
    total_time = time.time() - start_time
    logger.info(f"\n🏁 Test Results (completed in {total_time:.2f}s):")
    
    passed = 0
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        logger.info("🎉 All tests passed! The simplified comprehensive pipeline is working correctly.")
    else:
        logger.warning(f"⚠️  {len(test_results) - passed} tests failed. Check the logs for details.")

if __name__ == "__main__":
    main() 