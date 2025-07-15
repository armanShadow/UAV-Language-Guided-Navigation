#!/usr/bin/env python3
"""
Simple validation test to understand validation criteria and debugging.
"""

import logging
from comprehensive_contrastive_pipeline import ComprehensiveContrastivePipeline

# Configure logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Simple test of single instruction processing with detailed validation logging."""
    
    # Simple test instruction
    test_instruction = "Turn right and fly over the white building at 3 o'clock"
    
    # Initialize pipeline
    pipeline = ComprehensiveContrastivePipeline(
        target_positives=2,
        target_negatives=1
    )
    
    try:
        # Initialize components
        logger.info("🚀 Initializing pipeline for validation test...")
        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            return
        
        # Test single instruction processing with detailed validation
        logger.info(f"\n=== Testing Instruction: '{test_instruction}' ===")
        result = pipeline.process_instruction(test_instruction)
        
        print(f"\n📊 FINAL RESULT:")
        print(f"Success: {result['success']}")
        print(f"Valid Positives ({len(result['positives'])}): ")
        for i, pos in enumerate(result['positives'], 1):
            print(f"  {i}. {pos}")
        
        print(f"Valid Negatives ({len(result['negatives'])}): ")
        for i, neg in enumerate(result['negatives'], 1):
            print(f"  {i}. {neg}")
        
        # Show validation summary
        report = result['validation_report']
        print(f"\n📈 VALIDATION SUMMARY:")
        print(f"Generated: {report['total_generated']['positives']} positives, {report['total_generated']['negatives']} negatives")
        print(f"Valid: {len(report['valid_positives'])} positives, {len(report['valid_negatives'])} negatives")
        print(f"Success rates: {report['summary']['positive_success_rate']:.1%} positives, {report['summary']['negative_success_rate']:.1%} negatives")
        print(f"Quality score: {report['summary']['overall_quality_score']:.3f}")
        
        # Cleanup
        pipeline.cleanup()
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Error in test: {e}")

if __name__ == "__main__":
    main() 