#!/usr/bin/env python3
"""
Comprehensive Contrastive Pipeline
Simplified pipeline that uses existing components directly:
- ParaphraseGenerationPipeline for Mixtral-based generation (no iteration)
- ValidationPipeline for validation reporting only
- No iteration loop - single pass generation with validation report
"""

import os
import json
import time
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Import the core pipeline components directly
from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
from validation_pipeline import ValidationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveContrastivePipeline:
    """
    Simplified comprehensive pipeline that uses existing components directly.
    - Single-pass generation (no iteration)
    - Validation as reporting only
    - Sequential processing for reliability
    """
    
    def __init__(self, 
                 generation_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 validation_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 target_positives: int = 2,
                 target_negatives: int = 1):
        
        self.generation_model = generation_model
        self.validation_model = validation_model
        self.target_positives = target_positives
        self.target_negatives = target_negatives
        
        # Pipeline components (initialized lazily)
        self.generation_pipeline = None
        self.validation_pipeline = None
        
        # Processing statistics
        self.stats = {
            'total_instructions_processed': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'validation_reports': []
        }
        
        logger.info("Initialized Comprehensive Contrastive Pipeline (Simplified)")
        logger.info(f"Target: {target_positives} positives + {target_negatives} negatives")
        logger.info("Mode: Single-pass generation + validation reporting")
    
    def initialize(self) -> bool:
        """Initialize generation and validation pipeline components."""
        try:
            logger.info("Initializing pipeline components...")
            
            # Initialize ParaphraseGenerationPipeline
            logger.info("Loading ParaphraseGenerationPipeline...")
            self.generation_pipeline = ParaphraseGenerationPipeline(model_name=self.generation_model)
            
            success = self.generation_pipeline.load_model()
            if not success:
                logger.error("Failed to initialize ParaphraseGenerationPipeline")
                return False
            
            # Initialize ValidationPipeline
            logger.info("Loading ValidationPipeline...")
            self.validation_pipeline = ValidationPipeline(embedding_model=self.validation_model)
            
            success = self.validation_pipeline.load_model()
            if not success:
                logger.error("Failed to initialize ValidationPipeline")
                return False
            
            logger.info("✅ All pipeline components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def process_instruction(self, instruction: str, strategy: str = "combined") -> Dict:
        """
        Process a single instruction: generate paraphrases then validate and report.
        Returns generation results with validation report.
        """
        start_time = time.time()
        logger.info(f"Processing instruction: '{instruction}'")
        
        try:
            # Step 1: Generate paraphrases using ParaphraseGenerationPipeline
            logger.info("Generating paraphrases...")
            generation_result = self.generation_pipeline.generate_paraphrases(instruction, strategy=strategy)
            
            if not generation_result.get('success', False):
                logger.error(f"Generation failed: {generation_result.get('error', 'Unknown error')}")
                self.stats['failed_generations'] += 1
                return {
                    'success': False,
                    'failure_reason': f"Generation failed: {generation_result.get('error', 'Unknown')}",
                    'positives': [],
                    'negatives': [],
                    'validation_report': {},
                    'processing_time': time.time() - start_time
                }
            
            positives = generation_result.get('positives', [])
            negatives = generation_result.get('negatives', [])
            
            logger.info(f"Generated {len(positives)} positives, {len(negatives)} negatives")
            
            # Step 2: Validate and create report using ValidationPipeline
            logger.info("Creating validation report...")
            validation_report = self.create_validation_report(instruction, positives, negatives)
            
            # Step 3: Determine success based on validation
            valid_positives = validation_report.get('valid_positives', [])
            valid_negatives = validation_report.get('valid_negatives', [])
            
            success = (len(valid_positives) >= self.target_positives and 
                      len(valid_negatives) >= self.target_negatives)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['total_instructions_processed'] += 1
            if success:
                self.stats['successful_generations'] += 1
            else:
                self.stats['failed_generations'] += 1
            
            self.stats['validation_reports'].append(validation_report)
            
            # Log results
            if success:
                logger.info(f"✅ Success - Valid: {len(valid_positives)} positives, {len(valid_negatives)} negatives")
            else:
                logger.warning(f"❌ Insufficient valid samples - Valid: {len(valid_positives)}/{self.target_positives} positives, {len(valid_negatives)}/{self.target_negatives} negatives")
            
            logger.info(f"Processing time: {processing_time:.2f}s")
            
            return {
                'success': success,
                'positives': valid_positives,  # Return only validated positives
                'negatives': valid_negatives,  # Return only validated negatives
                'all_positives': positives,    # Include all generated for reference
                'all_negatives': negatives,    # Include all generated for reference
                'validation_report': validation_report,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing instruction: {e}")
            self.stats['failed_generations'] += 1
            return {
                'success': False,
                'failure_reason': f"Processing error: {str(e)}",
                'positives': [],
                'negatives': [],
                'validation_report': {},
                'processing_time': time.time() - start_time
            }
    
    def create_validation_report(self, original: str, positives: List[str], negatives: List[str]) -> Dict:
        """
        Create comprehensive validation report using ValidationPipeline.
        Returns detailed validation results for reporting purposes.
        """
        try:
            report = {
                'original_instruction': original,
                'total_generated': {
                    'positives': len(positives),
                    'negatives': len(negatives)
                },
                'valid_positives': [],
                'valid_negatives': [],
                'validation_details': {
                    'positive_results': [],
                    'negative_results': []
                },
                'summary': {
                    'positive_success_rate': 0.0,
                    'negative_success_rate': 0.0,
                    'overall_quality_score': 0.0
                }
            }
            
            # Validate positives
            positive_scores = []
            for positive in positives:
                result = self.validation_pipeline.validate_positive(original, positive)
                report['validation_details']['positive_results'].append({
                    'paraphrase': positive,
                    'is_valid': result['is_valid'],
                    'score': result['score'],
                    'failure_reasons': result.get('failure_reasons', [])
                })
                
                if result['is_valid']:
                    report['valid_positives'].append(positive)
                
                positive_scores.append(result['score'])
            
            # Validate negatives
            negative_scores = []
            for negative in negatives:
                result = self.validation_pipeline.validate_negative(original, negative)
                report['validation_details']['negative_results'].append({
                    'paraphrase': negative,
                    'is_valid': result['is_valid'],
                    'score': result['score'],
                    'failure_reasons': result.get('failure_reasons', [])
                })
                
                if result['is_valid']:
                    report['valid_negatives'].append(negative)
                
                negative_scores.append(result['score'])
            
            # Calculate summary statistics
            if positives:
                report['summary']['positive_success_rate'] = len(report['valid_positives']) / len(positives)
            if negatives:
                report['summary']['negative_success_rate'] = len(report['valid_negatives']) / len(negatives)
            
            all_scores = positive_scores + negative_scores
            if all_scores:
                report['summary']['overall_quality_score'] = sum(all_scores) / len(all_scores)
            
            logger.info(f"Validation report: {len(report['valid_positives'])}/{len(positives)} positives, {len(report['valid_negatives'])}/{len(negatives)} negatives valid")
            
            return report
            
        except Exception as e:
            logger.error(f"Error creating validation report: {e}")
            return {
                'error': str(e),
                'valid_positives': [],
                'valid_negatives': [],
                'summary': {'positive_success_rate': 0.0, 'negative_success_rate': 0.0, 'overall_quality_score': 0.0}
            }
    
    def process_instructions(self, instructions: List[str], strategy: str = "combined") -> List[Dict]:
        """
        Process multiple instructions sequentially.
        Returns list of results for each instruction.
        """
        logger.info(f"Processing {len(instructions)} instructions sequentially...")
        
        results = []
        total_start_time = time.time()
        
        for i, instruction in enumerate(instructions, 1):
            logger.info(f"\n--- Processing instruction {i}/{len(instructions)} ---")
            result = self.process_instruction(instruction, strategy=strategy)
            results.append(result)
            
            # Brief pause between instructions
            if i < len(instructions):
                time.sleep(0.5)
        
        total_time = time.time() - total_start_time
        logger.info(f"\n🏁 Batch processing complete in {total_time:.2f}s")
        logger.info(f"Average time per instruction: {total_time/len(instructions):.2f}s")
        
        # Summary statistics
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Success rate: {successful}/{len(instructions)} ({successful/len(instructions)*100:.1f}%)")
        
        return results
    
    def generate_paraphrases_only(self, instruction: str, strategy: str = "combined") -> Dict:
        """
        Generate paraphrases only (without validation) using the generation pipeline.
        Useful for testing generation in isolation.
        """
        if not self.generation_pipeline:
            logger.error("Generation pipeline not initialized")
            return {'success': False, 'error': 'Pipeline not initialized'}
        
        try:
            logger.info(f"Generating paraphrases for: '{instruction}' (strategy: {strategy})")
            result = self.generation_pipeline.generate_paraphrases(instruction, strategy=strategy)
            logger.info(f"Generated {len(result.get('positives', []))} positives, {len(result.get('negatives', []))} negatives")
            return result
            
        except Exception as e:
            logger.error(f"Error in generation-only mode: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_paraphrases_only(self, original: str, positives: List[str], negatives: List[str]) -> Dict:
        """
        Create validation report only (without generation).
        Useful for testing validation logic in isolation.
        """
        if not self.validation_pipeline:
            logger.error("Validation pipeline not initialized")
            return {'success': False, 'error': 'Pipeline not initialized'}
        
        try:
            logger.info(f"Creating validation report for {len(positives)} positives and {len(negatives)} negatives")
            report = self.create_validation_report(original, positives, negatives)
            return {'success': True, 'validation_report': report}
            
        except Exception as e:
            logger.error(f"Error in validation-only mode: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_statistics(self) -> Dict:
        """Get comprehensive processing statistics."""
        return {
            'processing_stats': self.stats.copy(),
            'generation_stats': getattr(self.generation_pipeline, 'stats', {}) if self.generation_pipeline else {},
            'validation_stats': getattr(self.validation_pipeline, 'stats', {}) if self.validation_pipeline else {}
        }
    
    def cleanup(self):
        """Cleanup pipeline resources."""
        logger.info("Cleaning up pipeline resources...")
        
        if self.generation_pipeline:
            if hasattr(self.generation_pipeline, 'cleanup'):
                self.generation_pipeline.cleanup()
        
        if self.validation_pipeline:
            if hasattr(self.validation_pipeline, 'cleanup'):
                self.validation_pipeline.cleanup()
        
        logger.info("✅ Pipeline cleanup complete")

def main():
    """Test the comprehensive pipeline with sample instructions."""
    
    # Test instructions
    test_instructions = [
        "Turn right and fly over the white building at 3 o'clock",
        "Head north towards the red house near the highway",
        "Navigate left around the tall structure and proceed straight"
    ]
    
    # Initialize pipeline
    pipeline = ComprehensiveContrastivePipeline(
        target_positives=2,
        target_negatives=1
    )
    
    try:
        # Initialize components
        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            return
        
        # Test single instruction processing
        logger.info("\n=== Testing Single Instruction Processing ===")
        result = pipeline.process_instruction(test_instructions[0])
        print(f"\nSingle Instruction Result:")
        print(f"Success: {result['success']}")
        if result['success']:
            print(f"Valid Positives: {result['positives']}")
            print(f"Valid Negatives: {result['negatives']}")
            print(f"Validation Report Summary: {result['validation_report']['summary']}")
        else:
            print(f"Failure reason: {result.get('failure_reason', 'Insufficient valid samples')}")
        
        # Show statistics
        stats = pipeline.get_statistics()
        print(f"\nProcessing Statistics:")
        print(f"Total processed: {stats['processing_stats']['total_instructions_processed']}")
        print(f"Success rate: {stats['processing_stats']['successful_generations']}/{stats['processing_stats']['total_instructions_processed']}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Cleanup
        pipeline.cleanup()

if __name__ == "__main__":
    main() 