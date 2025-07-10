#!/usr/bin/env python3
"""
Test Script for Simple Mixtral Paraphrasing
Validates current implementation before enhancements.
"""

import torch
import sys
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_environment():
    """Test the environment setup"""
    logger.info("=== Environment Testing ===")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check PyTorch
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Check disk space (for model downloads)
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        logger.info(f"Disk space: {free / 1024**3:.2f} GB free")
    except:
        logger.info("Could not check disk space")
    
    logger.info("Environment check complete\n")

def test_import():
    """Test importing the paraphraser"""
    logger.info("=== Import Testing ===")
    
    try:
        from simple_mixtral_paraphrasing import SimpleMixtralParaphraser
        logger.info("✅ Successfully imported SimpleMixtralParaphraser")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to import SimpleMixtralParaphraser: {e}")
        return False

def test_initialization():
    """Test model initialization"""
    logger.info("=== Initialization Testing ===")
    
    try:
        from simple_mixtral_paraphrasing import SimpleMixtralParaphraser
        
        logger.info("Attempting to initialize SimpleMixtralParaphraser...")
        paraphraser = SimpleMixtralParaphraser()
        
        logger.info("✅ Successfully initialized SimpleMixtralParaphraser")
        return paraphraser
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize SimpleMixtralParaphraser: {e}")
        return None

def test_paraphrasing(paraphraser):
    """Test paraphrasing functionality"""
    logger.info("=== Paraphrasing Testing ===")
    
    # Test sentences for UAV navigation
    test_sentences = [
        "Fly to the white building on your right.",
        "Turn left at the intersection.",
        "Go straight for 100 meters.",
        "Land near the parking lot.",
        "Follow the road north."
    ]
    
    results = []
    
    for i, sentence in enumerate(test_sentences, 1):
        logger.info(f"\nTest {i}: {sentence}")
        
        try:
            # Generate paraphrase
            paraphrases = paraphraser.paraphrase(sentence)
            
            if paraphrases:
                logger.info(f"✅ Paraphrase: {paraphrases[0]}")
                results.append({
                    'original': sentence,
                    'paraphrase': paraphrases[0],
                    'success': True
                })
            else:
                logger.error(f"❌ No paraphrase generated")
                results.append({
                    'original': sentence,
                    'paraphrase': None,
                    'success': False
                })
                
        except Exception as e:
            logger.error(f"❌ Paraphrasing failed: {e}")
            results.append({
                'original': sentence,
                'paraphrase': None,
                'success': False,
                'error': str(e)
            })
    
    return results

def test_gpu_memory():
    """Test GPU memory usage"""
    logger.info("=== GPU Memory Testing ===")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i} memory:")
            logger.info(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            logger.info(f"  Cached: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    else:
        logger.info("No CUDA devices available")

def main():
    """Run all tests"""
    logger.info("=== Simple Mixtral Paraphrasing Test Suite ===")
    logger.info(f"Test started at: {datetime.now()}")
    
    # Test 1: Environment
    test_environment()
    
    # Test 2: Import
    if not test_import():
        logger.error("Import test failed. Exiting.")
        return
    
    # Test 3: Initialization
    paraphraser = test_initialization()
    if paraphraser is None:
        logger.error("Initialization test failed. Exiting.")
        return
    
    # Test 4: GPU Memory (after initialization)
    test_gpu_memory()
    
    # Test 5: Paraphrasing
    results = test_paraphrasing(paraphraser)
    
    # Test 6: Final GPU Memory
    test_gpu_memory()
    
    # Summary
    logger.info("\n=== Test Summary ===")
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['success'])
    
    logger.info(f"Total paraphrasing tests: {total_tests}")
    logger.info(f"Successful tests: {successful_tests}")
    logger.info(f"Failed tests: {total_tests - successful_tests}")
    
    if successful_tests == total_tests:
        logger.info("🎉 All tests passed!")
    else:
        logger.info("⚠️  Some tests failed. Check logs above.")
    
    logger.info(f"Test completed at: {datetime.now()}")

if __name__ == "__main__":
    main() 