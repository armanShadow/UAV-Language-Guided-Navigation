#!/usr/bin/env python3
"""
Simple Runner for Headless Server Testing
Quick validation of basic Mixtral functionality.
"""

import sys
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Simple test runner for headless server"""
    logger.info("=== Simple Mixtral Test Runner ===")
    logger.info(f"Started at: {datetime.now()}")
    
    try:
        # Import and test
        from simple_mixtral_paraphrasing import SimpleMixtralParaphraser
        
        logger.info("✅ Import successful")
        
        # Initialize
        logger.info("🔄 Initializing Mixtral...")
        paraphraser = SimpleMixtralParaphraser()
        
        logger.info("✅ Initialization successful")
        
        # Test one sentence
        test_sentence = "Fly to the white building on your right."
        logger.info(f"🔄 Testing: {test_sentence}")
        
        paraphrases = paraphraser.paraphrase(test_sentence)
        
        if paraphrases:
            logger.info(f"✅ Success! Paraphrase: {paraphrases[0]}")
            logger.info("🎉 Basic functionality confirmed!")
        else:
            logger.error("❌ No paraphrase generated")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        sys.exit(1)
    
    logger.info(f"Completed at: {datetime.now()}")

if __name__ == "__main__":
    main() 