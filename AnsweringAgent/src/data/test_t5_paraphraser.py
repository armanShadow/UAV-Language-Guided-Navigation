#!/usr/bin/env python3
import sys
import logging
from transformers import pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_t5_paraphraser():
    """Test the T5-based Parrot paraphraser model."""
    logger.info("Testing T5 Parrot paraphraser")
    
    # Load the model
    try:
        logger.info("Loading prithivida/parrot_paraphraser_on_T5 model")
        paraphraser = pipeline(
            "text2text-generation",
            model="prithivida/parrot_paraphraser_on_T5",
            device=-1  # Use CPU
        )
        logger.info("Successfully loaded paraphraser model")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Test examples
    test_examples = [
        "Turn right and head towards the large blue building in front of you.",
        "Go north until you see a red-roofed structure, then turn left.",
        "Your destination is the tall white tower at the corner of the complex."
    ]
    
    for i, example in enumerate(test_examples):
        logger.info(f"\nTest Example {i+1}: {example}")
        
        try:
            # Format input for T5 paraphraser
            paraphrase_input = f"paraphrase: {example}"
            
            # Generate paraphrases
            logger.info("Generating paraphrases...")
            outputs = paraphraser(
                paraphrase_input,
                max_length=100,
                num_return_sequences=3,
                temperature=1.0,
                do_sample=True
            )
            
            # Display results
            logger.info("Generated paraphrases:")
            for j, output in enumerate(outputs):
                logger.info(f"  {j+1}: {output['generated_text']}")
        except Exception as e:
            logger.error(f"Error generating paraphrases: {e}")
    
    # Test negative generation
    logger.info("\nTesting negative generation")
    test_negatives = [
        "contradict: The building is on your right side.",
        "opposite: Go north to reach the destination.",
        "negate: The landmark has a blue roof."
    ]
    
    for i, example in enumerate(test_negatives):
        logger.info(f"\nNegative Test {i+1}: {example}")
        
        try:
            # Generate contraries
            logger.info("Generating contraries...")
            outputs = paraphraser(
                example,
                max_length=100,
                num_return_sequences=2,
                temperature=1.0,
                do_sample=True
            )
            
            # Display results
            logger.info("Generated contraries:")
            for j, output in enumerate(outputs):
                logger.info(f"  {j+1}: {output['generated_text']}")
        except Exception as e:
            logger.error(f"Error generating contraries: {e}")

if __name__ == "__main__":
    test_t5_paraphraser() 