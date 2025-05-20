#!/usr/bin/env python3
import json
import sys
import logging
import random
from pprint import pprint
from contrastive_sample_generator import ContrastiveSampleGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_mixed_approach():
    """Test the mixed approach with both language model and rule/template-based examples."""
    logger = logging.getLogger(__name__)
    logger.info("Testing mixed approach for contrastive learning examples")
    
    # Create sample generator with CPU for testing
    # Note: Language model generation will only work if you have the required models
    generator = ContrastiveSampleGenerator(device="cpu")
    
    # Sample navigation instructions
    sample_instructions = [
        "Turn right and head towards the large blue building in front of you.",
        "Go north until you see a red-roofed structure, then turn left.",
        "The destination is at your 3 o'clock direction, it's a tall rectangular tower.",
        "Head to the gray oval-shaped building across from the lake."
    ]
    
    for i, instruction in enumerate(sample_instructions):
        logger.info(f"\n{'='*80}\nTesting instruction {i+1}:\n{instruction}\n{'-'*80}")
        
        # Generate 3 positive examples (1 LM-based, 2 template-based)
        logger.info("\nGenerating positive examples:")
        positives = generator.generate_positive_examples(instruction, n=3)
        
        for j, pos in enumerate(positives):
            logger.info(f"Positive {j+1} ({pos['type']}, similarity: {pos['similarity']:.3f})")
            logger.info(f"  {pos['text']}")
        
        # Generate 3 negative examples (1 LM-based, 2 rule-based)
        logger.info("\nGenerating negative examples:")
        negatives = generator.generate_negative_examples(instruction, n=3)
        
        for j, neg in enumerate(negatives):
            logger.info(f"Negative {j+1} ({neg['type']}, similarity: {neg['similarity']:.3f})")
            logger.info(f"  {neg['text']}")
    
    # Test with a real sample from the dataset
    try:
        logger.info(f"\n{'='*80}\nTesting with real examples from dataset\n{'-'*80}")
        with open("processed_data/val_seen_data.json", "r") as f:
            data = json.load(f)
            
            # Find a good dialog turn with an answer
            real_answer = None
            for episode in data[:10]:  # Check first 10 episodes
                for dialog in episode.get("dialogs", []):
                    if dialog.get("answer") and len(dialog["answer"]) > 20:
                        real_answer = dialog["answer"]
                        break
                if real_answer:
                    break
            
            if real_answer:
                logger.info(f"\nReal answer from dataset:\n{real_answer}\n{'-'*80}")
                
                # Generate full augmentation
                augmentation = generator.augment_dialog_turn(
                    question=None,  # Question not needed for this test
                    answer=real_answer,
                    num_positives=3,
                    num_negatives=3
                )
                
                if augmentation and "contrastive_samples" in augmentation:
                    samples = augmentation["contrastive_samples"]
                    
                    # Print positive examples
                    logger.info("\nPositive Examples:")
                    for j, pos in enumerate(samples.get("positive_examples", [])):
                        logger.info(f"  {j+1}. {pos['text']} (similarity: {pos['similarity']:.3f}, type: {pos['type']})")
                    
                    # Print negative examples
                    logger.info("\nNegative Examples:")
                    for j, neg in enumerate(samples.get("negative_examples", [])):
                        logger.info(f"  {j+1}. {neg['text']} (similarity: {neg['similarity']:.3f}, type: {neg['type']})")
                    
                    # Print complexity metadata
                    if "complexity_metadata" in augmentation:
                        logger.info("\nComplexity Metadata:")
                        for key, value in augmentation["complexity_metadata"].items():
                            logger.info(f"  {key}: {value}")
    
    except Exception as e:
        logger.error(f"Error testing with real examples: {e}")

if __name__ == "__main__":
    test_mixed_approach() 