#!/usr/bin/env python3
import json
import sys
import logging
import random
import os
from pprint import pprint
from contrastive_sample_generator import ContrastiveSampleGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def find_dataset_files():
    """Find AVDN dataset files in the workspace."""
    possible_paths = [
        "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations",
        "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations",
        "../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations",
        "datasets/AVDN/annotations"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            dataset_files = []
            for filename in ["train_data.json", "val_seen_data.json", "val_unseen_data.json", "test_unseen_data.json"]:
                full_path = os.path.join(path, filename)
                if os.path.exists(full_path):
                    dataset_files.append(full_path)
            
            if dataset_files:
                return dataset_files
    
    return []

def extract_instructions_from_episode(episode):
    """Extract instructions from an AVDN dataset episode."""
    instructions = []
    
    # Extract main instruction
    if "instructions" in episode and isinstance(episode["instructions"], str):
        instruction = episode["instructions"]
        if "[INS]" in instruction:
            ins_start = instruction.find("[INS]")
            if ins_start >= 0:
                answer = instruction[ins_start+5:].strip()
                instructions.append(answer)
    
    # Extract from pre_dialogs
    if "pre_dialogs" in episode and isinstance(episode["pre_dialogs"], list):
        for dialog in episode["pre_dialogs"]:
            if isinstance(dialog, str) and "[INS]" in dialog:
                ins_start = dialog.find("[INS]")
                if ins_start >= 0:
                    answer = dialog[ins_start+5:].strip()
                    instructions.append(answer)
    
    return instructions

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
    
    # Test with hardcoded examples first
    logger.info("\nTesting with hardcoded examples:")
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
    
    # Test with real samples from the dataset
    try:
        logger.info(f"\n{'='*80}\nTesting with real examples from dataset\n{'-'*80}")
        
        # Find dataset files
        dataset_files = find_dataset_files()
        
        if not dataset_files:
            logger.error("No dataset files found. Skipping dataset testing.")
            return
            
        logger.info(f"Found dataset files: {dataset_files}")
        
        # Select a random dataset file
        dataset_path = random.choice(dataset_files)
        logger.info(f"Using dataset: {dataset_path}")
        
        with open(dataset_path, "r") as f:
            data = json.load(f)
            
            # Select 3 random episodes
            if len(data) > 3:
                selected_episodes = random.sample(data, 3)
            else:
                selected_episodes = data
                
            logger.info(f"Selected {len(selected_episodes)} random episodes for testing")
            
            # Process each selected episode
            for episode_idx, episode in enumerate(selected_episodes):
                # Extract instructions from the episode
                instructions = extract_instructions_from_episode(episode)
                
                if not instructions:
                    logger.warning(f"No instructions found in episode {episode_idx+1}")
                    continue
                
                # Select a random instruction
                instruction = random.choice(instructions)
                
                logger.info(f"\n{'='*80}\nTesting with real instruction {episode_idx+1}:\n{instruction}\n{'-'*80}")
                
                # Generate full augmentation
                augmentation = generator.augment_dialog_turn(
                    question=None,  # Question not needed for this test
                    answer=instruction,
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
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_mixed_approach() 