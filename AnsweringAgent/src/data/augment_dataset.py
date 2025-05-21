import argparse
import os
import logging
import sys
import random
import json
from contrastive_sample_generator import ContrastiveSampleGenerator

def setup_logging():
    """Setup logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def print_sample_examples(data_path, num_samples=2):
    """Print random samples from the augmented dataset."""
    logger = logging.getLogger()
    
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Collect all dialog turns with contrastive samples
        valid_dialogs = []
        for episode in data:
            for dialog in episode.get("dialogs", []):
                if "contrastive_samples" in dialog:
                    valid_dialogs.append(dialog)
        
        if not valid_dialogs:
            logger.info(f"No dialogs with contrastive samples found in {data_path}")
            return
        
        # Select random samples
        samples = random.sample(valid_dialogs, min(num_samples, len(valid_dialogs)))
        
        logger.info(f"\n{'='*80}\nRANDOM SAMPLES FROM {os.path.basename(data_path)}\n{'='*80}")
        
        for i, sample in enumerate(samples):
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            
            logger.info(f"\nSample {i+1}:")
            logger.info(f"Question: {question}")
            logger.info(f"Original Answer: {answer}")
            
            if "contrastive_samples" in sample:
                # Print positive examples
                logger.info("\nPositive Examples:")
                for j, pos in enumerate(sample["contrastive_samples"].get("positive_examples", [])):
                    logger.info(f"  {j+1}. {pos['text']} (similarity: {pos['similarity']:.3f}, type: {pos['type']})")
                
                # Print negative examples
                logger.info("\nNegative Examples:")
                for j, neg in enumerate(sample["contrastive_samples"].get("negative_examples", [])):
                    logger.info(f"  {j+1}. {neg['text']} (similarity: {neg['similarity']:.3f}, type: {neg['type']})")
            
            if "complexity_metadata" in sample:
                logger.info("\nComplexity Metadata:")
                for key, value in sample["complexity_metadata"].items():
                    logger.info(f"  {key}: {value}")
            
            logger.info(f"\n{'-'*80}")
    
    except Exception as e:
        logger.error(f"Error printing samples from {data_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Augment navigation dataset with contrastive samples')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data JSON')
    parser.add_argument('--val_seen_path', type=str, required=True, help='Path to val_seen data JSON')
    parser.add_argument('--val_unseen_path', type=str, required=True, help='Path to val_unseen data JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_name', type=str, default="sentence-transformers/all-MiniLM-L6-v2", 
                      help='Name of the sentence transformer model to use')
    parser.add_argument('--paraphrase_model', type=str, default="eugenesiow/bart-paraphrase",
                      help='Model to use for paraphrasing (supports longer sequences than Pegasus)')
    parser.add_argument('--pos_examples', type=int, default=3, help='Number of positive examples per dialog (default: 3, 1 LM-based + 2 template-based)')
    parser.add_argument('--neg_examples', type=int, default=3, help='Number of negative examples per dialog (default: 3, 1 LM-based + 2 rule-based)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                      help='Device to use for sentence embedding')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val_seen', 'val_unseen', 'all'],
                      help='Which dataset split to augment')
    parser.add_argument('--print_samples', type=int, default=2,
                      help='Number of random samples to print from each dataset (0 to disable)')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting dataset augmentation process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize contrastive sample generator
    generator = ContrastiveSampleGenerator(
        model_name=args.model_name, 
        paraphrase_model_name=args.paraphrase_model,
        device=args.device
    )
    
    # Process each dataset based on the specified split
    datasets_to_process = []
    
    if args.split == 'all' or args.split == 'train':
        datasets_to_process.append(('train', args.train_path))
    
    if args.split == 'all' or args.split == 'val_seen':
        datasets_to_process.append(('val_seen', args.val_seen_path))
    
    if args.split == 'all' or args.split == 'val_unseen':
        datasets_to_process.append(('val_unseen', args.val_unseen_path))
    
    for name, path in datasets_to_process:
        output_path = os.path.join(args.output_dir, f"{name}_contrastive.json")
        logger.info(f"Processing {name} dataset")
        
        augmented_count = generator.augment_dataset(
            path, 
            output_path, 
            num_positives=args.pos_examples,
            num_negatives=args.neg_examples
        )
        
        logger.info(f"Completed {name} dataset: {augmented_count} dialog turns augmented")
        
        # Print random samples if requested
        if args.print_samples > 0:
            print_sample_examples(output_path, args.print_samples)

if __name__ == "__main__":
    main() 


"""
Usage:
python -m AnsweringAgent.src.data.augment_dataset \
--train_path /app/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/processed_data/train_data.json \
--val_seen_path /app/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/processed_data/val_seen_data.json \
--val_unseen_path /app/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/processed_data/val_unseen_data.json \
--output_dir /app/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/augmented_data \
--model_name "sentence-transformers/all-mpnet-base-v2" \
--paraphrase_model "eugenesiow/bart-paraphrase" \
--pos_examples 3 \
--neg_examples 3 \
--device cuda \
--print_samples 3
"""