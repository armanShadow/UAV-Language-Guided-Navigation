import argparse
import os
import logging
import sys
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

def main():
    parser = argparse.ArgumentParser(description='Augment navigation dataset with contrastive samples')
    parser.add_argument('--train_path', type=str, required=True, help='Path to training data JSON')
    parser.add_argument('--val_seen_path', type=str, required=True, help='Path to val_seen data JSON')
    parser.add_argument('--val_unseen_path', type=str, required=True, help='Path to val_unseen data JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_name', type=str, default="sentence-transformers/all-MiniLM-L6-v2", 
                      help='Name of the sentence transformer model to use')
    parser.add_argument('--pos_examples', type=int, default=2, help='Number of positive examples per dialog')
    parser.add_argument('--neg_examples', type=int, default=3, help='Number of negative examples per dialog')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                      help='Device to use for sentence embedding')
    parser.add_argument('--split', type=str, default='all', choices=['train', 'val_seen', 'val_unseen', 'all'],
                      help='Which dataset split to augment')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting dataset augmentation process")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize contrastive sample generator
    generator = ContrastiveSampleGenerator(model_name=args.model_name, device=args.device)
    
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

if __name__ == "__main__":
    main() 


"""
Usage:
python -m augment_dataset \
--train_path /Users/arman/Desktop/UTA/Thesis/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/processed_data/train_data.json \
--val_seen_path /Users/arman/Desktop/UTA/Thesis/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/processed_data/val_seen_data.json \
--val_unseen_path /Users/arman/Desktop/UTA/Thesis/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/processed_data/val_unseen_data.json \
--output_dir /Users/arman/Desktop/UTA/Thesis/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/augmented_data \
--model_name "sentence-transformers/all-mpnet-base-v2" \
--pos_examples 2 \
--neg_examples 3 \
--device cpu    
  
"""