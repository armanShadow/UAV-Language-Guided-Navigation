#!/usr/bin/env python3
"""
Preprocess Datasets for Training
===============================

This script preprocesses all datasets (train, val_seen, val_unseen) and saves them
as .pkl files for faster loading during training.

USAGE:
    cd AnsweringAgent/src
    python preprocess_datasets.py
"""

import os
import sys
from transformers import T5Tokenizer
from config import Config
from data.dataset import AnsweringDataset
from utils.logger import setup_logger

def main():
    """Preprocess all datasets and save as .pkl files."""
    print("🚀 Starting Dataset Preprocessing...")
    
    # Load configuration
    config = Config()
    
    # Initialize logger
    logger = setup_logger('preprocessing', log_dir=config.log_dir)
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, model_max_length=config.data.max_seq_length)
    
    # Log dataset configuration
    status = "enabled" if config.data.use_augmented_data else "disabled"
    logger.info(f"Dataset configuration - Augmented data: {status}")
    logger.info(f"  Train: {config.data.get_json_path('train')}")
    logger.info(f"  Val Seen: {config.data.get_json_path('val_seen')}")
    logger.info(f"  Val Unseen: {config.data.get_json_path('val_unseen')}")
    
    # Check if augmented data files exist
    if config.data.use_augmented_data:
        missing_files = []
        for split in ['train', 'val_seen', 'val_unseen']:
            json_path = config.data.get_json_path(split)
            if not os.path.exists(json_path):
                missing_files.append(json_path)
        
        if missing_files:
            logger.error("❌ Missing augmented data files:")
            for file in missing_files:
                logger.error(f"  - {file}")
            logger.error("Run comprehensive_avdn_pipeline.py first to generate augmented data")
            sys.exit(1)
    
    # Preprocess all splits
    splits = ['train', 'val_seen', 'val_unseen']
    
    for split in splits:
        logger.info(f"\n📊 Processing {split} dataset...")
        try:
            processed_path = AnsweringDataset.preprocess_and_save(
                config, tokenizer, split=split, logger=logger
            )
            logger.info(f"✅ {split} dataset saved to {processed_path}")
        except Exception as e:
            logger.error(f"❌ Error processing {split} dataset: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            sys.exit(1)
    
    # Verify all processed files exist
    logger.info("\n🔍 Verifying processed files...")
    
    train_path = config.data.train_processed_path_dir
    val_seen_path = config.data.val_seen_processed_path
    val_unseen_path = config.data.val_unseen_processed_path
    
    if os.path.exists(train_path) and os.listdir(train_path):
        logger.info(f"✅ Train data: {train_path} (chunked)")
    else:
        logger.error(f"❌ Train data missing: {train_path}")
        sys.exit(1)
    
    if os.path.exists(val_seen_path):
        logger.info(f"✅ Val seen data: {val_seen_path}")
    else:
        logger.error(f"❌ Val seen data missing: {val_seen_path}")
        sys.exit(1)
    
    if os.path.exists(val_unseen_path):
        logger.info(f"✅ Val unseen data: {val_unseen_path}")
    else:
        logger.error(f"❌ Val unseen data missing: {val_unseen_path}")
        sys.exit(1)
    
    logger.info("\n🎉 All datasets preprocessed successfully!")
    logger.info("Ready to start training with:")
    logger.info("  python train.py")
    logger.info("  or")
    logger.info("  torchrun --nproc_per_node=2 train.py")

if __name__ == "__main__":
    main() 