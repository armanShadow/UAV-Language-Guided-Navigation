#!/bin/bash

# Enhanced contrastive sample generation script for UAV navigation

# Set paths
DATA_DIR="processed_data"
OUTPUT_DIR="augmented_data/enhanced"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run test script to verify enhanced approaches are working
echo "Testing enhanced approach..."
python test_enhanced_approach.py --device cpu

# Run augmentation with enhanced settings
echo "Running data augmentation with enhanced settings..."
python augment_dataset.py \
    --train_path $DATA_DIR/train_data.json \
    --val_seen_path $DATA_DIR/val_seen_data.json \
    --val_unseen_path $DATA_DIR/val_unseen_data.json \
    --output_dir $OUTPUT_DIR \
    --model_name "sentence-transformers/all-mpnet-base-v2" \
    --paraphrase_model "prithivida/parrot_paraphraser_on_T5" \
    --pos_examples 3 \
    --neg_examples 3 \
    --device cpu \
    --print_samples 2

echo "Augmentation complete! Enhanced samples saved to $OUTPUT_DIR"
echo "Sample output file structure:"
ls -la $OUTPUT_DIR

# Alternative options for CUDA users with sufficient memory
echo ""
echo "For CUDA-enabled systems, use the following command:"
echo "python augment_dataset.py \\
    --train_path $DATA_DIR/train_data.json \\
    --val_seen_path $DATA_DIR/val_seen_data.json \\
    --val_unseen_path $DATA_DIR/val_unseen_data.json \\
    --output_dir $OUTPUT_DIR \\
    --model_name \"sentence-transformers/all-mpnet-base-v2\" \\
    --paraphrase_model \"prithivida/parrot_paraphraser_on_T5\" \\
    --pos_examples 3 \\
    --neg_examples 3 \\
    --device cuda \\
    --print_samples 2" 