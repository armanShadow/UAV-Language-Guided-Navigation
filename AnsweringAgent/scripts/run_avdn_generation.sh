#!/bin/bash

# Distributed AVDN Dataset Generation Script
# This script generates new AVDN datasets using Answering Agent on 10 RTX 2080 GPUs

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Model checkpoint path (update this to your actual checkpoint path)
CHECKPOINT_PATH="./checkpoints/best_model.pth"

# AVDN dataset path
AVDN_DATA_DIR="../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations"

# Output directory
OUTPUT_DIR="./generated_avdn_dataset"

# Generation parameters
SPLITS=("train" "val_seen" "val_unseen")
SAMPLE_RATIO=0.1  # Process 10% of each split
MAX_SAMPLES=1000  # Limit to 1000 samples per split for testing

# Create output directory
mkdir -p $OUTPUT_DIR

echo "🚀 Starting Distributed AVDN Dataset Generation"
echo "================================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "AVDN Data Dir: $AVDN_DATA_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "Splits: ${SPLITS[@]}"
echo "Sample Ratio: $SAMPLE_RATIO"
echo "Max Samples: $MAX_SAMPLES"
echo "GPUs: 10 RTX 2080"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo "Please update the CHECKPOINT_PATH variable in this script."
    exit 1
fi

# Check if AVDN data directory exists
if [ ! -d "$AVDN_DATA_DIR" ]; then
    echo "❌ AVDN data directory not found: $AVDN_DATA_DIR"
    echo "Please ensure the AVDN dataset is available."
    exit 1
fi

# Function to run generation for all splits
run_avdn_generation() {
    echo "🔄 Processing all splits..."
    
    python -m torch.distributed.launch \
        --nproc_per_node=10 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=12355 \
        src/generate_avdn_with_agent.py \
        --checkpoint $CHECKPOINT_PATH \
        --avdn_data_dir $AVDN_DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --splits ${SPLITS[@]} \
        --sample_ratio $SAMPLE_RATIO \
        --max_samples $MAX_SAMPLES \

        --seed 42
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed processing all splits"
    else
        echo "❌ Failed processing all splits"
        return 1
    fi
}

# Main execution
echo "🎯 Starting AVDN generation process..."

# Process all splits at once
run_avdn_generation

echo ""
echo "🎉 AVDN Dataset Generation Complete!"
echo "📁 Generated dataset saved to: $OUTPUT_DIR"
echo ""
echo "📊 Generated files:"
ls -la $OUTPUT_DIR/*.json

echo ""
echo "🔍 Sample of generated instructions:"
if [ -f "$OUTPUT_DIR/val_seen_data.json" ]; then
    echo "First 3 samples from val_seen:"
    python3 -c "
import json
with open('$OUTPUT_DIR/val_seen_data.json', 'r') as f:
    data = json.load(f)
for i, sample in enumerate(data[:3]):
    print(f'Sample {i+1}: {sample[\"instructions\"]}')
"
fi

echo ""
echo "✅ Script completed successfully!" 