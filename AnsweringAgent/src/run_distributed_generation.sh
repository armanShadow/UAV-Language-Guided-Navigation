#!/bin/bash

# Simple Distributed Generation Pipeline Launch Script
# Runs generation across 10 GPUs without saving results

set -e  # Exit on any error

# Configuration
CHECKPOINT_PATH="${1:-/path/to/your/checkpoint.pth}"
NUM_SAMPLES_PER_GPU="${2:-5}"
BATCH_SIZE="${3:-4}"
NUM_GPUS=10

# Validate checkpoint path
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint file not found at $CHECKPOINT_PATH"
    echo "Usage: $0 <checkpoint_path> [num_samples_per_gpu] [batch_size]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/model_checkpoint.pth 10 4"
    exit 1
fi

echo "🚀 Simple Distributed Generation Pipeline"
echo "========================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Samples per GPU: $NUM_SAMPLES_PER_GPU"
echo "Batch Size: $BATCH_SIZE"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Set environment variables for optimal performance
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

# Launch distributed generation
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    distributed_generation_pipeline.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --num_samples "$NUM_SAMPLES_PER_GPU" \
    --batch_size "$BATCH_SIZE" \
    --splits train val_seen val_unseen

echo ""
echo "✅ Distributed generation completed!" 