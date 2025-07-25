#!/bin/bash

# Distributed UAV Navigation Evaluation Script
# Optimized for 10 RTX 2080 Ti GPUs

echo "🚀 Starting Distributed UAV Navigation Evaluation"
echo "📊 Configuration: 10 RTX 2080 Ti GPUs"
echo "⚙️  Batch Size: 8 per GPU, Total Effective Batch: 80"

# Check if checkpoint path is provided
if [ $# -eq 0 ]; then
    echo "❌ Error: Please provide the path to the checkpoint file"
    echo "Usage: ./run_distributed_evaluation.sh <checkpoint_path>"
    echo "Example: ./run_distributed_evaluation.sh /path/to/checkpoint_epoch_100_fp32.pth"
    exit 1
fi

CHECKPOINT_PATH=$1

# Check if checkpoint file exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "📂 Checkpoint: $CHECKPOINT_PATH"

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="distributed_evaluation_results_${TIMESTAMP}"

echo "📁 Output Directory: $OUTPUT_DIR"

# Set environment variables for distributed training
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

# Run distributed evaluation using torchrun
# Using 10 GPUs (world_size=10)
torchrun \
    --nproc_per_node=10 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    evaluate_distributed.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --batch-size 8 \
    --num-workers 4 \
    --max-samples 10 \
    --output-dir "$OUTPUT_DIR"

echo "✅ Distributed evaluation completed!"
echo "📊 Results saved in: $OUTPUT_DIR"
echo "📄 Check evaluation_results.json for detailed results" 