#!/bin/bash

# Distributed Evaluation with Hint Tags Script
# Runs evaluation across multiple GPUs with 10% sampling and hint tags

# Configuration
CHECKPOINT_PATH="/app/UAV-Language-Guided-Navigation/AnsweringAgent/src/checkpoints/best_model_450_fp32.pth"
NUM_GPUS=10  # Adjust based on available GPUs
SAMPLE_RATIO=0.1  # 10% sampling
OUTPUT_DIR="./evaluation_outputs"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "🚀 Starting Distributed Evaluation with Hint Tags"
echo "📊 Configuration:"
echo "   Checkpoint: $CHECKPOINT_PATH"
echo "   GPUs: $NUM_GPUS"
echo "   Sample Ratio: $SAMPLE_RATIO"
echo "   Output Dir: $OUTPUT_DIR"
echo ""

# Run distributed evaluation
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12356 \
    run_eval_generation.py \
    --checkpoint $CHECKPOINT_PATH \
    --sample_ratio $SAMPLE_RATIO \
    --splits train val_seen val_unseen \
    --hint_types spatial movement landmark navigation \
    --output_dir $OUTPUT_DIR

echo ""
echo "✅ Distributed evaluation completed!"
echo "📁 Results saved to: $OUTPUT_DIR" 