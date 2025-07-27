#!/bin/bash

# Simple Generation Pipeline Runner
# This script runs the generation pipeline on random samples from each dataset

# Check if checkpoint path is provided
if [ $# -eq 0 ]; then
    echo "❌ Please provide the checkpoint path as an argument"
    echo "Usage: $0 <checkpoint_path> [num_samples] [device]"
    echo "Example: $0 outputs/checkpoints/best_model.pth 3 cuda"
    exit 1
fi

# Get parameters
CHECKPOINT_PATH="$1"
NUM_SAMPLES="${2:-3}"  # Default to 3 if not provided
DEVICE="${3:-cuda}"    # Default to cuda if not provided

echo "🚀 Starting Simple Generation Pipeline..."

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found at: $CHECKPOINT_PATH"
    echo "Please provide a valid checkpoint path."
    exit 1
fi

echo "📂 Using checkpoint: $CHECKPOINT_PATH"
echo "🎯 Number of samples per split: $NUM_SAMPLES"
echo "🖥️  Device: $DEVICE"

# Run the generation pipeline (no output file - just print to console)
python simple_generation_pipeline.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --device "$DEVICE" \
    --splits train val_seen val_unseen \
    --no_save

echo "✅ Generation pipeline completed!" 