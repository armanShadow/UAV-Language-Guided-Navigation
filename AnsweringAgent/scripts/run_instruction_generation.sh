#!/bin/bash

# Distributed Instruction Generation Script
# This script generates new instructions for the formatted dataset on 10 RTX 2080 GPUs

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Model checkpoint path (update this to your actual checkpoint path)
CHECKPOINT_PATH="./checkpoints/best_model.pth"

# Output directory
OUTPUT_DIR="./generated_instruction_dataset"

# Generation parameters
SPLITS=("train" "val_seen" "val_unseen")
SAMPLE_RATIO=0.1  # Process 10% of each split
MAX_SAMPLES=1000  # Limit to 1000 samples per split for testing

# Create output directory
mkdir -p $OUTPUT_DIR

echo "🚀 Starting Distributed Instruction Generation"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT_PATH"
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

# Function to run generation for all splits
run_instruction_generation() {
    echo "🔄 Processing all splits..."
    
    python -m torch.distributed.launch \
        --nproc_per_node=10 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=12355 \
        src/generate_instruction_dataset.py \
        --checkpoint $CHECKPOINT_PATH \
        --output_dir $OUTPUT_DIR \
        --splits ${SPLITS[@]} \
        --sample_ratio $SAMPLE_RATIO \
        --max_samples $MAX_SAMPLES \
        --use_paraphrasing \
        --seed 42
    
    if [ $? -eq 0 ]; then
        echo "✅ Completed processing all splits"
    else
        echo "❌ Failed processing all splits"
        return 1
    fi
}

# Main execution
echo "🎯 Starting instruction generation process..."

# Process all splits at once
run_instruction_generation

echo ""
echo "🎉 Instruction Generation Complete!"
echo "📁 Generated dataset saved to: $OUTPUT_DIR"
echo ""
echo "📊 Generated files:"
ls -la $OUTPUT_DIR/*.pkl

echo ""
echo "🔍 Sample of generated instructions:"
if [ -f "$OUTPUT_DIR/val_seen_processed_data.pkl" ]; then
    echo "First 3 samples from val_seen:"
    python3 -c "
import pickle
with open('$OUTPUT_DIR/val_seen_processed_data.pkl', 'rb') as f:
    data = pickle.load(f)
for i, sample in enumerate(data[:3]):
    # Decode the answer
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    answer = tokenizer.decode(sample['text_label']['input_ids'], skip_special_tokens=True)
    print(f'Sample {i+1} Answer: {answer}')
"
fi

echo ""
echo "✅ Script completed successfully!" 