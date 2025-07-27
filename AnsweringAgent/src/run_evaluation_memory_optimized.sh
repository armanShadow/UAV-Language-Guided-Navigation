#!/bin/bash

# Memory-Optimized UAV Navigation Model Evaluation Script
# This script evaluates the model with reduced memory usage

set -e  # Exit on any error

# Check if checkpoint path is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path> [output_dir]"
    echo "Example: $0 outputs/checkpoints/best_model.pth evaluation_results"
    exit 1
fi

# Configuration - optimized for memory
CHECKPOINT_PATH="$1"
OUTPUT_DIR="${2:-evaluation_results_optimized}"
BATCH_SIZE=2  # Reduced batch size for memory
NUM_WORKERS=2  # Reduced workers
MAX_SAMPLES=5

echo "🔍 UAV Navigation Model Evaluation (Memory Optimized)"
echo "====================================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo "Please provide a valid checkpoint path as the first argument"
    exit 1
fi

echo "📂 Checkpoint: $CHECKPOINT_PATH"
echo "📁 Output Directory: $OUTPUT_DIR"
echo "⚙️  Batch Size: $BATCH_SIZE (reduced for memory)"
echo "🔧 Workers: $NUM_WORKERS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clear GPU memory before starting
echo "🧹 Clearing GPU memory..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Run evaluation with memory optimizations
echo "🚀 Starting memory-optimized evaluation..."
python evaluate_distributed.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    --validate-leakage

echo ""
echo "✅ Evaluation completed!"
echo "📊 Results saved to: $OUTPUT_DIR/evaluation_results.json"

# Display summary if results file exists
if [ -f "$OUTPUT_DIR/evaluation_results.json" ]; then
    echo ""
    echo "📈 EVALUATION SUMMARY:"
    echo "======================"
    
    # Extract and display key metrics using jq if available
    if command -v jq &> /dev/null; then
        echo "Train Dataset:"
        jq -r '.train | "  Loss: \(.loss) | Accuracy: \(.overall_accuracy)"' "$OUTPUT_DIR/evaluation_results.json" 2>/dev/null || echo "  Results not available"
        
        echo "Val Seen Dataset:"
        jq -r '.val_seen | "  Loss: \(.loss) | Accuracy: \(.overall_accuracy)"' "$OUTPUT_DIR/evaluation_results.json" 2>/dev/null || echo "  Results not available"
        
        echo "Val Unseen Dataset:"
        jq -r '.val_unseen | "  Loss: \(.loss) | Accuracy: \(.overall_accuracy)"' "$OUTPUT_DIR/evaluation_results.json" 2>/dev/null || echo "  Results not available"
    else
        echo "Install jq for better result formatting: sudo apt-get install jq"
        echo "Results available in: $OUTPUT_DIR/evaluation_results.json"
    fi
else
    echo "❌ Results file not found. Check the evaluation logs for errors."
fi

echo ""
echo "🎯 Next Steps:"
echo "1. Review the evaluation results in $OUTPUT_DIR/evaluation_results.json"
echo "2. Check the generated samples for quality assessment"
echo "3. Compare val_seen vs val_unseen performance for generalization"
echo "4. If memory issues persist, consider reducing batch size further"

# Clear GPU memory after completion
echo "🧹 Clearing GPU memory after evaluation..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 