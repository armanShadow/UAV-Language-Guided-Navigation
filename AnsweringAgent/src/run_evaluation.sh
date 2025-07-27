#!/bin/bash

# UAV Navigation Model Evaluation Script
# This script evaluates the best trained model on all datasets with data leakage validation

set -e  # Exit on any error

# Configuration
CHECKPOINT_PATH="outputs/checkpoints/best_model.pth"  # Update this path to your best checkpoint
OUTPUT_DIR="evaluation_results"
BATCH_SIZE=8
NUM_WORKERS=4
MAX_SAMPLES=10

echo "🔍 UAV Navigation Model Evaluation"
echo "=================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT_PATH"
    echo "Please update CHECKPOINT_PATH in this script to point to your best model checkpoint"
    exit 1
fi

echo "📂 Checkpoint: $CHECKPOINT_PATH"
echo "📁 Output Directory: $OUTPUT_DIR"
echo "⚙️  Batch Size: $BATCH_SIZE"
echo "🔧 Workers: $NUM_WORKERS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation with data leakage validation
echo "🚀 Starting evaluation with data leakage validation..."
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
echo "4. Validate that there's no significant data leakage between splits" 