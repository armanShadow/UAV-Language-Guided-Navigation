# UAV Navigation Model Evaluation

This document describes how to evaluate the UAV Navigation Answering Agent model with comprehensive loss calculations that match the training loop.

## Overview

The evaluation system calculates the same comprehensive loss as used during training, including:

1. **Cross-Entropy Loss** (weighted): Primary language modeling loss
2. **Feature Regularization**: Prevents feature explosion (1e-4 * feature_norm)
3. **Destination Loss** (weighted): Cosine similarity between current and destination features
4. **Contrastive Loss** (weighted): Multi-positive contrastive learning loss
5. **Knowledge Distillation Loss** (weighted): Teacher-student alignment loss

## Loss Calculation Details

### Final Loss Formula
```
total_loss = (ce_weight * ce_loss) + 
             feature_reg_loss + 
             (dest_weight * destination_loss) + 
             (contrastive_weight * contrastive_loss) + 
             (kd_weight * kd_loss)
```

### Default Weights (from config.py)
- `ce_loss_weight_end`: 0.5
- `destination_loss_weight_end`: 0.2  
- `contrastive_weight_end`: 10.0
- `kd_weight_end`: 0.5
- Feature regularization: 1e-4 (fixed)

## Distributed Evaluation (10 RTX 2080 GPUs)

### Quick Start
```bash
cd AnsweringAgent/src
chmod +x run_distributed_evaluation.sh
./run_distributed_evaluation.sh /path/to/your/checkpoint_epoch_X_fp32.pth
```

### Manual Distributed Evaluation
```bash
# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

# Run with torchrun
torchrun \
    --nproc_per_node=10 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    evaluate_distributed.py \
    --checkpoint /path/to/checkpoint.pth \
    --batch-size 8 \
    --num-workers 4 \
    --max-samples 10 \
    --output-dir evaluation_results
```

### Parameters
- `--checkpoint`: Path to model checkpoint (.pth file)
- `--batch-size`: Per-GPU batch size (default: 8, effective batch: 80 across 10 GPUs)
- `--num-workers`: Data loading workers per GPU (default: 4)
- `--max-samples`: Number of sample outputs to generate per dataset (default: 5)
- `--output-dir`: Directory to save results (default: evaluation_results)

## Single GPU Evaluation

For testing or smaller evaluations:
```bash
python evaluate.py \
    --checkpoint /path/to/checkpoint.pth \
    --batch-size 16 \
    --num-workers 4 \
    --max-samples 5 \
    --output-dir single_gpu_results
```

## Output Results

### JSON Output Format
```json
{
  "train": {
    "loss": 2.1234,
    "ce_loss": 1.8765,
    "destination_loss": 0.1234,
    "contrastive_loss": 0.0876,
    "kd_loss": 0.0543,
    "feature_reg_loss": 0.0012,
    "accuracy": 0.7234,
    "overall_accuracy": 0.7156,
    "total_tokens": 150000,
    "correct_tokens": 107340,
    "weighted_contributions": {
      "ce_weighted": 0.9383,
      "destination_weighted": 0.0247,
      "contrastive_weighted": 0.8760,
      "kd_weighted": 0.0272,
      "feature_reg": 0.0012
    },
    "samples": [...]
  },
  "val_seen": {...},
  "val_unseen": {...}
}
```

### Console Output
The evaluation will print:
1. **Total Loss**: Combined weighted loss matching training calculation
2. **Individual Loss Components**: CE, Destination, Contrastive, KD, Feature Reg
3. **Effective Contributions**: Each component multiplied by its weight
4. **Accuracy Metrics**: Token-level and overall accuracy
5. **Sample Outputs**: Example predictions for inspection

## Datasets Evaluated

The system evaluates on three splits:
- `train`: Training set performance
- `val_seen`: Validation on seen environments
- `val_unseen`: Validation on unseen environments

## Performance Optimization

### Memory Optimization
- Uses mixed precision (AMP) for consistency with training
- Gradient accumulation disabled during evaluation
- Efficient distributed data loading with DistributedSampler

### Recommended Settings for RTX 2080
- Batch size: 6-8 per GPU (depending on model size)
- Num workers: 4 per GPU
- Total effective batch: 60-80 across 10 GPUs

## Troubleshooting

### OOM Errors
```bash
# Reduce batch size
--batch-size 6

# Reduce workers
--num-workers 2
```

### NCCL Issues
```bash
# Add to environment
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

### Slow Data Loading
```bash
# Increase workers (if not hitting memory limits)
--num-workers 6

# Check data preprocessing is complete
```

## Model Requirements

- **Checkpoint Format**: PyTorch .pth file with `model_state_dict`
- **Config Compatibility**: Model config must match training configuration
- **Feature Support**: Model must output all required features:
  - `logits`: Language model outputs
  - `adapted_features`: Main feature representations
  - `feature_norm`: Feature magnitude for regularization
  - Contrastive features (if contrastive learning enabled)
  - Destination features (if destination loss enabled)

## Comparison with Training

The evaluation loss calculation exactly matches the training loop:
- Same loss components and weights
- Same feature normalization and clipping
- Same contrastive learning implementation
- Same mixed precision settings

This ensures evaluation metrics are directly comparable to training loss curves.

## Expected Runtime

For 10 RTX 2080 GPUs with batch size 8:
- ~5-10 minutes per dataset split
- ~15-30 minutes total evaluation time
- Depends on dataset size and model complexity

## Output Analysis

The comprehensive loss breakdown helps identify:
- **CE Loss**: Language modeling capability
- **Destination Loss**: Spatial reasoning performance  
- **Contrastive Loss**: Contextual understanding quality
- **KD Loss**: Knowledge transfer effectiveness
- **Feature Reg**: Feature stability

High total loss with low CE loss may indicate issues with other components.
High contrastive loss suggests poor contextual representations.
High destination loss indicates spatial alignment problems. 