# Training with Augmented Paraphrase Dataset

## Overview

The AnsweringAgent training pipeline has been updated to support augmented datasets with paraphrases for contrastive learning. This enables the model to learn better semantic representations by understanding synonymous and contrasting navigation instructions.

## Key Features

### 1. **Automatic Augmented Data Usage**
- **Default behavior**: The system now uses augmented datasets with paraphrases by default
- **Configuration**: `use_augmented_data: bool = True` in config
- **Contrastive learning**: Enabled by default (`use_contrastive_learning: bool = True`)

### 2. **Command Line Control**
New command line arguments for explicit dataset control:

```bash
# Use augmented dataset (default behavior)
python train.py --use-augmented-data

# Use original dataset without paraphrases  
python train.py --no-augmented-data

# Other existing arguments still work
python train.py --batch-size 16 --grad-steps 4 --use-augmented-data
```

### 3. **Dataset Path Selection**
The system automatically selects the correct dataset files:

**Augmented Data Paths** (default):
- Train: `AnsweringAgent/src/data/augmented_data/train_data_with_paraphrases.json`
- Val Seen: `AnsweringAgent/src/data/augmented_data/val_seen_data_with_paraphrases.json`
- Val Unseen: `AnsweringAgent/src/data/augmented_data/val_unseen_data_with_paraphrases.json`

**Original Data Paths** (with `--no-augmented-data`):
- Train: `AnsweringAgent/src/data/processed_data/train_data.json`
- Val Seen: `AnsweringAgent/src/data/processed_data/val_seen_data.json`  
- Val Unseen: `AnsweringAgent/src/data/processed_data/val_unseen_data.json`

## Training Commands

### Basic Training with Paraphrases (Recommended)
```bash
cd AnsweringAgent/src
python train.py
```

### Multi-GPU Training with Paraphrases
```bash
cd AnsweringAgent/src
torchrun --nproc_per_node=2 train.py
```

### Training with Original Dataset Only
```bash
cd AnsweringAgent/src
python train.py --no-augmented-data
```

### Custom Configuration
```bash
cd AnsweringAgent/src
python train.py --batch-size 8 --grad-steps 6 --use-augmented-data
```

## Logging and Monitoring

The training script now provides detailed logging about dataset usage:

```
Dataset configuration - Augmented data: enabled
  Train: /path/to/train_data_with_paraphrases.json
  Val Seen: /path/to/val_seen_data_with_paraphrases.json
  Val Unseen: /path/to/val_unseen_data_with_paraphrases.json
  Contrastive Learning: enabled
```

## Contrastive Learning Configuration

The system uses the following contrastive learning settings:

```python
# Training Configuration
use_contrastive_learning: bool = True
contrastive_loss_type: str = "triplet"  # Options: "triplet", "infonce", "supcon"
contrastive_margin: float = 0.5
contrastive_temperature: float = 0.07
contrastive_weight_start: float = 0.1
contrastive_weight_end: float = 0.5
```

## Data Processing Pipeline

### Paraphrase Structure
Each dialog turn in the augmented dataset contains:
```json
{
  "question": "Which direction should I go?",
  "answer": "Turn right towards the white building",
  "paraphrases": {
    "positives": [
      "Go right toward the white structure",
      "Head right to the white edifice"
    ],
    "negatives": [
      "Turn left towards the gray building"
    ]
  }
}
```

### Contrastive Learning Process
1. **Anchor**: Original answer
2. **Positives**: Paraphrased versions with same meaning
3. **Negatives**: Semantically different responses
4. **Loss**: Triplet loss pulls positives closer, pushes negatives apart

## Prerequisites

Before training with augmented data, ensure you have:

1. **Generated Augmented Dataset**:
   ```bash
   cd AnsweringAgent/src/data
   python comprehensive_avdn_pipeline.py
   ```

2. **Verified Integration**:
   ```bash
   cd AnsweringAgent/src
   python test_paraphrase_integration.py
   ```

## Expected Improvements

Training with paraphrases should provide:

- **Better semantic understanding**: Model learns to recognize equivalent navigation instructions
- **Improved generalization**: Better performance on unseen instruction phrasings
- **Robust representations**: More stable feature spaces through contrastive learning
- **Enhanced dialogue understanding**: Better handling of varied user language

## Troubleshooting

### Missing Augmented Data
If you see warnings about missing augmented data files:
```bash
cd AnsweringAgent/src/data
python comprehensive_avdn_pipeline.py
```

### Memory Issues
If you encounter GPU memory issues with contrastive learning:
```bash
python train.py --batch-size 4 --grad-steps 8
```

### Disable Contrastive Learning
To use augmented data without contrastive learning, modify `config.py`:
```python
use_contrastive_learning: bool = False
```

## Performance Monitoring

Monitor these additional metrics during training:
- **Contrastive Loss**: Should decrease over time
- **Triplet Accuracy**: Percentage of correct triplet rankings
- **Feature Similarity**: Cosine similarity between paraphrases
- **Negative Mining**: Difficulty of negative examples

## Integration Status

✅ **Normalizer**: Processes paraphrases from augmented dataset  
✅ **Config**: Handles augmented data paths correctly  
✅ **Dataset**: Loads augmented data when `use_augmented_data=True`  
✅ **Model**: Processes contrastive examples properly  
✅ **Training**: Uses paraphrases for contrastive learning  
✅ **Command Line**: Controls dataset selection via arguments  

The system is ready for production training with paraphrase-based contrastive learning! 