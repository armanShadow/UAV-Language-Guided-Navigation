# Hard Negative Mining Guide

This guide explains how to add a second negative sample to your AVDN dataset to improve contrastive learning.

## Overview

The negative mining script adds **one additional negative sample** per anchor to the existing dataset (in addition to the existing LM-generated negative), with a 50/50 split between:

### 1. Hard Negatives (50% of samples)
- **Visual K-NN**: Find K nearest visual neighbors for each anchor
- **MPNet embeddings**: Use MPNet for better text similarity (same as KD teacher)
- **Least-similar instruction**: Among neighbors, find instruction with lowest cosine similarity
- **Different goal**: Ensure hard negative has different first instruction (goal)
- **Answer quality filter**: Only select detailed answers (min 20 chars, no simple commands)

### 2. Diverse Negatives (50% of samples)
- **Visual clustering**: Use K-means to cluster visual features
- **Cross-cluster sampling**: Pick samples from different visual clusters
- **Semantic diversity**: Ensures semantic variety beyond visual similarity
- **MPNet embeddings**: Use MPNet for better text similarity
- **Answer quality filter**: Only select detailed answers (min 20 chars, no simple commands)

## Quick Start

### 1. Test the functionality (recommended)

```bash
cd AnsweringAgent/src
python test_hard_negatives.py
```

This tests the mining with a small subset (50 samples) to verify everything works.

### 2. Add negatives to train dataset

```bash
cd AnsweringAgent/src
python data/add_hard_negatives.py \
    --split train \
    --k-nn 50 \
    --cosine-threshold 0.3 \
    --image-dir ../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images \
    --use-diverse-negatives \
    --diverse-ratio 0.5 \
    --min-answer-length 20
```

### 3. Add negatives to validation dataset (optional)

```bash
python data/add_hard_negatives.py \
    --split val_seen \
    --k-nn 50 \
    --cosine-threshold 0.3 \
    --image-dir ../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images
```

## Parameters

- `--split`: Dataset split ('train', 'val_seen', 'val_unseen')
- `--k-nn`: Number of K-NN neighbors to consider (default: 50)
- `--cosine-threshold`: Threshold for considering instructions as dissimilar (default: 0.3)
- `--max-samples`: Maximum samples to process (for testing, default: None = all)
- `--image-dir`: Directory containing satellite images (required)
- `--use-diverse-negatives`: Whether to add diverse negatives from outside clusters (default: True)
- `--diverse-ratio`: Ratio of samples to use for diverse negative mining (default: 0.5 for 50/50 split)
- `--min-answer-length`: Minimum answer length to consider (default: 20 characters)

## How It Works

### 1. Visual Feature Extraction
- Uses simple average pooling (8x8) to extract lightweight visual features
- Normalizes features for cosine similarity

### 2. Text Feature Extraction
- Uses MPNet embeddings for better text similarity (same as KD teacher)
- Falls back to bag-of-words if MPNet unavailable
- Normalizes features for cosine similarity

### 3. Negative Selection (50/50 split)
For each anchor:
1. **Randomly decide**: 50% chance for hard negative, 50% for diverse negative
2. **If hard negative**: Find K nearest visual neighbors using K-NN
3. **If diverse negative**: Use K-means clustering, pick from different clusters
4. **For both**: Find instruction with lowest MPNet cosine similarity
5. **Ensure different goal**: Different first instruction (goal)
6. **Quality filter**: Only select detailed answers (min 20 chars, no simple commands)
7. **Add as negative**: Only one additional negative per sample

### 4. Dataset Integration
- Adds `negative_text_2` and `tokenized_negative_2` to each sample
- Preserves existing LM-generated `negative_text` and `tokenized_negative`
- Adds detailed validation metadata including:
  - **Hard negatives**: Text similarity (MPNet) + visual similarity (K-NN distance)
  - **Diverse negatives**: Anchor cluster, negative cluster, and visual similarity
  - **Map names**: Geographic location information
  - **Mining timestamps**: When the negative was mined
- Updates both main item and contrastive_data

## Training Integration

The hard negatives are automatically integrated into training:

1. **Dataset loading**: Hard negatives are loaded with other contrastive examples
2. **Model forward pass**: Hard negatives are processed alongside positives and LM negatives
3. **Contrastive loss**: Hard negatives provide additional challenging examples

## Expected Results

With second negative added (50% hard, 50% diverse) in addition to existing LM negative:

- **Raw contrastive loss**: Should jump to ≈ 1.1 initially, then fall to < 0.05
- **Validation loss**: Should improve from 2.33 to ≈ 2.1-2.2
- **Training stability**: More challenging examples prevent early convergence
- **Semantic diversity**: Diverse negatives provide broader semantic coverage
- **Visual-semantic alignment**: Better understanding of visual-semantic relationships
- **MPNet consistency**: Same embeddings used for KD and negative mining
- **Dual negatives**: Both LM-generated and mined negatives for comprehensive learning
- **Validation reporting**: Detailed text + visual similarity scores and cluster information for analysis

## Monitoring

Monitor these metrics during training:

```python
# In training logs, look for:
"Contrast: 1.1xxx"  # Initial jump with hard negatives
"Contrast: 0.0xxx"  # Should fall to < 0.05 within 30 epochs
"Validation Loss: 2.1x"  # Should improve from 2.33
```

## Troubleshooting

### Memory Issues
- Reduce `--k-nn` (try 20-30)
- Use `--max-samples` to test with subset first

### No Hard Negatives Found
- Increase `--cosine-threshold` (try 0.4-0.5)
- Check if dataset has sufficient diversity

### Poor Quality Hard Negatives
- Decrease `--cosine-threshold` (try 0.2-0.25)
- Increase `--k-nn` to consider more neighbors

## Training Recipe

After adding hard negatives:

1. **Load checkpoint 1300**
2. **Lower contrastive weight**: 25 → 10 for first 5-10 epochs
3. **Restore weight**: Back to 25 when raw loss < 0.2
4. **Monitor**: Raw contrastive should fall < 0.05 in ≤ 30 epochs
5. **Early stop**: When validation loss stops improving

## File Structure

```
AnsweringAgent/src/
├── data/
│   ├── add_hard_negatives.py      # Main mining script
│   ├── dataset.py                  # Updated to handle hard negatives
│   └── Normalizer.py              # Image processing utilities
├── models/
│   └── answering_agent.py         # Updated to process hard negatives
├── train.py                       # Updated training loop
├── test_hard_negatives.py         # Test script
└── HARD_NEGATIVE_GUIDE.md        # This guide
```

## Performance Impact

- **Memory**: +5-7% for one additional negative per batch
- **Time**: +5-7% per training step
- **Quality**: Significant improvement in contrastive learning effectiveness
- **Diversity**: Better semantic coverage with 50/50 hard/diverse split
- **MPNet reuse**: Leverages existing MPNet embeddings from KD

## Next Steps

1. Run the test script to verify functionality
2. Add hard negatives to train dataset
3. Resume training from epoch 1300 with updated contrastive weights
4. Monitor contrastive loss and validation improvement
5. Early stop when validation loss plateaus

This should give you the final, stable model for regenerating instructions and running the AVDN baseline. 