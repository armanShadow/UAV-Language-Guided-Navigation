# Hard Negative Mining Guide

## Overview

The enhanced `add_hard_negatives.py` now supports optimized single-GPU processing with comprehensive metrics reporting for hard negative mining across large datasets.

## Usage Modes

### 1. Single-GPU Mode (Default)
```bash
python add_hard_negatives.py \
    --image-dir /path/to/images \
    --split train \
    --gpu-id 0 \
    --k-nn 100 \
    --cosine-threshold 0.2 \
    --min-visual-similarity 0.15 \
    --diverse-ratio 0.0
```

### 2. Optimized Single-GPU Mode (Recommended)
```bash
python add_hard_negatives.py \
    --image-dir /path/to/images \
    --split train \
    --gpu-id 0 \
    --k-nn 100 \
    --cosine-threshold 0.2 \
    --min-visual-similarity 0.15 \
    --diverse-ratio 0.0 \
    --batch-size 64
```

## Key Features

### Optimized Single-GPU Processing
- Efficient batch processing for maximum GPU utilization
- Memory-optimized operations for RTX 2080/2080 Ti
- Automatic fallback to CPU if no GPU available

### Enhanced Metrics Reporting
- Success rate and processing statistics
- Hard vs diverse negative distribution
- Visual and text similarity statistics
- Phrase diversity analysis
- Performance timing breakdown
- Comprehensive quality assessment

## Command Line Arguments

### Performance Optimization
- `--batch-size`: Batch size for GPU processing (default: 64)
- `--num-workers`: Number of workers for data loading (default: 4)

### Mining Parameters
- `--k-nn`: Number of K-NN neighbors (default: 30)
- `--cosine-threshold`: Text similarity threshold (default: 0.3)
- `--min-visual-similarity`: Minimum visual similarity for hard negatives (default: 0.30)
- `--diverse-ratio`: Ratio of diverse negatives (default: 0.3)
- `--min-answer-length`: Minimum answer length (default: 20)
- `--fallback-phrase-reuse-limit`: Max phrase reuse in fallback mode (default: 3)

### Performance
- `--batch-size`: Batch size per GPU (default: 64)
- `--num-workers`: Workers per GPU (default: 4)

## Example Commands for Different Splits

### Training Split (Optimized)
```bash
python add_hard_negatives.py \
    --image-dir /path/to/train_images \
    --split train \
    --gpu-id 0 \
    --k-nn 100 \
    --cosine-threshold 0.2 \
    --min-visual-similarity 0.15 \
    --diverse-ratio 0.0 \
    --min-answer-length 20 \
    --batch-size 64
```

### Validation Seen (Optimized)
```bash
python add_hard_negatives.py \
    --image-dir /path/to/val_seen_images \
    --split val_seen \
    --gpu-id 0 \
    --k-nn 50 \
    --cosine-threshold 0.25 \
    --min-visual-similarity 0.20 \
    --diverse-ratio 0.0 \
    --batch-size 32
```

### Validation Unseen (Optimized)
```bash
python add_hard_negatives.py \
    --image-dir /path/to/val_unseen_images \
    --split val_unseen \
    --gpu-id 0 \
    --k-nn 50 \
    --cosine-threshold 0.25 \
    --min-visual-similarity 0.20 \
    --diverse-ratio 0.0 \
    --batch-size 32
```

## Expected Output

### Multi-GPU Mode Output
```
🔍 Detected 8 GPU(s): [0, 1, 2, 3, 4, 5, 6, 7]
   GPU 0: 11.0 GB
   GPU 1: 11.0 GB
   ...
🚀 Starting multi-GPU mining with 8 GPUs
📊 Dataset sharding: 8 shards of ~1250 samples each

⏱️  Monitoring 8 mining processes...
✅ GPU 0 (shard 0) completed successfully
   📊 Shard 0 results:
      Success rate: 100.0%
      Hard negatives: 875
      Diverse negatives: 375
      Avg hard text sim: 0.234
      Avg hard visual sim: 0.456
...

🎉 Multi-GPU mining completed!
⏱️  Total time: 1247.32s
📊 Total samples processed: 10000
🎯 Total hard negatives: 7340
🌈 Total diverse negatives: 2660
📈 Average success rate: 100.0%
🔤 Average hard text similarity: 0.241
👁️ Average hard visual similarity: 0.467
```

## Performance Tips

1. **GPU Memory**: RTX 2080 (8GB) can handle ~1000-1500 samples per GPU
2. **Batch Size**: Increase `--batch-size` for faster processing if memory allows
3. **K-NN**: Higher `--k-nn` values improve hard negative quality but slow processing
4. **Visual Similarity**: Lower `--min-visual-similarity` increases hard negative yield
5. **Diverse Ratio**: Set to 0.0 for maximum hard negatives, 0.3 for balanced approach

## Troubleshooting

### GPU Memory Issues
- Reduce `--batch-size` (try 32 or 16)
- Reduce `--max-samples` per GPU
- Use fewer GPUs with `--num-gpus`

### Low Hard Negative Yield
- Lower `--min-visual-similarity` (try 0.10-0.15)
- Increase `--fallback-phrase-reuse-limit` (try 5-8)
- Lower `--cosine-threshold` (try 0.15-0.20)

### Slow Processing
- Increase `--batch-size` if memory allows
- Reduce `--k-nn` (try 50-75)
- Use more GPUs with `--num-gpus`

## Metrics Explained

- **Success Rate**: Percentage of samples that found valid negatives
- **Hard Negatives**: Visually similar but semantically different samples
- **Diverse Negatives**: From different visual clusters
- **Text Similarity**: Lower is better (more semantically different)
- **Visual Similarity**: Higher is better for hard negatives
- **Phrase Diversity**: Ratio of unique phrases used
- **Cluster Diversity**: Percentage of diverse negatives from different clusters 