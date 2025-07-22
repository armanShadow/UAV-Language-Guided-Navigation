# Visual Similarity Filtering Improvements

## Overview

This document describes the improvements made to the hard negative mining process to ensure that hard negatives are both visually similar and textually dissimilar.

## Problem Statement

The original hard negative mining had these issues:

1. **Wide visual similarity range**: Hard negatives had visual similarities ranging from -0.865 to 0.851, with many being visually dissimilar
2. **Poor filtering**: The algorithm accepted any visual similarity as long as text similarity was below threshold
3. **Performance bottleneck**: Phrase diversity checking was taking 85% of processing time

## Solution

### 1. Minimum Visual Similarity Filter

Added a new parameter `min_visual_similarity` (default: 0.30) that enforces a minimum visual similarity for hard negatives.

```python
# Only consider neighbors with visual similarity >= min_visual_similarity
if visual_similarity < self.min_visual_similarity:
    break  # Stop scanning, remaining neighbors are too dissimilar
```

### 2. Visual Similarity Sorting

Changed the search order to prioritize visually similar neighbors:

```python
# Sort by visual similarity (descending)
valid_indices_with_sims.sort(key=lambda x: x[1], reverse=True)

# Search through neighbors in visual similarity order
for (i, sample_idx, neighbor_answer), visual_similarity in valid_indices_with_sims:
    # Process in order of visual similarity
```

### 3. Performance Optimizations

#### Phrase Diversity Optimization
- Only run phrase diversity check on candidates that pass text similarity
- Early exit when reuse limit is reached
- Skip expensive similarity checks for short phrases

#### Pre-computation
- Pre-compute all text similarities upfront
- Pre-compute all visual similarities upfront
- Sort once, then process in order

### 4. New Command Line Parameter

```bash
python add_hard_negatives.py --min-visual-similarity 0.30
```

## Expected Results

### Visual Similarity Statistics
- **Before**: avg 0.007 ± 0.404, range -0.865 → 0.851
- **After**: avg 0.4-0.6 ± 0.1, range 0.3 → 0.8

### Performance
- **Before**: Phrase diversity check was 85% of processing time
- **After**: Phrase diversity check should be <10% of processing time

### Success Rate
- Should maintain 90-95% success rate with stricter visual requirements
- 2-5% failure rate is acceptable for better quality

## Validation

The improvements include:

1. **Visual similarity validation**: Check that all hard negatives meet minimum visual similarity
2. **Statistics reporting**: Show mean, std, and range for visual similarities
3. **Warning system**: Alert if any hard negatives fall below threshold

## Usage

```bash
# Standard usage with default 0.30 minimum visual similarity
python add_hard_negatives.py --config config.py --split train --image-dir /path/to/images

# Stricter visual similarity requirement
python add_hard_negatives.py --config config.py --split train --image-dir /path/to/images --min-visual-similarity 0.50

# More lenient for small datasets
python add_hard_negatives.py --config config.py --split train --image-dir /path/to/images --min-visual-similarity 0.20
```

## Code Changes

### Key Files Modified
- `add_hard_negatives.py`: Main implementation
- `test_visual_filtering.py`: Test script for validation

### Key Methods Modified
- `HardNegativeMiner.__init__()`: Added `min_visual_similarity` parameter
- `HardNegativeMiner.find_hard_negative()`: Complete rewrite with visual similarity filtering
- `HardNegativeMiner._is_phrase_diverse()`: Performance optimizations
- `main()`: Added command line argument

### New Features
- Visual similarity pre-computation and sorting
- Minimum visual similarity enforcement
- Performance-optimized phrase diversity checking
- Enhanced statistics reporting
- Validation warnings for quality control 