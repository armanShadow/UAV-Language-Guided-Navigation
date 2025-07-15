# AVDN Dataset Processing Pipeline

This directory contains the essential components for processing the AVDN dataset and generating paraphrases for AnsweringAgent training.

## Core Pipeline Files

### `comprehensive_avdn_pipeline.py` 
**🚀 SINGLE COMPREHENSIVE PIPELINE for AVDN dataset augmentation**
- **Generation**: Mixtral-8x7B-Instruct paraphrases (GPUs 0-8, distributed)
- **Validation**: Comprehensive spatial validation (GPU 9) with:
  - Embedding similarity validation
  - Spatial feature analysis (directions, landmarks, movement verbs)  
  - Clock direction recognition (1 o'clock, 2 o'clock, etc.)
  - Synonym-aware validation (north/northern, building/structure)
  - Multi-word landmark handling (parking lot)
  - UAV navigation terminology awareness
- **Processing**: Sequential episode processing with memory optimization
- **Dataset Coverage**: ALL three AVDN splits (train, val_seen, val_unseen)
- **Output**: Augmented AVDN datasets with paraphrases field added to dialog turns
- **Modes**: Test mode (few episodes) and Full mode (entire dataset)
- **Usage**: `python comprehensive_avdn_pipeline.py`

**Dataset Processing:**
- **Input**: `processed_data/{train,val_seen,val_unseen}_data.json`
- **Output**: `augmented_data/{train,val_seen,val_unseen}_data_with_paraphrases.json`
- **Statistics**: Comprehensive tracking across all splits
- **Processing Time**: ~2-5 min (test), ~3-8 hours (full dataset)

*This replaces all previous separate components (avdn_dataset_augmenter, paraphrase_validator, paraphrase_generator) with one unified solution.*

### `paraphrase_generation_pipeline.py`
**Paraphrase generation component (cleaned)**
- Uses Mixtral-8x7B-Instruct model for text generation
- Generates positive and negative paraphrases with strategic spatial changes
- Memory-optimized for multi-GPU distributed inference
- **Cleaned**: Removed redundant spatial extraction (handled by validation pipeline)
- **Cleaned**: Removed test functions (handled by comprehensive pipeline)

### `validation_pipeline.py`
**Paraphrase validation component (comprehensive)**
- Uses sentence-transformers for embedding similarity
- **Comprehensive spatial feature extraction** with regex patterns and synonyms
- Validates positive paraphrases for spatial preservation
- Validates negative paraphrases for appropriate spatial changes
- Clock direction recognition and landmark synonym matching
- **Cleaned**: Removed test functions (handled by comprehensive pipeline)

## Architecture Benefits

### ✅ **Redundancy Elimination:**
- **Single spatial extraction**: Only in validation pipeline (comprehensive version)
- **Single testing interface**: Only in comprehensive pipeline
- **No duplicate dataset loading**: Centralized in comprehensive pipeline
- **Clean separation**: Generation focuses on text creation, validation on quality assessment

### 🔧 **Modular Design:**
- **ParaphraseGenerationPipeline**: Pure text generation (GPUs 0-8)
- **ValidationPipeline**: Pure validation logic (GPU 9)
- **ComprehensiveAVDNPipeline**: Orchestrates both + dataset handling

### 🚀 **Performance Optimized:**
- No redundant feature extraction during generation
- Comprehensive validation ensures quality
- Memory-optimized GPU usage across 10 GPUs

## Usage

### Quick Test
```bash
python comprehensive_avdn_pipeline.py  # TEST_MODE = True (default)
```

### Full Processing
Edit `comprehensive_avdn_pipeline.py`:
```python
TEST_MODE = False  # Process all splits
```
Then run:
```bash
python comprehensive_avdn_pipeline.py
```

## Dataset Analysis

### `analyze_dataset_patterns.py`
Analyzes AVDN dataset for spatial patterns and navigation terminology.

### `contrastive_sample_generator.py`
Generates contrastive learning samples for model training.

### Supporting Files
- `dataset.py` - Core dataset loading utilities
- `Normalizer.py` - Text normalization for consistency 