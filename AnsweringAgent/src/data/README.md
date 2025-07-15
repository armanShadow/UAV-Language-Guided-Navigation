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
- **Output**: Augmented AVDN dataset with paraphrases field added to dialog turns
- **Usage**: `python comprehensive_avdn_pipeline.py`

*This replaces all previous separate components (avdn_dataset_augmenter, paraphrase_validator, paraphrase_generator) with one unified solution.*

### `paraphrase_generation_pipeline.py`
**Paraphrase generation component**
- Uses Mixtral-8x7B-Instruct model for text generation
- Generates positive and negative paraphrases
- Handles spatial term preservation for navigation instructions
- Memory-optimized for multi-GPU distributed inference

### `validation_pipeline.py`
**Paraphrase validation component**
- Uses sentence-transformers for embedding similarity
- Validates positive paraphrases (preserve meaning)
- Validates negative paraphrases (change meaning appropriately)
- Includes spatial feature analysis for navigation context

## Dataset Processing Files

### `analyze_dataset_patterns.py`
**Dataset analysis utilities**
- Analyzes AVDN dataset structure and patterns
- Extracts statistics about dialog turns and answers
- Identifies spatial terms and navigation language patterns

### `format_avdn_dataset.py`
**Dataset formatting utilities**
- Preprocesses raw AVDN dataset
- Formats dialog structure for pipeline processing
- Handles coordinate transformations and metadata

### `dataset.py`
**Dataset loading and processing**
- Core dataset loading functionality
- Handles different dataset splits (train/val)
- Provides data iteration and batching utilities

### `Normalizer.py`
**Text normalization utilities**
- Normalizes spatial terms and navigation language
- Handles coordinate system conversions
- Standardizes text formatting for consistency

## Data Directories

### `processed_data/`
- `train_data.json` - Processed AVDN training data
- `val_seen_data.json` - Validation data (seen environments)
- `val_unseen_data.json` - Validation data (unseen environments)
- `metadata.json` - Dataset statistics and metadata

### `augmented_data/`
- Output directory for augmented datasets
- `avdn_dialog_answers_augmented.json` - Augmented dialog answers with paraphrases

## GPU Configuration

- **GPUs 0-8**: Mixtral model distributed inference (paraphrase generation)
- **GPU 9**: Validation model (embedding similarity and spatial analysis)

## Usage

1. **Process AVDN dataset**:
   ```bash
   python correct_avdn_pipeline.py
   ```

2. **Analyze dataset patterns**:
   ```bash
   python analyze_dataset_patterns.py
   ```

3. **Format raw dataset** (if needed):
   ```bash
   python format_avdn_dataset.py
   ```

## Output Format

The augmented dataset contains dialog answers with:
- Original question and answer
- Generated positive paraphrases (preserve meaning)
- Generated negative paraphrases (change meaning)
- Validation analysis and quality scores
- Processing metadata and timestamps

This augmented data is used to train the AnsweringAgent to generate appropriate responses to navigation questions. 