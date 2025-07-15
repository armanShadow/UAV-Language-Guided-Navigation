# Comprehensive AVDN Pipeline Usage

## Quick Start

### Testing Mode (Default)
Test the pipeline with a few episodes:
```bash
cd AnsweringAgent/src/data
python3 comprehensive_avdn_pipeline.py
```

### Full Processing Mode
To process all dataset splits (train, val_seen, val_unseen), edit `comprehensive_avdn_pipeline.py`:
```python
TEST_MODE = False  # Change this line in main()
```
Then run:
```bash
python3 comprehensive_avdn_pipeline.py
```

## What It Does

### Test Mode (Default):
1. **Loads few episodes** from `processed_data/train_data.json`
2. **Processes dialog turns** that have answers (skips Turn 0)
3. **Generates paraphrases** using Mixtral-8x7B-Instruct (GPUs 0-8)
4. **Validates paraphrases** using comprehensive spatial analysis (GPU 9)
5. **Saves augmented dataset** to `augmented_data/train_data_with_paraphrases.json`

### Full Processing Mode:
1. **Processes ALL three dataset splits**: train, val_seen, val_unseen
2. **Loads from multiple sources**:
   - `processed_data/train_data.json`
   - `processed_data/val_seen_data.json`
   - `processed_data/val_unseen_data.json`
3. **Saves to separate files**:
   - `augmented_data/train_data_with_paraphrases.json`
   - `augmented_data/val_seen_data_with_paraphrases.json`
   - `augmented_data/val_unseen_data_with_paraphrases.json`
4. **Comprehensive statistics** for all splits

## Features

✅ **Mixtral-8x7B-Instruct** paraphrase generation  
✅ **Comprehensive spatial validation** with UAV navigation awareness  
✅ **Clock direction recognition** (1 o'clock, 2 o'clock, etc.)  
✅ **Synonym-aware validation** (north/northern, building/structure)  
✅ **Multi-word landmark handling** (parking lot)  
✅ **Memory optimization** for 10-GPU setup  
✅ **AVDN structure preservation** (keeps Turn 0, adds paraphrases field)  
✅ **Multiple dataset splits** (train, val_seen, val_unseen)  
✅ **Comprehensive statistics** and progress tracking  

## Hardware Requirements

- **10 GPUs** (RTX 2080 Ti or equivalent)
- **110GB+ total GPU memory**
- **GPUs 0-8**: Mixtral model (distributed)
- **GPU 9**: Validation model (dedicated)

## Output Format

The pipeline adds a `paraphrases` field to dialog turns that have answers:

```json
{
  "episode_id": "1855_1",
  "dialogs": [
    {
      "turn": 0,
      "question": null,
      "answer": null
    },
    {
      "turn": 1, 
      "question": "Where should I go?",
      "answer": "Turn right towards the building",
      "paraphrases": {
        "positives": ["Make a right turn toward the structure", "Go right to the building"],
        "negatives": ["Turn left towards the house"],
        "valid_positives": ["Make a right turn toward the structure"],
        "valid_negatives": ["Turn left towards the house"],
        "validation_analysis": { ... }
      }
    }
  ]
}
```

## Configuration

### Test Mode Configuration
Edit these variables in `comprehensive_avdn_pipeline.py`:
```python
TEST_MODE = True  # Set to False for full processing
MAX_TEST_EPISODES = 2  # Number of episodes for testing
```

### Dataset Paths
The pipeline automatically uses these paths:
- **Input**: `processed_data/{split}_data.json`
- **Output**: `augmented_data/{split}_data_with_paraphrases.json`

### Processing Modes
1. **Test Mode**: Process 2 episodes from train split (fast testing)
2. **Full Mode**: Process all episodes from all 3 splits (production)

## Expected Processing Times

### Test Mode (2 episodes):
- **Time**: ~2-5 minutes
- **Purpose**: Testing pipeline functionality

### Full Mode (entire dataset):
- **Train split**: ~2-6 hours (thousands of episodes)
- **Val_seen split**: ~30-60 minutes  
- **Val_unseen split**: ~30-60 minutes
- **Total**: ~3-8 hours for complete dataset

## Dataset Splits

The pipeline processes three AVDN dataset splits:

1. **train**: Training data (largest split)
2. **val_seen**: Validation data with seen environments
3. **val_unseen**: Validation data with unseen environments

## Troubleshooting

**CUDA Out of Memory?**
- Use test mode first: `TEST_MODE = True`
- Ensure no other processes are using GPUs: `nvidia-smi`
- Check GPU memory status in pipeline logs

**Missing Dataset Files?**
- Ensure processed data exists: `ls processed_data/`
- Run dataset preprocessing if needed: `python format_avdn_dataset.py`

**Import Errors?**
- Ensure you're in the correct directory: `AnsweringAgent/src/data`
- Install required packages: `pip install -r ../../requirements.txt`

**Slow Processing?**
- Start with test mode to verify functionality
- Use full mode only when ready for production processing
- Monitor GPU utilization with `nvidia-smi`

## Statistics

The pipeline provides comprehensive statistics:
- Episodes processed per split
- Dialog turns with answers
- Successful/failed paraphrases
- Success rates
- Processing times
- Memory usage tracking 