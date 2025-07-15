# Comprehensive AVDN Pipeline Usage

## Quick Start

Run the comprehensive pipeline to augment your AVDN dataset:

```bash
cd AnsweringAgent/src/data
python3 comprehensive_avdn_pipeline.py
```

## What It Does

1. **Loads AVDN dataset** from `processed_data/train_data.json`
2. **Processes dialog turns** that have answers (skips Turn 0)
3. **Generates paraphrases** using Mixtral-8x7B-Instruct (GPUs 0-8)
4. **Validates paraphrases** using comprehensive spatial analysis (GPU 9)
5. **Saves augmented dataset** to `augmented_data/train_data_with_paraphrases.json`

## Features

✅ **Mixtral-8x7B-Instruct** paraphrase generation  
✅ **Comprehensive spatial validation** with UAV navigation awareness  
✅ **Clock direction recognition** (1 o'clock, 2 o'clock, etc.)  
✅ **Synonym-aware validation** (north/northern, building/structure)  
✅ **Multi-word landmark handling** (parking lot)  
✅ **Memory optimization** for 10-GPU setup  
✅ **AVDN structure preservation** (keeps Turn 0, adds paraphrases field)  

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

Edit these variables in `comprehensive_avdn_pipeline.py`:

- `self.dataset_path`: Input AVDN dataset path
- `self.output_path`: Output augmented dataset path
- `max_episodes=2`: Number of episodes to process (for testing)

## Troubleshooting

**CUDA Out of Memory?**
- Reduce `max_episodes` for testing
- Ensure no other processes are using GPUs
- Check GPU memory with `nvidia-smi`

**Import Errors?**
- Ensure you're in the correct directory: `AnsweringAgent/src/data`
- Install required packages: `pip install -r requirements.txt` 