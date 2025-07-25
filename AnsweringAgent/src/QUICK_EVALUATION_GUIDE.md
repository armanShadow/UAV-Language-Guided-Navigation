# Quick Evaluation Guide for 10 RTX 2080 Server

## 🚀 One-Command Evaluation

```bash
cd AnsweringAgent/src
chmod +x run_distributed_evaluation.sh
./run_distributed_evaluation.sh /path/to/your/checkpoint_epoch_X_fp32.pth
```

## 📊 What You'll Get

The evaluation will report comprehensive loss values **exactly as calculated during training**:

### Loss Components
1. **Total Loss**: Complete weighted loss matching training
2. **CE Loss**: Cross-entropy (language modeling)  
3. **Destination Loss**: Spatial reasoning
4. **Contrastive Loss**: Context understanding
5. **KD Loss**: Knowledge distillation
6. **Feature Reg Loss**: Feature regularization

### Outputs
- Console logs with detailed breakdown
- JSON file with all metrics: `evaluation_results.json`
- Sample predictions for inspection

## 🔧 If You Need to Customize

### Different Batch Size
```bash
# Edit run_distributed_evaluation.sh and change:
--batch-size 6  # If you get OOM errors
--batch-size 10 # If you have extra memory
```

### Manual Command
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

torchrun \
    --nproc_per_node=10 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    evaluate_distributed.py \
    --checkpoint /path/to/checkpoint.pth \
    --batch-size 8 \
    --output-dir my_results
```

## 📈 Expected Output Format

```
📈 TRAIN Results:
  Total Loss: 2.1234
  CE Loss: 1.8765
  Destination Loss: 0.1234
  Contrastive Loss: 0.0876
  KD Loss: 0.0543
  Feature Reg Loss: 0.0012

🔍 Effective Loss Components:
  CE (weighted): 0.9383      # 1.8765 * 0.5
  Destination (weighted): 0.0247  # 0.1234 * 0.2
  Contrastive (weighted): 0.8760  # 0.0876 * 10.0
  KD (weighted): 0.0272      # 0.0543 * 0.5
```

## ⏱️ Expected Runtime

- **Per Dataset**: ~5-10 minutes
- **Total Time**: ~15-30 minutes  
- **3 Datasets**: train, val_seen, val_unseen

## 🎯 Key Features

✅ **Training-Matched Calculation**: Loss calculated exactly as in training loop  
✅ **Multi-GPU Scaling**: 10x speedup with distributed evaluation  
✅ **Comprehensive Metrics**: All loss components reported separately  
✅ **Memory Optimized**: Efficient for RTX 2080 (11GB VRAM)  
✅ **Ready-to-Use**: No additional setup required  

## 📄 Output Files

After completion, check:
- `evaluation_results_TIMESTAMP/evaluation_results.json`: Complete metrics
- Console logs: Real-time progress and results
- Sample outputs: Model prediction examples

The JSON file contains weighted contributions for analysis:
```json
"weighted_contributions": {
  "ce_weighted": 0.9383,
  "destination_weighted": 0.0247, 
  "contrastive_weighted": 0.8760,
  "kd_weighted": 0.0272,
  "feature_reg": 0.0012
}
```

## 🔍 What These Numbers Mean

- **High CE Loss**: Language modeling issues
- **High Destination Loss**: Spatial reasoning problems  
- **High Contrastive Loss**: Poor context understanding
- **High KD Loss**: Teacher-student misalignment
- **High Feature Reg**: Feature instability

Compare these values to training curves to assess model convergence and identify areas for improvement. 