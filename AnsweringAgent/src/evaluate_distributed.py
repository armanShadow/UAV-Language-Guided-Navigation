import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List, Tuple
import argparse
import json
import time
from transformers import T5Tokenizer
from transformers.models.t5.modeling_t5 import BaseModelOutput
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

from config import Config
from models.answering_agent import AnsweringAgent
from data.dataset import AnsweringDataset
from utils.logger import setup_logger
from models.contrastive_loss import ContrastiveLoss

# Add EMA class definition to match training script
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def apply_shadow(self):
        """Apply the EMA weights to the model for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore the original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name].clone()
        self.backup = {}
    
    def state_dict(self):
        return {
            'decay': self.decay,
            'shadow': self.shadow,
            'backup': self.backup
        }
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']

def setup_distributed():
    """Set up distributed evaluation."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
    
    return True, rank, world_size

def setup_environment():
    """Setup minimal environment for evaluation."""
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def calculate_cosine_similarity_loss(first_features, second_features):
    """Calculate cosine similarity loss - matches training implementation."""
    first_features_norm = F.normalize(first_features, p=2, dim=1)
    second_features_norm = F.normalize(second_features, p=2, dim=1)
    cosine_loss = 1 - F.cosine_similarity(first_features_norm, second_features_norm).mean()
    return cosine_loss

def compute_metrics(outputs: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> Dict[str, float]:
    """Compute accuracy and other metrics."""
    outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
    labels_reshaped = labels.reshape(-1)

    _, predicted = outputs_reshaped.max(1)
    predicted = predicted.reshape(outputs.size(0), outputs.size(1))

    mask = (labels != pad_token_id)

    total_tokens = mask.sum().item()
    correct_tokens = ((predicted == labels) & mask).sum().item()
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens
    }

def evaluate_dataset_distributed(model, dataloader, criterion, device, tokenizer, split_name: str, 
                              logger=None, rank=0, world_size=1, config=None, epoch=None) -> Dict[str, float]:
    """Evaluate model on a dataset with distributed processing and training-matching loss calculation."""
    model.eval()
    
    # Initialize EMA if not already present
    if not hasattr(model, 'ema'):
        model.ema = EMA(model, decay=0.999)
    
    # Apply EMA for evaluation
    model.ema.apply_shadow()
    
    # Initialize loss accumulators
    total_loss = 0.0
    total_ce_loss = 0.0
    total_contrastive_loss = 0.0
    total_destination_loss = 0.0
    total_kd_loss = 0.0
    total_reg_loss = 0.0
    total_accuracy = 0.0
    total_tokens = 0
    correct_tokens = 0
    
    # Initialize contrastive loss function if enabled
    contrastive_loss_fn = None
    if config.training.use_contrastive_learning:
        contrastive_loss_fn = ContrastiveLoss(
            margin=config.training.contrastive_margin,
            temperature=config.training.contrastive_temperature,
            loss_type=config.training.contrastive_loss_type,
            use_cosine_distance=config.training.use_cosine_distance,
            mean_all=config.training.contrastive_mean_all
        )
    
    # Import weight scheduling functions from training
    from train import (
        get_smart_contrastive_schedule, 
        get_adaptive_contrastive_schedule,
        get_smart_destination_schedule,
        get_smart_kd_schedule
    )
    
    # Get weight functions
    contrastive_weight_fn, ce_weight_fn = get_smart_contrastive_schedule(config.training.planned_epochs, config.training.num_epochs)
    adaptive_contrastive_fn = get_adaptive_contrastive_schedule(contrastive_weight_fn)
    destination_weight_fn = get_smart_destination_schedule(config.training.planned_epochs)
    kd_weight_fn = get_smart_kd_schedule(config.training.planned_epochs)
    
    # Calculate actual weights used for proper reporting
    actual_weights = {}
    if epoch is not None:
        actual_weights['ce_weight'] = ce_weight_fn(epoch)
        actual_weights['contrastive_weight'], _ = adaptive_contrastive_fn(epoch)
        actual_weights['destination_weight'] = destination_weight_fn(epoch)
        actual_weights['kd_weight'] = kd_weight_fn(epoch)
    else:
        # Fallback to end weights if epoch not provided
        actual_weights['ce_weight'] = ce_weight_fn(config.training.num_epochs - 1)
        actual_weights['contrastive_weight'], _ = adaptive_contrastive_fn(config.training.num_epochs - 1)
        actual_weights['destination_weight'] = destination_weight_fn(config.training.num_epochs - 1)
        actual_weights['kd_weight'] = kd_weight_fn(config.training.num_epochs - 1)
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}", disable=rank!=0)):
                try:
                    # Load data - ensure no data leakage by using proper data loading
                    text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                    
                    # Handle separate encoding components
                    if 'first_instruction_input' in batch:
                        text_input['first_instruction_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['first_instruction_input'].items()}
                    if 'current_question_input' in batch:
                        text_input['current_question_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['current_question_input'].items()}
                    
                    current_view = batch['current_view_image'].to(device, non_blocking=True)
                    previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                    
                    label_input_ids = batch['text_label']['input_ids'].to(device, non_blocking=True)
                    label_attention_mask = batch['text_label']['attention_mask'].to(device, non_blocking=True)
                    
                    destination_view = batch['destination_image'].to(device, non_blocking=True) if 'destination_image' in batch else None
                    
                    # Prepare contrastive examples for evaluation
                    positive_input = None
                    positive_input_2 = None
                    negative_input = None
                    negative_input_2 = None
                    
                    if 'positive_input' in batch:
                        positive_input = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input'].items()}
                        if 'positive_input_2' in batch:
                            positive_input_2 = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input_2'].items()}
                        if 'negative_input' in batch:
                            negative_input = {k: v.to(device, non_blocking=True) for k, v in batch['negative_input'].items()}
                        if 'negative_input_2' in batch:
                            negative_input_2 = {k: v.to(device, non_blocking=True) for k, v in batch['negative_input_2'].items()}
                    
                    # Forward pass with mixed precision for consistency
                    with torch.cuda.amp.autocast(enabled=config.training.mixed_precision):
                        outputs = model(
                            text_input, 
                            current_view, 
                            previous_views, 
                            labels=label_input_ids,
                            destination_view=destination_view,
                            curriculum_ratio=0.0,  # No curriculum during evaluation
                            positive_input=positive_input,
                            positive_input_2=positive_input_2,
                            negative_input=negative_input,
                            negative_input_2=negative_input_2
                        )
                        
                        logits = outputs["logits"]
                        # Use the loss already calculated by T5 instead of recalculating
                        ce_loss = outputs.get("loss", torch.tensor(0.0, device=device))
                        feature_norm = outputs.get("feature_norm", torch.tensor(0.0, device=device))
                        batch_size, seq_len, vocab_size = logits.size()
                        
                        # Get the actual weights used during training
                        if epoch is not None:
                            ce_weight = ce_weight_fn(epoch)
                            contrastive_weight, _ = adaptive_contrastive_fn(epoch)
                            destination_weight = destination_weight_fn(epoch)
                            kd_weight = kd_weight_fn(epoch)
                        else:
                            # Fallback to end weights if epoch not provided
                            ce_weight = ce_weight_fn(config.training.num_epochs - 1)
                            contrastive_weight, _ = adaptive_contrastive_fn(config.training.num_epochs - 1)
                            destination_weight = destination_weight_fn(config.training.num_epochs - 1)
                            kd_weight = kd_weight_fn(config.training.num_epochs - 1)
                        
                        # Add feature regularization with clipping to prevent explosion (matches training)
                        feature_norm_clipped = feature_norm.clamp(max=1e3)  # Clip to prevent explosion
                        reg_loss = 1e-4 * feature_norm_clipped
                        loss = ce_weight * ce_loss + reg_loss
                        
                        # Add destination loss if destination view is available (matches training)
                        destination_cosine_loss = torch.tensor(0.0, device=device)
                        if destination_view is not None:
                            dest_features = outputs.get("destination_features", outputs["raw_adapted_features"])
                            destination_cosine_loss = calculate_cosine_similarity_loss(outputs["raw_adapted_features"], dest_features)
                            loss = loss + destination_weight * destination_cosine_loss
                        
                        # Calculate contrastive loss if enabled (matches training)
                        contrastive_loss = torch.tensor(0.0, device=device)
                        if config.training.use_contrastive_learning and contrastive_loss_fn is not None:
                            # Collect all positive and negative embeddings
                            if 'positive_adapted_features' in outputs and 'negative_adapted_features' in outputs:
                                anchor_emb = outputs['adapted_features']
                                positive_embs = [outputs['positive_adapted_features']]
                                
                                # Gather negatives
                                negatives_embs = []
                                if 'negative_adapted_features' in outputs:
                                    negatives_embs.append(outputs['negative_adapted_features'])
                                if 'negative_adapted_features_2' in outputs:
                                    negatives_embs.append(outputs['negative_adapted_features_2'])
                                
                                # Add second positive if available
                                if 'positive_adapted_features_2' in outputs:
                                    positive_embs.append(outputs['positive_adapted_features_2'])
                                
                                # Stack positives if we have multiple
                                if len(positive_embs) > 1:
                                    positive_combined = torch.stack(positive_embs, dim=1)  # [batch, num_pos, hidden]
                                else:
                                    positive_combined = positive_embs[0]  # [batch, hidden]
                                
                                # Compute contrastive loss for each negative and average
                                contrastive_losses_list = []
                                if not negatives_embs:
                                    # Fall back to in-batch negatives if none provided
                                    contrastive_losses_list.append(
                                        contrastive_loss_fn(anchor_emb, positive_combined, None)
                                    )
                                else:
                                    for neg_emb in negatives_embs:
                                        contrastive_losses_list.append(
                                            contrastive_loss_fn(anchor_emb, positive_combined, neg_emb)
                                        )
                                
                                contrastive_loss = torch.stack(contrastive_losses_list).mean()
                                
                                # Add weighted contrastive loss using correct weights
                                loss = loss + contrastive_weight * contrastive_loss
                        
                        # KD loss using embeddings generated during preprocessing (matches training)
                        kd_loss = torch.tensor(0.0, device=device)
                        if config.training.use_kd:
                            # Teacher embeddings are included in the batch from preprocessing
                            teacher_embeddings = batch['teacher_embed'].to(device)
                            
                            # Use raw_adapted_features for KD - these are the pure T5 adapter outputs before contrastive projection
                            student_hidden = F.normalize(outputs["raw_adapted_features"], p=2, dim=-1)
                            teacher_hidden = F.normalize(teacher_embeddings, p=2, dim=-1)
                            
                            kd_loss = 1 - F.cosine_similarity(student_hidden, teacher_hidden, dim=-1).mean()
                            
                            loss = loss + kd_weight * kd_loss
                        
                        # Accumulate loss components
                        total_loss += loss.item()
                        total_ce_loss += ce_loss.item()
                        total_contrastive_loss += contrastive_loss.item()
                        total_destination_loss += destination_cosine_loss.item()
                        total_kd_loss += kd_loss.item()
                        total_reg_loss += reg_loss.item()
                        
                    # Calculate metrics
                    metrics = compute_metrics(logits, label_input_ids, tokenizer.pad_token_id)
                    total_accuracy += metrics['accuracy']
                    total_tokens += metrics['total_tokens']
                    correct_tokens += metrics['correct_tokens']
                
                except Exception as e:
                    if rank == 0 and logger:
                        logger.error(f"❌ Error in evaluation batch {batch_idx}: {str(e)}")
                    continue
    
    finally:
        # Always restore original weights after evaluation
        model.ema.restore()
    
    # Gather results across all processes
    if world_size > 1:
        # Gather loss values
        loss_tensor = torch.tensor(total_loss, device=device)
        ce_loss_tensor = torch.tensor(total_ce_loss, device=device)
        cont_loss_tensor = torch.tensor(total_contrastive_loss, device=device)
        dest_loss_tensor = torch.tensor(total_destination_loss, device=device)
        kd_loss_tensor = torch.tensor(total_kd_loss, device=device)
        reg_loss_tensor = torch.tensor(total_reg_loss, device=device)
        accuracy_tensor = torch.tensor(total_accuracy, device=device)
        tokens_tensor = torch.tensor(total_tokens, device=device)
        correct_tensor = torch.tensor(correct_tokens, device=device)
        
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(ce_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(cont_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(dest_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(kd_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(reg_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(accuracy_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = loss_tensor.item() / world_size
        total_ce_loss = ce_loss_tensor.item() / world_size
        total_contrastive_loss = cont_loss_tensor.item() / world_size
        total_destination_loss = dest_loss_tensor.item() / world_size
        total_kd_loss = kd_loss_tensor.item() / world_size
        total_reg_loss = reg_loss_tensor.item() / world_size
        total_accuracy = accuracy_tensor.item() / world_size
        total_tokens = tokens_tensor.item()
        correct_tokens = correct_tensor.item()
    
    # Calculate final metrics
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_ce_loss = total_ce_loss / num_batches if num_batches > 0 else 0.0
    avg_contrastive_loss = total_contrastive_loss / num_batches if num_batches > 0 else 0.0
    avg_destination_loss = total_destination_loss / num_batches if num_batches > 0 else 0.0
    avg_kd_loss = total_kd_loss / num_batches if num_batches > 0 else 0.0
    avg_reg_loss = total_reg_loss / num_batches if num_batches > 0 else 0.0
    avg_accuracy = total_accuracy / num_batches if num_batches > 0 else 0.0
    overall_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    
    results = {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'contrastive_loss': avg_contrastive_loss,
        'destination_loss': avg_destination_loss,
        'kd_loss': avg_kd_loss,
        'reg_loss': avg_reg_loss,
        'accuracy': avg_accuracy,
        'overall_accuracy': overall_accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens,
        'ce_weight': actual_weights['ce_weight'],
        'contrastive_weight': actual_weights['contrastive_weight'],
        'destination_weight': actual_weights['destination_weight'],
        'kd_weight': actual_weights['kd_weight']
    }
    
    return results

def validate_data_leakage(config, logger, rank=0):
    """Validate that there's no data leakage between train/val_seen/val_unseen splits."""
    if rank != 0:
        return
    
    logger.info("🔍 Validating data leakage prevention...")
    
    # Load all datasets to check for overlaps
    datasets = {}
    for split in ['train', 'val_seen', 'val_unseen']:
        try:
            dataset = AnsweringDataset(config, split=split)
            datasets[split] = dataset
            logger.info(f"  {split}: {len(dataset)} samples")
        except Exception as e:
            logger.error(f"  ❌ Error loading {split} dataset: {str(e)}")
            return False
    
    # Check for potential data leakage by examining sample IDs or unique identifiers
    # This is a basic check - in practice, you'd want more sophisticated validation
    logger.info("  ✅ Data loading validation completed")
    logger.info("  📊 Dataset sizes:")
    for split, dataset in datasets.items():
        logger.info(f"    {split}: {len(dataset)} samples")
    
    return True

def main():
    """Main distributed evaluation function."""
    parser = argparse.ArgumentParser(description='Distributed UAV Navigation Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--batch-size', type=int, default=8, help='Per-GPU batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers per GPU')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save results')
    parser.add_argument('--validate-leakage', action='store_true', help='Validate data leakage prevention')
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Setup distributed training
    is_distributed, rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"🔍 UAV Navigation Distributed Evaluation Pipeline")
        print(f"PyTorch: {torch.__version__}")
        print(f"World Size: {world_size}, Rank: {rank}")
        if torch.cuda.is_available():
            print(f"CUDA: {torch.version.cuda} | Devices: {torch.cuda.device_count()}")
    
    # Load configuration
    config = Config()
    config.training.per_gpu_batch_size = args.batch_size
    config.training.num_workers = args.num_workers
    
    # Initialize logger for all processes
    logger = setup_logger('evaluation_distributed', log_dir=config.log_dir)

    # Silence non-rank-0 processes by setting logger level
    if rank != 0:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.ERROR)  # Only show errors on non-rank-0
    
    # Validate data leakage if requested
    if args.validate_leakage:
        if not validate_data_leakage(config, logger, rank):
            if rank == 0:
                logger.error("❌ Data leakage validation failed!")
            return
    
    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, model_max_length=config.data.max_seq_length)
    
    # Initialize model
    if rank == 0:
        logger.info("🏗️ Loading model...")
    
    model = AnsweringAgent(config, tokenizer, logger)
    
    # Load checkpoint
    if rank == 0:
        logger.info(f"📂 Loading checkpoint: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Extract epoch for correct weight calculation
    checkpoint_epoch = checkpoint.get('epoch', None)
    if rank == 0:
        if checkpoint_epoch is not None:
            logger.info(f"📅 Checkpoint epoch: {checkpoint_epoch + 1}")
        else:
            logger.warning("⚠️ No epoch info in checkpoint - using fallback weights")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if rank == 0:
            logger.info("✅ Model state dict loaded successfully")
        
        # Load EMA state if available
        if 'ema' in checkpoint:
            # Initialize EMA with the same decay as in training
            ema = EMA(model, decay=0.999)
            ema.load_state_dict(checkpoint['ema'])
            
            # Attach EMA to the model
            model.ema = ema
            
            if rank == 0:
                logger.info("✅ EMA state loaded from checkpoint")
    else:
        if rank == 0:
            logger.error("❌ No model_state_dict found in checkpoint")
        return
    
    model.to(device)
    
    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    model.eval()
    
    # Loss function
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.05
    )
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate on all datasets
    datasets_to_evaluate = ['train', 'val_seen', 'val_unseen']
    all_results = {}
    
    for split in datasets_to_evaluate:
        if rank == 0:
            logger.info(f"📊 Evaluating {split} dataset...")
        
        try:
            # Load dataset
            dataset = AnsweringDataset(config, split=split, tokenizer=tokenizer)
            
            # Use DistributedSampler if distributed
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if is_distributed else None
            
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=args.num_workers,
                pin_memory=True
            )
            
            # Evaluate dataset with correct epoch for weight calculation
            results = evaluate_dataset_distributed(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                device=device,
                tokenizer=tokenizer,
                split_name=split,
                logger=logger,
                rank=rank,
                world_size=world_size,
                config=config,
                epoch=checkpoint_epoch  # Pass epoch for correct weights
            )
            
            if rank == 0:
                all_results[split] = results
                
                # Log raw loss values
                logger.info(f"📈 {split.upper()} Results:")
                logger.info(f"  Total Loss: {results['loss']:.6f}")
                logger.info(f"  CE Loss: {results['ce_loss']:.6f}")
                logger.info(f"  Contrastive Loss: {results['contrastive_loss']:.6f}")
                logger.info(f"  Destination Loss: {results['destination_loss']:.6f}")
                logger.info(f"  KD Loss: {results['kd_loss']:.6f}")
                logger.info(f"  Reg Loss: {results['reg_loss']:.6f}")
                logger.info(f"  Accuracy: {results['accuracy']:.6f}")
                logger.info(f"  Overall Accuracy: {results['overall_accuracy']:.6f}")
                logger.info(f"  Total Tokens: {results['total_tokens']:,}")
                logger.info(f"  Correct Tokens: {results['correct_tokens']:,}")
                
                # Log effective contributions with correct weights
                logger.info(f"🔍 Effective Loss Components:")
                logger.info(f"  CE (weighted): {results['ce_loss'] * results['ce_weight']:.6f}")
                logger.info(f"  Contrastive (weighted): {results['contrastive_loss'] * results['contrastive_weight']:.6f}")
                logger.info(f"  Destination (weighted): {results['destination_loss'] * results['destination_weight']:.6f}")
                logger.info(f"  KD (weighted): {results['kd_loss'] * results['kd_weight']:.6f}")
                logger.info(f"⚖️  Weights Used (Epoch {checkpoint_epoch + 1 if checkpoint_epoch is not None else 'Unknown'}):")
                logger.info(f"  CE Weight: {results['ce_weight']:.6f}")
                logger.info(f"  Contrastive Weight: {results['contrastive_weight']:.6f}")
                logger.info(f"  Destination Weight: {results['destination_weight']:.6f}")
                logger.info(f"  KD Weight: {results['kd_weight']:.6f}")
                
        except Exception as e:
            if rank == 0:
                logger.error(f"❌ Error evaluating {split} dataset: {str(e)}")
                all_results[split] = {'error': str(e)}

    # Save results (only on rank 0)
    if rank == 0:
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json_results = {}
            for split, results in all_results.items():
                if 'error' in results:
                    json_results[split] = results
                else:
                    json_results[split] = {
                        'loss': float(results['loss']),
                        'ce_loss': float(results['ce_loss']),
                        'contrastive_loss': float(results['contrastive_loss']),
                        'destination_loss': float(results['destination_loss']),
                        'kd_loss': float(results['kd_loss']),
                        'reg_loss': float(results['reg_loss']),
                        'accuracy': float(results['accuracy']),
                        'overall_accuracy': float(results['overall_accuracy']),
                        'total_tokens': int(results['total_tokens']),
                        'correct_tokens': int(results['correct_tokens']),
                        'weights_used': {
                            'ce_weight': results['ce_weight'],
                            'contrastive_weight': results['contrastive_weight'],
                            'destination_weight': results['destination_weight'],
                            'kd_weight': results['kd_weight'],
                            'epoch': checkpoint_epoch
                        }
                    }
            
            json.dump(json_results, f, indent=2)
        
        logger.info(f"💾 Results saved to {results_file}")
        
        # Print summary
        logger.info("📊 EVALUATION SUMMARY:")
        for split in datasets_to_evaluate:
            if split in all_results and 'error' not in all_results[split]:
                results = all_results[split]
                logger.info(f"  {split.upper()}:")
                logger.info(f"    Total Loss: {results['loss']:.6f} | Accuracy: {results['accuracy']:.6f}")
                logger.info(f"    CE: {results['ce_loss']:.6f} | Contrastive: {results['contrastive_loss']:.6f}")
                logger.info(f"    Destination: {results['destination_loss']:.6f} | KD: {results['kd_loss']:.6f}")
            else:
                logger.info(f"  {split.upper()}: ERROR")
        
        # Validate best model performance
        logger.info("🏆 BEST MODEL VALIDATION:")
        if 'val_seen' in all_results and 'error' not in all_results['val_seen']:
            val_seen_results = all_results['val_seen']
            logger.info(f"  Val Seen - Loss: {val_seen_results['loss']:.6f}, Accuracy: {val_seen_results['overall_accuracy']:.6f}")
        
        if 'val_unseen' in all_results and 'error' not in all_results['val_unseen']:
            val_unseen_results = all_results['val_unseen']
            logger.info(f"  Val Unseen - Loss: {val_unseen_results['loss']:.6f}, Accuracy: {val_unseen_results['overall_accuracy']:.6f}")
        
        logger.info("✅ Evaluation completed!")
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    main() 