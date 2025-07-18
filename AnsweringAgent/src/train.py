import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from utils.logger import setup_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from config import Config
from models.answering_agent import AnsweringAgent
from data.dataset import AnsweringDataset
import traceback
import time
import gc
import signal
import sys
import datetime
import logging
import tempfile
import numpy as np
import threading
import math
from transformers import T5Tokenizer
import torch.nn.functional as F
from models.contrastive_loss import ContrastiveLoss

# Global flag to track if training should continue
TRAINING_FAILED = False
CHECKPOINT_LOCK = threading.Lock()

# Create a temporary file for error logging (minimal)
temp_error_file = tempfile.NamedTemporaryFile(prefix="training_error_", suffix=".log", delete=False)

# Exponential Moving Average Implementation
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
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply the EMA weights to the model for evaluation"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore the original weights for training"""
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

def setup_minimal_environment():
    """Setup minimal environment for training without excessive logging."""
    # Disable excessive PyTorch distributed logging
    os.environ["NCCL_DEBUG"] = "WARN"  # Only warnings, not INFO
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # Minimal distributed info
    
    # Memory optimizations
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        # Remove blocking for performance
        # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Only for debugging
        
        # Optimize NCCL for better multi-GPU communication (minimal)
        os.environ['NCCL_NSOCKS_PERTHREAD'] = '2'
        os.environ['NCCL_SOCKET_NTHREADS'] = '2'
    
    # Enable cuDNN benchmarking for performance
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 for better performance (unless debugging precision issues)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def log_system_info(rank=0):
    """Log essential system information only on rank 0."""
    if rank != 0:
        return
        
    print(f"🚀 UAV Navigation Training Pipeline")
    print(f"PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA: {torch.version.cuda} | Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA: Not available - using CPU")

def compute_metrics(outputs: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> Dict[str, float]:
    """Compute accuracy and other metrics."""
    # Reshape outputs and labels
    outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
    labels_reshaped = labels.reshape(-1)

    # Get predictions
    _, predicted = outputs_reshaped.max(1)
    predicted = predicted.reshape(outputs.size(0), outputs.size(1))

    # Create mask for non-padding tokens
    mask = (labels != pad_token_id)

    # Calculate metrics
    total_tokens = mask.sum().item()
    correct_tokens = ((predicted == labels) & mask).sum().item()
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        'accuracy': accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens
    }

def get_weight_schedule(start_weight: float, end_weight: float, total_epochs: int):
    """
    Returns a function that linearly increases or decreases a weight from start_weight to end_weight.

    Args:
        start_weight (float): Initial value of the weight.
        end_weight (float): Final value of the weight.
        total_epochs (int): Total number of epochs over which the weight changes.

    Returns:
        Callable[[int], float]: A function that returns the weight at a given epoch.
    """
    def weight_fn(epoch: int) -> float:
        # Clamp epoch within range
        epoch = max(0, min(epoch, total_epochs))
        return start_weight + (end_weight - start_weight) * (epoch / total_epochs)

    return weight_fn

def calculate_reconstruction_loss(reconstructed_features, original_features):
    reconstructed_features_norm = F.normalize(reconstructed_features, p=2, dim=1)
    original_features_norm = F.normalize(original_features, p=2, dim=1)
    reconstruction_loss = F.mse_loss(reconstructed_features_norm, original_features_norm)
    return reconstruction_loss

def calculate_cosine_similarity_loss(first_features, second_features):
    first_features_norm = F.normalize(first_features, p=2, dim=1)
    second_features_norm = F.normalize(second_features, p=2, dim=1)
    cosine_loss = 1 - F.cosine_similarity(first_features_norm, second_features_norm).mean()
    return cosine_loss

def calculate_distribution_similarity_loss(logits_reshaped, labels_reshaped, mask_flat, model, device):
    """
    Calculate sentence-level distribution similarity loss using embeddings.
    
    Args:
        logits_reshaped: Model logits [batch_size * seq_len, vocab_size]
        labels_reshaped: Token labels [batch_size * seq_len]
        mask_flat: Attention mask [batch_size * seq_len]
        model: The model (for accessing vocab size)
        device: Current device
        
    Returns:
        Sentence-level distribution similarity loss
    """
    distribution_loss = torch.tensor(0.0, device=device)
    
    # Only compute on non-padded tokens (where mask is 1)
    valid_positions = mask_flat.bool()
    if valid_positions.sum() > 0:
        # Get the vocabulary size
        model_to_use = model.module if hasattr(model, 'module') else model
        
        # Extract valid logits and labels
        valid_logits = logits_reshaped[valid_positions]  # [valid_count, vocab_size]
        valid_labels = labels_reshaped[valid_positions]  # [valid_count]
        
        # Get softmax of logits (predicted distribution)
        probs = F.softmax(valid_logits, dim=-1)
        
        # Get embeddings for both predicted and target distributions
        with torch.no_grad():
            embedding_layer = model_to_use.t5_model.decoder.embed_tokens
            
            # Get embeddings for target tokens
            target_embeddings = embedding_layer(valid_labels)  # [valid_count, hidden_dim]
            
            # Get embeddings for predicted distribution
            # Weight each embedding by its probability
            all_token_embeddings = embedding_layer.weight  # [vocab_size, hidden_dim]
            predicted_embeddings = torch.matmul(probs, all_token_embeddings)  # [valid_count, hidden_dim]
        
        # Normalize embeddings
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
        predicted_embeddings = F.normalize(predicted_embeddings, p=2, dim=1)
        
        # Calculate cosine similarity loss
        distribution_loss = (1 - F.cosine_similarity(
            predicted_embeddings,
            target_embeddings
        )).mean()
    
    return distribution_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, config, teacher_model=None, start_epoch=0,
                best_val_loss=float('inf'), rank=None, logger=None, is_distributed=False):
    """Train the model with mixed precision training and gradient accumulation."""
    global TRAINING_FAILED

    save_frequency = config.training.checkpoint_frequency
    log_frequency = max(10, len(train_loader) // 3)

    # Enable automatic mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.mixed_precision)
    use_amp = config.training.mixed_precision
    
    if rank == 0:
        logger.info(f"🎯 Training Configuration:")
        logger.info(f"  Mixed precision: {'✅' if use_amp else '❌'}")
        logger.info(f"  Gradient accumulation: {config.training.gradient_accumulation_steps}")
        logger.info(f"  Contrastive learning: {'✅' if config.training.use_contrastive_learning else '❌'}")
        
        # Note about darknet device mapping
        logger.info(f"📝 Note: Darknet weights loaded on CPU first to prevent OOM, then moved to GPU")
    
    # Initialize Exponential Moving Average
    ema = EMA(model, decay=0.999)
    
    # Initialize contrastive loss if enabled
    contrastive_loss_fn = None
    if config.training.use_contrastive_learning:
        contrastive_loss_fn = ContrastiveLoss(
            margin=config.training.contrastive_margin,
            temperature=config.training.contrastive_temperature,
            loss_type=config.training.contrastive_loss_type,
            use_cosine_distance=config.training.use_cosine_distance,
            mean_all=config.training.contrastive_mean_all
        )
        if rank == 0:
            distance_type = "cosine" if config.training.use_cosine_distance else "L2"
            mean_type = "all" if config.training.contrastive_mean_all else "non-zero"
            logger.info(f"🔗 Contrastive learning: {config.training.contrastive_loss_type} loss ({distance_type} distance, {mean_type} mean)")
    
    # Clear cache once at beginning
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Keep track of the last best model's epoch
    last_best_epoch = None
    
    # Early stopping variables
    early_stopping_counter = 0
    early_stopping_triggered = False
        
    try:
        for epoch in range(start_epoch, num_epochs):
            # Check if early stopping was triggered
            if early_stopping_triggered:
                if rank == 0:
                    logger.info(f"🛑 Early stopping triggered after {epoch} epochs")
                
                # Add synchronization barrier to ensure all processes exit together
                if is_distributed and dist.is_initialized():
                    try:
                        dist.barrier()
                        if rank == 0:
                            logger.info("✅ All processes synchronized before exit")
                    except Exception as e:
                        if rank == 0:
                            logger.error(f"❌ Error during synchronization: {e}")
                
                break
                
            if is_distributed:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)

            model.train()
            total_loss = 0
            total_ce_loss = 0
            total_destination_loss = 0
            total_contrastive_loss = 0
            total_kd_loss = 0
            optimizer.zero_grad(set_to_none=True)

            epoch_start_time = time.time()

            # Only rank 0 logs the results
            if rank == 0:
                logger.info(f"🚀 Epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Simple direct data loading - no prefetching
                    text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                    
                    # Add separate components if available (for hierarchical processing)
                    if 'first_instruction_input' in batch:
                        text_input['first_instruction_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['first_instruction_input'].items()}
                    if 'current_question_input' in batch:
                        text_input['current_question_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['current_question_input'].items()}
                    
                    current_view = batch['current_view_image'].to(device, non_blocking=True)
                    previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                    
                    # Handle text_label as a dictionary with input_ids and attention_mask
                    label_input_ids = batch['text_label']['input_ids'].to(device, non_blocking=True)
                    label_attention_mask = batch['text_label']['attention_mask'].to(device, non_blocking=True)
                    
                    # Set up destination view if available in batch and curriculum is active
                    destination_view = batch['destination_image'].to(device, non_blocking=True) if 'destination_image' in batch else None
                    
                    # Calculate curriculum learning ratio based on epochs
                    max_curriculum_epochs = config.training.curriculum_epochs
                    curriculum_ratio = max(0.0, 1.0 - (epoch / max_curriculum_epochs))
                    
                    # Prepare contrastive examples if enabled
                    positive_input = None
                    positive_input_2 = None
                    negative_input = None
                    contrastive_examples_found = False
                    
                    if config.training.use_contrastive_learning and contrastive_loss_fn is not None:
                        # Get tokenized contrastive inputs from dataset (normalizer always provides these)
                        if 'positive_input' in batch:
                            positive_input = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input'].items()}
                            contrastive_examples_found = True
                            
                            if 'positive_input_2' in batch:
                                positive_input_2 = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input_2'].items()}
                            
                            if 'negative_input' in batch:
                                negative_input = {k: v.to(device, non_blocking=True) for k, v in batch['negative_input'].items()}
                                   
                    # Debug log on first batch to check data loading
                    if batch_idx == 0 and epoch == 0 and rank == 0:
                        logger.info(f"🔍 Debug | Batch keys: {list(batch.keys())}")
                        logger.info(f"🔍 Debug | Context-aware contrastive examples found: {contrastive_examples_found}")
                        if 'positive_example' in batch:
                            logger.info(f"🔍 Debug | Positive context shape: {batch['positive_input']['input_ids'].shape}")
                        if 'positive_example_2' in batch:
                            logger.info(f"🔍 Debug | Positive context 2 shape: {batch['positive_input_2']['input_ids'].shape}")
                        if 'negative_example' in batch:
                            logger.info(f"🔍 Debug | Negative context shape: {batch['negative_input']['input_ids'].shape}")
                        logger.info(f"🔍 Debug | All contrastive examples now include full context for semantic alignment")
                        logger.info(f"🔍 Debug | Context includes: First Instruction + Dialog History + Current Question")
                    
                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(
                            text_input, 
                            current_view, 
                            previous_views, 
                            labels=label_input_ids,
                            destination_view=destination_view,
                            curriculum_ratio=curriculum_ratio,
                            positive_input=positive_input,
                            positive_input_2=positive_input_2,
                            negative_input=negative_input
                        )
                        
                        logits = outputs["logits"]
                        feature_norm = outputs.get("feature_norm", torch.tensor(0.0, device=device))

                        # Get batch and sequence dimensions
                        batch_size, seq_len, vocab_size = logits.size()

                        if torch.isnan(logits).any():
                            if rank == 0:
                                logger.error(f"❌ NaN detected in logits at batch {batch_idx}")
                            optimizer.zero_grad(set_to_none=True)
                            continue

                        # Reshape outputs and labels consistently
                        logits_reshaped = logits.contiguous().view(batch_size * seq_len, vocab_size)
                        labels_reshaped = label_input_ids.contiguous().view(batch_size * seq_len)

                        # Calculate cross-entropy loss
                        ce_loss = criterion(logits_reshaped, labels_reshaped)
                        ce_loss_weight = get_weight_schedule(config.training.ce_loss_weight_start, config.training.ce_loss_weight_end, max_curriculum_epochs)(epoch)

                        # Add feature regularization with clipping to prevent explosion
                        feature_norm_clipped = feature_norm.clamp(max=1e3)  # Clip to prevent explosion
                        reg_loss = 1e-4 * feature_norm_clipped
                        loss = ce_loss_weight * ce_loss + reg_loss
                            
                        # Add destination loss if destination view is available
                        if destination_view is not None:
                            dest_features = outputs.get("destination_features", outputs["adapted_features"])
                            destination_cosine_loss = calculate_cosine_similarity_loss(outputs["adapted_features"], dest_features)
                            
                            destination_weight = get_weight_schedule(config.training.destination_loss_weight_start, config.training.destination_loss_weight_end, max_curriculum_epochs)(epoch)
                            loss = loss + destination_weight * destination_cosine_loss
                        else:
                            destination_cosine_loss = torch.tensor(0.0, device=device)

                        
                        # Calculate contrastive loss if enabled
                        contrastive_loss = torch.tensor(0.0, device=device)
                        if config.training.use_contrastive_learning and contrastive_loss_fn is not None:
                            
                            # Collect all positive and negative embeddings
                            anchor_emb = None
                            positive_embs = []
                            negative_emb = None
                            
                            if 'positive_adapted_features' in outputs and 'negative_adapted_features' in outputs:
                                anchor_emb = outputs['adapted_features']
                                positive_embs.append(outputs['positive_adapted_features'])
                                negative_emb = outputs['negative_adapted_features']
                                
                                # Add second positive if available
                                if 'positive_adapted_features_2' in outputs:
                                    positive_embs.append(outputs['positive_adapted_features_2'])
                                
                                # Stack positives if we have multiple
                                if len(positive_embs) > 1:
                                    positive_combined = torch.stack(positive_embs, dim=1)  # [batch, num_pos, hidden]
                                else:
                                    positive_combined = positive_embs[0]  # [batch, hidden]
                                
                                # Calculate contrastive loss
                                contrastive_loss = contrastive_loss_fn(anchor_emb, positive_combined, negative_emb)
                                
                                
                                # Add weighted contrastive loss to total loss
                                contrastive_weight = get_weight_schedule(
                                    config.training.contrastive_weight_start,
                                    config.training.contrastive_weight_end,
                                    max_curriculum_epochs
                                )(epoch)
                                loss = loss + contrastive_weight * contrastive_loss
                                total_contrastive_loss += contrastive_loss.item()
                            elif rank == 0 and batch_idx == 0 and epoch == 0:
                                logger.warning(f"⚠️ Contrastive learning enabled but no adapted features found in outputs!")
                                logger.info(f"🔍 Available output keys: {list(outputs.keys())}")
                
                    # KD loss using embeddings generated during preprocessing
                    kd_loss = torch.tensor(0.0, device=device)
                    if config.training.use_kd:
                        # Teacher embeddings are included in the batch from preprocessing
                        teacher_embeddings = batch['teacher_embed'].to(device)
                        student_hidden = F.normalize(outputs["adapted_features"], p=2, dim=-1)
                        kd_loss = 1 - F.cosine_similarity(student_hidden, teacher_embeddings, dim=-1).mean()
                        
                        kd_weight = get_weight_schedule(
                            config.training.kd_weight_start,
                            config.training.kd_weight_end,
                            config.training.kd_epochs
                        )(epoch)
                        loss = loss + kd_weight * kd_loss
                        total_kd_loss += kd_loss.item()

                    # Apply gradient accumulation: normalize loss
                        loss = loss / config.training.gradient_accumulation_steps

                    # Accumulate statistics
                    total_ce_loss += ce_loss.item()
                    total_destination_loss += destination_cosine_loss.item()
                    
                    # Backpropagation with mixed precision
                    scaler.scale(loss).backward()
                    
                    # Gradient accumulation
                    if ((batch_idx + 1) % config.training.gradient_accumulation_steps == 0) or (batch_idx + 1 == len(train_loader)):
                        # Unscale gradients to apply custom gradient operations (like clipping)
                        scaler.unscale_(optimizer)

                        # Add gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                        
                        # Update parameters with scaler aware step
                        scaler.step(optimizer)
                        scaler.update()

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()      # release cached kernels
                            torch.cuda.ipc_collect()      # C++ side arena defrag

                        optimizer.zero_grad(set_to_none=True)
                        
                        # Update EMA
                        ema.update()

                    total_loss += loss.item() * config.training.gradient_accumulation_steps
                
                    # Only have rank 0 log progress
                    if rank == 0 and batch_idx % log_frequency == 0:
                        avg_loss = total_loss / (batch_idx + 1)
                        avg_ce = total_ce_loss / (batch_idx + 1)
                        avg_contrastive = total_contrastive_loss / (batch_idx + 1)
                        avg_destination = total_destination_loss / (batch_idx + 1)
                        avg_kd = total_kd_loss / (batch_idx + 1)
                        
                        logger.info(f"📊 Batch {batch_idx}/{len(train_loader)} | "
                                  f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | "
                                  f"Contrast: {avg_contrastive:.4f} | KD: {avg_kd:.4f} | Dest: {avg_destination:.4f}")
                        
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()   

                except Exception as e:
                    # Log and continue in case of batch failure
                    if rank == 0:
                        logger.error(f"❌ Error in batch {batch_idx}: {str(e)}")
                    
                    # Zero out gradients to avoid accumulation
                    optimizer.zero_grad(set_to_none=True)
                    continue

            # Calculate average losses across distributed processes
            if is_distributed:
                # Gather losses from all processes
                loss_tensor = torch.tensor(total_loss, device=device)
                total_ce_loss_tensor = torch.tensor(total_ce_loss, device=device)   
                total_destination_loss_tensor = torch.tensor(total_destination_loss, device=device)
                total_contrastive_loss_tensor = torch.tensor(total_contrastive_loss, device=device)
                total_kd_loss_tensor = torch.tensor(total_kd_loss, device=device)
                
                # All-reduce to sum losses across processes
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_ce_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_destination_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_contrastive_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_kd_loss_tensor, op=dist.ReduceOp.SUM)

                
                # Calculate averages
                total_loss = loss_tensor.item() / dist.get_world_size()
                total_ce_loss = total_ce_loss_tensor.item() / dist.get_world_size()
                total_destination_loss = total_destination_loss_tensor.item() / dist.get_world_size()
                total_contrastive_loss = total_contrastive_loss_tensor.item() / dist.get_world_size()
                total_kd_loss = total_kd_loss_tensor.item() / dist.get_world_size()
            
            avg_epoch_loss = total_loss / len(train_loader)
            avg_ce_loss = total_ce_loss / len(train_loader)
            avg_destination_loss = total_destination_loss / len(train_loader)
            avg_contrastive_loss = total_contrastive_loss / len(train_loader)
            avg_kd_loss = total_kd_loss / len(train_loader)
            
            # Log the epoch summary (only rank 0)
            if rank == 0:
                epoch_time = time.time() - epoch_start_time
                
                # Calculate current weights for this epoch
                if hasattr(config.training, 'log_loss_weights') and config.training.log_loss_weights:
                    max_curriculum_epochs = config.training.curriculum_epochs
                    current_ce_weight = get_weight_schedule(config.training.ce_loss_weight_start, config.training.ce_loss_weight_end, max_curriculum_epochs)(epoch)
                    current_contrastive_weight = get_weight_schedule(config.training.contrastive_weight_start, config.training.contrastive_weight_end, max_curriculum_epochs)(epoch)
                    current_dest_weight = get_weight_schedule(config.training.destination_loss_weight_start, config.training.destination_loss_weight_end, max_curriculum_epochs)(epoch)
                    current_kd_weight = get_weight_schedule(config.training.kd_weight_start, config.training.kd_weight_end, max_curriculum_epochs)(epoch)

                    
                    logger.info(f"✅ Epoch {epoch+1} | Loss: {avg_epoch_loss:.4f} | "
                              f"CE: {avg_ce_loss:.4f} | "
                              f"Contrast: {avg_contrastive_loss:.4f} | Dest: {avg_destination_loss:.4f} | KD: {avg_kd_loss:.4f} "
                              f"Time: {epoch_time:.1f}s")
                    logger.info(f"🎛️  Weights | CE: {current_ce_weight:.2f} | "
                              f"Contrastive: {current_contrastive_weight:.2f} | Dest: {current_dest_weight:.2f} | KD: {current_kd_weight} ")
                    
                    # Log effective contributions for debugging
                    effective_ce = avg_ce_loss * current_ce_weight
                    effective_contrastive = avg_contrastive_loss * current_contrastive_weight
                    effective_dest = avg_destination_loss * current_dest_weight
                    effective_kd = avg_kd_loss * current_kd_weight
                    effective_total = effective_ce + effective_contrastive + effective_dest + effective_kd
                    
                    logger.info(f"🔍 Effective | Total Loss: {effective_total:.4f} | CE: {effective_ce:.4f} | "
                              f"Contrastive: {effective_contrastive:.4f} | Dest: {effective_dest:.4f} | KD: {effective_kd:.4f} ")
                else:
                    logger.info(f"✅ Epoch {epoch+1} | Loss: {avg_epoch_loss:.4f} | "
                              f"CE: {avg_ce_loss:.4f} | "
                              f"Contrast: {avg_contrastive_loss:.4f} | Dest: {avg_destination_loss:.4f} | KD: {avg_kd_loss:.4f}"
                              f"Time: {epoch_time:.1f}s")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()          # throw away the giant arena
                torch.cuda.reset_peak_memory_stats()
                
            # Validation step
            val_loss = 0
                
            if (epoch + 1) % config.training.eval_freq == 0 or epoch == num_epochs - 1:
                model.eval()
                
                # Apply EMA for validation
                ema.apply_shadow()

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        try:
                            # Load validation data
                            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                            
                            # Add separate components if available (for hierarchical processing)
                            if 'first_instruction_input' in batch:
                                text_input['first_instruction_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['first_instruction_input'].items()}
                            if 'current_question_input' in batch:
                                text_input['current_question_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['current_question_input'].items()}
                            
                            current_view = batch['current_view_image'].to(device, non_blocking=True)
                            previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                            
                            label_input_ids = batch['text_label']['input_ids'].to(device, non_blocking=True)
                            label_attention_mask = batch['text_label']['attention_mask'].to(device, non_blocking=True)
                            
                            # Set up destination view if available
                            destination_view = batch['destination_image'].to(device, non_blocking=True) if 'destination_image' in batch else None
                            
                            # Prepare contrastive examples if available
                            positive_input = None
                            positive_input_2 = None
                            negative_input = None
                            
                            # Check for new format (both tokenized and raw text) vs old format
                            if 'positive_input' in batch:
                                # New format: tokenized inputs from normalizer
                                positive_input = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input'].items()}
                                
                                if 'positive_input_2' in batch:
                                    positive_input_2 = {k: v.to(device, non_blocking=True) for k, v in batch['positive_input_2'].items()}
                                
                                if 'negative_input' in batch:
                                    negative_input = {k: v.to(device, non_blocking=True) for k, v in batch['negative_input'].items()}
                                    
                        
                            # Use mixed precision for validation as well for consistent numerical behavior
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                outputs = model(
                                    text_input, 
                                    current_view, 
                                    previous_views, 
                                    labels=label_input_ids,
                                    destination_view=destination_view,
                                    curriculum_ratio=0.0,  # No curriculum during validation
                                    positive_input=positive_input,
                                    positive_input_2=positive_input_2,
                                    negative_input=negative_input
                                )
                                
                                logits = outputs["logits"]
                                batch_size, seq_len, vocab_size = logits.size()
                                
                                # Reshape logits and labels
                                logits_reshaped = logits.contiguous().view(batch_size * seq_len, vocab_size)
                                labels_reshaped = label_input_ids.contiguous().view(batch_size * seq_len)
                                
                                # Calculate validation losses
                                ce_loss = criterion(logits_reshaped, labels_reshaped)
                                loss = config.training.ce_loss_weight_end * ce_loss
                                
                                
                                # Add contrastive loss if enabled
                                if config.training.use_contrastive_learning and contrastive_loss_fn is not None:
                                    contrastive_losses = []
                                    
                                    # First triplet: anchor, positive1, negative
                                    if 'positive_adapted_features' in outputs and 'negative_adapted_features' in outputs:
                                        anchor_emb = outputs['adapted_features']
                                        positive_emb = outputs['positive_adapted_features']
                                        negative_emb = outputs['negative_adapted_features']
                                        
                                        # Add shape validation
                                        if anchor_emb.shape != positive_emb.shape or anchor_emb.shape != negative_emb.shape:
                                            logger.error(f"❌ Shape mismatch in validation contrastive loss: anchor={anchor_emb.shape}, "
                                                       f"positive={positive_emb.shape}, negative={negative_emb.shape}")
                                            continue
                                        
                                        contrastive_loss_1 = contrastive_loss_fn(anchor_emb, positive_emb, negative_emb)
                                        contrastive_losses.append(contrastive_loss_1)
                                    
                                    # Second triplet: anchor, positive2, negative (if available)
                                    if 'positive_adapted_features_2' in outputs and 'negative_adapted_features' in outputs:
                                        anchor_emb = outputs['adapted_features']
                                        positive_emb_2 = outputs['positive_adapted_features_2']
                                        negative_emb = outputs['negative_adapted_features']
                                        
                                        # Add shape validation
                                        if anchor_emb.shape != positive_emb_2.shape or anchor_emb.shape != negative_emb.shape:
                                            logger.error(f"❌ Shape mismatch in validation contrastive loss (triplet 2): anchor={anchor_emb.shape}, "
                                                       f"positive_2={positive_emb_2.shape}, negative={negative_emb.shape}")
                                            continue
                                        
                                        contrastive_loss_2 = contrastive_loss_fn(anchor_emb, positive_emb_2, negative_emb)
                                        contrastive_losses.append(contrastive_loss_2)
                                        
                                    # Average the contrastive losses and add to validation loss
                                    if contrastive_losses:
                                        contrastive_loss = torch.stack(contrastive_losses).mean()
                                        loss = loss + config.training.contrastive_weight_end * contrastive_loss
                            
                            val_loss += loss.item()
                        except Exception as e:
                            if rank == 0:
                                logger.error(f"❌ Error in validation batch {batch_idx}: {str(e)}")
                            continue
                
                # Restore original weights
                ema.restore()

                # Average validation loss across all processes if distributed
                if is_distributed:
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = val_loss_tensor.item() / dist.get_world_size()
                
                # Calculate average validation loss
                val_loss = val_loss / len(val_loader)

                if rank == 0:
                    logger.info(f"📋 Validation Loss: {val_loss:.4f}")
                    
                    # Check if this is the best model so far (only compare to best, not previous)
                    if val_loss < best_val_loss - best_val_loss * config.training.early_stopping_min_delta:
                        improvement = (best_val_loss - val_loss) / best_val_loss * 100
                        logger.info(f"🎯 New best model! Improved by {improvement:.2f}%")
                        
                        # Save best model
                        save_dict = {
                            'epoch': epoch,
                            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'ema': ema.state_dict(),
                            'val_loss': val_loss,
                            'config': config,
                        }
                        
                        with CHECKPOINT_LOCK:
                            best_model_path = os.path.join(checkpoint_dir, f'best_model_{epoch+1}.pth')
                            torch.save(save_dict, best_model_path)
                            logger.info(f"💾 Saved best model")
                        
                        best_val_loss = val_loss
                        last_best_epoch = epoch
                        
                        # Reset early stopping counter on improvement
                        early_stopping_counter = 0
                    else:
                        # Only increment counter if no improvement (don't compare to previous loss)
                        early_stopping_counter += 1
                        logger.info(f"🔍 Early stopping counter: {early_stopping_counter}/{config.training.early_stopping_patience}")
                        if config.training.early_stopping and early_stopping_counter >= config.training.early_stopping_patience:
                            early_stopping_triggered = True
                            logger.info(f"🛑 Early stopping triggered after {early_stopping_counter} epochs without improvement")
                            break
                    
            
            # Save checkpoint at regular intervals (only rank 0)
            if rank == 0 and (epoch + 1) % save_frequency == 0:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'ema': ema.state_dict(),
                    'val_loss': val_loss if 'val_loss' in locals() else None,
                    'config': config,
                }
                
                with CHECKPOINT_LOCK:
                    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                    torch.save(save_dict, checkpoint_path)
                    logger.info(f"💾 Checkpoint saved")
            
            # Step the scheduler based on validation loss if available
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau needs the validation loss
                    if 'val_loss' in locals():
                        scheduler.step(val_loss)
                else:
                    scheduler.step()
        
        # End of training - save final model
        if rank == 0:
            save_dict = {
                'epoch': num_epochs - 1,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'ema': ema.state_dict(),
                'val_loss': val_loss if 'val_loss' in locals() else None,
                'config': config,
            }
            
            with CHECKPOINT_LOCK:
                final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
                torch.save(save_dict, final_model_path)
                logger.info(f"💾 Final model saved")

            # Print training summary
            logger.info(f"🎉 Training complete! Best val loss: {best_val_loss:.4f} at epoch {last_best_epoch + 1}")
                
        return best_val_loss, last_best_epoch

    except Exception as e:
        if rank == 0:
            logger.error(f"❌ Training failed: {str(e)}")
        
        TRAINING_FAILED = True
        raise

def log_gpu_memory():
    """Log GPU memory usage (simplified)."""
    if not torch.cuda.is_available():
        return "No CUDA"
        
    try:
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024 ** 2
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024 ** 2
        
        return f"GPU {current_device}: {memory_allocated:.0f}MB/{memory_reserved:.0f}MB"
    except Exception:
        return "GPU memory error"

def setup_distributed():
    """Set up the distributed environment using environment variables set by torchrun."""
    # Get distributed training environment variables from torchrun
    if "LOCAL_RANK" not in os.environ:
        # Not running with torchrun, assume single-GPU
        return False, 0, 1
    
    # Check available GPU count
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("⚠️ No CUDA devices available! Running on CPU only.")
        return False, 0, 1
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Ensure local_rank is within the range of available devices
    if local_rank >= available_gpus:
        print(f"⚠️ local_rank ({local_rank}) >= available GPUs ({available_gpus})")
        local_rank = local_rank % available_gpus
    
    # Set the device
    torch.cuda.set_device(local_rank)
    
    # Ensure proper initialization of master address and port
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    # Initialize the process group
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=30)
            )
            print(f"✅ Process group initialized for rank {rank}")
        except Exception as e:
            print(f"❌ Error initializing process group: {e}")
            raise
    
    return True, rank, world_size

def cleanup():
    """Force cleanup of resources."""
    if dist.is_initialized():
        dist.destroy_process_group()
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    """Main training function that works with torchrun."""
    global TRAINING_FAILED
    
    # Setup minimal environment (no excessive logging)
    setup_minimal_environment()
    
    # Parse arguments first
    parser = argparse.ArgumentParser(description='UAV Navigation Training')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from', default=None)
    parser.add_argument('--single-gpu', action='store_true', help='Force running on a single GPU even with torchrun')
    parser.add_argument('--batch-size', type=int, help='Per-GPU batch size (overrides config value)', default=None)
    parser.add_argument('--grad-steps', type=int, help='Gradient accumulation steps (overrides config value)', default=None)
    parser.add_argument('--use-augmented-data', action='store_true', help='Use augmented dataset with paraphrases for contrastive learning')
    parser.add_argument('--no-augmented-data', action='store_true', help='Use original dataset without augmented paraphrases')
    args = parser.parse_args()
    
    # Check CUDA availability first
    if not torch.cuda.is_available():
        print("⚠️ CUDA is not available! Training will run on CPU only.")
        num_gpus = 0
    else:
        num_gpus = torch.cuda.device_count()
    
    if args.single_gpu or num_gpus <= 1:
        # Single GPU or CPU mode
        print("🖥️ Running in single device mode")
        is_distributed = False
        rank = 0
        world_size = 1
        local_rank = 0
        
        if num_gpus > 0:
            device = torch.device('cuda:0')
            torch.cuda.set_device(0)
        else:
            device = torch.device('cpu')
    else:
        # Multi-GPU mode with torchrun
        try:
            is_distributed, rank, world_size = setup_distributed()
            local_rank = int(os.environ.get("LOCAL_RANK", 0)) % max(1, torch.cuda.device_count())
            device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            print(f"❌ Error setting up distributed environment: {e}")
            print("🔄 Falling back to single GPU mode")
            is_distributed = False
            rank = 0
            world_size = 1
            local_rank = 0
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Log essential system info
    log_system_info(rank)
    
    # Set up signal handlers for proper cleanup
    def signal_handler(sig, frame):
        print(f"🛑 Process {rank} received signal {sig}, cleaning up...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = Config()
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    if rank == 0:
        print(f"🎲 Random seed: {config.training.seed}")

    # Initialize logger
    logger = setup_logger('training', log_dir=config.log_dir)

    # Override config values with command-line arguments if provided
    if args.batch_size is not None:
        config.training.per_gpu_batch_size = args.batch_size
        if rank == 0:
            logger.info(f"⚙️ Batch size override: {args.batch_size}")
            
    if args.grad_steps is not None:
        config.training.gradient_accumulation_steps = args.grad_steps
        if rank == 0:
            logger.info(f"⚙️ Gradient accumulation override: {args.grad_steps}")
    
    # Handle augmented data arguments
    if args.no_augmented_data:
        config.data.use_augmented_data = False
        if rank == 0:
            logger.info("📊 Using original dataset (no augmentation)")
    elif args.use_augmented_data:
        config.data.use_augmented_data = True
        if rank == 0:
            logger.info("📊 Using augmented dataset with paraphrases")
    
    # Log dataset configuration
    if rank == 0:
        status = "✅ enabled" if config.data.use_augmented_data else "❌ disabled"
        logger.info(f"📊 Augmented data: {status}")
        logger.info(f"🔗 Contrastive learning: {'✅' if config.training.use_contrastive_learning else '❌'}")
    
    # Silence non-rank-0 processes by setting logger level
    if rank != 0:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.ERROR)  # Only show errors on non-rank-0
    
    if rank == 0:
        logger.info(f"🚀 Starting UAV Navigation Training")
        logger.info(f"📍 Device: {device} | Rank: {rank} | World Size: {world_size}")

    try:
        # Initialize tokenizer
        tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, model_max_length=config.data.max_seq_length)
        
        if rank == 0:
            logger.info(f"🎯 Training on {max(1, num_gpus)} GPU(s), distributed: {is_distributed}")
        
        # Wait for rank 0 to finish preprocessing
        if is_distributed:
            dist.barrier()
        
        if rank == 0:
            logger.info("🏗️ Initializing model...")
            
        # Initialize model and move to correct GPU
        model = AnsweringAgent(config, tokenizer, logger)
        if is_distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)

        # Teacher embeddings are now generated during preprocessing in the normalizer
        # No need to load teacher model during training
        teacher_model = None
        if config.training.use_kd and rank == 0:
            logger.info(f"🧑‍🏫 Teacher embeddings generated during preprocessing for KD")

        # Initialize training variables
        start_epoch = 0
        best_val_loss = float('inf')

        # Wrap model with DDP if using distributed training
        if is_distributed:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                broadcast_buffers=True
            )

        # Resume training if checkpoint is provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            if rank == 0:
                logger.info(f"📂 Loading checkpoint: {args.checkpoint}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(args.checkpoint, map_location=map_location)
            
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            if rank == 0:
                logger.info(f"▶️ Resuming from epoch {start_epoch+1}")

        
        
        # Load dataset
        try:
            if rank == 0:
                logger.info("📊 Loading datasets...")

            datasets = AnsweringDataset.create_datasets(config, logger=logger, splits=['train', 'val_seen'], tokenizer=tokenizer)

        except Exception as e:
            logger.error(f"❌ Dataset loading failed: {str(e)}")
            TRAINING_FAILED = True
            raise e
        
        if rank == 0:
            logger.info(f"📊 Dataset: {len(datasets['train'])} train, {len(datasets['val_seen'])} validation")

        # Use DistributedSampler for distributed training
        if is_distributed:
            train_sampler = DistributedSampler(datasets['train'], num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(datasets['val_seen'], num_replicas=world_size, rank=rank)
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        
        if rank == 0:
            batch_str = f"📊 Per-GPU batch size: {config.training.per_gpu_batch_size}"
            if is_distributed:
                effective_batch_size = config.training.per_gpu_batch_size * world_size * config.training.gradient_accumulation_steps
                batch_str += f" (effective: {effective_batch_size})"
            logger.info(batch_str)
            
            # Log validation batch size
            val_batch_str = f"📊 Validation batch size: {config.training.per_gpu_batch_size_val}"
            if is_distributed:
                effective_val_batch_size = config.training.per_gpu_batch_size_val * world_size
                val_batch_str += f" (effective: {effective_val_batch_size})"
            logger.info(val_batch_str)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()          # throw away the giant arena
            torch.cuda.reset_peak_memory_stats()

        if rank == 0:
            logger.info(f"💾 Memory usage: {log_gpu_memory()}")

        train_loader = DataLoader(
            datasets['train'],
            batch_size=config.training.per_gpu_batch_size,
            sampler=train_sampler,
            shuffle=shuffle,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=(config.training.num_workers > 0)
        )

        val_loader = DataLoader(
            datasets['val_seen'],
            batch_size=config.training.per_gpu_batch_size_val,  # Smaller validation batch size
            sampler=val_sampler,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=False,
            persistent_workers=False  # Disable persistent workers for validation to save VRAM
        )

        # Create warmup then decay scheduler
        def get_lr_schedule(optimizer, warmup_steps, total_steps):
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # Curriculum-aware decay
                    curriculum_phase_steps = int(total_steps * 0.15)
                    if current_step < warmup_steps + curriculum_phase_steps:
                        progress = float(current_step - warmup_steps) / float(max(1, curriculum_phase_steps))
                        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    else:
                        progress = float(current_step - warmup_steps - curriculum_phase_steps) / float(
                            max(1, total_steps - warmup_steps - curriculum_phase_steps))
                        return max(0.0, 0.3 * (1.0 + math.cos(math.pi * progress)))
            return LambdaLR(optimizer, lr_lambda)
        
        # Optimizer, loss, and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            label_smoothing=0.05
        )
        
        # Calculate total steps
        total_steps = len(train_loader) * config.training.num_epochs // config.training.gradient_accumulation_steps
        
        # Create scheduler with warmup
        scheduler = get_lr_schedule(
            optimizer, 
            warmup_steps=config.training.warmup_steps,
            total_steps=total_steps
        )

        # Training
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=config.training.num_epochs,
            device=device,
            checkpoint_dir=config.checkpoint_dir,
            config=config,
            teacher_model=teacher_model,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            rank=rank,
            logger=logger,
            is_distributed=is_distributed
        )

        # Normal cleanup
        cleanup()
        
        if rank == 0:
            logger.info("🎉 Training completed successfully!")

    except Exception as e:
        # Mark training as failed
        TRAINING_FAILED = True
        if logger:
            error_msg = f"❌ Fatal error: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Write to error file
            with open(temp_error_file.name, 'a') as f:
                f.write(f"RANK {rank}: {error_msg}\n")
                f.write(traceback.format_exc())
        else:
            print(f"❌ Fatal error: {str(e)}")
            print(traceback.format_exc())
    finally:
        # Proper cleanup for distributed environment
        if is_distributed and dist.is_initialized():
            try:
                dist.barrier()
                dist.destroy_process_group()
                if rank == 0 and logger:
                    logger.info("✅ Distributed cleanup complete")
            except Exception as e:
                if rank == 0 and logger:
                    logger.error(f"❌ Cleanup error: {e}")
        
        cleanup()
        
        if rank == 0:
            if TRAINING_FAILED:
                if logger:
                    logger.error("❌ Training failed")
                sys.exit(1)
            else:
                if logger:
                    logger.info("✅ Training completed successfully")

if __name__ == '__main__':
    import argparse
    main()
