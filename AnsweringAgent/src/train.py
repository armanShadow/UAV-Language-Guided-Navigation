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
import faulthandler
import tempfile
import numpy as np
import threading
import math  # Add math import for cosine decay function
from transformers import T5Tokenizer
import torch.nn.functional as F

# Enable Python fault handler for segfaults
faulthandler.enable()

# Enable detailed distributed debug info
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# Create a temporary file for error logging
temp_error_file = tempfile.NamedTemporaryFile(prefix="torch_elastic_error_", suffix=".log", delete=False)
os.environ["TORCHELASTIC_ERROR_FILE"] = temp_error_file.name
print(f"Torch elastic error file: {temp_error_file.name}")

# Global flag to track if training should continue
TRAINING_FAILED = False
CHECKPOINT_LOCK = threading.Lock()

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

def set_debug_mode():
    """Enable various debugging and error reporting options"""
    # Enable Python's detailed traceback
    sys.tracebacklimit = 100
    
    # Set NCCL debug logging
    os.environ["NCCL_DEBUG"] = "INFO"
    
    # Enable tensor core debug mode
    torch.backends.cuda.matmul.allow_tf32 = False
    
    # Print all system information
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Check for other important variables
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Environment variables: PATH, LD_LIBRARY_PATH, PYTHONPATH set: "
          f"{all(var in os.environ for var in ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH'])}")

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

def calculate_embedings_cosine_similarity_loss(labels_flat, mask_flat, decoder_hidden_states, model, device):
    cosine_sim_loss = torch.tensor(0.0, device=device)                                          
    # Calculate cosine similarity between decoder hidden states and labels
    # Get decoder hidden states [batch_size, seq_len, hidden_dim
    # Use label attention mask directly from batch
    # Reshape all to 2D tensors for easier processing
    decoder_hidden_flat = decoder_hidden_states.reshape(-1, decoder_hidden_states.size(-1))
    
    # Only compute similarity on non-padded tokens (where mask is 1)
    valid_positions = mask_flat.bool()
    if valid_positions.sum() > 0:  # Check that we have at least one valid position
        valid_hidden = decoder_hidden_flat[valid_positions]
        valid_labels = labels_flat[valid_positions]
        
        # Normalize the hidden states
        valid_hidden_norm = F.normalize(valid_hidden, p=2, dim=1)
        
        with torch.no_grad():
            model_to_use = model.module if hasattr(model, 'module') else model
            embedding_layer = model_to_use.t5_model.decoder.embed_tokens
            label_embeddings = embedding_layer(valid_labels)

        label_embeddings = F.normalize(label_embeddings, p=2, dim=1)
        
        # Calculate decoder hidden states cosine similarity loss
        cosine_sim_loss = (1 - F.cosine_similarity(
            valid_hidden_norm, 
            label_embeddings
        )).mean()

    return cosine_sim_loss

def calculate_distribution_similarity_loss(logits_reshaped, labels_reshaped, mask_flat, model, device):
    """
    Calculate KL divergence between predicted token distribution and smoothed label distribution.
    
    Args:
        logits_reshaped: Model logits [batch_size * seq_len, vocab_size]
        labels_reshaped: Token labels [batch_size * seq_len]
        mask_flat: Attention mask [batch_size * seq_len]
        model: The model (for accessing vocab size)
        device: Current device
        
    Returns:
        KL divergence loss between distributions
    """
    distribution_loss = torch.tensor(0.0, device=device)
    
    # Only compute on non-padded tokens (where mask is 1)
    valid_positions = mask_flat.bool()
    if valid_positions.sum() > 0:
        # Get the vocabulary size
        model_to_use = model.module if hasattr(model, 'module') else model
        vocab_size = model_to_use.t5_model.config.vocab_size
        
        # Extract valid logits and labels
        valid_logits = logits_reshaped[valid_positions]  # [valid_count, vocab_size]
        valid_labels = labels_reshaped[valid_positions]  # [valid_count]
        
        # Convert labels to one-hot and apply label smoothing
        smoothing = 0.1
        one_hot = F.one_hot(valid_labels, vocab_size).float()
        smoothed_targets = one_hot * (1 - smoothing) + smoothing / vocab_size
        
        # Get softmax of logits (predicted distribution)
        log_probs = F.log_softmax(valid_logits, dim=-1)
        
        # Calculate KL divergence
        # Note: kl_div expects log-probabilities for the first argument
        distribution_loss = F.kl_div(
            log_probs, 
            smoothed_targets,
            reduction='batchmean',
            log_target=False
        )
    
    return distribution_loss

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, config, start_epoch=0, best_val_loss=float('inf'), rank=None,
                logger=None, is_distributed=False):
    """Train the model with mixed precision training and gradient accumulation."""
    global TRAINING_FAILED

    save_frequency = config.training.checkpoint_frequency
    log_frequency = max(10, len(train_loader) // 3)

    # Enable automatic mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.mixed_precision)
    use_amp = config.training.mixed_precision
    
    if rank == 0:
        logger.info(f"Mixed precision training: {'enabled' if use_amp else 'disabled'}")

    # Enable cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    # Add memory tracking for debugging
    if torch.cuda.is_available() and rank == 0:
        logger.info(f"Initial GPU memory: {log_gpu_memory()}")
    
    # Initialize Exponential Moving Average
    ema = EMA(model, decay=0.999)
    
    # Clear cache once at beginning rather than every iteration
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Keep track of the last best model's epoch
    last_best_epoch = None
    
    # Early stopping variables
    early_stopping_counter = 0
    early_stopping_triggered = False
    prev_val_loss = float('inf')  # Track previous validation loss for comparison
    
    
    # Setup gradient buckets for efficient all-reduce if using distributed training
    gradient_buckets = None
    if is_distributed:
        if hasattr(model, 'module'):
            gradient_buckets = setup_gradient_buckets(model.module)
        else:
            gradient_buckets = setup_gradient_buckets(model)
        if rank == 0:
            logger.info(f"Created {len(gradient_buckets)} gradient buckets for efficient all-reduce")
        
    try:
        for epoch in range(start_epoch, num_epochs):
            # Check if early stopping was triggered
            if early_stopping_triggered:
                if rank == 0:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                
                # Add synchronization barrier to ensure all processes exit together
                if is_distributed and dist.is_initialized():
                    try:
                        # Use barrier to synchronize processes before exiting
                        dist.barrier()
                        if rank == 0:
                            logger.info("All processes synchronized before exit")
                    except Exception as e:
                        if rank == 0:
                            logger.error(f"Error during synchronization barrier: {e}")
                
                break
                
            if is_distributed:
                train_loader.sampler.set_epoch(epoch)
                val_loader.sampler.set_epoch(epoch)

            model.train()
            total_loss = 0
            total_ce_loss = 0
            total_distribution_similarity_loss = 0
            total_destination_loss = 0
            total_visual_reconstruction_loss = 0
            total_destination_reconstruction_loss = 0
            optimizer.zero_grad(set_to_none=True)

            epoch_start_time = time.time()

            # Only rank 0 logs the results
            if rank == 0:
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Simple direct data loading - no prefetching
                    text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                    current_view = batch['current_view_image'].to(device, non_blocking=True)
                    previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                    
                    # Handle text_label as a dictionary with input_ids and attention_mask
                    label_input_ids = batch['text_label']['input_ids'].to(device, non_blocking=True)
                    label_attention_mask = batch['text_label']['attention_mask'].to(device, non_blocking=True)
                    # Set up destination view if available in batch and curriculum is active
                    destination_view = batch['destination_image'].to(device, non_blocking=True)
                    
                    # Calculate curriculum learning ratio based on epochs
                    # Start with high ratio (rely more on destination) and gradually reduce to 0
                    max_curriculum_epochs = config.training.curriculum_epochs
                    curriculum_ratio = max(0.0, 1.0 - (epoch / max_curriculum_epochs))
                    
                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(
                            text_input, 
                            current_view, 
                            previous_views, 
                            labels=label_input_ids,
                            destination_view=destination_view,
                            curriculum_ratio=curriculum_ratio
                        )
                        
                        logits = outputs["logits"]
                        feature_norm = outputs.get("feature_norm", torch.tensor(0.0, device=device))

                        # Get batch and sequence dimensions
                        batch_size, seq_len, vocab_size = logits.size()

                        if torch.isnan(logits).any():
                            logger.error(f"[NaN Logits] NaN detected in logits on rank {rank}, batch {batch_idx}")
                            logger.error(f"Logits shape: {logits.shape}, stats: min={logits.detach().min().item():.4f}, max={logits.detach().max().item():.4f}")
                            # Don't store logits.mean() as it could cause OOM
                            optimizer.zero_grad(set_to_none=True)
                            continue

                        # Reshape outputs and labels consistently
                        logits_reshaped = logits.contiguous().view(batch_size * seq_len, vocab_size)
                        labels_reshaped = label_input_ids.contiguous().view(batch_size * seq_len)

                        # Calculate cross-entropy loss
                        ce_loss = criterion(logits_reshaped, labels_reshaped)
                        ce_loss_weight = get_weight_schedule(config.training.ce_loss_weight_start, config.training.ce_loss_weight_end, max_curriculum_epochs)(epoch)

                        # Add feature regularization with weight 0.0001
                        reg_loss = 0.0001 * feature_norm
                        loss = ce_loss_weight * ce_loss + reg_loss

                        distribution_similarity_loss = calculate_distribution_similarity_loss(logits_reshaped, labels_reshaped, label_attention_mask.reshape(-1), model, device)

                        # Combine losses with weighting from config
                        distribution_similarity_weight = get_weight_schedule(config.training.distribution_loss_weight_start, config.training.distribution_loss_weight_end, max_curriculum_epochs)(epoch)
                        loss = loss + distribution_similarity_weight * distribution_similarity_loss
                            
                        dest_features = outputs["destination_features"]
                        
                        destination_cosine_loss = calculate_cosine_similarity_loss(outputs["adapted_features"], dest_features)
                        destination_weight = get_weight_schedule(config.training.destination_loss_weight_start, config.training.destination_loss_weight_end, max_curriculum_epochs)(epoch)
                        loss = loss + destination_weight * destination_cosine_loss

                        #calculate visual reconstruction loss
                        visual_reconstruction_loss = calculate_reconstruction_loss(outputs["reconstructed_visual_features"], outputs["visual_context_target"])
                        destination_reconstruction_loss = calculate_reconstruction_loss(outputs["reconstructed_destination_features"], dest_features)
                        
                        # Base reconstruction weight from config
                        reconstruction_weight = get_weight_schedule(config.training.reconstruction_weight_start, config.training.reconstruction_weight_end, max_curriculum_epochs)(epoch)
                        
                        # Apply the weighted reconstruction losses
                        loss = loss + reconstruction_weight * (visual_reconstruction_loss + destination_reconstruction_loss)
                        
                        if torch.isnan(loss):
                            logger.error(f"[NaN Detected] NaN in loss before backward on rank {rank}, epoch {epoch}, batch {batch_idx}")
                            logger.error(f"Loss value: nan")
                            continue
                        
                        # Scale loss by gradient accumulation steps
                        loss = loss / config.training.gradient_accumulation_steps

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Check for NaNs in gradients right after backward
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            logger.error(f"[NaN Gradient] NaN detected in gradient for {name} on rank {rank}, batch {batch_idx}")
                            break
        
                    # Update weights if we've accumulated enough gradients
                    if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                        # Synchronize gradients across processes in distributed mode
                        if is_distributed and dist.is_initialized():
                            # Use bucket view approach for more efficient all-reduce
                            all_reduce_bucketed(gradient_buckets, dist.get_world_size())

                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)

                        for name, param in model.named_parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                logger.error(f"[NaN Gradient] NaN detected in gradient for {name} on rank {rank}, batch {batch_idx}")
                                break
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                        # Optimizer step with scaling
                        scaler.step(optimizer)
                        scaler.update()
                        # Step the scheduler after each update
                        scheduler.step()
                        # Update EMA weights
                        ema.update()
                        optimizer.zero_grad(set_to_none=True)

                    total_loss += loss.item() * config.training.gradient_accumulation_steps
                    total_ce_loss += ce_loss.item()
                    total_distribution_similarity_loss += distribution_similarity_loss.item()
                    total_destination_loss += destination_cosine_loss.item()
                    total_visual_reconstruction_loss += visual_reconstruction_loss.item()
                    total_destination_reconstruction_loss += destination_reconstruction_loss.item()
                
                    # Log at specified frequency
                    if batch_idx % log_frequency == 0 and rank == 0:
                        avg_loss = total_loss / (batch_idx + 1)
                        # Calculate throughput
                        elapsed = time.time() - epoch_start_time
                        batch_size = train_loader.batch_size
                        if is_distributed and dist.is_initialized():
                            samples_processed = (batch_idx + 1) * batch_size * dist.get_world_size()
                        else:
                            samples_processed = (batch_idx + 1) * batch_size
                        throughput = samples_processed / elapsed
                        
                        memory_stats = log_gpu_memory()
                        logger.info(f'Memory: {memory_stats}')
                        logger.info(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                                   f'Loss: {avg_loss:.4f}, Throughput: {throughput:.2f} samples/sec')

                    # Free variables but don't empty cache every iteration - too expensive
                    del text_input, current_view, previous_views, outputs, logits_reshaped, labels_reshaped
                    # Force garbage collection after every batch for debugging
                    gc.collect()
                    
                    # Clear CUDA cache more frequently when debugging memory issues
                    if batch_idx % 5 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    TRAINING_FAILED = True
                    logger.error(f"Critical error in training batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise e

            # Synchronize loss across processes in distributed mode
            if is_distributed and dist.is_initialized():
                loss_tensor = torch.tensor(total_loss, device=device)
                total_ce_loss_tensor = torch.tensor(total_ce_loss, device=device)   
                total_distribution_similarity_loss_tensor = torch.tensor(total_distribution_similarity_loss, device=device)
                total_destination_loss_tensor = torch.tensor(total_destination_loss, device=device)
                total_visual_reconstruction_loss_tensor = torch.tensor(total_visual_reconstruction_loss, device=device)
                total_destination_reconstruction_loss_tensor = torch.tensor(total_destination_reconstruction_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_ce_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_distribution_similarity_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_destination_loss_tensor, op=dist.ReduceOp.SUM)
                total_loss = loss_tensor.item() / dist.get_world_size()
                total_ce_loss = total_ce_loss_tensor.item() / dist.get_world_size()
                total_distribution_similarity_loss = total_distribution_similarity_loss_tensor.item() / dist.get_world_size()
                total_destination_loss = total_destination_loss_tensor.item() / dist.get_world_size()
                total_visual_reconstruction_loss = total_visual_reconstruction_loss_tensor.item() / dist.get_world_size()   
                total_destination_reconstruction_loss = total_destination_reconstruction_loss_tensor.item() / dist.get_world_size()
            # Normalize training loss
            avg_epoch_loss = total_loss / len(train_loader)
            avg_ce_loss = total_ce_loss / len(train_loader)
            avg_distribution_similarity_loss = total_distribution_similarity_loss / len(train_loader)
            avg_destination_loss = total_destination_loss / len(train_loader)
            avg_visual_reconstruction_loss = total_visual_reconstruction_loss / len(train_loader)
            avg_destination_reconstruction_loss = total_destination_reconstruction_loss / len(train_loader)
            epoch_time = time.time() - epoch_start_time
            
            if rank == 0:
                logger.info(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {avg_epoch_loss:.4f}, CE loss: {avg_ce_loss:.4f}, Distribution similarity loss: {avg_distribution_similarity_loss:.4f}, Destination loss: {avg_destination_loss:.4f}, Visual reconstruction loss: {avg_visual_reconstruction_loss:.4f}, Destination reconstruction loss: {avg_destination_reconstruction_loss:.4f}')

            # Validation phase - only run periodically to save time
            if (epoch + 1) % config.training.eval_freq == 0:
                model.eval()
                val_loss = 0
                if rank == 0:
                    logger.info("Starting validation...")
                
                # Apply EMA weights for validation
                ema.apply_shadow()

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        try:
                            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                            current_view = batch['current_view_image'].to(device, non_blocking=True)
                            previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                            
                            # Handle text_label as a dictionary with input_ids and attention_mask
                            label_input_ids = batch['text_label']['input_ids'].to(device, non_blocking=True)
                            label_attention_mask = batch['text_label']['attention_mask'].to(device, non_blocking=True)
                            
                            # Calculate curriculum learning ratio based on epochs
                            max_curriculum_epochs = config.training.curriculum_epochs
                            curriculum_ratio = max(0.0, 1.0 - (epoch / max_curriculum_epochs))
                            
                            # Set up destination view if available in batch and curriculum is active
                            destination_view = batch['destination_image'].to(device, non_blocking=True)

                            
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                outputs = model(
                                    text_input, 
                                    current_view, 
                                    previous_views, 
                                    labels=label_input_ids,
                                    destination_view=destination_view,
                                    curriculum_ratio=curriculum_ratio
                                )
                                logits = outputs["logits"]
                                batch_size, seq_len, vocab_size = logits.size()
                                logits_reshaped = logits.contiguous().view(batch_size * seq_len, vocab_size)
                                labels_reshaped = label_input_ids.contiguous().view(batch_size * seq_len)
                                
                                # Calculate cross-entropy loss
                                ce_loss = criterion(logits_reshaped, labels_reshaped)
                                ce_loss_weight = get_weight_schedule(config.training.ce_loss_weight_start, config.training.ce_loss_weight_end, max_curriculum_epochs)(epoch)
                                loss = ce_loss_weight * ce_loss
                                
                                distribution_similarity_loss = calculate_distribution_similarity_loss(logits_reshaped, labels_reshaped, label_attention_mask.reshape(-1), model, device)
                                # Combine losses with weighting from config
                                distribution_similarity_weight = get_weight_schedule(config.training.distribution_loss_weight_start, config.training.distribution_loss_weight_end, max_curriculum_epochs)(epoch)
                                loss = loss + distribution_similarity_weight * distribution_similarity_loss
                                
                                dest_features = outputs["destination_features"]
                                destination_cosine_loss = calculate_cosine_similarity_loss(outputs["adapted_features"], dest_features)
                                
                                # Apply destination loss weight
                                destination_weight = get_weight_schedule(config.training.destination_loss_weight_start, config.training.destination_loss_weight_end, max_curriculum_epochs)(epoch)
                                loss = loss + destination_weight * destination_cosine_loss
                                
                                visual_reconstruction_loss = calculate_reconstruction_loss(outputs["reconstructed_visual_features"], outputs["visual_context_target"])
                                destination_reconstruction_loss = calculate_reconstruction_loss(outputs["reconstructed_destination_features"], dest_features)
                                reconstruction_weight = get_weight_schedule(config.training.reconstruction_weight_start, config.training.reconstruction_weight_end, max_curriculum_epochs)(epoch)
                                
                                # Apply the weighted reconstruction losses
                                loss = loss + reconstruction_weight * (
                                    visual_reconstruction_loss + 
                                    destination_reconstruction_loss
                                )
                                
                            val_loss += loss.item()

                        except Exception as e:
                            TRAINING_FAILED = True
                            logger.error(f"Critical error in validation batch {batch_idx}: {str(e)}")
                            logger.error(traceback.format_exc())
                            raise e
                
                # Restore original weights after validation
                ema.restore()

                # Synchronize validation loss across processes in distributed mode
                if is_distributed and dist.is_initialized():
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = val_loss_tensor.item() / dist.get_world_size()
                val_loss /= len(val_loader)

                if rank == 0:
                    logger.info(f'Validation Loss: {val_loss:.4f}')
                    
                    # Early stopping check
                    if config.training.early_stopping:
                        # Compare against best val loss first (for significant improvements)
                        if val_loss < best_val_loss * (1.0 - config.training.early_stopping_min_delta):
                            # Validation loss improved significantly compared to best
                            early_stopping_counter = 0
                            logger.info(f"Validation loss improved significantly from {best_val_loss:.4f} to {val_loss:.4f}")
                        elif val_loss < prev_val_loss:
                            # Not better than best, but better than previous - halve the counter
                            early_stopping_counter = max(0, early_stopping_counter // 2)
                            logger.info(f"Validation loss improved from previous ({prev_val_loss:.4f} to {val_loss:.4f}) but not better than best ({best_val_loss:.4f}). Counter reduced to {early_stopping_counter}.")
                        else:
                            # Validation loss got worse
                            early_stopping_counter += 1
                            logger.info(f"Validation loss did not improve. Counter: {early_stopping_counter}/{config.training.early_stopping_patience}")
                            
                            if early_stopping_counter >= config.training.early_stopping_patience:
                                logger.info(f"Early stopping triggered! No sufficient improvement for {config.training.early_stopping_patience} epochs.")
                                early_stopping_triggered = True

                    # Save best model
                    if val_loss < best_val_loss:
                        # Remove previous best model if it exists
                        if last_best_epoch is not None:
                            prev_best_model = os.path.join(checkpoint_dir, f'best_model_epoch_{last_best_epoch}.pt')
                            if os.path.exists(prev_best_model):
                                try:
                                    os.remove(prev_best_model)
                                    logger.info(f"Removed previous best model from epoch {last_best_epoch}")
                                except Exception as e:
                                    logger.warning(f"Failed to remove previous best model: {str(e)}")

                        best_val_loss = val_loss
                        last_best_epoch = epoch + 1

                        # Save model with EMA weights
                        ema.apply_shadow()  # Apply EMA weights for saving
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save({
                            'model_state_dict': model_to_save.state_dict(),
                            'epoch': epoch + 1,
                            'val_loss': val_loss,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'ema': ema.state_dict()
                        }, os.path.join(checkpoint_dir, f'best_model_epoch_{epoch + 1}.pt'))
                        ema.restore()  # Restore original weights for training
                        logger.info(f'New best model saved at epoch {epoch + 1} (val_loss: {val_loss:.4f})')

                # Update previous validation loss for next comparison
                prev_val_loss = val_loss
                    
                # Save periodic checkpoint
                if (epoch + 1) % save_frequency == 0 and rank == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                        'ema': ema.state_dict()
                    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
                    logger.info(f'Checkpoint saved at epoch {epoch + 1}')

                # Clear cache after validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        TRAINING_FAILED = True
        logger.error(f"Fatal error in training loop: {str(e)}")
        logger.error(traceback.format_exc())
        raise e


def log_gpu_memory():
    """Log GPU memory usage for the current device only (no distributed operations)."""
    if not torch.cuda.is_available():
        return "No CUDA devices available"
        
    try:
        # Only log the current device's memory
        current_device = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024 ** 2
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1024 ** 2
        
        # Simple format that's easy to read
        result = f"GPU {current_device}: {memory_allocated:.1f}MB allocated, {memory_reserved:.1f}MB reserved"
        
        return result
    except Exception as e:
        # Do not fail training due to memory logging error
        return f"Error logging GPU memory: {str(e)}"


def setup_distributed():
    """Set up the distributed environment using environment variables set by torchrun."""
    # Get distributed training environment variables from torchrun
    if "LOCAL_RANK" not in os.environ:
        # Not running with torchrun, assume single-GPU
        return False, 0, 1
    
    # Check available GPU count
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        print("No CUDA devices available! Running on CPU only.")
        return False, 0, 1
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Ensure local_rank is within the range of available devices
    if local_rank >= available_gpus:
        print(f"WARNING: local_rank ({local_rank}) >= available GPUs ({available_gpus})")
        print(f"Remapping local_rank to available device: {local_rank % available_gpus}")
        local_rank = local_rank % available_gpus
    
    # Set the device
    torch.cuda.set_device(local_rank)
    
    # Ensure proper initialization of master address and port
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}, MASTER_PORT: {os.environ['MASTER_PORT']}")
    
    # Initialize the process group - Always use env:// for torchrun
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(minutes=30)
            )
            print(f"Successfully initialized process group for rank {rank}")
        except Exception as e:
            print(f"Error initializing process group: {e}")
            traceback.print_exc()
            raise
    
    return True, rank, world_size


def cleanup():
    """Force cleanup of multiprocessing resources"""
    if dist.is_initialized():
        dist.destroy_process_group()
    
    # Force garbage collection
    gc.collect()
    
    # Clean up CUDA resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():
    """Main training function that works with torchrun."""
    global TRAINING_FAILED
    
    # Set memory allocation optimizations early, before any CUDA operations
    if torch.cuda.is_available():
        # Set max split size to reduce memory fragmentation - good for all training
        # Using only parameters supported in PyTorch 1.11.0
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # Optimize NCCL for better multi-GPU communication
        os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
        os.environ['NCCL_SOCKET_NTHREADS'] = '4'
    
    # Parse arguments first
    parser = argparse.ArgumentParser(description='Train AnsweringAgent with DDP')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from', default=None)
    parser.add_argument('--single-gpu', action='store_true', help='Force running on a single GPU even with torchrun')
    parser.add_argument('--batch-size', type=int, help='Per-GPU batch size (overrides config value)', default=None)
    parser.add_argument('--grad-steps', type=int, help='Gradient accumulation steps (overrides config value)', default=None)
    args = parser.parse_args()
    
    # Check CUDA availability first
    if not torch.cuda.is_available():
        print("CUDA is not available! Training will run on CPU only.")
        num_gpus = 0
    else:
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s)")
    
    if args.single_gpu or num_gpus <= 1:
        # Single GPU or CPU mode
        print("Running in single device mode")
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
            # Initialize distributed environment using torchrun environment variables
            is_distributed, rank, world_size = setup_distributed()
            local_rank = int(os.environ.get("LOCAL_RANK", 0)) % max(1, torch.cuda.device_count())
            device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            print(f"Error setting up distributed environment: {e}")
            traceback.print_exc()
            print("Falling back to single GPU mode")
            is_distributed = False
            rank = 0
            world_size = 1
            local_rank = 0
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    print(f"Process {rank} using device: {device}, Local rank: {local_rank}, World size: {world_size}, Distributed: {is_distributed}")
    
    
    # Enable detailed debug and error reporting only on main process
    try:
        if rank == 0:
            set_debug_mode()
    except Exception as e:
        print(f"Error in debug mode setup: {e}")
    
    # Set up signal handlers for proper cleanup
    def signal_handler(sig, frame):
        print(f"Process {rank} received signal {sig}, cleaning up and exiting...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration
    config = Config()
    
    # Set seed for reproducibility - must happen after config is loaded but before any random operations
    set_seed(config.training.seed)
    if rank == 0:
        print(f"Set random seed to {config.training.seed} for all processes")

    # Initialize logger - only rank 0 should log to console
    logger = setup_logger('training', log_dir=config.log_dir)

    # Override config values with command-line arguments if provided
    if args.batch_size is not None:
        config.training.per_gpu_batch_size = args.batch_size
        if rank == 0:
            logger.info(f"Overriding batch size with command-line value: {args.batch_size}")
            
    if args.grad_steps is not None:
        config.training.gradient_accumulation_steps = args.grad_steps
        if rank == 0:
            logger.info(f"Overriding gradient accumulation steps with command-line value: {args.grad_steps}")
    
    # Silence non-rank-0 processes by setting logger level
    if rank != 0:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(logging.ERROR)  # Only show errors on non-rank-0
    
    if rank == 0:
        logger.info(f"Starting training with debugging enabled. Error file: {temp_error_file.name}")
        logger.info(f"Process information: Rank {rank}, Local Rank {local_rank}, World Size {world_size}")
        logger.info(f"Device: {device}")

    try:
        # Initialize tokenizer
        tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name, model_max_length=config.data.max_seq_length, add_special_tokens=True)
        
        if rank == 0:
            logger.info(f"Training on {max(1, num_gpus)} GPUs, distributed mode: {is_distributed}")
        
        # Wait for rank 0 to finish preprocessing
        if is_distributed:
            dist.barrier()
        
        if rank == 0:
            logger.info("Starting model initialization...")
            
        # Initialize model and move to correct GPU
        model = AnsweringAgent(config, tokenizer, logger)
        if is_distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)

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
                logger.info(f"Loading checkpoint from {args.checkpoint}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(args.checkpoint, map_location=map_location)
            
            if hasattr(model, 'module'):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            if rank == 0:
                logger.info(f"Resuming training from epoch {start_epoch}")
            
            # Load EMA state from checkpoint
            if 'ema' in checkpoint:
                ema.load_state_dict(checkpoint['ema'])
                if rank == 0:
                    logger.info("Loaded EMA state from checkpoint")
        
        # Load dataset
        try:
            if rank == 0:
                logger.info("Creating datasets...")
                for split in ['train', 'val_seen', 'val_unseen']:
                    AnsweringDataset.preprocess_and_save(config, tokenizer, split=split, logger=logger)

            if dist.is_initialized():
                dist.barrier()

            datasets = AnsweringDataset.create_datasets(config, logger=logger, splits=['train', 'val_seen'], tokenizer=tokenizer, exhuastive_loading=False)

        except Exception as e:
            logger.error(f"Critical error loading dataset: {str(e)}")
            logger.error(traceback.format_exc())
            TRAINING_FAILED = True
            raise e
        
        
        
        if rank == 0:
            logger.info(f"Dataset split: {len(datasets['train'])} training, {len(datasets['val_seen'])} validation")
        

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
            batch_str = f"Per-GPU batch size: {config.training.per_gpu_batch_size}"
            if is_distributed:
                effective_batch_size = config.training.per_gpu_batch_size * world_size * config.training.gradient_accumulation_steps
                batch_str += f" (effective batch size: {effective_batch_size}, with gradient_accumulation_steps={config.training.gradient_accumulation_steps})"
            logger.info(batch_str)

        train_loader = DataLoader(
            datasets['train'],
            batch_size=config.training.per_gpu_batch_size,
            sampler=train_sampler,
            shuffle=shuffle,  # Only shuffle if not using sampler
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=(config.training.num_workers > 0)
        )

        val_loader = DataLoader(
            datasets['val_seen'],
            batch_size=config.training.per_gpu_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=(config.training.num_workers > 0)
        )

        # Create warmup then decay scheduler
        def get_lr_schedule(optimizer, warmup_steps, total_steps):
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    # Introduce curriculum-aware decay
                    curriculum_phase_steps = int(total_steps * 0.15)  # e.g., first 15% of training
                    if current_step < warmup_steps + curriculum_phase_steps:
                        # Faster decay during curriculum learning
                        progress = float(current_step - warmup_steps) / float(max(1, curriculum_phase_steps))
                        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    else:
                        # Slower decay after curriculum phase
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
            start_epoch=start_epoch,
            best_val_loss=best_val_loss,
            rank=rank,
            logger=logger,
            is_distributed=is_distributed
        )

        # Normal cleanup
        cleanup()
        
        if rank == 0:
            logger.info("Training completed successfully.")

    except Exception as e:
        # Mark training as failed
        TRAINING_FAILED = True
        if logger:
            # Log all details of the exception
            error_msg = f"Fatal error in main function: {str(e)}"
            tb_str = traceback.format_exc()
            logger.error(error_msg)
            logger.error(tb_str)
            
            # Also print to stderr for torchrun to capture
            print(error_msg, file=sys.stderr)
            print(tb_str, file=sys.stderr)
            
            # Write to the elastic error file directly as well
            with open(temp_error_file.name, 'a') as f:
                f.write(f"RANK {rank}: {error_msg}\n")
                f.write(tb_str)
        else:
            print(f"Fatal error: {str(e)}")
            print(traceback.format_exc())
    finally:
        # Proper cleanup for distributed environment
        if is_distributed and dist.is_initialized():
            try:
                # Use barrier to synchronize processes before cleanup
                dist.barrier()
                # Destroy process group
                dist.destroy_process_group()
                if rank == 0 and logger:
                    logger.info("Distributed process group destroyed successfully")
            except Exception as e:
                if rank == 0 and logger:
                    logger.error(f"Error during distributed cleanup: {e}")
                else:
                    print(f"Error during distributed cleanup: {e}")
        
        # General cleanup
        cleanup()
        
        if rank == 0:
            if TRAINING_FAILED:
                if logger:
                    logger.error("Training failed with errors")
                else:
                    print("Training failed with errors")
                # Exit with error code
                sys.exit(1)
            else:
                if logger:
                    logger.info("Training completed successfully.")
                else:
                    print("Training completed successfully.")


# Helper functions for efficient gradient all_reduce
def _flatten_dense_tensors(tensors):
    """Flatten and concatenate dense tensors."""
    flat = torch.cat([t.contiguous().view(-1) for t in tensors])
    return flat

def _unflatten_dense_tensors(flat, tensors):
    """View the flattened tensor as tensors with original shapes."""
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return outputs

# New, more efficient approach using bucket views
def setup_gradient_buckets(model, bucket_size_mb=25):
    """Setup gradient buckets for efficient all-reduce.
    Args:
        model: The model to setup buckets for
        bucket_size_mb: Target bucket size in MB
    Returns:
        List of buckets, where each bucket is a list of parameters
    """
    buckets = []
    current_bucket = []
    current_size = 0
    target_size = bucket_size_mb * 1024 * 1024 / 4  # Convert MB to number of float32 elements

    # Group parameters with gradients
    for param in model.parameters():
        if param.requires_grad:
            param_size = param.numel()
            
            # If adding this parameter exceeds target bucket size, start a new bucket
            if current_size + param_size > target_size and current_bucket:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            
            current_bucket.append(param)
            current_size += param_size
    
    # Add last bucket if it has parameters
    if current_bucket:
        buckets.append(current_bucket)
    
    return buckets

def all_reduce_bucketed(buckets, world_size):
    """Perform all-reduce on parameter buckets more efficiently.
    Args:
        buckets: List of bucket lists, where each bucket is a list of parameters
        world_size: Number of processes in the distributed group
    """
    for bucket in buckets:
        # Get grad buffer views from all parameters in bucket
        grads = [param.grad.data for param in bucket if param.grad is not None]
        
        if not grads:
            continue
            
        # Create a single flat buffer to hold all grads in this bucket
        # We first determine the required dtype and device
        first_grad = grads[0]
        flat_buffer = torch.zeros(
            sum(grad.numel() for grad in grads),
            dtype=first_grad.dtype,
            device=first_grad.device
        )
        
        # Copy grads to flat buffer with views
        offset = 0
        grad_views = []
        for grad in grads:
            grad_numel = grad.numel()
            grad_view = flat_buffer[offset:offset+grad_numel].view_as(grad)
            grad_view.copy_(grad)
            grad_views.append(grad_view)
            offset += grad_numel
        
        # All-reduce on the flat buffer
        dist.all_reduce(flat_buffer, op=dist.ReduceOp.SUM)
        
        # Scale by world size
        flat_buffer.div_(world_size)
        
        # The grad views automatically update the original grads
        # We don't need to copy back


if __name__ == '__main__':
    import argparse
    main()
