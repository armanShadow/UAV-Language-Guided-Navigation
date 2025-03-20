import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from transformers import BertTokenizerFast, BertModel, BertTokenizer
from utils.logger import setup_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from config import Config
from models.answering_agent import AnsweringAgent
from data.dataset import AnsweringDataset
import traceback
import datetime
import time
import sys
import logging
from pathlib import Path


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


def log_gpu_memory(should_gather=True, should_log=False, logger=None):
    """
    Log GPU memory statistics for all processes or just the current one.
    
    Args:
        should_gather: If True, gather stats from all processes
        should_log: If True, log the stats to console
        logger: Logger to use for logging
    
    Returns:
        String with memory information
    """
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    # Get memory allocated for current device
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    
    if not should_gather:
        mem_info = f"GPU:{torch.cuda.current_device()} - Allocated: {allocated:.2f}MB, Peak: {max_allocated:.2f}MB, Reserved: {reserved:.2f}MB"
        if should_log and logger:
            logger.info(mem_info)
        return mem_info
    
    # Gather information from all processes if needed
    if dist.is_initialized():
        world_size = dist.get_world_size()
        mem_tensor = torch.tensor([allocated, max_allocated, reserved], device=torch.cuda.current_device())
        gathered_mem = [torch.zeros_like(mem_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_mem, mem_tensor)
        
        mem_info = "GPU Memory Usage:\n"
        for i, mem in enumerate(gathered_mem):
            mem_info += f"GPU:{i} - Allocated: {mem[0]:.2f}MB, Peak: {mem[1]:.2f}MB, Reserved: {mem[2]:.2f}MB\n"
    else:
        mem_info = f"GPU:{torch.cuda.current_device()} - Allocated: {allocated:.2f}MB, Peak: {max_allocated:.2f}MB, Reserved: {reserved:.2f}MB"
    
    if should_log and logger:
        logger.info(mem_info)
    
    return mem_info


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, config, start_epoch=0, best_val_loss=float('inf'), rank=None,
                logger=None):
    """Train the model with mixed precision training and gradient accumulation."""

    save_frequency = config.training.checkpoint_frequency
    log_frequency = max(1, len(train_loader) // 3)  # Log approximately 3 times per epoch
  

    # Enable automatic mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=config.training.enable_amp)

    # Enable cuDNN benchmarking for faster convolutions if using consistent input sizes
    # but disable it if input sizes vary a lot as it can cause memory spikes
    torch.backends.cudnn.benchmark = True
    
    # For deterministic behavior (might be slower but more consistent)
    if config.training.seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Keep track of the last best model's epoch
    last_best_epoch = None
    
    # Start with local memory logging (no sync)
    if rank == 0:
        logger.info(f"Initial GPU Memory: {log_gpu_memory(should_gather=False)}")
        
    # Clear cache before starting training (this is strategic and helpful)
    if config.training.optimize_memory_usage:
        torch.cuda.empty_cache()

    try:
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for samplers
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

            # Clear memory at the start of each epoch (strategic point)
            if config.training.optimize_memory_usage:
                torch.cuda.empty_cache()

            model.train()
            total_loss = 0
            optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()
            
            # Always log epoch start
            if rank == 0:
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
                logger.info(f"Creating data loader iterator...")
            
            # Create iterator with timeout handling
            train_iter = iter(train_loader)
            
            if rank == 0:
                logger.info(f"Data loader iterator created successfully. Starting batch processing...")

            # Accumulate gradients locally before synchronizing
            batch_idx = 0
            while batch_idx < len(train_loader):
                try:
                    # Set a timeout for fetching batch to prevent hanging
                    start_time = time.time()
                    timeout = 60  # 60 seconds timeout per batch
                    
                    # Try to get next batch with timeout
                    while True:
                        if time.time() - start_time > timeout:
                            if rank == 0:
                                logger.warning(f"Timeout fetching batch {batch_idx}, skipping to next batch")
                            break
                            
                        try:
                            batch = next(train_iter)
                            break  # Successfully got batch
                        except StopIteration:
                            # End of epoch
                            if rank == 0:
                                logger.info(f"Reached end of data loader at batch {batch_idx}")
                            break
                        except Exception as e:
                            if rank == 0:
                                logger.error(f"Error fetching batch: {str(e)}")
                            time.sleep(0.1)  # Short sleep before retry
                    
                    # Check if we timed out or reached end of iterator
                    if time.time() - start_time > timeout:
                        batch_idx += 1
                        continue
                    
                    # Move data to device (non-blocking for async transfer)
                    text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                    current_view = batch['current_view_image'].to(device, non_blocking=True)
                    
                    # previous_views is already properly sized by the dataset, just move to device
                    previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                                        
                    labels = batch['text_label'].to(device, non_blocking=True)

                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast(enabled=config.training.enable_amp):
                        outputs = model(text_input, current_view, previous_views)

                        # Get batch and sequence dimensions
                        batch_size, seq_len, vocab_size = outputs.size()

                        # Reshape outputs and labels consistently
                        outputs_reshaped = outputs.contiguous().view(batch_size * seq_len, vocab_size)
                        labels_reshaped = labels.contiguous().view(batch_size * seq_len)

                        loss = criterion(outputs_reshaped, labels_reshaped)
                        # Scale loss by gradient accumulation steps
                        loss = loss / config.training.gradient_accumulation_steps

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()

                    # Clean up tensors we don't need anymore
                    del outputs, outputs_reshaped, labels_reshaped

                    # Update weights if we've accumulated enough gradients
                    if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                        # Synchronize gradients across processes
                        if dist.is_initialized():
                            for param in model.parameters():
                                if param.grad is not None:
                                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                                    param.grad.data /= dist.get_world_size()

                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                        # Optimizer step with scaling
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)  # More efficient

                    # Clean up input tensors
                    del text_input, current_view, previous_views, labels

                    total_loss += loss.item() * config.training.gradient_accumulation_steps
                    del loss

                    # Log at specified frequency - no memory logging here to reduce synchronization
                    if batch_idx % log_frequency == 0 and rank == 0:
                        avg_loss = total_loss / (batch_idx + 1)
                        memory_stats = log_gpu_memory(should_gather=True)
                        logger.info(f'Epoch {epoch + 1} GPU Memory: {memory_stats}')
                        logger.info(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {avg_loss:.4f}')
                    
                    # Move to next batch
                    batch_idx += 1

                except RuntimeError as e:
                    # Specialized handling for OOM errors
                    if "CUDA out of memory" in str(e):
                        logger.error(f"CUDA OOM in batch {batch_idx}. Attempting to recover...")
                        # Try to free memory - this is a good place to clear cache
                        torch.cuda.empty_cache()
                        # Wait a bit
                        time.sleep(2)
                        # Skip this batch
                        logger.error(f"Skipping batch {batch_idx} due to OOM error.")
                        continue
                    else:
                        logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # Synchronize loss across processes at the end of the epoch
            if dist.is_initialized():
                loss_tensor = torch.tensor(total_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                total_loss = loss_tensor.item() / dist.get_world_size()

            # Normalize training loss
            avg_epoch_loss = total_loss / len(train_loader)
            if rank == 0:
                logger.info(f'Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}')

            # Clear memory before validation (strategic point)
            if config.training.optimize_memory_usage:
                torch.cuda.empty_cache()

            # Validation phase - only run at specified frequency to reduce overhead
            if (epoch + 1) % config.training.eval_freq == 0:
                model.eval()
                val_loss = 0
                if rank == 0:
                    logger.info("Starting validation...")

                with torch.no_grad():
                    # Use same approach as training loop for validation with timeout handling
                    val_iter = iter(val_loader)
                    batch_idx = 0
                    
                    while batch_idx < len(val_loader):
                        try:
                            # Set a timeout for fetching batch
                            start_time = time.time()
                            timeout = 30  # 30 seconds timeout per validation batch
                            
                            # Try to get next batch with timeout
                            while True:
                                if time.time() - start_time > timeout:
                                    if rank == 0:
                                        logger.warning(f"Timeout fetching validation batch {batch_idx}, skipping")
                                    break
                                    
                                try:
                                    batch = next(val_iter)
                                    break  # Successfully got batch
                                except StopIteration:
                                    # End of validation data
                                    if rank == 0:
                                        logger.info(f"Reached end of validation data at batch {batch_idx}")
                                    break
                                except Exception as e:
                                    if rank == 0:
                                        logger.error(f"Error fetching validation batch: {str(e)}")
                                    time.sleep(0.1)  # Short sleep before retry
                            
                            # Check if we timed out or reached end of iterator
                            if time.time() - start_time > timeout:
                                batch_idx += 1
                                continue
                            
                            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                            current_view = batch['current_view_image'].to(device, non_blocking=True)
                            
                            # previous_views is already properly sized by the dataset, just move to device
                            previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                                                
                            labels = batch['text_label'].to(device, non_blocking=True)

                            with torch.cuda.amp.autocast(enabled=config.training.enable_amp):
                                outputs = model(text_input, current_view, previous_views)
                                batch_size, seq_len, vocab_size = outputs.size()
                                outputs_reshaped = outputs.contiguous().view(batch_size * seq_len, vocab_size)
                                labels_reshaped = labels.contiguous().view(batch_size * seq_len)
                                loss = criterion(outputs_reshaped, labels_reshaped)
                            val_loss += loss.item()
                            
                            # Clean up tensors
                            del text_input, current_view, previous_views, labels, outputs, outputs_reshaped, labels_reshaped, loss
                            
                            # Move to next batch
                            batch_idx += 1

                        except Exception as e:
                            logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                            batch_idx += 1
                            continue

                # Synchronize validation loss across processes
                if dist.is_initialized():
                    val_loss_tensor = torch.tensor(val_loss, device=device)
                    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                    val_loss = val_loss_tensor.item() / dist.get_world_size()
                val_loss /= len(val_loader)

                if rank == 0:
                    logger.info(f'Validation Loss: {val_loss:.4f}')
                    scheduler.step(val_loss)

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

                        # Save model
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save({
                            'model_state_dict': model_to_save.state_dict(),
                            'epoch': epoch + 1,
                            'val_loss': val_loss,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                        }, os.path.join(checkpoint_dir, f'best_model_epoch_{epoch + 1}.pt'))
                        logger.info(f'New best model saved at epoch {epoch + 1} (val_loss: {val_loss:.4f})')

                # Save periodic checkpoint
                if (epoch + 1) % save_frequency == 0 and rank == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt'))
                    logger.info(f'Checkpoint saved at epoch {epoch + 1}')

                # Clear cache after validation (strategic point)
                if config.training.optimize_memory_usage:
                    torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in training loop: {str(e)}")
        logger.error(traceback.format_exc())
        raise e


def setup(rank, world_size):
    """Setup process groups for distributed training."""
    print(f"Setting up process {rank}/{world_size}")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Process {rank}/{world_size} setup completed")


def main(rank, world_size, checkpoint_path=None, config=Config(), tokenizer=None):
    # Initialize logger for this process
    logger = setup_logger('training', log_dir=config.log_dir)

    try:
        # Set environment variables for DDP
        setup(rank, world_size)

        # Explicitly set the CUDA device for this process
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        # Apply memory optimization settings
        if config.training.optimize_memory_usage:
            # Empty cache before model initialization
            torch.cuda.empty_cache()
            
            # Set PyTorch memory allocator for fragmentation handling
            if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
                # Set allocation strategy to reduce fragmentation
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        if rank == 0:
            logger.info(f"Training on {world_size} GPUs")
            logger.info(f"Initial GPU Memory (before model load): {log_gpu_memory(should_gather=False)}")
            
        # Initialize model and move to correct GPU
        model = AnsweringAgent(config)
        
        if config.training.use_gradient_checkpointing:
            # Enable gradient checkpointing
            if hasattr(model, 'bert') and hasattr(model.bert, 'gradient_checkpointing_enable'):
                model.bert.gradient_checkpointing_enable()
                if rank == 0:
                    logger.info("Gradient checkpointing enabled for BERT")
        
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)
        
        if rank == 0:
            logger.info(f"Model loaded to GPU: {log_gpu_memory(should_gather=False)}")

        # Initialize training variables
        start_epoch = 0
        best_val_loss = float('inf')

        # Wrap model with DDP
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
            broadcast_buffers=True
        )

        # Resume training if checkpoint is provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            if rank == 0:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            if rank == 0:
                logger.info(f"Resuming training from epoch {start_epoch}")

        # Optimizer, loss, and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            verbose=config.training.scheduler_verbose
        )

        # Load dataset and ensure deterministic splitting
        dataset = AnsweringDataset(config=config, tokenizer=tokenizer)
        generator = torch.Generator().manual_seed(config.training.seed)
        train_size = int(config.data.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

        # Use DistributedSampler for DDP
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        
        if rank == 0:
            logger.info(f"Per-GPU batch size: {config.training.batch_size} (global batch size: {config.training.batch_size * world_size})")
            logger.info(f"Gradient accumulation steps: {config.training.gradient_accumulation_steps}")
            logger.info(f"Effective batch size: {config.training.batch_size * world_size * config.training.gradient_accumulation_steps}")

        # Use pinned memory only if sufficient system RAM is available
        pin_memory = config.training.pin_memory
        
        # Consider reducing workers if memory is an issue
        num_workers = config.training.num_workers
        
        # For debugging hangs, try with zero workers
        debug_mode = True
        if debug_mode and rank == 0:
            logger.info("DEBUG MODE: Using 0 workers for DataLoader to troubleshoot hanging")
            num_workers = 0
            persistent_workers = False
        else:
            persistent_workers = (num_workers > 0)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers  # Keep workers alive between batches
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
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
            logger=logger
        )

        # Clean up after training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Cleanup
        dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            # Attempt to clean up on error
            dist.destroy_process_group()
            torch.cuda.empty_cache()
        except:
            pass
        raise e


if __name__ == '__main__':
    import argparse
    import torch.multiprocessing as mp

    config = Config()

    # Initialize tokenizer for dataset processing
    tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)

    parser = argparse.ArgumentParser(description='Train AnsweringAgent with DDP')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from', default=None)
    args = parser.parse_args()

    # Set up distributed training
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA GPUs available for training")

    # Clean up any stale file handles or shared memory
    try:
        dist.destroy_process_group()
    except:
        pass

    try:
        mp.spawn(
            main,
            args=(world_size, args.checkpoint, config, tokenizer),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Error in main process: {str(traceback.format_exc())}")
        # Try to clean up on error
        try:
            dist.destroy_process_group()
        except:
            pass
        # Make sure all processes are terminated
        import sys
        sys.exit(1)