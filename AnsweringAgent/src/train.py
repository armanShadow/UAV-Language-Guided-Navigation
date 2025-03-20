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
import pickle
import gc
import signal


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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, config, start_epoch=0, best_val_loss=float('inf'), rank=None,
                logger=None):
    """Train the model with mixed precision training and gradient accumulation."""

    save_frequency = config.training.checkpoint_frequency
    # Log less frequently to reduce overhead - adjust based on dataset size
    log_frequency = max(10, len(train_loader) // 3)  # Log approximately 10 times per epoch

    # Enable automatic mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Enable cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True

    # Keep track of the last best model's epoch
    last_best_epoch = None

    try:
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for samplers
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

            model.train()
            total_loss = 0
            optimizer.zero_grad(set_to_none=True)

            # Track start time for per-epoch metrics
            epoch_start_time = time.time()

            # Only rank 0 logs the results
            if rank == 0:
                logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Move data to device (non-blocking for async transfer)
                    text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                    current_view = batch['current_view_image'].to(device, non_blocking=True)
                    previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                    labels = batch['text_label'].to(device, non_blocking=True)

                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast():
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
                        optimizer.zero_grad(set_to_none=True)

                    total_loss += loss.item() * config.training.gradient_accumulation_steps
                
                    memory_stats = log_gpu_memory()

                    # Log at specified frequency
                    if batch_idx % log_frequency == 0 and rank == 0:
                        avg_loss = total_loss / (batch_idx + 1)
                        # Calculate throughput
                        elapsed = time.time() - epoch_start_time
                        samples_processed = (batch_idx + 1) * train_loader.batch_size * dist.get_world_size()
                        throughput = samples_processed / elapsed
                        
                        logger.info(f'GPU Memory: {memory_stats}')
                        logger.info(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                                    f'Loss: {avg_loss:.4f}, Throughput: {throughput:.2f} samples/sec')

                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # Synchronize loss across processes
            if dist.is_initialized():
                loss_tensor = torch.tensor(total_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                total_loss = loss_tensor.item() / dist.get_world_size()

            # Normalize training loss
            avg_epoch_loss = total_loss / len(train_loader)
            epoch_time = time.time() - epoch_start_time
            
            if rank == 0:
                logger.info(f'Epoch {epoch + 1} completed in {epoch_time:.2f}s. Average loss: {avg_epoch_loss:.4f}')

            # Validation phase - only run periodically to save time
            if (epoch + 1) % config.training.eval_freq == 0:
                model.eval()
                val_loss = 0
                if rank == 0:
                    logger.info("Starting validation...")

                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        try:
                            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                            current_view = batch['current_view_image'].to(device, non_blocking=True)
                            previous_views = batch['previous_views_image'].to(device, non_blocking=True)
                            labels = batch['text_label'].to(device, non_blocking=True)

                            with torch.cuda.amp.autocast():
                                outputs = model(text_input, current_view, previous_views)
                                batch_size, seq_len, vocab_size = outputs.size()
                                outputs_reshaped = outputs.contiguous().view(batch_size * seq_len, vocab_size)
                                labels_reshaped = labels.contiguous().view(batch_size * seq_len)
                                loss = criterion(outputs_reshaped, labels_reshaped)
                            val_loss += loss.item()

                        except Exception as e:
                            logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
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

                # Clear cache after validation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in training loop: {str(e)}")
        logger.error(traceback.format_exc())
        raise e


def log_gpu_memory():
    """Log GPU memory usage for all available GPUs with consolidated statistics."""
        
    if not torch.cuda.is_available():
        return "No CUDA devices available"
        
    current_device = torch.cuda.current_device()
    memory_allocated = torch.cuda.memory_allocated(current_device) / 1024 ** 2
    memory_reserved = torch.cuda.memory_reserved(current_device) / 1024 ** 2
    
    # Create tensors to gather memory stats from all processes
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # If we're not in distributed mode, just return the current device stats
    if not dist.is_initialized():
        return f"GPU {current_device}: {memory_allocated:.1f}MB/{memory_reserved:.1f}MB"
    
    # For distributed mode, gather stats from all processes
    gathered_allocated = torch.zeros(world_size, device=f'cuda:{current_device}')
    gathered_reserved = torch.zeros(world_size, device=f'cuda:{current_device}')
    
    # Prepare tensors for total and mean statistics
    total_allocated = torch.tensor([memory_allocated], device=f'cuda:{current_device}')
    total_reserved = torch.tensor([memory_reserved], device=f'cuda:{current_device}')
    
    # Each process puts its memory stats in the corresponding index
    gathered_allocated[rank] = memory_allocated
    gathered_reserved[rank] = memory_reserved
    
    # Use all_gather instead of all_reduce to avoid potential deadlocks
    # This is safer as each process only needs to provide its own data
    all_allocated = [torch.zeros(1, device=f'cuda:{current_device}') for _ in range(world_size)]
    all_reserved = [torch.zeros(1, device=f'cuda:{current_device}') for _ in range(world_size)]
    
    # Put local data in a tensor
    local_allocated = torch.tensor([memory_allocated], device=f'cuda:{current_device}')
    local_reserved = torch.tensor([memory_reserved], device=f'cuda:{current_device}')
    
    dist.all_gather(all_allocated, local_allocated)
    dist.all_gather(all_reserved, local_reserved)
   
    # Format the memory stats string
    memory_stats = []
    total_allocated_sum = 0
    total_reserved_sum = 0
    
    for i in range(world_size):
        gpu_allocated = all_allocated[i].item()
        gpu_reserved = all_reserved[i].item()
        memory_stats.append(f'GPU {i}: {gpu_allocated:.1f}MB/{gpu_reserved:.1f}MB')
        total_allocated_sum += gpu_allocated
        total_reserved_sum += gpu_reserved
    
    # Add total and mean statistics
    memory_stats.append(f'Total: {total_allocated_sum:.1f}MB allocated, {total_reserved_sum:.1f}MB reserved')
    memory_stats.append(f'Mean: {(total_allocated_sum/world_size):.1f}MB allocated, {(total_reserved_sum/world_size):.1f}MB reserved')
    
    result = ', '.join(memory_stats)
    
    return result



def setup(rank, world_size):
    # Set basic environment variables
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group with NCCL backend
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )


def main(rank, world_size, checkpoint_path=None, config=Config(), tokenizer=None):
    # Initialize logger for this process
    logger = setup_logger('training', log_dir=config.log_dir)

    try:
        # Set environment variables for DDP
        setup(rank, world_size)

        # Explicitly set the CUDA device for this process
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        if rank == 0:
            logger.info(f"Training on {world_size} GPUs")
            
            # Preprocess dataset on rank 0 only - use more efficient preprocess_all_data
            logger.info("Starting dataset preprocessing on rank 0...")
            start_time = time.time()
            AnsweringDataset.preprocess_and_save(config, tokenizer, logger)
            preprocess_time = time.time() - start_time
            logger.info(f"Dataset preprocessing complete. Took {preprocess_time:.2f} seconds.")
        
        # Wait for rank 0 to finish preprocessing
        dist.barrier()
        
        if rank == 0:
            logger.info("All ranks synced after preprocessing. Starting model initialization...")
            
        # Initialize model and move to correct GPU
        model = AnsweringAgent(config)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(device)

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

        # Load dataset
        dataset = AnsweringDataset(config=config)
        
        generator = torch.Generator().manual_seed(config.training.seed)
        train_size = int(config.data.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

        # Use DistributedSampler for DDP
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        
        if rank == 0:
            logger.info(f"Per-GPU batch size: {config.training.batch_size} (global batch size: {config.training.batch_size * world_size})")

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=config.training.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            sampler=val_sampler,
            num_workers=config.training.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn
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

        # Cleanup
        cleanup()

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        raise e


# Worker initialization function for DataLoader
def worker_init_fn(worker_id):
    """
    Initialize each worker process properly to avoid semaphore leaks.
    """
    # Set a different seed for each worker to ensure different random samples
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    
    # Clean up any previous resources and reset signal handlers
    # This helps to avoid semaphore leaks
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # Force garbage collection at start
    gc.collect()


# Function to properly clean up multiprocessing resources
def cleanup():
    """Force cleanup of multiprocessing resources"""
    if dist.is_initialized():
        dist.destroy_process_group()
    
    # Force garbage collection
    gc.collect()
    
    # Try to clean up any stray semaphores
    if torch.multiprocessing.get_start_method() == 'fork':
        try:
            torch.multiprocessing.resource_tracker.getfd()
        except:
            pass


if __name__ == '__main__':
    import argparse
    import torch.multiprocessing as mp

    # Set appropriate multiprocessing start method to avoid semaphore leaks
    if torch.cuda.is_available():
        # Spawn is recommended for CUDA operations to avoid sharing CUDA context
        mp.set_start_method('spawn', force=True)
    
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
        cleanup()
    except:
        pass

    try:
        mp.spawn(
            main,
            args=(world_size, args.checkpoint, config, tokenizer),
            nprocs=world_size,
            join=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error in main process: {str(traceback.format_exc())}")
    finally:
        # Make sure all cleanup is performed
        try:
            cleanup()
        except:
            pass