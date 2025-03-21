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
import time
import gc
import signal
import sys
import datetime
import logging
import faulthandler
import tempfile

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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, checkpoint_dir, config, start_epoch=0, best_val_loss=float('inf'), rank=None,
                logger=None, is_distributed=False):
    """Train the model with mixed precision training and gradient accumulation."""
    global TRAINING_FAILED

    save_frequency = config.training.checkpoint_frequency
    # Log less frequently to reduce overhead - adjust based on dataset size
    log_frequency = max(10, len(train_loader) // 3)  # Log approximately 10 times per epoch

    # Enable automatic mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Enable cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    # Reserve less memory on startup to avoid OOM
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.8)  # Use only 80% of GPU memory
    
    # Optimize memory allocation
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()

    # Keep track of the last best model's epoch
    last_best_epoch = None

    try:
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for samplers if distributed
            if is_distributed:
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
                        # Synchronize gradients across processes in distributed mode
                        if is_distributed and dist.is_initialized():
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
                        batch_size = train_loader.batch_size
                        if is_distributed and dist.is_initialized():
                            samples_processed = (batch_idx + 1) * batch_size * dist.get_world_size()
                        else:
                            samples_processed = (batch_idx + 1) * batch_size
                        throughput = samples_processed / elapsed
                        
                        logger.info(f'GPU Memory: {memory_stats}')
                        logger.info(f'Epoch: {epoch + 1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                                   f'Loss: {avg_loss:.4f}, Throughput: {throughput:.2f} samples/sec')

                    # Free up memory
                    if batch_idx % 10 == 0:
                        del text_input, current_view, previous_views, outputs, outputs_reshaped, labels_reshaped

                except Exception as e:
                    TRAINING_FAILED = True
                    logger.error(f"Critical error in training batch {batch_idx}: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise e

            # Synchronize loss across processes in distributed mode
            if is_distributed and dist.is_initialized():
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
                            TRAINING_FAILED = True
                            logger.error(f"Critical error in validation batch {batch_idx}: {str(e)}")
                            logger.error(traceback.format_exc())
                            raise e

                # Synchronize validation loss across processes in distributed mode
                if is_distributed and dist.is_initialized():
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
        TRAINING_FAILED = True
        logger.error(f"Fatal error in training loop: {str(e)}")
        logger.error(traceback.format_exc())
        raise e


def log_gpu_memory():
    """Log GPU memory usage for all available GPUs with consolidated statistics."""
    if not torch.cuda.is_available():
        return "No CUDA devices available"
        
    try:
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


def main():
    """Main training function that works with torchrun."""
    global TRAINING_FAILED
    
    # Parse arguments first
    parser = argparse.ArgumentParser(description='Train AnsweringAgent with DDP')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from', default=None)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with enhanced error reporting')
    parser.add_argument('--single-gpu', action='store_true', help='Force running on a single GPU even with torchrun')
    parser.add_argument('--reduce-memory', action='store_true', help='Enable aggressive memory optimization')
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
    
    # Apply memory optimizations if requested
    if args.reduce_memory:
        if torch.cuda.is_available():
            # Set memory fraction
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.7)  # More conservative
            
            # Empty cache
            torch.cuda.empty_cache()
            
            # Set max split size to reduce memory fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
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
    
    # Initialize logger - only rank 0 should log to console
    logger = setup_logger('training', log_dir=config.log_dir)
    
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
        tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)
        
        if rank == 0:
            logger.info(f"Training on {max(1, num_gpus)} GPUs, distributed mode: {is_distributed}")
            
            # Preprocess dataset on rank 0 only
            logger.info("Starting dataset preprocessing on rank 0...")
            start_time = time.time()
            AnsweringDataset.preprocess_and_save(config, tokenizer, logger)
            preprocess_time = time.time() - start_time
            logger.info(f"Dataset preprocessing complete. Took {preprocess_time:.2f} seconds.")
        
        # Wait for rank 0 to finish preprocessing
        if is_distributed:
            dist.barrier()
        
        if rank == 0:
            logger.info("Starting model initialization...")
            
        # Initialize model and move to correct GPU
        model = AnsweringAgent(config)
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
            verbose=config.training.scheduler_verbose if rank == 0 else False
        )

        # Load dataset
        try:
            dataset = AnsweringDataset(config=config)
        except Exception as e:
            logger.error(f"Critical error loading dataset: {str(e)}")
            logger.error(traceback.format_exc())
            TRAINING_FAILED = True
            raise e
        
        generator = torch.Generator().manual_seed(config.training.seed)
        train_size = int(config.data.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

        # Use DistributedSampler for distributed training
        if is_distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        
        if rank == 0:
            batch_str = f"Per-GPU batch size: {config.training.batch_size}"
            if is_distributed:
                batch_str += f" (global batch size: {config.training.batch_size * world_size})"
            logger.info(batch_str)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            shuffle=shuffle,  # Only shuffle if not using sampler
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=(config.training.num_workers > 0)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
            persistent_workers=(config.training.num_workers > 0)
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
        
        # Clean up resources
        cleanup()
        
        # Exit with error code
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    main()
