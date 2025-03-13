import os
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict
from transformers import BertTokenizerFast
from utils.logger import setup_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from config import Config
from models.answering_agent import AnsweringAgent
from data.dataset import AnsweringDataset



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
                num_epochs, device, checkpoint_dir, config, start_epoch=0, best_val_loss=float('inf'), rank=None):
    """Train the model with mixed precision training and gradient accumulation."""
    save_frequency = config.training.checkpoint_frequency

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

            if rank == 0:
                logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
                logger.info(f'GPU Memory at epoch start: {log_gpu_memory()}')

            for batch_idx, batch in enumerate(train_loader):
                try:
                    # Move data to device (non-blocking for async transfer)
                    text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                    current_view = batch['current_view_image'].to(device, non_blocking=True)
                    previous_views = [view.to(device, non_blocking=True) for view in batch['previous_views_image']]
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

                    # Log every 100 batches on rank 0
                    if batch_idx % 100 == 0 and rank == 0:
                        avg_loss = total_loss / (batch_idx + 1)
                        logger.info(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {avg_loss:.4f}')
                        logger.info(f'GPU Memory: {log_gpu_memory()}')

                except Exception as e:
                    logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                    continue

            # Synchronize loss across processes
            if dist.is_initialized():
                loss_tensor = torch.tensor(total_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                total_loss = loss_tensor.item() / dist.get_world_size()

            # Normalize training loss
            avg_epoch_loss = total_loss / len(train_loader)
            if rank == 0:
                logger.info(f'Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}')

            # Validation phase
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
                            previous_views = [view.to(device, non_blocking=True) for view in batch['previous_views_image']]
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
                        }, os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pt'))
                        logger.info(f'New best model saved at epoch {epoch+1} (val_loss: {val_loss:.4f})')

                # Save periodic checkpoint
                if (epoch + 1) % save_frequency == 0 and rank == 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': val_loss,
                    }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
                    logger.info(f'Checkpoint saved at epoch {epoch+1}')

                if rank == 0:
                    logger.info(f'After validation GPU Memory: {log_gpu_memory()}')

                # Clear cache periodically
                if torch.cuda.is_available() and epoch % 5 == 0:
                    torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in training loop: {str(e)}")
        raise e

    
def log_gpu_memory():
    """Log GPU memory usage for all available GPUs."""
    memory_stats = []
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
        memory_stats.append(f'GPU {i}: {memory_allocated:.1f}MB/{memory_reserved:.1f}MB')
    return ', '.join(memory_stats)



def main(rank, world_size, checkpoint_path=None, config=Config(), logger=setup_logger()):
    try:
        # Set environment variables for DDP
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize the process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device and ensure it's the correct one
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        logger.info(f"Process {rank}: Running on GPU {torch.cuda.current_device()} / {world_size}")
        
        # Set random seed for reproducibility
        torch.manual_seed(config.training.seed + rank)  # Different seed per process
        torch.cuda.manual_seed_all(config.training.seed + rank)
        
        # Initialize tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(config.model.bert_model_name)
        
        # Initialize model and move to correct GPU
        model = AnsweringAgent(config, device)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Convert batch norm
        model.to(device)
        
        # Resume training if checkpoint is provided
        start_epoch = 0
        best_val_loss = float('inf')
        
        # Wrap model with DDP after loading checkpoint
        model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True,
            broadcast_buffers=True
        )
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Process {rank}: Loading checkpoint from {checkpoint_path}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            logger.info(f"Process {rank}: Resuming training from epoch {start_epoch}")

        # Optimizer, loss, and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            verbose=config.training.scheduler_verbose
        )

        # Load dataset and ensure deterministic splitting
        dataset = AnsweringDataset(config=config, tokenizer=tokenizer)
        generator = torch.Generator().manual_seed(config.training.seed)  # Ensure same split on resume
        train_size = int(config.data.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)

        # Use DistributedSampler for DDP
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=train_sampler,
            num_workers=config.training.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            sampler=val_sampler,
            num_workers=config.training.num_workers,
            pin_memory=True
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
            rank=rank  # Pass rank to train_model
        )

        # Cleanup
        dist.destroy_process_group()
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}, Full Traceback:\n {traceback.format_exc()}")

if __name__ == '__main__':
    import argparse
    import torch.multiprocessing as mp

    config = Config()
    logger = setup_logger(config.log_dir)

    parser = argparse.ArgumentParser(description='Train AnsweringAgent with DDP')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from', default=None)
    parser.add_argument('--port', type=str, default='12355', help='Port number for DDP communication')
    args = parser.parse_args()

    # Set up distributed training
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA GPUs available for training")

    # Update port
    os.environ['MASTER_PORT'] = args.port

    try:
        mp.spawn(
            main,
            args=(world_size, args.checkpoint, config, logger),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise e