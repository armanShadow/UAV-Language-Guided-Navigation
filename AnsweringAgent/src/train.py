import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any
from transformers import BertTokenizerFast
from utils.logger import setup_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DataParallel
from datetime import datetime

from config import Config
config = Config()
logger = setup_logger(config.log_dir)

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
                num_epochs, device, checkpoint_dir, config, start_epoch=0, best_val_loss=float('inf')):
    """Train the model with mixed precision training and gradient accumulation."""
    save_frequency = config.training.checkpoint_frequency
    
    # Enable automatic mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Enable cuDNN benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        logger.info(f'GPU Memory at epoch start: {log_gpu_memory()}')
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device (non-blocking for async transfer)
            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
            current_view = batch['current_view_image'].to(device, non_blocking=True)
            previous_views = [view.to(device, non_blocking=True) for view in batch['previous_views_image']]
            labels = batch['text_label'].to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(text_input, current_view, previous_views)
                outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
                labels_reshaped = labels.reshape(-1)
                loss = criterion(outputs_reshaped, labels_reshaped)
                # Scale loss by gradient accumulation steps
                loss = loss / config.training.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients
            if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                # Optimizer step with scaling
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * config.training.gradient_accumulation_steps
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {avg_loss:.4f}')
                logger.info(f'GPU Memory: {log_gpu_memory()}')
                
                # Clear GPU cache if memory usage is high
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Log epoch summary
        avg_epoch_loss = total_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}')
        
        # Validation phase
        if (epoch + 1) % config.training.eval_freq == 0:
            model.eval()
            val_loss = 0
            logger.info("Starting validation...")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
                    current_view = batch['current_view_image'].to(device, non_blocking=True)
                    previous_views = [view.to(device, non_blocking=True) for view in batch['previous_views_image']]
                    labels = batch['text_label'].to(device, non_blocking=True)
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(text_input, current_view, previous_views)
                        outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
                        labels_reshaped = labels.reshape(-1)
                        loss = criterion(outputs_reshaped, labels_reshaped)
                    val_loss += loss.item()
                    
                    # Clear GPU cache periodically during validation
                    if batch_idx % 100 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            val_loss /= len(val_loader)
            logger.info(f'Validation Loss: {val_loss:.4f}')
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save model in a way that handles both single/multi-GPU
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                }, os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pt'))
                logger.info(f'New best model saved at epoch {epoch+1} (val_loss: {val_loss:.4f})')
        
        # Save periodic checkpoint based on save_frequency
        if (epoch + 1) % save_frequency == 0:
            # Save model in a way that handles both single/multi-GPU
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt'))
            logger.info(f'Checkpoint saved at epoch {epoch+1}')
        
        # Log GPU memory after validation
        logger.info(f'After validation GPU Memory: {log_gpu_memory()}')
        
        # Clear GPU cache at the end of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def log_gpu_memory():
    """Log GPU memory usage for all available GPUs."""
    memory_stats = []
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**2
        memory_stats.append(f'GPU {i}: {memory_allocated:.1f}MB/{memory_reserved:.1f}MB')
    return ', '.join(memory_stats)

def main(checkpoint_path=None):
    # Initialize config first to set up logging
    logger.info("Starting training script")
    
    # Initialize device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{config.training.primary_gpu}')
        logger.info(f"Using single GPU: {torch.cuda.get_device_name(config.training.primary_gpu)}")
        
        # Enable cuDNN benchmarking and deterministic behavior
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        
        # Log GPU information
        gpu_name = torch.cuda.get_device_name(config.training.primary_gpu)
        memory_total = torch.cuda.get_device_properties(config.training.primary_gpu).total_memory / 1024**2
        logger.info(f"GPU {config.training.primary_gpu}: {gpu_name} - Total Memory: {memory_total:.1f}MB")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU")

    # Set seeds for reproducibility
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config.model.bert_model_name)
    logger.info(f"Tokenizer initialized")
    
    # Initialize model and move to device
    model = AnsweringAgent(config)
    model = model.to(device)
    logger.info("Model initialized and moved to device")
    
    # Force synchronization to ensure GPU is initialized
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        logger.info(f"GPU synchronization complete")

    # Initialize optimizer with gradient clipping
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Initialize criterion on device
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device)
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        verbose=config.training.scheduler_verbose
    )
    
    # Log initial memory state
    logger.info("Initial GPU memory after model initialization:")
    logger.info(log_gpu_memory())
    
    # Create dataset with scaled batch size and workers
    dataset = AnsweringDataset(
        config=config,
        tokenizer=tokenizer
    )
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    train_indices = None
    val_indices = None
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Handle loading state dict for both single/multi-GPU scenarios
            if config.training.num_gpus > 1 and not isinstance(model, nn.DataParallel):
                # If loading multi-GPU checkpoint into single GPU
                state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
            elif config.training.num_gpus <= 1 and 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
                # If loading single GPU checkpoint into multi-GPU
                state_dict = {f'module.{k}': v for k, v in checkpoint['model_state_dict'].items()}
            else:
                state_dict = checkpoint['model_state_dict']
                
            model.load_state_dict(state_dict)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            # Load data split indices
            train_indices = checkpoint.get('train_indices')
            val_indices = checkpoint.get('val_indices')
            
            # Load optimizer and scheduler states if they exist
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer states to GPU if needed
                if torch.cuda.is_available():
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(device)
                                
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
            logger.info(f"Resuming training from epoch {start_epoch} (checkpoint: {checkpoint_path})")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise e
    
    # Create train/val split
    if train_indices is None or val_indices is None:
        # Create new split
        train_size = int(config.data.train_val_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        logger.info(f"Created new train/val split. Train size: {train_size}, Val size: {val_size}")
    else:
        # Use loaded split
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        logger.info(f"Using loaded train/val split. Train size: {len(train_indices)}, Val size: {len(val_indices)}")
    
    # Create data loaders with GPU settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,  # This helps speed up data transfer to GPU
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True  # Ensure consistent batch sizes for DataParallel
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    
    logger.info("Data loaders created with GPU optimization settings")
    
    # Train the model
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
        best_val_loss=best_val_loss
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train the AnsweringAgent model')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from', default=None)
    
    args = parser.parse_args()
    main(args.checkpoint) 