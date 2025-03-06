import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any
from transformers import BertTokenizerFast
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, checkpoint_dir):
    """Train the model with mixed precision training."""
    # Set up logger
    logger = logging.getLogger(__name__)
    
    best_val_loss = float('inf')
    best_model_path = None
    
    # Initialize gradient scaler for mixed precision training
    scaler = None  # No scaler needed for CPU training
        
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            text_input = {k: v.to(device) for k, v in batch['text_input'].items()}
            current_view = batch['current_view_image'].to(device)
            labels = batch['text_label'].to(device)
            
            # Ensure all model components are on the correct device
            model.to_device(device)
            
            # Forward pass with mixed precision
            if device.type in ['cuda', 'mps'] and scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(text_input, current_view)
                    outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
                    labels_reshaped = labels.reshape(-1)
                    loss = criterion(outputs_reshaped, labels_reshaped)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training for CPU
                outputs = model(text_input, current_view)
                outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
                labels_reshaped = labels.reshape(-1)
                loss = criterion(outputs_reshaped, labels_reshaped)
                loss.backward()
                optimizer.step()
            
            optimizer.zero_grad()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                # Move data to device
                text_input = {k: v.to(device) for k, v in batch['text_input'].items()}
                current_view = batch['current_view_image'].to(device)
                labels = batch['text_label'].to(device)
                
                # Ensure all model components are on the correct device
                model.to_device(device)
                
                outputs = model(text_input, current_view)
                outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
                labels_reshaped = labels.reshape(-1)
                loss = criterion(outputs_reshaped, labels_reshaped)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': val_loss,
            }, best_model_path, _use_new_zipfile_serialization=True)
            logger.info(f'New best model saved (val_loss: {val_loss:.4f})')
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path, _use_new_zipfile_serialization=True)
        logger.info(f'Saved checkpoint at epoch {epoch+1}')

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set device (force CPU)
    device = torch.device("cpu")
    logging.info(f'Using device: {device}')
    
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Create dataset
    dataset = AnsweringDataset(
        csv_path='data/train_data.csv',
        avdn_image_dir='../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images',
        tokenizer=tokenizer,
        max_previous_views=4
    )
    
    # Split dataset into train and validation
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = AnsweringAgent(
        bert_model_name='bert-base-uncased',
        hidden_size=768,
        dropout=0.5,
        feat_dropout=0.4,
        darknet_config_path='../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/yolo_v3.cfg',
        darknet_weights_path='../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/best.pt'
    )
    model.to_device(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean')
    
    # Create save directory
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=200000,  # Match AVDN's training iterations
        device=device,
        checkpoint_dir=save_dir
    )

if __name__ == '__main__':
    main() 