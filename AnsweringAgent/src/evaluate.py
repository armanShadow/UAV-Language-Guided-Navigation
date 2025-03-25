import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import argparse
import logging
from transformers import BertTokenizer
from models.answering_agent import AnsweringAgent
from data.dataset import AnsweringDataset
from config import Config
from utils.logger import setup_logger

def compute_metrics(outputs, labels, pad_token_id):
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

def evaluate_model(model, test_loader, criterion, device, logger):
    """Evaluate the model on the test set."""
    model.eval()
    total_loss = 0
    total_metrics = {
        'accuracy': 0,
        'total_tokens': 0,
        'correct_tokens': 0
    }
    
    logger.info("Starting evaluation on test set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
            current_view = batch['current_view_image'].to(device, non_blocking=True)
            previous_views = batch['previous_views_image'].to(device, non_blocking=True)
            labels = batch['text_label'].to(device, non_blocking=True)

            # Forward pass
            outputs = model(text_input, current_view, previous_views)
            
            # Calculate loss
            batch_size, seq_len, vocab_size = outputs.size()
            outputs_reshaped = outputs.contiguous().view(batch_size * seq_len, vocab_size)
            labels_reshaped = labels.contiguous().view(batch_size * seq_len)
            loss = criterion(outputs_reshaped, labels_reshaped)
            total_loss += loss.item()
            
            # Calculate metrics
            batch_metrics = compute_metrics(outputs, labels, pad_token_id=tokenizer.pad_token_id)
            for k, v in batch_metrics.items():
                total_metrics[k] += v
                
            if batch_idx % 10 == 0:
                logger.info(f"Evaluated {batch_idx}/{len(test_loader)} batches")
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    avg_accuracy = total_metrics['correct_tokens'] / total_metrics['total_tokens'] if total_metrics['total_tokens'] > 0 else 0
    
    logger.info(f"Test Loss: {avg_loss:.4f}")
    logger.info(f"Test Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Correct/Total Tokens: {total_metrics['correct_tokens']}/{total_metrics['total_tokens']}")
    
    return avg_loss, avg_accuracy, total_metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate AnsweringAgent on test set')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    args = parser.parse_args()
    
    # Setup
    config = Config()
    logger = setup_logger('evaluation', log_dir=config.log_dir)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Initialize tokenizer
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)
    
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = AnsweringAgent(config, tokenizer)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded. Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = AnsweringDataset(config=config)
    
    # Load test indices
    test_indices_path = os.path.join(config.log_dir, "test_indices.pt")
    if os.path.exists(test_indices_path):
        logger.info(f"Loading test indices from {test_indices_path}")
        test_indices = torch.load(test_indices_path)
        test_dataset = Subset(dataset, test_indices)
    else:
        # Fallback: recreate the splits if test indices aren't available
        logger.warning("Test indices not found. Recreating dataset splits.")
        generator = torch.Generator().manual_seed(config.training.seed)
        
        train_size = int(config.data.train_val_split * len(dataset))
        remaining_size = len(dataset) - train_size
        val_size = int(config.data.val_test_split * remaining_size)
        test_size = remaining_size - val_size
        
        _, val_test_dataset = torch.utils.data.random_split(
            dataset, [train_size, remaining_size], generator=generator
        )
        
        _, test_dataset = torch.utils.data.random_split(
            val_test_dataset, [val_size, test_size], generator=generator
        )
    
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Evaluate
    loss, accuracy, metrics = evaluate_model(model, test_loader, criterion, device, logger)
    
    logger.info("Evaluation completed")
    logger.info(f"Final Test Loss: {loss:.4f}")
    logger.info(f"Final Test Accuracy: {accuracy:.4f}")
    
    # Save results
    results = {
        'loss': loss,
        'accuracy': accuracy,
        'metrics': metrics,
        'checkpoint': args.checkpoint,
        'epoch': checkpoint['epoch']
    }
    
    results_path = os.path.join(config.log_dir, f"test_results_epoch_{checkpoint['epoch']}.pt")
    torch.save(results, results_path)
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main() 