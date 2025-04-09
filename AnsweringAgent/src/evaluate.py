import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import argparse
import logging
from transformers import T5Tokenizer
from models.answering_agent import AnsweringAgent
from data.dataset import AnsweringDataset
from config import Config
from utils.logger import setup_logger
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import json
from tqdm import tqdm

# Download required NLTK data for evaluation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_bleu(references, hypothesis):
    """Calculate BLEU score."""
    if not hypothesis:
        return 0.0
    
    # Tokenize hypothesis and references
    hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
    references_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
    
    # Use smoothing function to handle edge cases
    smoothie = SmoothingFunction().method1
    
    # Calculate BLEU score
    try:
        return sentence_bleu(references_tokens, hypothesis_tokens, 
                           weights=(0.25, 0.25, 0.25, 0.25), 
                           smoothing_function=smoothie)
    except:
        return 0.0

def calculate_rouge(references, hypothesis):
    """Calculate ROUGE scores."""
    if not hypothesis or not references:
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    # Initialize rouge scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate scores against each reference and take the best
    scores = {metric: 0.0 for metric in ['rouge1', 'rouge2', 'rougeL']}
    
    for ref in references:
        if not ref:
            continue
        
        results = scorer.score(ref, hypothesis)
        
        # Update with better scores
        for metric in scores.keys():
            scores[metric] = max(scores[metric], results[metric].fmeasure)
    
    return scores

def calculate_f1(references, hypothesis):
    """Calculate word-level F1 score."""
    if not hypothesis or not references:
        return 0.0
    
    hypothesis_tokens = set(nltk.word_tokenize(hypothesis.lower()))
    
    # Calculate F1 against each reference and take the best
    best_f1 = 0.0
    
    for ref in references:
        if not ref:
            continue
            
        reference_tokens = set(nltk.word_tokenize(ref.lower()))
        
        # Calculate precision, recall, and F1
        common_tokens = hypothesis_tokens.intersection(reference_tokens)
        
        # Avoid division by zero
        if not hypothesis_tokens or not reference_tokens:
            continue
            
        precision = len(common_tokens) / len(hypothesis_tokens) if hypothesis_tokens else 0
        recall = len(common_tokens) / len(reference_tokens) if reference_tokens else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            best_f1 = max(best_f1, f1)
    
    return best_f1

def decode_predictions(tokenizer, predictions):
    """Convert token IDs to text."""
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in predictions]

def compute_text_metrics(predictions, targets, tokenizer):
    """Compute text generation metrics including BLEU, ROUGE, and F1."""
    # Decode predictions and targets
    decoded_preds = decode_predictions(tokenizer, predictions)
    decoded_targets = decode_predictions(tokenizer, targets)
    
    # Calculate metrics
    metrics = {
        'bleu': [], 
        'rouge1': [], 
        'rouge2': [], 
        'rougeL': [],
        'f1': []
    }
    
    for pred, target in zip(decoded_preds, decoded_targets):
        # Each target is treated as a single reference
        bleu = calculate_bleu([target], pred)
        rouge = calculate_rouge([target], pred)
        f1 = calculate_f1([target], pred)
        
        metrics['bleu'].append(bleu)
        metrics['rouge1'].append(rouge['rouge1'])
        metrics['rouge2'].append(rouge['rouge2'])
        metrics['rougeL'].append(rouge['rougeL'])
        metrics['f1'].append(f1)
    
    # Average metrics
    return {k: np.mean(v) for k, v in metrics.items()}, decoded_preds, decoded_targets

def compute_accuracy_metrics(outputs, labels, pad_token_id, attention_mask=None):
    """Compute token-level accuracy metrics."""
    # Reshape outputs and labels
    outputs_reshaped = outputs.reshape(-1, outputs.size(-1))
    labels_reshaped = labels.reshape(-1)

    # Get predictions
    _, predicted = outputs_reshaped.max(1)
    predicted = predicted.reshape(outputs.size(0), outputs.size(1))

    # Create mask for non-padding tokens
    if attention_mask is not None:
        # Use provided attention mask
        mask = attention_mask.bool()
    else:
        # Fall back to inferring mask from labels
        mask = (labels != pad_token_id)

    # Calculate metrics
    total_tokens = mask.sum().item()
    correct_tokens = ((predicted == labels) & mask).sum().item()
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    return {
        'token_accuracy': accuracy,
        'total_tokens': total_tokens,
        'correct_tokens': correct_tokens
    }

def generate_examples(model, data_loader, tokenizer, device, logger, dataset_name, num_examples=10):
    """Generate and save language examples from each dataset."""
    model.eval()
    all_examples = []
    
    # Get a subset of random examples from the dataset
    total_batches = len(data_loader)
    batch_indices = np.random.choice(total_batches, min(num_examples // data_loader.batch_size + 1, total_batches), replace=False)
    
    logger.info(f"Generating {num_examples} random examples from {dataset_name} dataset...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Skip batches not in our random selection
            if batch_idx not in batch_indices:
                continue
                
            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
            current_view = batch['current_view_image'].to(device, non_blocking=True)
            previous_views = batch['previous_views_image'].to(device, non_blocking=True)
            
            # Handle text_label as either tensor or dict
            if isinstance(batch['text_label'], dict):
                labels = batch['text_label']['input_ids']
            else:
                labels = batch['text_label']
            
            # Get input dialog history
            input_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in text_input['input_ids']]
            
            # Generate text
            outputs = model(
                text_input, 
                current_view, 
                previous_views,
                generate=True
            )
            
            # Decode predictions and references
            generated_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs["sequences"]]
            reference_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
            
            # Store examples
            for i in range(len(generated_text)):
                # Only collect up to num_examples
                if len(all_examples) >= num_examples:
                    break
                    
                example = {
                    "dataset": dataset_name,
                    "input_text": input_text[i],
                    "reference": reference_text[i],
                    "generated": generated_text[i],
                    # Calculate individual metrics for this example
                    "bleu": calculate_bleu([reference_text[i]], generated_text[i]),
                    "rouge": calculate_rouge([reference_text[i]], generated_text[i]),
                    "f1": calculate_f1([reference_text[i]], generated_text[i])
                }
                all_examples.append(example)
    
    return all_examples

def format_example_for_display(example, index):
    """Format a single example for readable display."""
    formatted = f"Example {index} ({example['dataset']})\n"
    formatted += f"Input: {example['input_text']}\n"
    formatted += f"Reference: {example['reference']}\n"
    formatted += f"Generated: {example['generated']}\n"
    formatted += f"Metrics: BLEU={example['bleu']:.4f}, "
    formatted += f"ROUGE-1={example['rouge']['rouge1']:.4f}, "
    formatted += f"ROUGE-L={example['rouge']['rougeL']:.4f}, "
    formatted += f"F1={example['f1']:.4f}\n"
    return formatted

def analyze_examples(examples, logger):
    """Analyze generated examples and identify patterns."""
    # Group examples by dataset
    by_dataset = {}
    for ex in examples:
        dataset = ex['dataset']
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append(ex)
    
    # Calculate average metrics per dataset
    dataset_metrics = {}
    for dataset, dataset_examples in by_dataset.items():
        dataset_metrics[dataset] = {
            'bleu': np.mean([ex['bleu'] for ex in dataset_examples]),
            'rouge1': np.mean([ex['rouge']['rouge1'] for ex in dataset_examples]),
            'rouge2': np.mean([ex['rouge']['rouge2'] for ex in dataset_examples]),
            'rougeL': np.mean([ex['rouge']['rougeL'] for ex in dataset_examples]),
            'f1': np.mean([ex['f1'] for ex in dataset_examples]),
        }
    
    # Log analysis
    logger.info("\n" + "="*50)
    logger.info("EXAMPLE GENERATION ANALYSIS")
    logger.info("="*50)
    
    for dataset, metrics in dataset_metrics.items():
        logger.info(f"\n{dataset} Average Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    # Return correlation analysis
    return dataset_metrics

def evaluate_generation(model, data_loader, tokenizer, device, logger, dataset_name):
    """Evaluate model's text generation capabilities."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    logger.info(f"Generating text on {dataset_name} dataset...")
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
            current_view = batch['current_view_image'].to(device, non_blocking=True)
            previous_views = batch['previous_views_image'].to(device, non_blocking=True)
            
            # Handle text_label as either tensor or dict
            if isinstance(batch['text_label'], dict):
                labels = batch['text_label']['input_ids']
            else:
                labels = batch['text_label']
            
            # Generate text
            outputs = model(
                text_input, 
                current_view, 
                previous_views,
                generate=True
            )
            
            # Collect predictions and targets
            all_predictions.extend(outputs["sequences"].cpu())
            all_targets.extend(labels.cpu())
    
    # Calculate metrics
    metrics, decoded_preds, decoded_targets = compute_text_metrics(all_predictions, all_targets, tokenizer)
    
    # Log metrics
    for name, value in metrics.items():
        logger.info(f"{dataset_name} {name.upper()}: {value:.4f}")
    
    # Save some examples
    examples = []
    for i in range(min(5, len(decoded_preds))):
        examples.append({
            "input": tokenizer.decode(all_targets[i], skip_special_tokens=True),
            "target": decoded_targets[i],
            "prediction": decoded_preds[i]
        })
    
    return metrics, examples

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

def get_weight_schedule(start_weight, end_weight, total_epochs):
    """
    Returns a function that linearly changes a weight from start_weight to end_weight.
    
    Args:
        start_weight: Initial weight value
        end_weight: Final weight value
        total_epochs: Total number of epochs for transition
        
    Returns:
        A function that calculates weight for a given epoch
    """
    def weight_fn(epoch):
        # Clamp epoch within range
        epoch = max(0, min(epoch, total_epochs))
        return start_weight + (end_weight - start_weight) * (epoch / total_epochs)
    
    return weight_fn

def evaluate_classification(model, data_loader, criterion, tokenizer, device, logger, dataset_name, curriculum_ratio=0.0):
    """Evaluate model using classification metrics and detailed loss components."""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_distribution_loss = 0
    total_destination_loss = 0
    total_visual_recon_loss = 0
    total_dest_recon_loss = 0
    
    total_metrics = {
        'token_accuracy': 0,
        'total_tokens': 0,
        'correct_tokens': 0
    }
    
    logger.info(f"Evaluating classification on {dataset_name} dataset...")
    
    # Get config values for loss weighting
    config = model.config
    # Use epoch 1 for validation
    epoch = 1  
    max_curriculum_epochs = config.training.curriculum_epochs
    
    # Get the weight schedules
    ce_loss_weight = config.training.ce_loss_weight_end
    
    distribution_similarity_weight = config.training.distribution_loss_weight_end
    
    destination_weight = config.training.destination_loss_weight_end
    
    reconstruction_weight = config.training.reconstruction_weight_end
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}"):
            text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
            current_view = batch['current_view_image'].to(device, non_blocking=True)
            previous_views = batch['previous_views_image'].to(device, non_blocking=True)
            labels_input_ids = batch['text_label']['input_ids'].to(device, non_blocking=True)
            labels_attention_mask = batch['text_label']['attention_mask'].to(device, non_blocking=True)
            
            # Get destination view if available
            destination_view = batch.get('destination_image')
            if destination_view is not None:
                destination_view = destination_view.to(device, non_blocking=True)
            
            # Forward pass for classification metrics
            outputs = model(
                text_input, 
                current_view, 
                previous_views, 
                labels=labels_input_ids,
                destination_view=destination_view,
                curriculum_ratio=curriculum_ratio
            )
            
            # Calculate cross-entropy loss
            logits = outputs["logits"]
            batch_size, seq_len, vocab_size = logits.size()
            logits_reshaped = logits.contiguous().view(batch_size * seq_len, vocab_size)
            labels_reshaped = labels_input_ids.contiguous().view(batch_size * seq_len)
            ce_loss = criterion(logits_reshaped, labels_reshaped)

            if labels_attention_mask is not None:
                labels_attention_mask = labels_attention_mask.reshape(-1)
                distribution_loss = calculate_distribution_similarity_loss(
                    logits_reshaped, 
                    labels_reshaped, 
                    labels_attention_mask, 
                    model, 
                    device
                )
            else:
                distribution_loss = torch.tensor(0.0, device=device)
            
            # Calculate destination cosine similarity loss if available
            adapted_features = outputs.get("adapted_features")
            dest_features = outputs.get("destination_features")
            if adapted_features is not None and dest_features is not None:
                destination_loss = calculate_cosine_similarity_loss(
                    adapted_features, 
                    dest_features
                )
            else:
                destination_loss = torch.tensor(0.0, device=device)
            
            # Calculate reconstruction losses if available
            reconstructed_visual = outputs.get("reconstructed_visual_features")
            visual_context_target = outputs.get("visual_context_target")
            reconstructed_dest = outputs.get("reconstructed_destination_features")
            
            if reconstructed_visual is not None and visual_context_target is not None:
                visual_recon_loss = calculate_reconstruction_loss(
                    reconstructed_visual, 
                    visual_context_target
                )
            else:
                visual_recon_loss = torch.tensor(0.0, device=device)
                
            if reconstructed_dest is not None and dest_features is not None:
                dest_recon_loss = calculate_reconstruction_loss(
                    reconstructed_dest, 
                    dest_features
                )
            else:
                dest_recon_loss = torch.tensor(0.0, device=device)
            
            # Combine all losses with appropriate weights
            weighted_ce_loss = ce_loss_weight * ce_loss
            weighted_distribution_loss = distribution_similarity_weight * distribution_loss
            weighted_destination_loss = destination_weight * destination_loss
            weighted_recon_loss = reconstruction_weight * (visual_recon_loss + dest_recon_loss)
            
            combined_loss = weighted_ce_loss + weighted_distribution_loss + weighted_destination_loss + weighted_recon_loss
            
            # Accumulate losses
            total_loss += combined_loss.item()
            total_ce_loss += ce_loss.item()
            total_distribution_loss += distribution_loss.item()
            total_destination_loss += destination_loss.item()
            total_visual_recon_loss += visual_recon_loss.item()
            total_dest_recon_loss += dest_recon_loss.item()
            
    
    # Calculate average losses
    num_batches = len(data_loader)
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_distribution_loss = total_distribution_loss / num_batches
    avg_destination_loss = total_destination_loss / num_batches
    avg_visual_recon_loss = total_visual_recon_loss / num_batches
    avg_dest_recon_loss = total_dest_recon_loss / num_batches
    
    # Log metrics
    logger.info(f"{dataset_name} Combined Loss: {avg_loss:.4f}")
    logger.info(f"{dataset_name} Cross-Entropy Loss: {avg_ce_loss:.4f}")
    logger.info(f"{dataset_name} Distribution Loss: {avg_distribution_loss:.4f}")
    logger.info(f"{dataset_name} Destination Loss: {avg_destination_loss:.4f}")
    logger.info(f"{dataset_name} Visual Recon Loss: {avg_visual_recon_loss:.4f}")
    logger.info(f"{dataset_name} Dest Recon Loss: {avg_dest_recon_loss:.4f}")    
    # Return detailed metrics
    loss_metrics = {
        'combined_loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'distribution_loss': avg_distribution_loss,
        'destination_loss': avg_destination_loss,
        'visual_recon_loss': avg_visual_recon_loss,
        'dest_recon_loss': avg_dest_recon_loss
    }
    
    return loss_metrics

def analyze_reconstruction_tradeoff(classification_metrics, reconstruction_losses, logger):
    """
    Analyze the tradeoff between reconstruction quality and task performance.
    
    Args:
        classification_metrics: Dictionary containing classification metrics
        reconstruction_losses: Dictionary containing reconstruction losses
        logger: Logger object for logging results
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Calculate ratio of visual reconstruction loss to classification loss
    if 'ce_loss' in classification_metrics and 'visual_recon_loss' in reconstruction_losses:
        results['visual_recon_to_task_ratio'] = reconstruction_losses['visual_recon_loss'] / classification_metrics['ce_loss']
    
    # Calculate ratio of destination reconstruction loss to destination prediction loss
    if 'destination_loss' in classification_metrics and 'dest_recon_loss' in reconstruction_losses:
        results['dest_recon_to_dest_task_ratio'] = reconstruction_losses['dest_recon_loss'] / classification_metrics['destination_loss']
    
    # Log findings
    logger.info(f"Reconstruction-Task Tradeoff Analysis: {results}")
    
    # Interpret ratios
    if 'visual_recon_to_task_ratio' in results:
        ratio = results['visual_recon_to_task_ratio']
        if ratio > 2.0:
            logger.info("Visual reconstruction is significantly more difficult than classification")
        elif ratio < 0.5:
            logger.info("Classification is significantly more difficult than visual reconstruction")
        else:
            logger.info("Visual reconstruction and classification tasks are fairly balanced")
    
    if 'dest_recon_to_dest_task_ratio' in results:
        ratio = results['dest_recon_to_dest_task_ratio']
        if ratio > 2.0:
            logger.info("Destination reconstruction is significantly more difficult than destination prediction")
        elif ratio < 0.5:
            logger.info("Destination prediction is significantly more difficult than destination reconstruction")
        else:
            logger.info("Destination reconstruction and prediction tasks are fairly balanced")
    
    return results

def evaluate_all_datasets(model, tokenizer, config, device, logger, args):
    """Evaluate on all three datasets: train, val_seen, val_unseen with curriculum learning phases."""
    # Create dataset instances
    logger.info("Loading datasets...")
    
    # Set up smaller subsets for quicker evaluation if needed
    subset_size = args.subset_size if hasattr(args, 'subset_size') else None
    
    # Create train dataset
    train_dataset = AnsweringDataset(config=config, split='train')
    if subset_size and subset_size < len(train_dataset):
        # Use a fixed subset for consistent evaluation
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = Subset(train_dataset, indices)
    
    # Create validation seen dataset
    val_seen_dataset = AnsweringDataset(config=config, split='val_seen')
    
    # Create validation unseen dataset
    val_unseen_dataset = AnsweringDataset(config=config, split='val_unseen')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_seen_loader = DataLoader(
        val_seen_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_unseen_loader = DataLoader(
        val_unseen_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    # Set up criterion for classification evaluation
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Results dictionary - we'll only use post_curriculum since we're validating in a single epoch
    results = {
        'post_curriculum': {'classification': {}, 'generation': {}, 'examples': {}}
    }
    
    # List of datasets to evaluate
    datasets = [
        ('train', train_loader),
        ('val_seen', val_seen_loader),
        ('val_unseen', val_unseen_loader)
    ]

    # Evaluate post-curriculum (no destination features)
    curriculum_ratio = 0.0  # No curriculum (later in training)
    logger.info(f"\n{'='*50}\nEvaluating with curriculum ratio: {curriculum_ratio}\n{'='*50}")
    
    for name, loader in datasets:
        logger.info(f"\n{'='*50}\nEvaluating {name} dataset\n{'='*50}")
        
        # Classification metrics post-curriculum
        loss_metrics, class_metrics = evaluate_classification(
            model, loader, criterion, tokenizer, device, logger, name, 
            curriculum_ratio=curriculum_ratio
        )
        
        results['post_curriculum']['classification'][name] = {
            **loss_metrics,
            **class_metrics
        }
        
        # Generation metrics (BLEU, ROUGE, F1) - only in post-curriculum phase
        gen_metrics, examples = evaluate_generation(
            model, loader, tokenizer, device, logger, name
        )
        results['post_curriculum']['generation'][name] = gen_metrics
        results['post_curriculum']['examples'][name] = examples
    
    # Calculate overfitting metrics for post-curriculum phase
    classification_gap = {}
    generation_gap = {}
    
    # Calculate gaps between train and validation metrics
    for name in ['val_seen', 'val_unseen']:
        # Classification gaps
        classification_gap[name] = {
            'combined_loss_gap': results['post_curriculum']['classification'][name]['combined_loss'] - 
                                results['post_curriculum']['classification']['train']['combined_loss'],
            'ce_loss_gap': results['post_curriculum']['classification'][name]['ce_loss'] - 
                          results['post_curriculum']['classification']['train']['ce_loss'],
            'distribution_loss_gap': results['post_curriculum']['classification'][name]['distribution_loss'] - 
                                    results['post_curriculum']['classification']['train']['distribution_loss'],
            'destination_loss_gap': results['post_curriculum']['classification'][name]['destination_loss'] - 
                                   results['post_curriculum']['classification']['train']['destination_loss'],
            'visual_recon_loss_gap': results['post_curriculum']['classification'][name]['visual_recon_loss'] - 
                                    results['post_curriculum']['classification']['train']['visual_recon_loss'],
            'dest_recon_loss_gap': results['post_curriculum']['classification'][name]['dest_recon_loss'] - 
                                  results['post_curriculum']['classification']['train']['dest_recon_loss'],
            'accuracy_gap': results['post_curriculum']['classification']['train']['token_accuracy'] - 
                           results['post_curriculum']['classification'][name]['token_accuracy']
        }
        
        # Generation gaps
        generation_gap[name] = {
            metric: results['post_curriculum']['generation']['train'][metric] - 
                   results['post_curriculum']['generation'][name][metric]
            for metric in results['post_curriculum']['generation']['train'].keys()
        }
    
    # Add gaps to results
    results['post_curriculum']['overfitting'] = {
        'classification': classification_gap,
        'generation': generation_gap
    }
    
    # Log overfitting metrics
    logger.info("\n" + "="*50)
    logger.info("OVERFITTING ANALYSIS")
    logger.info("="*50)
    
    for name in ['val_seen', 'val_unseen']:
        logger.info(f"\nGap between train and {name}:")
        for loss_type, gap in classification_gap[name].items():
            logger.info(f"{loss_type}: {gap:.4f}")
        
        for metric, gap in generation_gap[name].items():
            logger.info(f"{metric.upper()} gap: {gap:.4f}")
    
    return results

def explain_metrics():
    """Return a dictionary explaining what each metric means and how to interpret it."""
    explanations = {
        'classification': {
            'accuracy': 'Token-level classification accuracy - percentage of tokens correctly predicted.',
            'cross_entropy_loss': 'Standard cross-entropy loss from token classification.',
            'distribution_loss': 'KL divergence between predicted and target token distributions - lower is better.',
            'destination_loss': 'Cosine similarity loss between predicted and target destination features - lower is better.',
            'visual_recon_loss': 'Mean squared error loss for reconstructing visual features - lower is better.',
            'dest_recon_loss': 'Mean squared error loss for reconstructing destination features - lower is better.',
            'total_loss': 'Combined loss with weights applied to each component.'
        },
        'generation': {
            'bleu': 'BLEU score (0-1) measuring n-gram precision between generated and reference texts. Higher is better.',
            'rouge': 'ROUGE score (0-1) measuring recall of n-grams between generated and reference texts. Higher is better.',
            'meteor': 'METEOR score (0-1) measuring semantic similarity, including synonyms. Higher is better.',
            'f1': 'F1 score (0-1) balancing precision and recall of tokens. Higher is better.',
            'exact_match': 'Percentage of exact matches between generated and reference texts.',
            'semantic_sim': 'Semantic similarity between generated and reference texts using embeddings. Higher is better.'
        },
        'overfitting': {
            'classification': 'Difference between train and validation metrics for classification. Lower gap is better.',
            'generation': 'Difference between train and validation metrics for generation. Lower gap is better.'
        },
        'curriculum_analysis': {
            'dest_recon_ratio': 'Ratio of destination reconstruction loss after curriculum vs. with curriculum. Higher values (>1.0) indicate the model is properly using the destination when available.',
            'visual_recon_ratio': 'Ratio of visual reconstruction loss after curriculum vs. with curriculum. Values close to 1.0 suggest consistent visual processing.'
        },
        'visual_context_analysis': {
            'lexical_diversity': 'Ratio of unique tokens to total tokens - higher values indicate more diverse language.',
            'landmark_references': 'Frequency of visual landmark terms normalized by text length.',
            'diversity_ratio': 'Ratio of lexical diversity gaps between unseen and seen environments. Higher values suggest generic responses in unseen environments.',
            'landmark_ratio': 'Ratio of landmark reference gaps between unseen and seen environments. Higher values suggest the model mentions fewer landmarks in unseen environments.'
        }
    }
    return explanations

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enhanced evaluation for AnsweringAgent')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--subset-size', type=int, default=500, help='Size of training subset to evaluate on')
    parser.add_argument('--num-examples', type=int, default=10, help='Number of examples to generate per dataset')
    args = parser.parse_args()
    
    # Setup
    config = Config()
    logger = setup_logger('enhanced_evaluation', log_dir=config.log_dir)
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    # Initialize tokenizer
    global tokenizer
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name)
    
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = AnsweringAgent(config, tokenizer, logger)
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded. Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    logger.info("Running validation for a single epoch (epoch=1)")
    
    # Evaluate on all datasets
    results = evaluate_all_datasets(model, tokenizer, config, device, logger, args)
    
    # Generate examples from all datasets
    logger.info("\n" + "="*50)
    logger.info("ANALYZING GENERATED LANGUAGE EXAMPLES")
    logger.info("="*50)
    
    # Create dataloaders for example generation
    train_dataset = AnsweringDataset(config=config, split='train')
    val_seen_dataset = AnsweringDataset(config=config, split='val_seen')
    val_unseen_dataset = AnsweringDataset(config=config, split='val_unseen')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  # Set to True to get different random examples
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_seen_loader = DataLoader(
        val_seen_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    val_unseen_loader = DataLoader(
        val_unseen_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    
    all_examples = []
    for name, loader in [('train', train_loader), ('val_seen', val_seen_loader), ('val_unseen', val_unseen_loader)]:
        examples = generate_examples(model, loader, tokenizer, device, logger, name, num_examples=args.num_examples)
        all_examples.extend(examples)
        
        # Log examples
        logger.info(f"\nGenerated {len(examples)} examples from {name} dataset:")
        for i, example in enumerate(examples):
            logger.info("\n" + format_example_for_display(example, i+1))
    
    # Analyze examples
    example_metrics = analyze_examples(all_examples, logger)
    results['example_analysis'] = example_metrics
    
    # Add examples to results
    results['generated_examples'] = all_examples
    
    # Perform specific visual context analysis
    visual_context_analysis = {}
    
    # Check for hallucination patterns in val_unseen vs. val_seen
    seen_refs = " ".join([ex['reference'] for ex in all_examples if ex['dataset'] == 'val_seen'])
    seen_gens = " ".join([ex['generated'] for ex in all_examples if ex['dataset'] == 'val_seen'])
    unseen_refs = " ".join([ex['reference'] for ex in all_examples if ex['dataset'] == 'val_unseen'])
    unseen_gens = " ".join([ex['generated'] for ex in all_examples if ex['dataset'] == 'val_unseen'])
    
    # Calculate lexical diversity as a proxy for hallucination/generic responses
    def lexical_diversity(text):
        if not text:
            return 0
        tokens = nltk.word_tokenize(text.lower())
        return len(set(tokens)) / len(tokens) if tokens else 0
    
    visual_context_analysis['lexical_diversity'] = {
        'val_seen_reference': lexical_diversity(seen_refs),
        'val_seen_generated': lexical_diversity(seen_gens),
        'val_unseen_reference': lexical_diversity(unseen_refs),
        'val_unseen_generated': lexical_diversity(unseen_gens),
    }
    
    # Check for reference to visual landmarks in generated text
    # This is a simplified heuristic - in real implementation, you'd have a more sophisticated
    # landmark detection approach
    landmark_terms = ['building', 'tree', 'road', 'path', 'house', 'wall', 'roof', 'window', 
                     'door', 'field', 'mountain', 'hill', 'river', 'lake', 'bridge', 'street', 'car']
    
    def count_landmark_references(text):
        if not text:
            return 0
        text_lower = text.lower()
        return sum(1 for term in landmark_terms if term in text_lower)
    
    # Count landmark references in generated text vs references
    visual_context_analysis['landmark_references'] = {
        'val_seen_reference': count_landmark_references(seen_refs) / (len(seen_refs.split()) + 1),
        'val_seen_generated': count_landmark_references(seen_gens) / (len(seen_gens.split()) + 1),
        'val_unseen_reference': count_landmark_references(unseen_refs) / (len(unseen_refs.split()) + 1),
        'val_unseen_generated': count_landmark_references(unseen_gens) / (len(unseen_gens.split()) + 1),
    }
    
    # Log visual context analysis
    logger.info("\n" + "="*50)
    logger.info("VISUAL CONTEXT ANALYSIS")
    logger.info("="*50)
    
    # Lexical diversity analysis
    ld = visual_context_analysis['lexical_diversity']
    logger.info("\nLexical Diversity (higher is better):")
    logger.info(f"Val_Seen References: {ld['val_seen_reference']:.4f}")
    logger.info(f"Val_Seen Generated: {ld['val_seen_generated']:.4f}")
    logger.info(f"Val_Unseen References: {ld['val_unseen_reference']:.4f}")
    logger.info(f"Val_Unseen Generated: {ld['val_unseen_generated']:.4f}")
    
    # Calculate diversity gaps
    seen_diversity_gap = ld['val_seen_reference'] - ld['val_seen_generated']
    unseen_diversity_gap = ld['val_unseen_reference'] - ld['val_unseen_generated']
    logger.info(f"Seen Diversity Gap: {seen_diversity_gap:.4f}")
    logger.info(f"Unseen Diversity Gap: {unseen_diversity_gap:.4f}")
    
    # Check for generic responses in unseen environments
    diversity_ratio = unseen_diversity_gap / (seen_diversity_gap + 1e-10)
    if diversity_ratio > 1.5:
        logger.info("FINDING: More generic responses on unseen environments suggests visual overfitting.")
    
    # Landmark reference analysis
    lr = visual_context_analysis['landmark_references']
    logger.info("\nLandmark References (normalized by text length):")
    logger.info(f"Val_Seen References: {lr['val_seen_reference']:.4f}")
    logger.info(f"Val_Seen Generated: {lr['val_seen_generated']:.4f}")
    logger.info(f"Val_Unseen References: {lr['val_unseen_reference']:.4f}")
    logger.info(f"Val_Unseen Generated: {lr['val_unseen_generated']:.4f}")
    
    # Calculate landmark reference gaps
    seen_landmark_gap = lr['val_seen_reference'] - lr['val_seen_generated']
    unseen_landmark_gap = lr['val_unseen_reference'] - lr['val_unseen_generated']
    logger.info(f"Seen Landmark Gap: {seen_landmark_gap:.4f}")
    logger.info(f"Unseen Landmark Gap: {unseen_landmark_gap:.4f}")
    
    # Check for fewer landmark references in unseen environments
    landmark_ratio = unseen_landmark_gap / (seen_landmark_gap + 1e-10)
    if landmark_ratio > 1.5:
        logger.info("FINDING: Fewer landmark references in unseen environments suggests visual context overfitting.")
    
    # Add to results
    results['visual_context_analysis'] = visual_context_analysis
    results['visual_context_overfitting'] = {
        'diversity_ratio': diversity_ratio,
        'landmark_ratio': landmark_ratio
    }
    
    # Add metric explanations
    results['metric_explanations'] = explain_metrics()
    
    # Save results
    results_path = os.path.join(config.log_dir, f"enhanced_evaluation_epoch_{checkpoint['epoch']}.json")
    
    # Convert torch tensors to Python types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        else:
            return obj
    
    results_json = convert_for_json(results)
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Enhanced evaluation completed. Results saved to {results_path}")
    
    # Print summary of key findings
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    # Calculate the average gap for all metrics
    avg_val_seen_gap = np.mean([gap for gap in results['post_curriculum']['overfitting']['generation']['val_seen'].values()])
    avg_val_unseen_gap = np.mean([gap for gap in results['post_curriculum']['overfitting']['generation']['val_unseen'].values()])
    
    logger.info(f"Average Generation Metric Gap (Train-Val_Seen): {avg_val_seen_gap:.4f}")
    logger.info(f"Average Generation Metric Gap (Train-Val_Unseen): {avg_val_unseen_gap:.4f}")
    
    # Key findings
    logger.info("\nKEY FINDINGS:")
    
    # Visual context overfitting assessment
    visual_context_overfitting = avg_val_unseen_gap > (1.5 * avg_val_seen_gap)
    
    if visual_context_overfitting:
        logger.info("1. SIGNIFICANT VISUAL CONTEXT OVERFITTING DETECTED")
        logger.info("   - The model performs much worse on unseen environments than on seen ones")
        logger.info("   - Recommendation: Increase visual feature regularization and augmentation")
    else:
        logger.info("1. No significant visual context overfitting detected")
        logger.info("   - The gap between seen and unseen environments is proportional")
    
    # Lexical diversity assessment
    if diversity_ratio > 1.5:
        logger.info("2. GENERIC RESPONSES ON UNSEEN ENVIRONMENTS")
        logger.info("   - Less lexical diversity on unseen environments suggests falling back to generic responses")
        logger.info("   - Recommendation: Enhance visual feature processing pathway")
    
    # Landmark reference assessment
    if landmark_ratio > 1.5:
        logger.info("3. REDUCED LANDMARK REFERENCES ON UNSEEN ENVIRONMENTS")
        logger.info("   - Model mentions fewer visual landmarks in unseen environments")
        logger.info("   - Recommendation: Improve cross-modal integration between visual and text features")

    # Analyze reconstruction-task tradeoff for last evaluated dataset (could be any of train, val_seen, val_unseen)
    logger.info("4. RECONSTRUCTION-TASK TRADEOFF ANALYSIS")
    classification_metrics = results['post_curriculum']['classification'][list(datasets)[-1][0]]
    reconstruction_losses = results['post_curriculum']['classification'][list(datasets)[-1][0]]
    analyze_reconstruction_tradeoff(classification_metrics, reconstruction_losses, logger)

if __name__ == "__main__":
    main() 