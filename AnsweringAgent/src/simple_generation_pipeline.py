#!/usr/bin/env python3
"""
Simple Generation Pipeline for UAV Navigation
Picks random samples from each dataset and generates answers.
"""

import os
import torch
import argparse
import json
import random
from typing import Dict, List
from transformers import T5Tokenizer
from tqdm import tqdm
import pickle

from config import Config
from models.answering_agent import AnsweringAgent
from data.dataset import AnsweringDataset
from utils.logger import setup_logger

class SimpleGenerationPipeline:
    """Simple generator for testing model performance on random samples."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = device
        self.config = Config()
        
        # Initialize logger
        self.logger = setup_logger('simple_generation', log_dir=self.config.log_dir)
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.model.t5_model_name, 
            model_max_length=self.config.data.max_seq_length
        )
        
        # Load model
        self.logger.info("🏗️ Loading model for generation...")
        self.model = AnsweringAgent(self.config, self.tokenizer, self.logger)
        
        # Load checkpoint
        self.logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info("✅ Model state loaded successfully")
        else:
            self.logger.warning("⚠️ No model_state_dict found in checkpoint, trying direct load")
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
    def generate_for_sample(self, sample: dict) -> Dict:
        """
        Generate answer for a single sample.
        
        Args:
            sample: Dataset sample with text_input, current_view, previous_views, etc.
            
        Returns:
            Dictionary with generated text and metadata
        """
        with torch.no_grad():
            # Move data to device
            text_input = {k: v.to(self.device) if torch.is_tensor(v) else v 
                         for k, v in sample['text_input'].items()}
            current_view = sample['current_view'].to(self.device)
            previous_views = sample['previous_views'].to(self.device)
            
            # Generate answer
            outputs = self.model(
                text_input=text_input,
                current_view=current_view,
                previous_views=previous_views,
                generate=True
            )
            
            # Decode generated text
            generated_ids = outputs['sequences']
            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Get original question and answer for comparison
            original_question = sample.get('question', 'N/A')
            original_answer = sample.get('answer', 'N/A')
            
            return {
                'sample_id': sample.get('id', 'unknown'),
                'original_question': original_question,
                'original_answer': original_answer,
                'generated_answer': generated_text[0] if generated_text else 'No generation',
                'generated_ids': generated_ids[0].cpu().tolist() if len(generated_ids) > 0 else []
            }
    
    def get_random_samples(self, dataset_split: str, num_samples: int = 5) -> List[dict]:
        """
        Get random samples from a dataset split.
        
        Args:
            dataset_split: Dataset split ('train', 'val_seen', 'val_unseen')
            num_samples: Number of random samples to get
            
        Returns:
            List of random samples
        """
        self.logger.info(f"📊 Loading {dataset_split} dataset...")
        
        try:
            # Create dataset
            dataset = AnsweringDataset(self.config, split=dataset_split, tokenizer=self.tokenizer)
            self.logger.info(f"✅ Loaded {len(dataset)} samples from {dataset_split}")
            
            # Get random indices
            total_samples = len(dataset)
            if num_samples > total_samples:
                self.logger.warning(f"⚠️ Requested {num_samples} samples but only {total_samples} available")
                num_samples = total_samples
            
            random_indices = random.sample(range(total_samples), num_samples)
            
            # Load samples
            samples = []
            for idx in tqdm(random_indices, desc=f"Loading {dataset_split} samples"):
                try:
                    sample = dataset[idx]
                    # Add some metadata for easier identification
                    sample['id'] = f"{dataset_split}_{idx}"
                    sample['question'] = self.tokenizer.decode(
                        sample['text_input']['input_ids'][0], 
                        skip_special_tokens=True
                    )
                    sample['answer'] = self.tokenizer.decode(
                        sample['labels'][0], 
                        skip_special_tokens=True
                    ) if 'labels' in sample else 'N/A'
                    samples.append(sample)
                except Exception as e:
                    self.logger.error(f"❌ Error loading sample {idx}: {e}")
                    continue
            
            return samples
            
        except Exception as e:
            self.logger.error(f"❌ Error loading {dataset_split} dataset: {e}")
            return []
    
    def run_generation_pipeline(self, num_samples_per_split: int = 3, 
                               splits: List[str] = None) -> Dict:
        """
        Run generation pipeline on random samples from each dataset split.
        
        Args:
            num_samples_per_split: Number of random samples per split
            splits: List of dataset splits to process
            
        Returns:
            Dictionary with results for each split
        """
        if splits is None:
            splits = ['train', 'val_seen', 'val_unseen']
        
        results = {}
        
        for split in splits:
            self.logger.info(f"🚀 Processing {split} split...")
            
            # Get random samples
            samples = self.get_random_samples(split, num_samples_per_split)
            
            if not samples:
                self.logger.warning(f"⚠️ No samples found for {split}")
                results[split] = []
                continue
            
            # Generate answers for each sample
            split_results = []
            for sample in tqdm(samples, desc=f"Generating for {split}"):
                try:
                    result = self.generate_for_sample(sample)
                    split_results.append(result)
                except Exception as e:
                    self.logger.error(f"❌ Error generating for sample {sample.get('id', 'unknown')}: {e}")
                    continue
            
            results[split] = split_results
            self.logger.info(f"✅ Generated {len(split_results)} answers for {split}")
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save generation results to file.
        
        Args:
            results: Generation results dictionary
            output_path: Path to save results
        """
        # Convert tensors to lists for JSON serialization
        serializable_results = {}
        for split, split_results in results.items():
            serializable_results[split] = []
            for result in split_results:
                serializable_result = {
                    'sample_id': result['sample_id'],
                    'original_question': result['original_question'],
                    'original_answer': result['original_answer'],
                    'generated_answer': result['generated_answer'],
                    'generated_ids': result['generated_ids']
                }
                serializable_results[split].append(serializable_result)
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"💾 Results saved to {output_path}")
    
    def print_results(self, results: Dict):
        """
        Print generation results in a readable format.
        
        Args:
            results: Generation results dictionary
        """
        print("\n" + "="*80)
        print("🎯 GENERATION RESULTS")
        print("="*80)
        
        for split, split_results in results.items():
            print(f"\n📊 {split.upper()} SPLIT ({len(split_results)} samples)")
            print("-" * 60)
            
            for i, result in enumerate(split_results, 1):
                print(f"\n🔍 Sample {i} (ID: {result['sample_id']})")
                print(f"❓ Question: {result['original_question'][:100]}...")
                print(f"✅ Original Answer: {result['original_answer'][:100]}...")
                print(f"🤖 Generated Answer: {result['generated_answer'][:100]}...")
                print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Simple Generation Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="Number of random samples per split")
    parser.add_argument("--splits", nargs="+", 
                       default=['train', 'val_seen', 'val_unseen'],
                       help="Dataset splits to process")
    parser.add_argument("--output", type=str, default="generation_results.json",
                       help="Output file path for results (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--no_save", action="store_true",
                       help="Don't save results to file, just print to console")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Initialize pipeline
    pipeline = SimpleGenerationPipeline(args.checkpoint, device=args.device)
    
    # Run generation
    results = pipeline.run_generation_pipeline(
        num_samples_per_split=args.num_samples,
        splits=args.splits
    )
    
    # Print results
    pipeline.print_results(results)
    
    # Save results only if not --no_save flag
    if not args.no_save:
        pipeline.save_results(results, args.output)
        print(f"\n🎉 Generation pipeline completed! Results saved to {args.output}")
    else:
        print(f"\n🎉 Generation pipeline completed! Results printed to console only.")

if __name__ == "__main__":
    main() 