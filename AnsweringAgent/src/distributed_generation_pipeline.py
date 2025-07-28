#!/usr/bin/env python3
"""
Simple Distributed Generation Pipeline for UAV Navigation
Runs generation across 10 GPUs without saving results.
"""

import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List
import argparse
import random
from transformers import T5Tokenizer
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import Config
from models.answering_agent import AnsweringAgent
from data.dataset import AnsweringDataset
from utils.logger import setup_logger

def setup_distributed():
    """Set up distributed generation."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
    
    return True, rank, world_size

def setup_environment():
    """Setup environment for distributed generation."""
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

class SimpleDistributedGeneration:
    """Simple distributed generator for 10 GPUs."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', rank: int = 0, world_size: int = 1):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.config = Config()
        
        # Initialize logger
        self.logger = setup_logger('simple_distributed_generation', log_dir=self.config.log_dir)
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.model.t5_model_name, 
            model_max_length=self.config.data.max_seq_length
        )
        
        # Load model
        if rank == 0:
            self.logger.info("🏗️ Loading model for distributed generation...")
        self.model = AnsweringAgent(self.config, self.tokenizer, self.logger)
        
        # Load checkpoint
        if rank == 0:
            self.logger.info(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if rank == 0:
                self.logger.info("✅ Model state loaded successfully")
        else:
            if rank == 0:
                self.logger.warning("⚠️ No model_state_dict found in checkpoint, trying direct load")
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
    def generate_for_batch(self, batch: dict) -> List[str]:
        """Generate answers for a batch of samples."""
        with torch.no_grad():
            # Move data to device
            text_input = {k: v.to(self.device, non_blocking=True) for k, v in batch['text_input'].items()}
            
            # Handle separate encoding components
            if 'first_instruction_input' in batch:
                text_input['first_instruction_input'] = {k: v.to(self.device, non_blocking=True) for k, v in batch['first_instruction_input'].items()}
            if 'current_question_input' in batch:
                text_input['current_question_input'] = {k: v.to(self.device, non_blocking=True) for k, v in batch['current_question_input'].items()}
            
            current_view = batch['current_view_image'].to(self.device, non_blocking=True)
            previous_views = batch['previous_views_image'].to(self.device, non_blocking=True)
            
            destination_view = batch['destination_image'].to(self.device, non_blocking=True) if 'destination_image' in batch else None
            
            # Generate answers
            outputs = self.model(
                text_input=text_input,
                current_view=current_view,
                previous_views=previous_views,
                destination_view=destination_view,
                generate=True,
                curriculum_ratio=0.0
            )
            
            # Decode generated text
            generated_ids = outputs['sequences']
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Remove spatial prompt if it appears at the beginning
            cleaned_texts = []
            spatial_prompt = "Provide precise navigation with clock directions (1-12 o'clock), landmark colors and shapes, clear movement instructions."
            for text in generated_texts:
                if text.startswith(spatial_prompt.strip()):
                    text = text[len(spatial_prompt.strip()):].strip()
                cleaned_texts.append(text)
            
            return cleaned_texts
    
    def run_generation(self, num_samples_per_split: int = 5, 
                      splits: List[str] = None,
                      batch_size: int = 4) -> Dict[str, int]:
        """Run generation on samples from each dataset split."""
        if splits is None:
            splits = ['train', 'val_seen', 'val_unseen']
        
        results = {}
        
        for split in splits:
            if self.rank == 0:
                self.logger.info(f"🚀 Processing {split} split across {self.world_size} GPUs...")
            
            # Create dataset
            dataset = AnsweringDataset(self.config, split=split, tokenizer=self.tokenizer)
            
            # Use DistributedSampler for proper distribution
            sampler = DistributedSampler(
                dataset, 
                num_replicas=self.world_size, 
                rank=self.rank,
                shuffle=False
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=2,
                pin_memory=True
            )
            
            # Generate answers for batches
            samples_generated = 0
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating for {split} (GPU {self.rank})", disable=self.rank != 0)):
                try:
                    generated_texts = self.generate_for_batch(batch)
                    samples_generated += len(generated_texts)
                    
                    # Limit to requested number of samples
                    if samples_generated >= num_samples_per_split:
                        break
                        
                except Exception as e:
                    self.logger.error(f"❌ Error generating for batch {batch_idx}: {e}")
                    continue
            
            results[split] = samples_generated
            
            if self.rank == 0:
                self.logger.info(f"✅ Generated {samples_generated} answers for {split}")
        
        return results

def main():
    """Main distributed generation function."""
    parser = argparse.ArgumentParser(description="Simple Distributed Generation Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples per GPU per split")
    parser.add_argument("--splits", nargs="+", 
                       default=['train', 'val_seen', 'val_unseen'],
                       help="Dataset splits to process")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for generation")
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Setup distributed training
    is_distributed, rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"🚀 Simple Distributed UAV Navigation Generation Pipeline")
        print(f"World Size: {world_size}, Rank: {rank}")
        if torch.cuda.is_available():
            print(f"CUDA Devices: {torch.cuda.device_count()}")
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Initialize distributed pipeline
    pipeline = SimpleDistributedGeneration(
        args.checkpoint, 
        device=device, 
        rank=rank, 
        world_size=world_size
    )
    
    # Wrap with DDP if distributed
    if is_distributed:
        pipeline.model = DDP(pipeline.model, device_ids=[local_rank], output_device=local_rank)
    
    # Run distributed generation
    local_results = pipeline.run_generation(
        num_samples_per_split=args.num_samples,
        splits=args.splits,
        batch_size=args.batch_size
    )
    
    # Print summary (only on rank 0)
    if rank == 0:
        print("\n" + "="*60)
        print("🎯 GENERATION SUMMARY")
        print("="*60)
        total_samples = sum(local_results.values())
        print(f"Total samples generated: {total_samples}")
        for split, count in local_results.items():
            print(f"{split}: {count} samples")
        print("✅ Generation completed!")
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 