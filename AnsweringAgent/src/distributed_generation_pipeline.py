#!/usr/bin/env python3
"""
Optimized Distributed Generation Pipeline for UAV Navigation
Runs generation across 10 GPUs with enhanced performance and quality monitoring.
"""

import os
import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, List
import argparse
import random
import time
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

class OptimizedDistributedGeneration:
    """Optimized distributed generator for 10 GPUs with enhanced quality monitoring."""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', rank: int = 0, world_size: int = 1):
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.config = Config()
        self.start_time = time.time()
        
        # Performance tracking
        self.generation_stats = {
            'total_samples': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'total_generation_time': 0.0,
            'avg_generation_length': 0.0,
            'spatial_keywords_used': 0
        }
        
        # Initialize logger only on rank 0
        if rank == 0:
            self.logger = setup_logger('optimized_distributed_generation', log_dir=self.config.log_dir)
        else:
            # Create a dummy logger for non-rank-0 processes
            class DummyLogger:
                def info(self, msg): pass
                def warning(self, msg): pass
                def error(self, msg): pass
                def debug(self, msg): pass
            self.logger = DummyLogger()
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.model.t5_model_name, 
            model_max_length=self.config.data.max_seq_length
        )
        
        # Load model
        if rank == 0:
            self.logger.info("🏗️ Loading model for optimized distributed generation...")
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
        
        # Set model to use best generation settings
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for better performance
        
        if rank == 0:
            self.logger.info("🎯 Model optimized for best generation quality")
        
    def analyze_generation_quality(self, generated_text: str, original_question: str) -> Dict:
        """Analyze the quality of generated text."""
        # Spatial keywords to look for
        spatial_keywords = [
            "o'clock", "clock", "direction", "building", "turn", "move", "go", 
            "north", "south", "east", "west", "left", "right", "forward", "straight",
            "1 o'clock", "2 o'clock", "3 o'clock", "4 o'clock", "5 o'clock", "6 o'clock",
            "7 o'clock", "8 o'clock", "9 o'clock", "10 o'clock", "11 o'clock", "12 o'clock"
        ]
        
        # Count spatial keywords
        spatial_count = sum(1 for keyword in spatial_keywords if keyword.lower() in generated_text.lower())
        
        # Calculate metrics
        metrics = {
            'length': len(generated_text),
            'spatial_keywords_count': spatial_count,
            'has_direction': any(word in generated_text.lower() for word in ['left', 'right', 'forward', 'straight', 'turn']),
            'has_clock_direction': any(word in generated_text.lower() for word in ['o\'clock', 'clock']),
            'has_landmarks': any(word in generated_text.lower() for word in ['building', 'structure', 'landmark']),
            'completeness_score': min(len(generated_text) / 50.0, 1.0),  # Based on reasonable answer length
        }
        
        return metrics
        
    def generate_for_batch(self, batch: dict, show_samples: bool = False) -> List[Dict]:
        """Generate answers for a batch of samples with quality analysis."""
        batch_start_time = time.time()
        
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
            
            # Generate answers with optimized settings
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
            
            # Process results with quality analysis
            batch_results = []
            batch_size = len(generated_texts)
            generation_time = time.time() - batch_start_time
            
            for i in range(batch_size):
                # Get original question for comparison
                original_question = self.tokenizer.decode(
                    text_input['input_ids'][i], 
                    skip_special_tokens=True
                )
                
                # Try to get original answer if available
                original_answer = 'N/A'
                if 'text_label' in batch:
                    original_answer = self.tokenizer.decode(
                        batch['text_label']['input_ids'][i], 
                        skip_special_tokens=True
                    )
                
                # Clean generated text (remove spatial prompt if present)
                generated_text = generated_texts[i]
                spatial_prompt = "Provide precise navigation with clock directions (1-12 o'clock), landmark colors and shapes, clear movement instructions."
                if generated_text.startswith(spatial_prompt.strip()):
                    generated_text = generated_text[len(spatial_prompt.strip()):].strip()
                
                # Analyze generation quality
                quality_metrics = self.analyze_generation_quality(generated_text, original_question)
                
                result = {
                    'sample_id': f'gpu_{self.rank}_batch_{i}',
                    'original_question': original_question,
                    'original_answer': original_answer,
                    'generated_answer': generated_text,
                    'generation_time': generation_time / batch_size,
                    'quality_metrics': quality_metrics,
                    'gpu_rank': self.rank
                }
                
                batch_results.append(result)
                
                # Update stats
                self.generation_stats['spatial_keywords_used'] += quality_metrics['spatial_keywords_count']
                self.generation_stats['avg_generation_length'] += quality_metrics['length']
                
                # Show sample on rank 0 if requested
                if show_samples and self.rank == 0 and i == 0:
                    self.logger.info(f"🔍 Sample Generation:")
                    self.logger.info(f"   Question: {original_question[:100]}...")
                    self.logger.info(f"   Generated: {generated_text}")
                    self.logger.info(f"   Quality: {quality_metrics['spatial_keywords_count']} spatial keywords, "
                                   f"Length: {quality_metrics['length']}, "
                                   f"Clock directions: {quality_metrics['has_clock_direction']}")
            
            # Update generation stats
            self.generation_stats['total_samples'] += batch_size
            self.generation_stats['successful_generations'] += batch_size
            self.generation_stats['total_generation_time'] += generation_time
            
            return batch_results
    
    def print_comprehensive_results(self, results: Dict[str, List]):
        """Print comprehensive results with quality analysis."""
        if self.rank != 0:
            return
        
        total_runtime = time.time() - self.start_time
        
        print("\n" + "="*80)
        print("🎯 OPTIMIZED DISTRIBUTED GENERATION RESULTS")
        print("="*80)
        print(f"⏱️  Total Runtime: {total_runtime:.2f}s")
        print(f"🚀 World Size: {self.world_size} GPUs")
        print(f"📊 Overall Performance:")
        
        # Calculate overall statistics
        all_results = []
        for split_results in results.values():
            all_results.extend(split_results)
        
        if all_results:
            total_samples = len(all_results)
            avg_length = sum(r['quality_metrics']['length'] for r in all_results) / total_samples
            total_spatial_keywords = sum(r['quality_metrics']['spatial_keywords_count'] for r in all_results)
            clock_direction_usage = sum(1 for r in all_results if r['quality_metrics']['has_clock_direction']) / total_samples
            landmark_usage = sum(1 for r in all_results if r['quality_metrics']['has_landmarks']) / total_samples
            avg_generation_time = sum(r['generation_time'] for r in all_results) / total_samples
            
            print(f"   Total Samples: {total_samples}")
            print(f"   Avg Generation Time: {avg_generation_time:.3f}s per sample")
            print(f"   Throughput: {total_samples / total_runtime:.2f} samples/second")
            print(f"   Avg Answer Length: {avg_length:.1f} characters")
            print(f"   Total Spatial Keywords: {total_spatial_keywords}")
            print(f"   Clock Direction Usage: {clock_direction_usage:.1%}")
            print(f"   Landmark Usage: {landmark_usage:.1%}")
        
        print(f"\n📈 Quality Analysis by Split:")
        print("-" * 60)
        
        for split, split_results in results.items():
            if not split_results:
                continue
                
            print(f"\n📊 {split.upper()} SPLIT ({len(split_results)} samples)")
            
            # Calculate split metrics
            avg_length = sum(r['quality_metrics']['length'] for r in split_results) / len(split_results)
            spatial_keywords = sum(r['quality_metrics']['spatial_keywords_count'] for r in split_results)
            clock_usage = sum(1 for r in split_results if r['quality_metrics']['has_clock_direction']) / len(split_results)
            direction_usage = sum(1 for r in split_results if r['quality_metrics']['has_direction']) / len(split_results)
            completeness = sum(r['quality_metrics']['completeness_score'] for r in split_results) / len(split_results)
            
            print(f"   📏 Avg Length: {avg_length:.1f} chars")
            print(f"   🧭 Spatial Keywords: {spatial_keywords} total")
            print(f"   🕐 Clock Directions: {clock_usage:.1%}")
            print(f"   ➡️  Movement Directions: {direction_usage:.1%}")
            print(f"   ✅ Completeness Score: {completeness:.2f}/1.0")
            
            # Show detailed sample generations with exact target and generated answers
            print(f"\n🔍 Detailed Sample Generations:")
            for i, result in enumerate(split_results[:5]):  # Show first 5 samples
                print(f"\n     === Sample {i+1} (ID: {result['sample_id']}) ===")
                print(f"     ❓ QUESTION:")
                print(f"        {result['original_question']}")
                print(f"     ✅ TARGET ANSWER:")
                print(f"        {result['original_answer']}")
                print(f"     🤖 GENERATED ANSWER:")
                print(f"        {result['generated_answer']}")
                print(f"     📊 QUALITY METRICS:")
                print(f"        - Length: {result['quality_metrics']['length']} chars")
                print(f"        - Spatial Keywords: {result['quality_metrics']['spatial_keywords_count']}")
                print(f"        - Has Clock Direction: {result['quality_metrics']['has_clock_direction']}")
                print(f"        - Has Movement Direction: {result['quality_metrics']['has_direction']}")
                print(f"        - Has Landmarks: {result['quality_metrics']['has_landmarks']}")
                print(f"        - Completeness Score: {result['quality_metrics']['completeness_score']:.2f}")
                print(f"        - Generation Time: {result['generation_time']:.3f}s")
                print()
        
        print("✅ Optimized generation analysis completed!")
        print(f"💡 Prompt engineering effectiveness: {total_spatial_keywords} spatial keywords used across {len(all_results)} samples")
        print(f"🎯 Average spatial keywords per sample: {total_spatial_keywords / len(all_results):.1f}" if all_results else "")

    def run_generation(self, num_samples_per_split: int = 10, 
                      splits: List[str] = None,
                      batch_size: int = 4,
                      show_samples: bool = True) -> Dict[str, List]:
        """Run optimized generation on samples from each dataset split."""
        if splits is None:
            splits = ['train', 'val_seen', 'val_unseen']
        
        results = {}
        
        for split in splits:
            if self.rank == 0:
                self.logger.info(f"🚀 Processing {split} split across {self.world_size} GPUs...")
                self.logger.info(f"   Target: {num_samples_per_split} samples per GPU")
            
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
            split_results = []
            samples_generated = 0
            total_batches = len(dataloader)
            
            if self.rank == 0:
                self.logger.info(f"   Processing {total_batches} batches...")
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Generating for {split} (GPU {self.rank})", disable=self.rank != 0)):
                try:
                    batch_results = self.generate_for_batch(batch, show_samples=(batch_idx == 0 and show_samples))
                    
                    # Add results but respect the sample limit
                    for result in batch_results:
                        if samples_generated < num_samples_per_split:
                            split_results.append(result)
                            samples_generated += 1
                        else:
                            break
                    
                    # Break if we've reached the target
                    if samples_generated >= num_samples_per_split:
                        if self.rank == 0:
                            self.logger.info(f"   Reached target of {num_samples_per_split} samples, stopping...")
                        break
                        
                except Exception as e:
                    if self.rank == 0:
                        self.logger.error(f"❌ Error generating for batch {batch_idx}: {e}")
                    self.generation_stats['failed_generations'] += batch_size
                    continue
            
            results[split] = split_results
            
            if self.rank == 0:
                # Calculate split-specific quality metrics
                total_spatial_keywords = sum(r['quality_metrics']['spatial_keywords_count'] for r in split_results)
                avg_length = sum(r['quality_metrics']['length'] for r in split_results) / len(split_results) if split_results else 0
                clock_direction_usage = sum(1 for r in split_results if r['quality_metrics']['has_clock_direction']) / len(split_results) if split_results else 0
                
                self.logger.info(f"✅ {split} completed: {len(split_results)} samples generated")
                self.logger.info(f"   Avg length: {avg_length:.1f} chars")
                self.logger.info(f"   Total spatial keywords: {total_spatial_keywords}")
                self.logger.info(f"   Clock direction usage: {clock_direction_usage:.1%}")
                
                # Show a sample result
                if split_results:
                    sample = split_results[0]
                    self.logger.info(f"📝 Sample from {split}:")
                    self.logger.info(f"   Q: {sample['original_question'][:80]}...")
                    self.logger.info(f"   Target: {sample['original_answer'][:60]}...")
                    self.logger.info(f"   Generated: {sample['generated_answer'][:60]}...")
        
        return results

def main():
    """Main optimized distributed generation function."""
    parser = argparse.ArgumentParser(description="Optimized Distributed Generation Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples per GPU per split")
    parser.add_argument("--splits", nargs="+", 
                       default=['train', 'val_seen', 'val_unseen'],
                       help="Dataset splits to process")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for generation")
    parser.add_argument("--show_samples", action="store_true",
                       help="Show sample generations during processing")
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Setup distributed training
    is_distributed, rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"🚀 Optimized Distributed UAV Navigation Generation Pipeline")
        print(f"🎯 Focus: Maximum generation quality and prompt engineering analysis")
        print(f"World Size: {world_size}, Rank: {rank}")
        if torch.cuda.is_available():
            print(f"CUDA Devices: {torch.cuda.device_count()}")
        print()
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Initialize optimized distributed pipeline
    pipeline = OptimizedDistributedGeneration(
        args.checkpoint, 
        device=device, 
        rank=rank, 
        world_size=world_size
    )
    
    # Wrap with DDP if distributed
    if is_distributed:
        pipeline.model = DDP(pipeline.model, device_ids=[local_rank], output_device=local_rank)
    
    # Run optimized distributed generation
    local_results = pipeline.run_generation(
        num_samples_per_split=args.num_samples,
        splits=args.splits,
        batch_size=args.batch_size,
        show_samples=args.show_samples
    )
    
    # Print comprehensive results (only on rank 0)
    if rank == 0:
        pipeline.print_comprehensive_results(local_results)
    
    # Cleanup
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 