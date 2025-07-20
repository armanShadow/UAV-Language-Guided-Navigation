#!/usr/bin/env python3
"""
Multi-GPU Hard Negative Mining Script

This script distributes mining across multiple GPUs for faster processing.
Usage: python multi_gpu_mining.py --image-dir /path/to/images --split train
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def run_mining_on_gpu(gpu_id, image_dir, split, batch_size=64, max_samples=None, num_shards=10):
    """Run mining on a specific GPU with sharding."""
    cmd = [
        sys.executable, "add_hard_negatives.py",
        "--image-dir", image_dir,
        "--split", split,
        "--gpu-id", str(gpu_id),
        "--batch-size", str(batch_size),
        "--num-shards", str(num_shards),
        "--shard-id", str(gpu_id)
    ]
    
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    
    print(f"🚀 Starting mining on GPU {gpu_id}")
    print(f"Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU hard negative mining')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing satellite images')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val_seen', 'val_unseen'],
                       help='Dataset split to process')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size per GPU')
    parser.add_argument('--max-samples-per-gpu', type=int, default=None,
                       help='Maximum samples to process per GPU')
    parser.add_argument('--num-gpus', type=int, default=10,
                       help='Number of GPUs to use (default: 10 for RTX 2080s)')
    parser.add_argument('--diverse-ratio', type=float, default=0.3,
                       help='Ratio of diverse vs hard negatives (default: 0.3)')
    parser.add_argument('--k-nn', type=int, default=30,
                       help='Number of K-NN neighbors (default: 30)')
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists("add_hard_negatives.py"):
        print("❌ Error: add_hard_negatives.py not found in current directory")
        print("Please run this script from AnsweringAgent/src/data/")
        sys.exit(1)
    
    print(f"🎯 Multi-GPU Mining Setup:")
    print(f"  Image directory: {args.image_dir}")
    print(f"  Split: {args.split}")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Number of GPUs: {args.num_gpus}")
    print(f"  Diverse ratio: {args.diverse_ratio}")
    print(f"  K-NN neighbors: {args.k_nn}")
    if args.max_samples_per_gpu:
        print(f"  Max samples per GPU: {args.max_samples_per_gpu}")
    
    # Start mining processes on each GPU
    processes = []
    start_time = time.time()
    
    try:
        for gpu_id in range(args.num_gpus):
            process = run_mining_on_gpu(
                gpu_id=gpu_id,
                image_dir=args.image_dir,
                split=args.split,
                batch_size=args.batch_size,
                max_samples=args.max_samples_per_gpu,
                num_shards=args.num_gpus  # Each GPU gets its own shard
            )
            processes.append((gpu_id, process))
        
        print(f"\n⏱️  Started {len(processes)} mining processes...")
        print("📊 Monitoring progress:")
        print("  Each GPU will process its own shard of the dataset")
        print("  This avoids duplicate work across GPUs")
        
        # Monitor processes
        completed = 0
        while processes:
            for i, (gpu_id, process) in enumerate(processes):
                if process.poll() is not None:
                    # Process completed
                    stdout, stderr = process.communicate()
                    
                    if process.returncode == 0:
                        print(f"✅ GPU {gpu_id} completed successfully")
                        if stdout:
                            # Extract key metrics from output
                            for line in stdout.split('\n'):
                                if "Mined" in line and "negatives total" in line:
                                    print(f"    {line.strip()}")
                                elif "Hard negatives:" in line and "Diverse negatives:" in line:
                                    print(f"    {line.strip()}")
                        completed += 1
                    else:
                        print(f"❌ GPU {gpu_id} failed with return code {process.returncode}")
                        if stderr:
                            print(f"Error: {stderr}")
                    
                    # Remove completed process
                    processes.pop(i)
                    break
            
            if processes:
                print(f"⏳ Still running: {len(processes)} GPUs, Completed: {completed}")
                time.sleep(10)  # Check every 10 seconds
        
        total_time = time.time() - start_time
        print(f"\n🎉 All mining processes completed!")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"📊 Successfully completed: {completed}/{args.num_gpus} GPUs")
        print(f"💡 Each GPU processed ~1/{args.num_gpus} of the dataset")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Stopping all processes...")
        for gpu_id, process in processes:
            process.terminate()
            print(f"🛑 Terminated GPU {gpu_id}")
        
        # Wait for processes to terminate
        for gpu_id, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"💀 Force killed GPU {gpu_id}")

if __name__ == '__main__':
    main() 