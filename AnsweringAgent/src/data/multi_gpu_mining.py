#!/usr/bin/env python3
"""
Multi-GPU Hard Negative Mining Script

This script distributes hard negative mining across multiple GPUs for faster processing.
Each GPU processes a unique shard of the dataset to avoid duplicate work.

Usage: 
    python multi_gpu_mining.py --image-dir /path/to/images --split train --num-gpus 4
"""

import os
import sys
import subprocess
import argparse
import time
import signal
from pathlib import Path

def run_mining_on_gpu(gpu_id, image_dir, split, batch_size=64, max_samples=None, 
                     num_shards=10, k_nn=30, diverse_ratio=0.3, min_answer_length=20):
    """Run mining on a specific GPU with dataset sharding."""
    
    cmd = [
        sys.executable, "add_hard_negatives.py",
        "--image-dir", image_dir,
        "--split", split,
        "--gpu-id", str(gpu_id),
        "--batch-size", str(batch_size),
        "--num-shards", str(num_shards),
        "--shard-id", str(gpu_id),
        "--k-nn", str(k_nn),
        "--diverse-ratio", str(diverse_ratio),
        "--min-answer-length", str(min_answer_length),
        "--use-diverse-negatives"
    ]
    
    if max_samples:
        cmd.extend(["--max-samples", str(max_samples)])
    
    print(f"🚀 Starting mining on GPU {gpu_id}")
    if max_samples:
        print(f"   Processing max {max_samples} samples from shard {gpu_id}/{num_shards}")
    else:
        print(f"   Processing full shard {gpu_id}/{num_shards}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    return process

def monitor_processes(processes, show_progress=True):
    """Monitor mining processes and return results."""
    
    completed = []
    failed = []
    
    print(f"\n⏱️  Monitoring {len(processes)} mining processes...")
    
    while processes:
        for i, (gpu_id, process) in enumerate(processes):
            if process.poll() is not None:
                # Process completed
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    print(f"✅ GPU {gpu_id} completed successfully")
                    
                    # Extract key metrics from output
                    if stdout and show_progress:
                        lines = stdout.strip().split('\n')
                        for line in lines[-20:]:  # Show last 20 lines for summary
                            if any(keyword in line for keyword in [
                                "Mined", "negatives total", "Hard negatives:", "Success rate:",
                                "Phrase diversity:", "Hard negative quality:"
                            ]):
                                print(f"  GPU {gpu_id}: {line.strip()}")
                    
                    completed.append((gpu_id, stdout, stderr))
                else:
                    print(f"❌ GPU {gpu_id} failed with return code {process.returncode}")
                    if stderr:
                        error_lines = stderr.strip().split('\n')
                        print(f"  Error: {error_lines[-1] if error_lines else 'Unknown error'}")
                    failed.append((gpu_id, stdout, stderr))
                
                # Remove completed process
                processes.pop(i)
                break
        
        if processes and show_progress:
            print(f"⏳ Still running: {len(processes)} GPUs, Completed: {len(completed)}, Failed: {len(failed)}")
            time.sleep(15)  # Check every 15 seconds
    
    return completed, failed

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU hard negative mining for AVDN dataset')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='Directory containing satellite images')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val_seen', 'val_unseen'],
                       help='Dataset split to process')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size per GPU')
    parser.add_argument('--max-samples-per-gpu', type=int, default=None,
                       help='Maximum samples to process per GPU (for testing)')
    parser.add_argument('--num-gpus', type=int, default=4,
                       help='Number of GPUs to use')
    parser.add_argument('--k-nn', type=int, default=30,
                       help='Number of K-NN neighbors for hard negative mining')
    parser.add_argument('--diverse-ratio', type=float, default=0.3,
                       help='Ratio of diverse vs hard negatives')
    parser.add_argument('--min-answer-length', type=int, default=20,
                       help='Minimum answer length for negative mining')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Validate environment
    if not os.path.exists("add_hard_negatives.py"):
        print("❌ Error: add_hard_negatives.py not found in current directory")
        print("Please run this script from AnsweringAgent/src/data/")
        sys.exit(1)
    
    if not os.path.exists(args.image_dir):
        print(f"❌ Error: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    print(f"🎯 Multi-GPU Hard Negative Mining Setup:")
    print(f"  Image directory: {args.image_dir}")
    print(f"  Dataset split: {args.split}")
    print(f"  Number of GPUs: {args.num_gpus}")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  K-NN neighbors: {args.k_nn}")
    print(f"  Diverse ratio: {args.diverse_ratio}")
    print(f"  Min answer length: {args.min_answer_length}")
    if args.max_samples_per_gpu:
        print(f"  Max samples per GPU: {args.max_samples_per_gpu}")
    
    # Start mining processes
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
                num_shards=args.num_gpus,
                k_nn=args.k_nn,
                diverse_ratio=args.diverse_ratio,
                min_answer_length=args.min_answer_length
            )
            processes.append((gpu_id, process))
        
        print(f"\n📊 Dataset Distribution:")
        print(f"  Each GPU processes ~1/{args.num_gpus} of the dataset")
        print(f"  No overlap between GPU workloads")
        print(f"  Results will be automatically saved when each GPU completes")
        
        # Monitor processes
        completed, failed = monitor_processes(processes, show_progress=not args.quiet)
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n🎉 Multi-GPU mining completed!")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Results: {len(completed)} successful, {len(failed)} failed")
        
        if completed:
            print(f"\n✅ Successful GPUs: {[gpu_id for gpu_id, _, _ in completed]}")
        
        if failed:
            print(f"\n❌ Failed GPUs: {[gpu_id for gpu_id, _, _ in failed]}")
            print("Check individual GPU logs for error details.")
        
        # Performance summary
        if completed and total_time > 0:
            throughput = len(completed) / (total_time / 60)  # GPUs per minute
            print(f"🚀 Performance: {throughput:.2f} GPUs completed per minute")
            if args.max_samples_per_gpu:
                total_samples = len(completed) * args.max_samples_per_gpu
                sample_rate = total_samples / total_time
                print(f"📈 Sample processing rate: {sample_rate:.1f} samples/second")
        
        # Return appropriate exit code
        if failed:
            print(f"\n⚠️  {len(failed)} GPU(s) failed. Check logs and consider rerunning failed shards.")
            sys.exit(1)
        else:
            print(f"\n🎊 All {len(completed)} GPUs completed successfully!")
            print("Hard negatives have been added to the dataset.")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Stopping all processes...")
        
        # Terminate all processes gracefully
        for gpu_id, process in processes:
            if process.poll() is None:  # Still running
                print(f"🛑 Terminating GPU {gpu_id}...")
                process.terminate()
        
        # Wait for graceful termination
        time.sleep(3)
        
        # Force kill any remaining processes
        for gpu_id, process in processes:
            if process.poll() is None:
                print(f"💀 Force killing GPU {gpu_id}...")
                process.kill()
        
        print("🛑 All processes stopped.")
        sys.exit(130)  # Standard exit code for SIGINT
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        
        # Clean up processes
        for gpu_id, process in processes:
            if process.poll() is None:
                process.terminate()
        
        sys.exit(1)

if __name__ == '__main__':
    main() 