#!/usr/bin/env python3
"""
Test script for AVDN dataset generation.
This script tests the generation process with a small sample to ensure everything works correctly.
"""

import torch
import json
import os
import argparse
from transformers import T5Tokenizer
from tqdm import tqdm
from config import Config
from data.dataset import AnsweringDataset
from models.answering_agent import AnsweringAgent
from utils.logger import setup_logger

# Import evaluation functions
import sys
sys.path.append('../scripts')
from run_eval_generation import composite_score, direction_score, yesno_score, attribute_score, landmark_score, movement_score

def test_avdn_structure():
    """Test understanding of AVDN dataset structure."""
    print("🔍 Testing AVDN dataset structure...")
    
    # Load a small sample from val_seen
    avdn_file = "/app/UAV-Language-Guided-Navigation/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/val_seen_data.json"
    
    if not os.path.exists(avdn_file):
        print(f"❌ AVDN file not found: {avdn_file}")
        return False
    
    with open(avdn_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} samples from val_seen")
    
    # Examine first sample
    sample = data[0]
    print(f"\n📋 AVDN Sample structure:")
    print(f"Keys: {list(sample.keys())}")
    print(f"Instruction: {sample['instructions']}")
    print(f"Map: {sample['map_name']}")
    print(f"Route: {sample['route_index']}")
    print(f"Angle: {sample['angle']}")
    
    return True

def test_formatted_dataset_structure():
    """Test understanding of formatted dataset structure."""
    print("\n🔍 Testing formatted dataset structure...")
    
    # Load config
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name)
    
    # Load a small sample from val_seen
    try:
        dataset = AnsweringDataset(config, split='val_seen', tokenizer=tokenizer)
        print(f"✅ Loaded {len(dataset)} samples from formatted val_seen")
        
        # Examine first sample
        sample = dataset[0]
        print(f"\n📋 Formatted Sample structure:")
        print(f"Keys: {list(sample.keys())}")
        
        # Decode some components
        first_instruction = tokenizer.decode(sample['first_instruction_input']['input_ids'], skip_special_tokens=True)
        current_question = tokenizer.decode(sample['current_question_input']['input_ids'], skip_special_tokens=True)
        current_answer = tokenizer.decode(sample['text_label']['input_ids'], skip_special_tokens=True)
        dialog_context = tokenizer.decode(sample['text_input']['input_ids'], skip_special_tokens=True)
        
        print(f"First Instruction: {first_instruction}")
        print(f"Current Question: {current_question}")
        print(f"Current Answer: {current_answer}")
        print(f"Dialog Context: {dialog_context[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading formatted dataset: {e}")
        return False

def test_answering_agent_generation(checkpoint_path: str, max_samples: int = 5):
    """Test Answering Agent generation with actual checkpoint and calculate metrics."""
    print(f"\n🧪 Testing Answering Agent generation with checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config and tokenizer
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name)
    
    # Initialize logger
    logger = setup_logger('test_generation', log_dir=config.log_dir)
    
    # Load Answering Agent model
    print("🏗️ Loading Answering Agent model...")
    model = AnsweringAgent(config, tokenizer, logger)
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model state loaded successfully")
    else:
        print("⚠️ No model_state_dict found in checkpoint, trying direct load")
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load formatted dataset
    print("📊 Loading formatted dataset...")
    dataset = AnsweringDataset(config, split='val_seen', tokenizer=tokenizer)
    
    # Test generation on a few samples
    print(f"🧪 Testing generation on {max_samples} samples...")
    
    # Track metrics
    all_scores = []
    successful_generations = 0
    
    for i in range(min(max_samples, len(dataset))):
        try:
            sample = dataset[i]
            
            # Get text input and visual features
            text_input = {
                'input_ids': sample['text_input']['input_ids'].unsqueeze(0).to(device),
                'attention_mask': sample['text_input']['attention_mask'].unsqueeze(0).to(device)
            }
            
            current_view = sample['current_view_image'].unsqueeze(0).to(device)
            previous_views = sample['previous_views_image'].unsqueeze(0).to(device)
            
            # Generation parameters (same as evaluation)
            generation_params = {
                'task_type': 'precision_short',
                'num_beams': 4,
                'do_sample': False,
                'repetition_penalty': 1.1,
                'length_penalty': 0.8,
                'min_new_tokens': 8,
                'max_new_tokens': 70,
                'early_stopping': True,
            }
            
            # Generate answer
            with torch.no_grad():
                generated_seq = model.generate_answer(
                    text_input, current_view, previous_views,
                    **generation_params
                )
            
            # Decode results
            original_answer = tokenizer.decode(sample['text_label']['input_ids'], skip_special_tokens=True)
            generated_answer = tokenizer.decode(generated_seq[0], skip_special_tokens=True)
            dialog_context = tokenizer.decode(sample['text_input']['input_ids'], skip_special_tokens=True)
            
            # Calculate composite score
            scores = composite_score(generated_answer, original_answer, task_type="precision_short")
            all_scores.append(scores)
            successful_generations += 1
            
            print(f"\n📝 Sample {i+1}:")
            print(f"Context: {dialog_context[:100]}...")
            print(f"Original Answer: {original_answer}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Composite Score: {scores['total']:.4f}")
            print(f"Direction Score: {scores['direction']:.4f}")
            print(f"Movement Score: {scores['movement']:.4f}")
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error generating sample {i}: {e}")
            continue
    
    # Calculate and report final metrics
    if all_scores:
        avg_composite = sum(s['total'] for s in all_scores) / len(all_scores)
        avg_direction = sum(s['direction'] for s in all_scores) / len(all_scores)
        avg_movement = sum(s['movement'] for s in all_scores) / len(all_scores)
        avg_landmark = sum(s['landmark'] for s in all_scores) / len(all_scores)
        avg_attribute = sum(s['attribute'] for s in all_scores) / len(all_scores)
        
        print(f"\n📊 GENERATION METRICS SUMMARY:")
        print(f"Successful Generations: {successful_generations}/{max_samples}")
        print(f"Average Composite Score: {avg_composite:.4f}")
        print(f"Average Direction Score: {avg_direction:.4f}")
        print(f"Average Movement Score: {avg_movement:.4f}")
        print(f"Average Landmark Score: {avg_landmark:.4f}")
        print(f"Average Attribute Score: {avg_attribute:.4f}")
        
        # Individual scores for detailed analysis
        print(f"\n📈 INDIVIDUAL SCORES:")
        for i, scores in enumerate(all_scores):
            print(f"Sample {i+1}: Composite={scores['total']:.4f}, Direction={scores['direction']:.4f}, Movement={scores['movement']:.4f}")
    
    print("✅ Answering Agent generation test completed!")
    return True

def test_small_generation(max_samples: int = 5):
    """Test AVDN generation on a small sample."""
    print(f"\n🚀 Testing AVDN generation on {max_samples} samples...")
    
    # Load AVDN data
    avdn_file = "../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/val_seen_data.json"
    
    if not os.path.exists(avdn_file):
        print(f"❌ AVDN file not found: {avdn_file}")
        return False
    
    with open(avdn_file, 'r') as f:
        avdn_data = json.load(f)
    
    # Load formatted dataset
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name)
    formatted_dataset = AnsweringDataset(config, split='val_seen', tokenizer=tokenizer)
    
    # Process a few samples
    processed_samples = []
    
    for i in tqdm(range(min(max_samples, len(avdn_data))), desc="Processing samples"):
        avdn_sample = avdn_data[i]
        
        # Get original instruction
        original_instruction = avdn_sample['instructions']
        
        # For testing, we'll just use the original instruction
        # In the real script, this would be generated by the Answering Agent
        new_instruction = original_instruction  # Placeholder for testing
        
        # Create new AVDN sample
        new_sample = avdn_sample.copy()
        new_sample['instructions'] = new_instruction
        
        processed_samples.append(new_sample)
        
        # Print example
        if i < 3:
            print(f"\nSample {i+1}:")
            print(f"Map: {avdn_sample['map_name']}, Route: {avdn_sample['route_index']}")
            print(f"Original: {original_instruction}")
            print(f"Generated: {new_sample['instructions']}")
            print("-" * 80)
    
    # Save test results
    output_dir = "./test_generated_avdn"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "test_val_seen_data.json")
    with open(output_file, 'w') as f:
        json.dump(processed_samples, f, indent=2)
    
    print(f"\n✅ Test completed successfully!")
    print(f"📁 Test results saved to: {output_file}")
    print(f"📊 Processed {len(processed_samples)} samples")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test AVDN generation")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to Answering Agent checkpoint")
    parser.add_argument("--max_samples", type=int, default=5,
                       help="Maximum samples to process for testing")
    args = parser.parse_args()

    print("🚀 AVDN Dataset Generation Test Suite")
    print("=" * 50)
    
    # Test 1: AVDN structure
    if not test_avdn_structure():
        print("❌ AVDN structure test failed")
        return
    
    # Test 2: Formatted dataset structure
    if not test_formatted_dataset_structure():
        print("❌ Formatted dataset structure test failed")
        return
    
    # Test 3: Answering Agent generation (if checkpoint provided)
    if args.checkpoint:
        if not test_answering_agent_generation(args.checkpoint, args.max_samples):
            print("❌ Answering Agent generation test failed")
            return
    else:
        print("\n⚠️ No checkpoint provided, skipping Answering Agent generation test")
        print("Use --checkpoint path/to/checkpoint.pth to test generation")
    
    # Test 4: Small generation
    if not test_small_generation(args.max_samples):
        print("❌ Small generation test failed")
        return
    
    print("\n✅ All tests completed successfully!")
    print("\n📝 Next steps:")
    print("1. Run the full generation script with --checkpoint path")
    print("2. Test with a small sample first: --max_samples 10")
    print("3. Scale up to full dataset when ready")

if __name__ == "__main__":
    main() 