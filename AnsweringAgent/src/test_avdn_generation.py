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

import json
import torch
import os
import sys
from typing import Dict, List, Optional
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from data.dataset import AnsweringDataset
from models.answering_agent import AnsweringAgent
from run_eval_generation import composite_score, direction_score, yesno_score, attribute_score, landmark_score, movement_score

def load_preprocessed_json(config, split: str) -> Optional[List[Dict]]:
    """Load preprocessed JSON data with fallback paths"""
    json_paths = [
        getattr(config.data, f'{split}_json_path', None),
        f'data/processed_data/{split}_data.json',
        f'processed_data/{split}_data.json',
        f'../data/processed_data/{split}_data.json'
    ]
    
    for path in json_paths:
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    print(f"❌ Could not load preprocessed JSON for {split}")
    return None

def load_avdn_dataset(split: str) -> List[Dict]:
    """Load original AVDN dataset"""
    avdn_paths = [
        f'../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/{split}_data.json',
        f'../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/{split}_data.json',
        f'../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/{split}_data.json',
        f'Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/{split}_data.json'
    ]
    
    for path in avdn_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    raise FileNotFoundError(f"Could not find AVDN dataset for {split}")

def test_avdn_structure():
    """Test the structure of AVDN dataset"""
    print("🔍 Testing AVDN dataset structure...")
    
    try:
        avdn_file = "/app/UAV-Language-Guided-Navigation/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/val_seen_data.json"
        if not os.path.exists(avdn_file):
            avdn_file = "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/val_seen_data.json"
        
        with open(avdn_file, 'r') as f:
            avdn_data = json.load(f)
        
        print(f"✅ AVDN dataset loaded: {len(avdn_data)} samples")
        
        # Print first few samples
        for i in range(min(3, len(avdn_data))):
            sample = avdn_data[i]
            print(f"\n📋 Sample {i}:")
            print(f"  Map: {sample.get('map_name', 'N/A')}")
            print(f"  Route Index: {sample.get('route_index', 'N/A')}")
            print(f"  Last Round: {sample.get('last_round_idx', 'N/A')}")
            print(f"  Instructions: {sample.get('instructions', 'N/A')}")
            print(f"  Pre-dialogs: {len(sample.get('pre_dialogs', []))} entries")
            
    except Exception as e:
        print(f"❌ Error testing AVDN structure: {e}")

def test_formatted_dataset_structure():
    """Test the structure of formatted dataset"""
    print("\n🔍 Testing formatted dataset structure...")
    
    try:
        config = Config()
        dataset = AnsweringDataset(config, split='val_seen')
        
        print(f"✅ Formatted dataset loaded: {len(dataset)} samples")
        
        # Print first few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n📋 Formatted Sample {i}:")
            print(f"  Text input keys: {list(sample['text_input'].keys())}")
            if 'first_instruction_input' in sample:
                print(f"  First instruction input: {list(sample['first_instruction_input'].keys())}")
            if 'current_question_input' in sample:
                print(f"  Current question input: {list(sample['current_question_input'].keys())}")
            
            # Safely print text label (avoid tensor slice issues)
            text_label = sample['text_label']
            if hasattr(text_label, 'shape'):
                print(f"  Text label shape: {text_label.shape}")
            else:
                print(f"  Text label: {str(text_label)[:100]}...")
            
    except Exception as e:
        print(f"❌ Error testing formatted dataset structure: {e}")

def test_instruction_comparison():
    """Compare instructions between AVDN and formatted datasets"""
    print("\n🔍 Testing instruction comparison between AVDN and formatted datasets...")
    
    try:
        # Load AVDN dataset
        avdn_data = load_avdn_dataset('val_seen')
        print(f"✅ Loaded AVDN dataset: {len(avdn_data)} samples")
        
        # Load preprocessed JSON
        config = Config()
        preprocessed_json = load_preprocessed_json(config, 'val_seen')
        if not preprocessed_json:
            print("❌ Could not load preprocessed JSON for comparison")
            return
        
        print(f"✅ Loaded preprocessed JSON: {len(preprocessed_json)} episodes")
        
        # Load formatted dataset
        dataset = AnsweringDataset(config, split='val_seen')
        print(f"✅ Loaded formatted dataset: {len(dataset)} samples")
        
        # Compare first few samples
        sample_count = min(5, len(avdn_data))
        
        for i in range(sample_count):
            print(f"\n{'='*80}")
            print(f"📋 COMPARISON SAMPLE {i}")
            print(f"{'='*80}")
            
            # AVDN sample
            avdn_sample = avdn_data[i]
            print(f"\n🔴 AVDN DATASET:")
            print(f"  Map: {avdn_sample.get('map_name', 'N/A')}")
            print(f"  Route Index: {avdn_sample.get('route_index', 'N/A')}")
            print(f"  Last Round: {avdn_sample.get('last_round_idx', 'N/A')}")
            print(f"  Instructions: {avdn_sample.get('instructions', 'N/A')}")
            
            # Parse AVDN instructions
            instructions = avdn_sample.get('instructions', '')
            if '[QUE]' in instructions and '[INS]' in instructions:
                # Extract question and instruction
                que_start = instructions.find('[QUE]')
                ins_start = instructions.find('[INS]')
                question = instructions[que_start+5:ins_start].strip()
                instruction = instructions[ins_start+5:].strip()
                print(f"  Extracted Question: {question}")
                print(f"  Extracted Instruction: {instruction}")
            elif '[INS]' in instructions:
                # First instruction only
                instruction = instructions.replace('[INS]', '').strip()
                print(f"  First Instruction: {instruction}")
                print(f"  Question: None (first instruction)")
            else:
                print(f"  Raw Instructions: {instructions}")
            
            # Find corresponding formatted sample
            if i < len(dataset):
                formatted_sample = dataset[i]
                print(f"\n🟢 FORMATTED DATASET:")
                
                # Safely print text label
                text_label = formatted_sample['text_label']
                if hasattr(text_label, 'shape'):
                    print(f"  Text label shape: {text_label.shape}")
                else:
                    print(f"  Text label: {str(text_label)[:200]}...")
                
                # Try to decode the inputs
                try:
                    from transformers import T5Tokenizer
                    tokenizer = T5Tokenizer.from_pretrained('t5-base')
                    
                    # Decode text input
                    text_input_ids = formatted_sample['text_input']['input_ids']
                    decoded_text = tokenizer.decode(text_input_ids, skip_special_tokens=True)
                    print(f"  Decoded text input: {decoded_text[:200]}...")
                    
                    # Decode first instruction if present
                    if 'first_instruction_input' in formatted_sample:
                        first_ins_ids = formatted_sample['first_instruction_input']['input_ids']
                        decoded_first_ins = tokenizer.decode(first_ins_ids, skip_special_tokens=True)
                        print(f"  Decoded first instruction: {decoded_first_ins}")
                    
                    # Decode current question if present
                    if 'current_question_input' in formatted_sample:
                        current_q_ids = formatted_sample['current_question_input']['input_ids']
                        decoded_current_q = tokenizer.decode(current_q_ids, skip_special_tokens=True)
                        print(f"  Decoded current question: {decoded_current_q}")
                        
                except Exception as e:
                    print(f"  Error decoding formatted sample: {e}")
            
            print(f"\n{'='*80}")
            
    except Exception as e:
        print(f"❌ Error in instruction comparison: {e}")

def test_answering_agent_generation():
    """Test Answering Agent generation with real data"""
    print("\n🔍 Testing Answering Agent generation...")
    
    try:
        config = Config()
        
        # Load model
        model = AnsweringAgent(config)
        checkpoint_path = config.model.checkpoint_path
        
        print(f"🔍 Looking for checkpoint at: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found at: {checkpoint_path}")
            print("🔍 Checking for alternative checkpoint paths...")
            
            # Try alternative paths
            alt_paths = [
                "checkpoints/best_model.pth",
                "checkpoints/latest_model.pth", 
                "models/checkpoint.pth",
                "../checkpoints/best_model.pth",
                "../models/checkpoint.pth"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    print(f"✅ Found checkpoint at: {checkpoint_path}")
                    break
            else:
                print("❌ No checkpoint found. Please provide a valid checkpoint path.")
                return
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model state from {checkpoint_path}")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded model directly from {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"✅ Model loaded on device: {device}")
        
        # Load datasets
        dataset = AnsweringDataset(config, split='val_seen')
        preprocessed_json = load_preprocessed_json(config, 'val_seen')
        
        if not preprocessed_json:
            print("❌ Could not load preprocessed JSON")
            return
        
        print(f"✅ Loaded dataset: {len(dataset)} samples")
        print(f"✅ Loaded preprocessed JSON: {len(preprocessed_json)} episodes")
        
        # Test first few samples
        sample_count = min(3, len(dataset))
        all_scores = []
        
        for i in range(sample_count):
            print(f"\n{'='*60}")
            print(f"🎯 GENERATION SAMPLE {i}")
            print(f"{'='*60}")
            
            try:
                sample = dataset[i]
                
                # Get metadata info
                metadata_info = "Unknown"
                if i < len(preprocessed_json):
                    episode = preprocessed_json[i]
                    metadata_info = f"Episode: {episode.get('episode_id', 'N/A')}, Map: {episode.get('map_name', 'N/A')}"
                
                print(f"📋 Metadata: {metadata_info}")
                
                # Construct text_input exactly as in training
                text_input = {
                    'input_ids': sample['text_input']['input_ids'].unsqueeze(0).to(device),
                    'attention_mask': sample['text_input']['attention_mask'].unsqueeze(0).to(device)
                }
                if 'first_instruction_input' in sample:
                    text_input['first_instruction_input'] = {k: v.unsqueeze(0).to(device) for k, v in sample['first_instruction_input'].items()}
                if 'current_question_input' in sample:
                    text_input['current_question_input'] = {k: v.unsqueeze(0).to(device) for k, v in sample['current_question_input'].items()}
                
                # Get visual features
                current_view = sample['current_view_image'].unsqueeze(0).to(device)
                previous_views = sample['previous_views_image'].unsqueeze(0).to(device)
                destination = sample['destination_image'].unsqueeze(0).to(device)
                
                # Generate answer
                with torch.no_grad():
                    generated_ids = model.generate_answer(
                        text_input=text_input,
                        current_view=current_view,
                        previous_views=previous_views,
                        destination=destination
                    )
                
                # Decode generated text
                from transformers import T5Tokenizer
                tokenizer = T5Tokenizer.from_pretrained('t5-base')
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Decode original text
                original_text = tokenizer.decode(sample['text_label'], skip_special_tokens=True)
                
                print(f"🔄 Original: {original_text}")
                print(f"🎯 Generated: {generated_text}")
                
                # Calculate scores
                scores = {
                    'direction': direction_score(generated_text),
                    'yesno': yesno_score(generated_text),
                    'attribute': attribute_score(generated_text),
                    'landmark': landmark_score(generated_text),
                    'movement': movement_score(generated_text)
                }
                composite = composite_score(scores)
                all_scores.append(composite)
                
                print(f"📊 Scores: {scores}")
                print(f"📈 Composite: {composite:.4f}")
                
            except Exception as e:
                print(f"❌ Error generating sample {i}: {e}")
        
        if all_scores:
            avg_score = np.mean(all_scores)
            print(f"\n📊 Average Composite Score: {avg_score:.4f}")
        
    except Exception as e:
        print(f"❌ Error in Answering Agent generation: {e}")

def test_direct_generation():
    """Test direct generation without preprocessed JSON"""
    print("\n🔍 Testing direct generation...")
    
    try:
        config = Config()
        
        # Load model
        model = AnsweringAgent(config)
        checkpoint_path = config.model.checkpoint_path
        
        print(f"🔍 Looking for checkpoint at: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found at: {checkpoint_path}")
            print("🔍 Checking for alternative checkpoint paths...")
            
            # Try alternative paths
            alt_paths = [
                "checkpoints/best_model.pth",
                "checkpoints/latest_model.pth", 
                "models/checkpoint.pth",
                "../checkpoints/best_model.pth",
                "../models/checkpoint.pth"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    print(f"✅ Found checkpoint at: {checkpoint_path}")
                    break
            else:
                print("❌ No checkpoint found. Please provide a valid checkpoint path.")
                return
        
        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded model state from {checkpoint_path}")
            else:
                model.load_state_dict(checkpoint)
                print(f"✅ Loaded model directly from {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return
        
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        print(f"✅ Model loaded on device: {device}")
        
        # Load dataset
        dataset = AnsweringDataset(config, split='val_seen')
        print(f"✅ Loaded dataset: {len(dataset)} samples")
        
        # Test first few samples
        sample_count = min(3, len(dataset))
        all_scores = []
        
        for i in range(sample_count):
            print(f"\n{'='*60}")
            print(f"🎯 DIRECT GENERATION SAMPLE {i}")
            print(f"{'='*60}")
            
            try:
                sample = dataset[i]
                
                # Construct text_input exactly as in training
                text_input = {
                    'input_ids': sample['text_input']['input_ids'].unsqueeze(0).to(device),
                    'attention_mask': sample['text_input']['attention_mask'].unsqueeze(0).to(device)
                }
                if 'first_instruction_input' in sample:
                    text_input['first_instruction_input'] = {k: v.unsqueeze(0).to(device) for k, v in sample['first_instruction_input'].items()}
                if 'current_question_input' in sample:
                    text_input['current_question_input'] = {k: v.unsqueeze(0).to(device) for k, v in sample['current_question_input'].items()}
                
                # Get visual features
                current_view = sample['current_view_image'].unsqueeze(0).to(device)
                previous_views = sample['previous_views_image'].unsqueeze(0).to(device)
                destination = sample['destination_image'].unsqueeze(0).to(device)
                
                # Generate answer
                with torch.no_grad():
                    generated_ids = model.generate_answer(
                        text_input=text_input,
                        current_view=current_view,
                        previous_views=previous_views,
                        destination=destination
                    )
                
                # Decode generated text
                from transformers import T5Tokenizer
                tokenizer = T5Tokenizer.from_pretrained('t5-base')
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Decode original text
                original_text = tokenizer.decode(sample['text_label'], skip_special_tokens=True)
                
                print(f"🔄 Original: {original_text}")
                print(f"🎯 Generated: {generated_text}")
                
                # Calculate scores
                scores = {
                    'direction': direction_score(generated_text),
                    'yesno': yesno_score(generated_text),
                    'attribute': attribute_score(generated_text),
                    'landmark': landmark_score(generated_text),
                    'movement': movement_score(generated_text)
                }
                composite = composite_score(scores)
                all_scores.append(composite)
                
                print(f"📊 Scores: {scores}")
                print(f"📈 Composite: {composite:.4f}")
                
            except Exception as e:
                print(f"❌ Error generating sample {i}: {e}")
        
        if all_scores:
            avg_score = np.mean(all_scores)
            print(f"\n📊 Average Composite Score: {avg_score:.4f}")
        
    except Exception as e:
        print(f"❌ Error in direct generation: {e}")

def test_small_generation():
    """Test small generation with AVDN data"""
    print("\n🔍 Testing small generation with AVDN data...")
    
    try:
        # Load AVDN data
        avdn_file = "/app/UAV-Language-Guided-Navigation/Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/val_seen_data.json"
        if not os.path.exists(avdn_file):
            avdn_file = "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/val_seen_data.json"
        
        if not os.path.exists(avdn_file):
            print(f"❌ AVDN file not found: {avdn_file}")
            return
        
        with open(avdn_file, 'r') as f:
            avdn_data = json.load(f)
        
        print(f"✅ Loaded AVDN data: {len(avdn_data)} samples")
        
        # Print first few samples with instruction parsing
        for i in range(min(3, len(avdn_data))):
            sample = avdn_data[i]
            print(f"\n📋 AVDN Sample {i}:")
            print(f"  Map: {sample.get('map_name', 'N/A')}")
            print(f"  Route Index: {sample.get('route_index', 'N/A')}")
            print(f"  Last Round: {sample.get('last_round_idx', 'N/A')}")
            
            instructions = sample.get('instructions', '')
            print(f"  Raw Instructions: {instructions}")
            
            # Parse instructions
            if '[QUE]' in instructions and '[INS]' in instructions:
                que_start = instructions.find('[QUE]')
                ins_start = instructions.find('[INS]')
                question = instructions[que_start+5:ins_start].strip()
                instruction = instructions[ins_start+5:].strip()
                print(f"  📝 Question: {question}")
                print(f"  📝 Instruction: {instruction}")
            elif '[INS]' in instructions:
                instruction = instructions.replace('[INS]', '').strip()
                print(f"  📝 First Instruction: {instruction}")
                print(f"  📝 Question: None (first instruction)")
            else:
                print(f"  📝 Unrecognized format: {instructions}")
            
    except Exception as e:
        print(f"❌ Error in small generation test: {e}")

if __name__ == "__main__":
    print("🧪 Starting AVDN Generation Tests...")
    
    # Test AVDN structure
    test_avdn_structure()
    
    # Test formatted dataset structure
    test_formatted_dataset_structure()
    
    # Test instruction comparison
    test_instruction_comparison()
    
    # Test Answering Agent generation
    test_answering_agent_generation()
    
    # Test direct generation
    test_direct_generation()
    
    # Test small generation
    test_small_generation()
    
    print("\n✅ All tests completed!") 