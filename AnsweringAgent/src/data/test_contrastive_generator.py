#!/usr/bin/env python3
"""
Test script for the updated contrastive sample generator.
This script loads a random dialog turn from train_data.json and tests the new approach.
"""

import json
import random
import sys
import os
import logging

# Add the AnsweringAgent path to sys.path
sys.path.append('AnsweringAgent')

from contrastive_sample_generator import ContrastiveSampleGenerator
# Import config to use proper paths
sys.path.append('../')
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_train_data(file_path):
    """Load train data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} episodes from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading train data: {e}")
        return None

def extract_dialog_turns(data):
    """Extract all dialog turns from the dataset."""
    dialog_turns = []
    
    for episode_idx, episode in enumerate(data):
        # Handle AVDN format
        if "instructions" in episode:
            instruction = episode["instructions"]
            if isinstance(instruction, str) and "[INS]" in instruction:
                # Extract instruction part
                ins_start = instruction.find("[INS]")
                if ins_start >= 0:
                    answer = instruction[ins_start+5:].strip()
                    if answer:
                        dialog_turns.append({
                            "episode_idx": episode_idx,
                            "question": None,
                            "answer": answer,
                            "type": "main_instruction"
                        })
        
        # Handle pre_dialogs
        if "pre_dialogs" in episode and isinstance(episode["pre_dialogs"], list):
            for dialog_idx, dialog in enumerate(episode["pre_dialogs"]):
                if isinstance(dialog, str) and "[INS]" in dialog:
                    ins_start = dialog.find("[INS]")
                    if ins_start >= 0:
                        answer = dialog[ins_start+5:].strip()
                        if answer:
                            dialog_turns.append({
                                "episode_idx": episode_idx,
                                "dialog_idx": dialog_idx,
                                "question": None,
                                "answer": answer,
                                "type": "pre_dialog"
                            })
        
        # Handle standard format with dialogs list
        if "dialogs" in episode:
            for dialog_idx, dialog in enumerate(episode["dialogs"]):
                if dialog.get("answer"):
                    dialog_turns.append({
                        "episode_idx": episode_idx,
                        "dialog_idx": dialog_idx,
                        "question": dialog.get("question"),
                        "answer": dialog["answer"],
                        "type": "standard_dialog"
                    })
    
    logger.info(f"Extracted {len(dialog_turns)} dialog turns")
    return dialog_turns

def collect_episode_answers(data, target_episode_idx):
    """Collect all answers from a specific episode to use as alternative answers."""
    episode_answers = []
    episode = data[target_episode_idx]
    
    # Collect from main instructions
    if "instructions" in episode:
        instruction = episode["instructions"]
        if isinstance(instruction, str) and "[INS]" in instruction:
            ins_start = instruction.find("[INS]")
            if ins_start >= 0:
                answer = instruction[ins_start+5:].strip()
                if answer:
                    episode_answers.append(answer)
    
    # Collect from pre_dialogs
    if "pre_dialogs" in episode and isinstance(episode["pre_dialogs"], list):
        for dialog in episode["pre_dialogs"]:
            if isinstance(dialog, str) and "[INS]" in dialog:
                ins_start = dialog.find("[INS]")
                if ins_start >= 0:
                    answer = dialog[ins_start+5:].strip()
                    if answer:
                        episode_answers.append(answer)
    
    # Collect from standard dialogs
    if "dialogs" in episode:
        for dialog in episode["dialogs"]:
            if dialog.get("answer"):
                episode_answers.append(dialog["answer"])
    
    return episode_answers

def test_contrastive_generator():
    """Test the contrastive sample generator with a random dialog turn."""
    
    # Initialize config to get proper paths
    config = Config()
    
    # Try to load train data from config paths
    train_data_paths = [
        config.data.train_json_path,
        "/app/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/processed_data/train_data.json"  # Alternative AVDN path
    ]
    
    data = None
    for path in train_data_paths:
        if os.path.exists(path):
            logger.info(f"Found train data at: {path}")
            data = load_train_data(path)
            if data:
                break
    
    if not data:
        logger.error(f"Could not load train data from any of the expected locations: {train_data_paths}")
        return
    
    # Extract all dialog turns
    dialog_turns = extract_dialog_turns(data)
    
    if not dialog_turns:
        logger.error("No dialog turns found in the data")
        return
    
    # Select a random dialog turn
    random_turn = random.choice(dialog_turns)
    logger.info(f"Selected random dialog turn from episode {random_turn['episode_idx']}")
    
    print("="*80)
    print("SELECTED DIALOG TURN:")
    print(f"Episode: {random_turn['episode_idx']}")
    print(f"Type: {random_turn['type']}")
    if random_turn['question']:
        print(f"Question: {random_turn['question']}")
    print(f"Answer: {random_turn['answer']}")
    print("="*80)
    
    # Collect alternative answers from other episodes for negative generation
    alternative_answers = []
    other_episodes = [i for i in range(len(data)) if i != random_turn['episode_idx']]
    sample_episodes = random.sample(other_episodes, min(100, len(other_episodes)))
    
    for ep_idx in sample_episodes:
        ep_answers = collect_episode_answers(data, ep_idx)
        alternative_answers.extend(ep_answers)
    
    # Remove duplicates and the current answer
    alternative_answers = list(set(alternative_answers))
    if random_turn['answer'] in alternative_answers:
        alternative_answers.remove(random_turn['answer'])
    
    logger.info(f"Collected {len(alternative_answers)} alternative answers for negative generation")
    
    # Initialize contrastive sample generator
    try:
        generator = ContrastiveSampleGenerator()
        generator.alternative_answers = alternative_answers
        
        print("\nTesting Contrastive Sample Generator...")
        print("-" * 50)
        
        # Generate positive examples (2 samples with 70-90% similarity)
        print("GENERATING POSITIVE EXAMPLES:")
        positives = generator.generate_positive_examples(random_turn['answer'], n=2)
        
        if positives:
            for i, pos in enumerate(positives, 1):
                print(f"\nPositive Example {i}:")
                print(f"  Text: {pos['text']}")
                print(f"  Similarity: {pos['similarity']:.3f}")
                print(f"  Type: {pos['type']}")
        else:
            print("  No positive examples generated")
        
        # Generate negative examples (3 samples with 40-70% similarity)
        print(f"\nGENERATING NEGATIVE EXAMPLES:")
        negatives = generator.generate_negative_examples(random_turn['answer'], n=3)
        
        if negatives:
            for i, neg in enumerate(negatives, 1):
                print(f"\nNegative Example {i}:")
                print(f"  Text: {neg['text']}")
                print(f"  Similarity: {neg['similarity']:.3f}")
                print(f"  Type: {neg['type']}")
        else:
            print("  No negative examples generated")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY:")
        print(f"Original Answer: {random_turn['answer']}")
        print(f"Positive Examples Generated: {len(positives)}")
        print(f"Negative Examples Generated: {len(negatives)}")
        
        if positives:
            avg_pos_sim = sum(p['similarity'] for p in positives) / len(positives)
            print(f"Average Positive Similarity: {avg_pos_sim:.3f}")
        
        if negatives:
            avg_neg_sim = sum(n['similarity'] for n in negatives) / len(negatives)
            print(f"Average Negative Similarity: {avg_neg_sim:.3f}")
        
        print("="*80)
        
    except Exception as e:
        logger.error(f"Error testing contrastive generator: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_contrastive_generator() 