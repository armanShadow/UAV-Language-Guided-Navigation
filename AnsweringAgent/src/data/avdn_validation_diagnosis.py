#!/usr/bin/env python3
"""
AVDN Dataset Validation Diagnosis
Test validation framework with actual AVDN dataset instructions.
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict
from validation_pipeline import ValidationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def load_random_avdn_examples(num_examples: int = 10) -> List[str]:
    """Load random examples from the AVDN dataset."""
    dataset_paths = [
        "processed_data/train_data.json",
        "src/data/processed_data/train_data.json",
        "AnsweringAgent/src/data/processed_data/train_data.json",
        "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json",
        "../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json",
        "../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json",
        "../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json"
    ]
    
    for path in dataset_paths:
        if Path(path).exists():
            logger.info(f"📂 Loading dataset from: {path}")
            try:
                with open(path, 'r') as f:
                    episodes = json.load(f)
                
                # Extract all answers from dialogs
                all_instructions = []
                for episode in episodes:
                    dialogs = episode.get('dialogs', [])
                    for dialog in dialogs:
                        if dialog and dialog.get('answer'):
                            answer = dialog['answer'].strip()
                            if answer and len(answer.split()) >= 3:
                                all_instructions.append(answer)
                
                # Get random examples
                if len(all_instructions) >= num_examples:
                    random.seed()  # True randomness
                    random_examples = random.sample(all_instructions, num_examples)
                    logger.info(f"📊 Selected {num_examples} random examples from {len(all_instructions)} total")
                    return random_examples
                else:
                    logger.warning(f"Only {len(all_instructions)} examples available, returning all")
                    return all_instructions
                    
            except Exception as e:
                logger.error(f"Error loading dataset from {path}: {e}")
                continue
    
    logger.error("❌ Could not find AVDN dataset. Using fallback examples.")
    return [
        "Turn right and fly over the white building at 3 o'clock",
        "Go straight ahead towards the gray road near the parking area",
        "Navigate to the brown house at 6 o'clock position",
        "Fly north over the highway and turn left at the intersection"
    ]

def create_simple_paraphrases(instruction: str) -> Dict[str, List[str]]:
    """Create simple positive and negative paraphrases for testing."""
    # Simple positive paraphrases (preserve meaning)
    positives = []
    negatives = []
    
    # Basic synonym replacements for positives
    positive_replacements = {
        'go': 'move',
        'turn': 'rotate',
        'fly': 'navigate',
        'head': 'proceed',
        'straight': 'forward',
        'building': 'structure',
        'house': 'building',
        'road': 'street'
    }
    
    # Create positive by replacing words
    positive = instruction.lower()
    for original, replacement in positive_replacements.items():
        if original in positive:
            positive = positive.replace(original, replacement, 1)
            break
    positives.append(positive.capitalize())
    
    # Create another positive with different phrasing
    if 'turn' in instruction.lower():
        positives.append(instruction.replace('turn', 'make a turn'))
    elif 'go' in instruction.lower():
        positives.append(instruction.replace('go', 'proceed'))
    else:
        positives.append(f"Head {instruction.lower().replace('head', '').strip()}")
    
    # Create negatives by changing spatial elements
    negative = instruction.lower()
    
    # Direction changes
    direction_changes = {
        'left': 'right',
        'right': 'left',
        'north': 'south',
        'south': 'north',
        'east': 'west',
        'west': 'east',
        'forward': 'backward',
        'straight': 'backward'
    }
    
    for original, replacement in direction_changes.items():
        if original in negative:
            negative = negative.replace(original, replacement, 1)
            break
    
    # Clock direction changes
    import re
    clock_match = re.search(r'(\d+)\s*o\'?clock', negative)
    if clock_match:
        hour = int(clock_match.group(1))
        opposite_hour = (hour + 6) % 12
        if opposite_hour == 0:
            opposite_hour = 12
        negative = re.sub(r'\d+\s*o\'?clock', f'{opposite_hour} o\'clock', negative)
    
    negatives.append(negative.capitalize())
    
    # Create another negative with landmark changes
    landmark_changes = {
        'white': 'black',
        'black': 'white',
        'red': 'blue',
        'blue': 'red',
        'building': 'house',
        'house': 'building',
        'road': 'path'
    }
    
    negative2 = instruction.lower()
    for original, replacement in landmark_changes.items():
        if original in negative2:
            negative2 = negative2.replace(original, replacement, 1)
            break
    negatives.append(negative2.capitalize())
    
    return {
        'positives': positives[:2],  # Take first 2
        'negatives': negatives[:2]   # Take first 2
    }

def analyze_instruction_validation(val_pipeline: ValidationPipeline, instruction: str, paraphrases: Dict[str, List[str]]) -> Dict:
    """Analyze validation results for a single instruction."""
    results = {
        'instruction': instruction,
        'positive_results': [],
        'negative_results': [],
        'spatial_features': val_pipeline.extract_spatial_features(instruction)
    }
    
    print(f"\n🔍 ANALYZING INSTRUCTION:")
    print(f"Original: {instruction}")
    print(f"Spatial features: {results['spatial_features']}")
    
    # Test positive paraphrases
    print(f"\n✅ POSITIVE PARAPHRASES:")
    for i, positive in enumerate(paraphrases['positives']):
        print(f"  {i+1}. {positive}")
        
        pos_result = val_pipeline.validate_positive_paraphrase(instruction, positive)
        
        # Compute similarities for display
        embedding_sim = val_pipeline.compute_embedding_similarity(instruction, positive)
        orig_dirs = results['spatial_features'].get('directions', [])
        pos_dirs = val_pipeline.extract_spatial_features(positive).get('directions', [])
        direction_sim = val_pipeline._compute_direction_similarity(orig_dirs, pos_dirs)
        
        print(f"     Embedding: {embedding_sim:.3f}, Direction: {direction_sim:.3f}, Valid: {pos_result['is_valid']}")
        
        pos_result['text'] = positive
        pos_result['embedding_similarity'] = embedding_sim
        results['positive_results'].append(pos_result)
    
    # Test negative paraphrases
    print(f"\n❌ NEGATIVE PARAPHRASES:")
    for i, negative in enumerate(paraphrases['negatives']):
        print(f"  {i+1}. {negative}")
        
        neg_result = val_pipeline.validate_negative_paraphrase(instruction, negative)
        
        # Compute similarities for display
        embedding_sim = val_pipeline.compute_embedding_similarity(instruction, negative)
        orig_dirs = results['spatial_features'].get('directions', [])
        neg_dirs = val_pipeline.extract_spatial_features(negative).get('directions', [])
        direction_sim = val_pipeline._compute_direction_similarity(orig_dirs, neg_dirs)
        
        print(f"     Embedding: {embedding_sim:.3f}, Direction: {direction_sim:.3f}, Valid: {neg_result['is_valid']}")
        
        neg_result['text'] = negative
        neg_result['embedding_similarity'] = embedding_sim
        results['negative_results'].append(neg_result)
    
    return results

def test_avdn_validation_framework():
    """Test validation framework with actual AVDN dataset instructions."""
    print("🧪 AVDN DATASET VALIDATION DIAGNOSIS")
    print("=" * 60)
    
    # Load actual AVDN instructions
    avdn_instructions = load_random_avdn_examples(num_examples=8)
    
    if not avdn_instructions:
        print("❌ No AVDN instructions loaded. Exiting.")
        return
    
    print(f"📋 Testing with {len(avdn_instructions)} AVDN instructions")
    
    # Initialize validation pipeline
    val_pipeline = ValidationPipeline()
    
    # Test each instruction
    all_results = []
    positive_valid = 0
    positive_total = 0
    negative_valid = 0
    negative_total = 0
    
    for i, instruction in enumerate(avdn_instructions, 1):
        print(f"\n🎯 TEST CASE {i}/{len(avdn_instructions)}")
        print("=" * 40)
        
        # Create paraphrases
        paraphrases = create_simple_paraphrases(instruction)
        
        # Analyze validation
        result = analyze_instruction_validation(val_pipeline, instruction, paraphrases)
        all_results.append(result)
        
        # Update counters
        for pos_result in result['positive_results']:
            positive_total += 1
            if pos_result['is_valid']:
                positive_valid += 1
        
        for neg_result in result['negative_results']:
            negative_total += 1
            if neg_result['is_valid']:
                negative_valid += 1
    
    # Overall summary
    print(f"\n📊 OVERALL AVDN VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Instructions tested: {len(avdn_instructions)}")
    print(f"Positive validation: {positive_valid}/{positive_total} ({positive_valid/positive_total*100:.1f}%)")
    print(f"Negative validation: {negative_valid}/{negative_total} ({negative_valid/negative_total*100:.1f}%)")
    
    # Detailed failure analysis
    print(f"\n🔍 DETAILED FAILURE ANALYSIS")
    print("=" * 30)
    
    failed_positives = []
    failed_negatives = []
    
    for result in all_results:
        for pos_result in result['positive_results']:
            if not pos_result['is_valid']:
                failed_positives.append({
                    'instruction': result['instruction'],
                    'paraphrase': pos_result['text'],
                    'embedding_sim': pos_result['embedding_similarity'],
                    'direction_sim': pos_result['direction_similarity']
                })
        
        for neg_result in result['negative_results']:
            if not neg_result['is_valid']:
                failed_negatives.append({
                    'instruction': result['instruction'],
                    'paraphrase': neg_result['text'],
                    'embedding_sim': neg_result['embedding_similarity'],
                    'direction_sim': neg_result['direction_similarity']
                })
    
    if failed_positives:
        print(f"\n❌ FAILED POSITIVES ({len(failed_positives)}):")
        for fail in failed_positives[:3]:  # Show first 3
            print(f"  Original: {fail['instruction']}")
            print(f"  Positive: {fail['paraphrase']}")
            print(f"  Embedding: {fail['embedding_sim']:.3f}, Direction: {fail['direction_sim']:.3f}")
            print()
    
    if failed_negatives:
        print(f"\n❌ FAILED NEGATIVES ({len(failed_negatives)}):")
        for fail in failed_negatives[:3]:  # Show first 3
            print(f"  Original: {fail['instruction']}")
            print(f"  Negative: {fail['paraphrase']}")
            print(f"  Embedding: {fail['embedding_sim']:.3f}, Direction: {fail['direction_sim']:.3f}")
            print()
    
    # Threshold recommendations
    print(f"\n💡 THRESHOLD RECOMMENDATIONS")
    print("=" * 30)
    
    pos_rate = positive_valid / positive_total if positive_total > 0 else 0
    neg_rate = negative_valid / negative_total if negative_total > 0 else 0
    
    print("Current thresholds:")
    print("  Positive: embedding > 0.75, direction > 0.8 OR landmark > 0.7")
    print("  Negative: 0.3 < embedding < 0.95, direction < 0.7 OR landmark < 0.7")
    print()
    
    if pos_rate < 0.7:
        print("🔧 Positive validation too strict:")
        print("  - Consider lowering embedding threshold from 0.75 to 0.7")
        print("  - Consider lowering direction threshold from 0.8 to 0.75")
    
    if neg_rate < 0.7:
        print("🔧 Negative validation too strict:")
        print("  - Consider raising direction change threshold from 0.7 to 0.75")
        print("  - Consider adjusting embedding range from 0.3-0.95 to 0.25-0.97")
    
    if pos_rate >= 0.8 and neg_rate >= 0.8:
        print("✅ Thresholds appear well-calibrated for AVDN dataset!")
    elif pos_rate >= 0.8 and neg_rate >= 0.6:
        print("✅ Thresholds are reasonably calibrated for AVDN dataset!")
        print("    Positive validation excellent, negative validation acceptable.")
    
    return all_results

if __name__ == "__main__":
    results = test_avdn_validation_framework() 