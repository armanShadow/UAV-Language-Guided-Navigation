#!/usr/bin/env python3
"""
Comprehensive test script for enhanced contrastive learning improvements.
Tests both spatial-aware positive generation and enhanced spatial negative generation
using real AVDN dataset samples.
"""

import sys
import os
import logging
import json
import random

# Add the AnsweringAgent path to sys.path
sys.path.append('AnsweringAgent')
sys.path.append('../')

from contrastive_sample_generator import ContrastiveSampleGenerator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset_samples(num_samples=6):
    """Load diverse real samples from the dataset for comprehensive testing."""
    
    # Try to load train data from common paths
    train_data_paths = [
        "processed_data/train_data.json",
        "../processed_data/train_data.json",
        "../../processed_data/train_data.json"
    ]
    
    data = None
    for path in train_data_paths:
        if os.path.exists(path):
            logger.info(f"Loading dataset from: {path}")
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                break
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
                continue
    
    if not data:
        logger.warning(f"Could not load dataset from any path: {train_data_paths}")
        # Use fallback samples with diverse spatial patterns
        return get_comprehensive_fallback_samples()
    
    logger.info(f"Loaded dataset with {len(data)} episodes")
    
    # Extract all meaningful dialog turns
    all_samples = []
    
    for episode_idx, episode in enumerate(data):
        # Extract first instruction
        first_instruction = episode.get("first_instruction", "").strip()
        if first_instruction and len(first_instruction) > 15:
            all_samples.append({
                "text": first_instruction,
                "source": f"Episode {episode_idx} - First Instruction",
                "episode_id": episode.get("episode_id", f"ep_{episode_idx}")
            })
        
        # Extract dialog answers
        for dialog_idx, dialog in enumerate(episode.get("dialogs", [])):
            answer = dialog.get("answer", "")
            question = dialog.get("question", "")
            
            if answer and len(answer.strip()) > 15:
                all_samples.append({
                    "text": answer.strip(),
                    "source": f"Episode {episode_idx} - Dialog {dialog_idx}",
                    "episode_id": episode.get("episode_id", f"ep_{episode_idx}"),
                    "question": question.strip() if question else None
                })
    
    logger.info(f"Extracted {len(all_samples)} total dialog samples")
    
    # Sample diverse examples with different spatial characteristics
    random.seed(42)  # For reproducible results
    
    # Categorize samples by spatial content
    clock_samples = []
    direction_samples = []
    landmark_samples = []
    multi_element_samples = []
    
    for sample in all_samples:
        text_lower = sample["text"].lower()
        
        # Categorize by spatial content
        has_clock = any(f"{i} o'clock" in text_lower or f"{i}:30" in text_lower for i in range(1, 13))
        has_direction = any(direction in text_lower for direction in ["north", "south", "east", "west", "left", "right", "forward", "turn"])
        has_landmark = any(landmark in text_lower for landmark in ["building", "container", "road", "tree", "house", "structure", "parking", "field"])
        has_color = any(color in text_lower for color in ["white", "gray", "grey", "red", "blue", "green", "black", "brown"])
        
        # Count spatial elements
        spatial_count = sum([has_clock, has_direction, has_landmark, has_color])
        
        if has_clock:
            clock_samples.append(sample)
        if has_direction:
            direction_samples.append(sample)
        if has_landmark:
            landmark_samples.append(sample)
        if spatial_count >= 3:
            multi_element_samples.append(sample)
    
    logger.info(f"Found {len(clock_samples)} samples with clock directions")
    logger.info(f"Found {len(direction_samples)} samples with directions")
    logger.info(f"Found {len(landmark_samples)} samples with landmarks")
    logger.info(f"Found {len(multi_element_samples)} samples with multiple spatial elements")
    
    # Select diverse samples
    selected_samples = []
    
    # Prioritize samples with multiple spatial elements
    if multi_element_samples:
        selected_samples.extend(random.sample(multi_element_samples, min(2, len(multi_element_samples))))
    
    # Add clock direction samples (UAV-specific)
    if clock_samples:
        remaining_clock = [s for s in clock_samples if s not in selected_samples]
        selected_samples.extend(random.sample(remaining_clock, min(2, len(remaining_clock))))
    
    # Add direction samples
    if direction_samples:
        remaining_direction = [s for s in direction_samples if s not in selected_samples]
        selected_samples.extend(random.sample(remaining_direction, min(1, len(remaining_direction))))
    
    # Add landmark samples
    if landmark_samples:
        remaining_landmark = [s for s in landmark_samples if s not in selected_samples]
        selected_samples.extend(random.sample(remaining_landmark, min(1, len(remaining_landmark))))
    
    return selected_samples[:num_samples]

def get_comprehensive_fallback_samples():
    """Comprehensive fallback samples covering all spatial patterns we want to test."""
    return [
        {
            "text": "Head forward towards 6 o'clock direction. After passing a road and few buildings, the destination is a white building.",
            "source": "AVDN-style Sample 1 (Clock + Multi-element)",
            "episode_id": "test_1",
            "description": "Clock direction + landmarks + colors + multi-step"
        },
        {
            "text": "Turn right and proceed straight forward passing the parking lot.",
            "source": "AVDN-style Sample 2 (Action + Direction)",
            "episode_id": "test_2", 
            "description": "Action verbs + directions + landmarks"
        },
        {
            "text": "Go north to the large gray building.",
            "source": "AVDN-style Sample 3 (Cardinal + Descriptors)",
            "episode_id": "test_3",
            "description": "Cardinal direction + size + color + landmark"
        },
        {
            "text": "Move towards your 3:30. You will reach a red structure.",
            "source": "AVDN-style Sample 4 (Time Format)",
            "episode_id": "test_4",
            "description": "Time format + color + landmark + future tense"
        },
        {
            "text": "The destination is near the highway.",
            "source": "AVDN-style Sample 5 (Spatial Relations)",
            "episode_id": "test_5",
            "description": "Spatial relation + landmark"
        },
        {
            "text": "Fly over the field and turn left at the intersection.",
            "source": "AVDN-style Sample 6 (Multi-step)",
            "episode_id": "test_6",
            "description": "Multi-step with spatial relations + UAV-specific verb"
        },
        {
            "text": "Navigate to the building next to the parking area.",
            "source": "AVDN-style Sample 7 (Multiple Landmarks)",
            "episode_id": "test_7",
            "description": "Multiple landmarks with spatial relation"
        },
        {
            "text": "Head towards 12 o'clock direction and you will see the target.",
            "source": "AVDN-style Sample 8 (Clock + Future)",
            "episode_id": "test_8",
            "description": "Clock direction + future tense + target reference"
        }
    ]

def analyze_spatial_content(text):
    """Analyze and categorize the spatial content of a text sample."""
    text_lower = text.lower()
    
    spatial_features = {
        "clock_directions": [],
        "cardinal_directions": [],
        "relative_directions": [],
        "landmarks": [],
        "colors": [],
        "spatial_relations": [],
        "action_verbs": []
    }
    
    # Clock directions
    for i in range(1, 13):
        if f"{i} o'clock" in text_lower:
            spatial_features["clock_directions"].append(f"{i} o'clock")
        if f"{i}:30" in text_lower:
            spatial_features["clock_directions"].append(f"{i}:30")
    
    # Cardinal directions
    cardinals = ["north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"]
    for direction in cardinals:
        if direction in text_lower:
            spatial_features["cardinal_directions"].append(direction)
    
    # Relative directions
    relatives = ["left", "right", "forward", "backward", "ahead", "behind", "straight"]
    for direction in relatives:
        if direction in text_lower:
            spatial_features["relative_directions"].append(direction)
    
    # Action verbs
    actions = ["turn", "move", "head", "go", "proceed", "fly", "navigate", "travel"]
    for action in actions:
        if action in text_lower:
            spatial_features["action_verbs"].append(action)
    
    # Landmarks
    landmarks = ["building", "container", "road", "tree", "house", "structure", "stadium", "field", "parking", "intersection", "highway"]
    for landmark in landmarks:
        if landmark in text_lower:
            spatial_features["landmarks"].append(landmark)
    
    # Colors
    colors = ["red", "blue", "green", "white", "black", "gray", "grey", "brown", "yellow"]
    for color in colors:
        if color in text_lower:
            spatial_features["colors"].append(color)
    
    # Spatial relations
    relations = ["near", "behind", "in front", "next to", "above", "below", "between", "across", "over", "under"]
    for relation in relations:
        if relation in text_lower:
            spatial_features["spatial_relations"].append(relation)
    
    return spatial_features

def test_comprehensive_contrastive():
    """Test comprehensive contrastive learning with both positives and negatives."""
    
    print("="*100)
    print("COMPREHENSIVE CONTRASTIVE LEARNING TEST")
    print("Testing Enhanced Spatial Negatives + Spatial-Aware Positives")
    print("="*100)
    
    # Load real dataset samples
    samples = load_dataset_samples(num_samples=6)
    
    if not samples:
        logger.error("No samples loaded")
        return
    
    # Initialize generator
    try:
        generator = ContrastiveSampleGenerator(device="cpu")  # Use CPU for container compatibility
        logger.info("Successfully initialized ContrastiveSampleGenerator")
    except Exception as e:
        logger.error(f"Error initializing generator: {e}")
        return
    
    for i, sample in enumerate(samples, 1):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i}: {sample['source']}")
        print(f"Episode ID: {sample['episode_id']}")
        print(f"{'='*80}")
        print(f"Original: {sample['text']}")
        
        if sample.get('question'):
            print(f"Context Question: {sample['question']}")
        
        # Analyze spatial content
        spatial_analysis = analyze_spatial_content(sample['text'])
        print(f"\nSPATIAL CONTENT ANALYSIS:")
        for feature_type, features in spatial_analysis.items():
            if features:
                print(f"  {feature_type.replace('_', ' ').title()}: {features}")
        
        # Extract spatial information using generator
        print(f"\nGENERATOR SPATIAL EXTRACTION:")
        spatial_info = generator._extract_navigation_info(sample['text'])
        for key, values in spatial_info.items():
            if values:
                print(f"  {key.replace('_', ' ').title()}: {values}")
        
        # ===== SPATIAL-AWARE POSITIVES =====
        print(f"\n🟢 SPATIAL-AWARE POSITIVES:")
        try:
            positives = generator.generate_positive_examples(sample['text'], n=3)
            
            if positives:
                for j, pos in enumerate(positives, 1):
                    print(f"\n  Positive {j}:")
                    print(f"    Text: {pos['text']}")
                    print(f"    Similarity: {pos['similarity']:.3f}")
                    print(f"    Type: {pos['type']}")
                    
                    # Show specific changes
                    if pos['type'] == 'spatial_synonym' and 'substitution' in pos:
                        print(f"    Change: {pos['substitution']}")
                    elif pos['type'] == 'spatial_structure' and 'transformation' in pos:
                        print(f"    Transform: {pos['transformation']}")
                    elif pos['type'] == 'clock_format_variation':
                        print(f"    Format: {pos.get('original_format', 'N/A')} → {pos.get('new_format', 'N/A')}")
            else:
                print("  No spatial-aware positives generated")
                
        except Exception as e:
            logger.error(f"Error generating spatial-aware positives: {e}")
        
        # ===== ENHANCED SPATIAL NEGATIVES =====
        print(f"\n🔴 ENHANCED SPATIAL NEGATIVES:")
        try:
            # Use the public API method instead of internal method
            negatives = generator.generate_negative_examples(sample['text'], n=4)
            
            if negatives:
                for j, neg in enumerate(negatives, 1):
                    print(f"\n  Negative {j}:")
                    print(f"    Text: {neg['text']}")
                    print(f"    Similarity: {neg['similarity']:.3f}")
                    print(f"    Type: {neg['type']}")
                    
                    # Show specific changes based on type
                    if neg['type'] == 'clock_shift':
                        print(f"    Change: {neg.get('original_hour', 'N/A')} o'clock → {neg.get('new_hour', 'N/A')} o'clock ({neg.get('shift_degrees', 0)}°)")
                    elif neg['type'] == 'enhanced_direction_reversal':
                        print(f"    Change: {neg.get('original_direction', 'N/A')} → {neg.get('new_direction', 'N/A')}")
                    elif neg['type'] == 'contextual_landmark':
                        print(f"    Change: {neg.get('original_landmark', 'N/A')} → {neg.get('new_landmark', 'N/A')}")
                    elif neg['type'] == 'spatial_relation_perturbation':
                        print(f"    Change: {neg.get('original_relation', 'N/A')} → {neg.get('new_relation', 'N/A')}")
            else:
                print("    No enhanced spatial negatives generated for this sample")
                
        except Exception as e:
            logger.error(f"Error generating enhanced spatial negatives: {e}")
            print(f"    ❌ Error: {e}")
            
        print("=" * 80)
        
        # ===== STRATEGY BREAKDOWN =====
        print(f"\n📊 STRATEGY BREAKDOWN:")
        
        # Count positive strategies
        positive_types = {}
        for pos in positives:
            pos_type = pos.get('type', 'unknown')
            positive_types[pos_type] = positive_types.get(pos_type, 0) + 1
        
        print(f"\n  Positive Strategies:")
        strategy_mapping = {
            'spatial_synonym': 'Spatial Synonyms',
            'spatial_structure': 'Structure Variations', 
            'spatial_elaboration': 'Spatial Elaborations',
            'clock_format_variation': 'Clock Format Variations',
            'simple_variation': 'Simple Variations'
        }
        
        for strategy_type, strategy_name in strategy_mapping.items():
            count = positive_types.get(strategy_type, 0)
            print(f"    {strategy_name}: {count} generated")
        
        # Count negative strategies
        negative_types = {}
        for neg in negatives:
            neg_type = neg.get('type', 'unknown')
            negative_types[neg_type] = negative_types.get(neg_type, 0) + 1
        
        print(f"\n  Negative Strategies:")
        neg_strategy_mapping = {
            'clock_shift': 'Clock Shift Negatives',
            'enhanced_direction_reversal': 'Direction Negatives',
            'contextual_landmark': 'Landmark Negatives',
            'spatial_relation_perturbation': 'Spatial Relation Negatives'
        }
        
        for strategy_type, strategy_name in neg_strategy_mapping.items():
            count = negative_types.get(strategy_type, 0)
            print(f"    {strategy_name}: {count} generated")
        
        # ===== QUALITY METRICS =====
        print(f"\n📈 QUALITY METRICS:")
        try:
            # Calculate average similarities
            if positives:
                avg_pos_sim = sum(p['similarity'] for p in positives) / len(positives)
                print(f"    Average Positive Similarity: {avg_pos_sim:.3f}")
            
            if negatives:
                avg_neg_sim = sum(n['similarity'] for n in negatives) / len(negatives)
                print(f"    Average Negative Similarity: {avg_neg_sim:.3f}")
                
                # Check similarity distribution
                high_sim_negs = [n for n in negatives if n['similarity'] > 0.6]
                med_sim_negs = [n for n in negatives if 0.3 <= n['similarity'] <= 0.6]
                low_sim_negs = [n for n in negatives if n['similarity'] < 0.3]
                
                print(f"    Hard Negatives (>0.6): {len(high_sim_negs)}")
                print(f"    Medium Negatives (0.3-0.6): {len(med_sim_negs)}")
                print(f"    Easy Negatives (<0.3): {len(low_sim_negs)}")
        except Exception as e:
            print(f"    Quality Metrics Error: {e}")
    
    print(f"\n{'='*100}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*100}")
    print(f"Total samples tested: {len(samples)}")
    print("\n✅ FEATURES TESTED:")
    print("🟢 Spatial-Aware Positives:")
    print("   • Spatial synonym substitution (preserves spatial semantics)")
    print("   • Spatial structure variation (same info, different structure)")
    print("   • Spatial elaboration (adds compatible spatial context)")
    print("   • Clock direction format variations (preserves exact direction)")
    print("\n🔴 Enhanced Spatial Negatives:")
    print("   • Clock direction shifting (UAV-specific 90°/180°/270° rotations)")
    print("   • Enhanced direction reversal (multiple spatial alternatives)")
    print("   • Contextual landmark substitution (frequency-based)")
    print("   • Multi-element spatial perturbation (combines 2-3 changes)")
    print("\n📊 Quality Control:")
    print("   • Similarity filtering and ranking")
    print("   • Diversity enforcement")
    print("   • Dataset-driven terminology")
    print("   • Real AVDN sample testing")
    print(f"{'='*100}")

if __name__ == "__main__":
    test_comprehensive_contrastive() 