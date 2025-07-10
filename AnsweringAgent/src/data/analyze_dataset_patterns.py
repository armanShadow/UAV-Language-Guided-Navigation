#!/usr/bin/env python3
"""
Focused AVDN Dataset Pattern Analysis
Extract core patterns from 300 random samples for Strategy 1 expansion
"""

import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

def load_300_random_samples():
    """Load 300 random samples from the AVDN dataset."""
    dataset_paths = [
        "src/data/processed_data/train_data.json",
        "../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json"
    ]
    
    for path in dataset_paths:
        if Path(path).exists():
            print(f"📂 Loading dataset from: {path}")
            with open(path, 'r') as f:
                episodes = json.load(f)
            
            # Extract all answers
            all_samples = []
            for episode in episodes:
                dialogs = episode.get('dialogs', [])
                for dialog in dialogs:
                    if dialog and dialog.get('answer'):
                        answer = dialog['answer'].strip()
                        if answer and len(answer.split()) >= 3:
                            all_samples.append(answer)
            
            # Get 300 random samples
            random.seed(42)  # Consistent results
            if len(all_samples) >= 300:
                random_samples = random.sample(all_samples, 300)
                print(f"📊 Selected 300 random samples from {len(all_samples)} total")
                return random_samples
            else:
                return all_samples
    
    print("❌ Could not find dataset.")
    return []

def extract_avdn_spatial_examples(samples):
    """Extract high-quality AVDN examples for Mixtral few-shot prompting."""
    print("\n🎯 EXTRACTING AVDN SPATIAL EXAMPLES FOR MIXTRAL PROMPTS")
    print("="*60)
    
    # Categories for few-shot examples
    spatial_categories = {
        'clock_directions': [],
        'cardinal_directions': [],
        'spatial_relations': [],
        'landmark_descriptions': [],
        'complex_instructions': []
    }
    
    # Pattern matching for each category
    for sample in samples:
        sample_lower = sample.lower()
        
        # Clock directions (UAV-critical)
        if re.search(r'\d+\s*o\'?clock', sample_lower):
            spatial_categories['clock_directions'].append(sample)
        
        # Cardinal directions
        elif any(direction in sample_lower for direction in ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']):
            spatial_categories['cardinal_directions'].append(sample)
        
        # Spatial relations
        elif any(relation in sample_lower for relation in ['next to', 'in front of', 'behind', 'above', 'below', 'near', 'across from']):
            spatial_categories['spatial_relations'].append(sample)
        
        # Landmark descriptions with colors/shapes
        elif re.search(r'(red|blue|green|white|gray|grey|black|brown)\s+(building|structure|house|container)', sample_lower):
            spatial_categories['landmark_descriptions'].append(sample)
        
        # Complex multi-step instructions
        elif len(sample.split('.')) > 1 or 'then' in sample_lower or 'and' in sample_lower:
            spatial_categories['complex_instructions'].append(sample)
    
    print("\n📋 CATEGORY EXAMPLES FOR MIXTRAL FEW-SHOT PROMPTS:")
    
    # Select best examples from each category
    few_shot_examples = {}
    
    for category, examples in spatial_categories.items():
        if examples:
            # Sort by length and complexity (prefer medium-length, clear examples)
            sorted_examples = sorted(examples, key=lambda x: len(x.split()))
            
            # Select diverse examples from different length ranges
            selected = []
            for example in sorted_examples:
                word_count = len(example.split())
                if 8 <= word_count <= 25 and len(selected) < 3:  # Medium-length, clear examples
                    selected.append(example)
            
            few_shot_examples[category] = selected[:3]  # Top 3 per category
            
            print(f"\n  {category.upper()}:")
            for i, example in enumerate(selected[:3], 1):
                print(f"    {i}. \"{example}\"")
    
    return few_shot_examples

def generate_mixtral_prompts(few_shot_examples):
    """Generate Mixtral-specific prompts using real AVDN examples."""
    print("\n🤖 GENERATED MIXTRAL PROMPTS")
    print("="*60)
    
    # Base prompt template
    base_prompt = """Paraphrase this UAV navigation instruction while preserving ALL spatial information:
- Keep exact directions (clock positions, cardinal directions)
- Maintain all landmarks and their spatial relationships  
- Preserve step order and navigation target
- Use different wording but same meaning

Examples:"""
    
    # Select best examples across categories
    selected_examples = []
    
    for category, examples in few_shot_examples.items():
        if examples:
            selected_examples.extend(examples[:1])  # One from each category
    
    # Create prompt with real AVDN examples
    prompt_with_examples = base_prompt
    
    for i, example in enumerate(selected_examples[:3], 1):
        # Create a simple paraphrase for demonstration
        paraphrase = create_simple_paraphrase(example)
        prompt_with_examples += f"\n\nExample {i}:\nInput: {example}\nOutput: {paraphrase}"
    
    prompt_with_examples += "\n\nNow paraphrase: {original_instruction}"
    
    print("\n📝 COMPLETE MIXTRAL PROMPT:")
    print(prompt_with_examples)
    
    return prompt_with_examples

def create_simple_paraphrase(original):
    """Create a simple paraphrase for demonstration purposes."""
    # Basic transformations for demonstration
    paraphrase = original
    
    # Simple substitutions
    substitutions = {
        'move towards': 'head toward',
        'turn to': 'rotate to',
        'destination': 'target',
        'building': 'structure',
        'you will see': 'you\'ll observe',
        'go to': 'proceed to',
        'looks like': 'appears to be'
    }
    
    for original_term, replacement in substitutions.items():
        if original_term in paraphrase.lower():
            paraphrase = re.sub(original_term, replacement, paraphrase, flags=re.IGNORECASE)
            break  # Only one substitution per example
    
    return paraphrase

def extract_spatial_tokens(samples):
    """Extract critical spatial tokens that must be preserved."""
    print("\n🔒 CRITICAL SPATIAL TOKENS (Must be preserved)")
    print("="*60)
    
    spatial_tokens = {
        'clock_positions': set(),
        'cardinal_directions': set(),
        'spatial_prepositions': set(),
        'landmarks': set()
    }
    
    for sample in samples:
        sample_lower = sample.lower()
        
        # Extract clock positions
        clock_matches = re.findall(r'\d+\s*o\'?clock', sample_lower)
        spatial_tokens['clock_positions'].update(clock_matches)
        
        # Extract cardinal directions
        cardinal_matches = re.findall(r'\b(north|south|east|west|northeast|northwest|southeast|southwest)\b', sample_lower)
        spatial_tokens['cardinal_directions'].update(cardinal_matches)
        
        # Extract spatial prepositions
        spatial_prep_matches = re.findall(r'\b(next to|in front of|behind|above|below|near|across from|over|under)\b', sample_lower)
        spatial_tokens['spatial_prepositions'].update(spatial_prep_matches)
        
        # Extract common landmarks
        landmark_matches = re.findall(r'\b(building|structure|house|container|parking|road|field|tree|tower)\b', sample_lower)
        spatial_tokens['landmarks'].update(landmark_matches)
    
    print("\n📊 TOKEN FREQUENCIES:")
    for category, tokens in spatial_tokens.items():
        if tokens:
            print(f"\n  {category.upper()}:")
            for token in sorted(tokens):
                print(f"    - {token}")
    
    return spatial_tokens

def analyze_core_patterns(samples):
    """Analyze the most frequent and important patterns in 300 samples."""
    print("\n🔍 ANALYZING CORE PATTERNS (300 samples)")
    print("="*60)
    
    # Focus on the most important patterns based on what we saw
    core_patterns = {
        # Clock patterns - very important for UAV
        r"(\d+)\s*o'?clock": "X o'clock",
        r"(\d+):(\d+)": "X:Y time format",
        r"turn\s+to\s+your\s+(\d+)\s*o'?clock": "turn to your X o'clock",
        r"move\s+towards\s+your\s+(\d+)\s*o'?clock": "move towards your X o'clock",
        r"turn\s+on\s+(\d+)\s*o'?clock": "turn on X o'clock",
        
        # Movement patterns
        r"head\s+(\w+)": "head X",
        r"turn\s+(\w+)": "turn X",
        r"move\s+(\w+)": "move X", 
        r"go\s+(\w+)": "go X",
        
        # Destination patterns
        r"destination\s+is\s+(.+)": "destination is X",
        r"your\s+destination\s+is\s+(.+)": "your destination is X",
        r"that'?s\s+your\s+destination": "that's your destination",
        
        # Positional descriptions
        r"you\s+will\s+see\s+(.+)": "you will see X",
        r"looks?\s+like\s+(.+)": "looks like X",
        r"it\s+is\s+(.+)": "it is X",
        
        # Spatial relations
        r"pass\s+(.+)": "pass X",
        r"cross\s+(.+)": "cross X",
        r"over\s+(.+)": "over X",
        r"next\s+to\s+(.+)": "next to X",
        r"in\s+front\s+of\s+(.+)": "in front of X",
        
        # Building descriptions
        r"(\w+)\s+building": "X building",
        r"building\s+with\s+(.+)": "building with X",
        r"structure\s+(.+)": "structure X",
        
        # Distance/proximity
        r"you\s+are\s+(near|close)": "you are near/close",
        r"very\s+close": "very close",
        r"nearly\s+there": "nearly there",
    }
    
    pattern_counts = defaultdict(list)
    
    for sample in samples:
        sample_lower = sample.lower()
        for pattern, description in core_patterns.items():
            matches = re.findall(pattern, sample_lower)
            if matches:
                pattern_counts[description].append((sample, matches))
    
    print("\n📊 CORE PATTERN FREQUENCY:")
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: len(x[1]), reverse=True)
    
    for description, matches in sorted_patterns:
        if len(matches) >= 2:  # Only show patterns that appear at least twice
            print(f"  {description}: {len(matches)} matches")
            # Show top 2 examples
            for i, (sample, match_groups) in enumerate(matches[:2]):
                if i == 0:
                    print(f"    Examples:")
                print(f"      → \"{sample[:70]}{'...' if len(sample) > 70 else ''}\"")

def analyze_key_vocabulary(samples):
    """Analyze key vocabulary for synonyms."""
    print("\n🔍 ANALYZING KEY VOCABULARY")
    print("="*60)
    
    # Count key spatial terms
    key_terms = {
        'movement_verbs': ['turn', 'move', 'go', 'head', 'fly', 'proceed'],
        'direction_words': ['north', 'south', 'east', 'west', 'left', 'right', 'forward', 'straight'],
        'landmark_types': ['building', 'structure', 'house', 'parking', 'road', 'street'],
        'spatial_relations': ['next', 'front', 'behind', 'over', 'under', 'near', 'close'],
        'descriptors': ['large', 'small', 'long', 'rectangular', 'gray', 'grey']
    }
    
    term_counts = defaultdict(Counter)
    
    for sample in samples:
        words = sample.lower().split()
        for category, terms in key_terms.items():
            for word in words:
                if word in terms:
                    term_counts[category][word] += 1
    
    for category, counts in term_counts.items():
        if counts:
            print(f"\n  {category.upper()}:")
            for term, count in counts.most_common(10):
                print(f"    {term}: {count}")

def find_missing_patterns(samples):
    """Find patterns we might have missed in Strategy 1."""
    print("\n🔍 POTENTIAL MISSING PATTERNS")
    print("="*60)
    
    # Look for common sentence starters and structures we might have missed
    sentence_starters = Counter()
    
    for sample in samples:
        # Get first few words
        words = sample.lower().split()
        if len(words) >= 2:
            starter = ' '.join(words[:2])
            sentence_starters[starter] += 1
        if len(words) >= 3:
            starter = ' '.join(words[:3])
            sentence_starters[starter] += 1
    
    print("\n📊 COMMON SENTENCE STARTERS:")
    for starter, count in sentence_starters.most_common(15):
        if count >= 3:  # Only show if appears 3+ times
            print(f"  '{starter}': {count} times")

def generate_strategy1_recommendations(samples):
    """Generate specific recommendations for Strategy 1 based on analysis."""
    print("\n🚀 STRATEGY 1 RECOMMENDATIONS")
    print("="*60)
    
    print("📋 HIGH-PRIORITY PATTERNS TO ADD:")
    
    # Analyze most common patterns not yet covered
    high_priority = []
    
    # Check for clock direction variations
    clock_samples = [s for s in samples if re.search(r'\d+\s*o\'?clock', s.lower())]
    if clock_samples:
        high_priority.append(f"Clock directions: {len(clock_samples)} samples")
    
    # Check for informal language
    informal_samples = [s for s in samples if any(word in s.lower() for word in ['you are', 'you will', 'you should'])]
    if informal_samples:
        high_priority.append(f"Informal language: {len(informal_samples)} samples")
    
    # Check for building descriptions  
    building_samples = [s for s in samples if 'building' in s.lower()]
    if building_samples:
        high_priority.append(f"Building descriptions: {len(building_samples)} samples")
    
    for priority in high_priority:
        print(f"  • {priority}")
    
    print("\n📋 SUGGESTED NEW PATTERNS:")
    suggested_patterns = [
        ("you will see X", "you'll observe X / you can spot X"),
        ("head X", "go X / move X"),
        ("turn on X o'clock", "rotate to X o'clock"),
        ("looks like X", "appears to be X / resembles X"), 
        ("your destination", "your target / your goal"),
        ("pass X", "go past X / move beyond X"),
        ("cross X", "go across X / traverse X"),
    ]
    
    for original, suggestion in suggested_patterns:
        print(f"  {original} → {suggestion}")

if __name__ == "__main__":
    print("🔍 FOCUSED AVDN PATTERN ANALYSIS (300 samples)")
    print("="*60)
    
    # Load 300 random samples
    samples = load_300_random_samples()
    if not samples:
        exit(1)
    
    # NEW: Extract AVDN spatial examples for Mixtral
    few_shot_examples = extract_avdn_spatial_examples(samples)
    mixtral_prompt = generate_mixtral_prompts(few_shot_examples)
    spatial_tokens = extract_spatial_tokens(samples)
    
    # Run focused analyses
    analyze_core_patterns(samples)
    analyze_key_vocabulary(samples)
    find_missing_patterns(samples)
    generate_strategy1_recommendations(samples)
    
    print(f"\n{'='*60}")
    print("🎯 FOCUSED ANALYSIS COMPLETE")
    print(f"Based on {len(samples)} random samples")
    print('='*60) 