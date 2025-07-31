import json
import os
from typing import List, Dict

def load_json_file(file_path: str) -> List[Dict]:
    """Load JSON file containing AVDN data."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {file_path}")
    return data

def save_json_file(data: List[Dict], file_path: str):
    """Save updated AVDN data to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} samples to {file_path}")

def update_pre_dialogs(data: List[Dict]) -> List[Dict]:
    """Update pre_dialogs for all samples based on generated instructions."""
    print("🔄 Starting pre_dialogs update...")
    
    # Group samples by episode for processing
    episodes = {}
    for i, sample in enumerate(data):
        map_name = sample['map_name']
        route_index = sample['route_index']
        
        episode_key = route_index.rsplit('_', 1)[0]  # Remove turn number
        full_episode_key = f"{map_name}_{episode_key}"
        
        if full_episode_key not in episodes:
            episodes[full_episode_key] = []
        episodes[full_episode_key].append((i, sample))
    
    # Sort episodes by turn number for proper processing order
    for episode_key in episodes:
        episodes[episode_key].sort(key=lambda x: int(x[1]['route_index'].split('_')[-1]))
    
    print(f"📊 Found {len(episodes)} episodes to process")
    
    # Create copy of data for updates
    updated_data = [sample.copy() for sample in data]
    
    # Process each episode
    for episode_key, episode_samples in episodes.items():
        print(f"🔄 Processing episode {episode_key} with {len(episode_samples)} turns")
        
        # TODO: IMPLEMENT YOUR LOGIC HERE
        for turn_idx, (sample_idx, sample) in enumerate(episode_samples):
            if turn_idx > 0 and turn_idx < len(episode_samples) - 1:
                new_instruction = sample['instructions']
                updated_data[sample_idx+1]['pre_dialogs'][turn_idx] = new_instruction
        
    
    return updated_data

def main():
    """Main function to update pre_dialogs in generated AVDN dataset."""
    import argparse
    
    args = argparse.ArgumentParser()
    args.add_argument("--split", type=str, default="train")
    args = args.parse_args()
    
    # File paths
    input_file = f"./generated_avdn_dataset/{args.split}_data.json"
    output_file = f"./generated_avdn_dataset/{args.split}_data_updated_pre_dialogs.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Load data
    print(f"📂 Loading data from: {input_file}")
    data = load_json_file(input_file)
    
    # Update pre_dialogs
    updated_data = update_pre_dialogs(data)
    
    # Save updated data
    print(f"💾 Saving updated data to: {output_file}")
    save_json_file(updated_data, output_file)
    
    print("✅ Pre_dialogs update complete!")

if __name__ == "__main__":
    main() 