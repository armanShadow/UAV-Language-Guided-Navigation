import json
import os
from collections import defaultdict, Counter
import copy

def load_data(path, output_dir="processed_data", augment=True, max_augmented_per_episode=5):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the JSON file as an array
    with open(path, "r") as file:
        data_array = json.load(file)

    # Group by episodes (trajectories)
    episodes = defaultdict(list)
    i = 0
    original_episode_count = 0
    augmented_episode_count = 0
    
    # Track trajectory lengths for stats
    trajectory_lengths = Counter()
    map_names = Counter()
    
    while i < len(data_array):
        map_name = data_array[i]["map_name"]
        map_names[map_name] += 1
        number_of_rounds = int(data_array[i]["last_round_idx"])
        episode_id = f"{map_name}_{map_names[map_name]}"
        original_episode_count += 1
        
        # Track statistics
        trajectory_lengths[number_of_rounds] += 1
        
        # Create episode structure
        episode = {
            "episode_id": episode_id,
            "map_name": map_name,
            "gps_botm_left": data_array[i]["gps_botm_left"],
            "gps_top_right": data_array[i]["gps_top_right"],
            "lng_ratio": data_array[i]["lng_ratio"],
            "lat_ratio": data_array[i]["lat_ratio"],
            "first_instruction": data_array[i]["instructions"].replace("[INS] ", ""),
            "destination": data_array[i].get("destination", None),  # Keep destination info
            "dialogs": []
        }
        
        # Process each dialog turn
        all_observations = []
        all_dialog_turns = []
        
        for j in range(number_of_rounds):
            current_data = data_array[i+j]
            
            # For the first round, there's no question/answer
            if j == 0:
                dialog_turn = {
                    "turn_id": j,
                    "question": None,
                    "answer": None,
                    "observation": {
                        "view_area_coords": current_data["gt_path_corners"][0]
                    },
                    "dialog_history": [],
                    "previous_observations": []
                }
                all_observations.append(current_data["gt_path_corners"][0])
            else:
                # Extract question and answer from instructions
                instruction_text = current_data["instructions"]
                instruction_start_index = instruction_text.index("[INS]")
                question = instruction_text[5:instruction_start_index].strip()
                answer = instruction_text[instruction_start_index+5:].strip()
                
                # Get previous turn's final observation
                current_observation = data_array[i+j-1]["gt_path_corners"][-1]
                all_observations.append(current_observation)
                
                # Convert pre_dialogs to T5 format
                t5_dialog_history = []
                for dialog in current_data["pre_dialogs"]:
                    # Find [INS] marker in pre_dialogs
                    ins_start = dialog.find("[INS]")
                    
                    if ins_start != -1:
                        que_start = dialog.find("[QUE]")
                        
                        # Handle first instruction case (no [QUE] marker)
                        if que_start == -1:
                            # This is a first instruction without a question
                            a_text = dialog[ins_start + 5:].strip()  # Remove [INS] prefix
                            t5_dialog = f"First Instruction: {a_text}"
                            t5_dialog_history.append(t5_dialog)
                        else:
                            # Normal Q&A case
                            q_text = dialog[que_start + 5:ins_start].strip()  # Remove [QUE] prefix
                            a_text = dialog[ins_start + 5:].strip()  # Remove [INS] prefix
                            t5_dialog = f"Question: {q_text} Answer: {a_text}"
                            t5_dialog_history.append(t5_dialog)
                
                dialog_turn = {
                    "turn_id": j,
                    "question": question,
                    "answer": answer,
                    "observation": {
                        "view_area_coords": current_observation
                    },
                    "dialog_history": t5_dialog_history,
                    "previous_observations": all_observations[:-1]  # All observations except the current one
                }
            
            all_dialog_turns.append(dialog_turn)
            episode["dialogs"].append(dialog_turn)
        
        episodes[episode_id] = episode
        
        # Data augmentation: Create sub-trajectories for trajectories with 3+ turns
        if augment and number_of_rounds >= 3:
            # Generate sub-trajectories
            augmented_episodes = create_optimized_sub_trajectories(
                episode, 
                all_dialog_turns, 
                max_augmented=max_augmented_per_episode
            )
            
            # Add augmented episodes
            for aug_episode in augmented_episodes:
                aug_episode_id = aug_episode["episode_id"]
                episodes[aug_episode_id] = aug_episode
                augmented_episode_count += 1
                
        i += number_of_rounds

    # Convert to list for JSON serialization
    episodes_list = list(episodes.values())
    
    # Save as JSON for better nested structure support
    output_file = os.path.join(output_dir, os.path.basename(path))
    with open(output_file, 'w') as f:
        json.dump(episodes_list, f, indent=2)
    
    print(f"Processed {original_episode_count} original episodes and created {augmented_episode_count} augmented episodes")
    print(f"Total: {len(episodes_list)} episodes with {sum(len(ep['dialogs']) for ep in episodes_list)} dialog turns")
    
    # Print trajectory length statistics
    print("\nTrajectory length statistics:")
    for length, count in sorted(trajectory_lengths.items()):
        print(f"  {length} turns: {count} episodes ({count/original_episode_count*100:.1f}%)")
    
    return episodes_list

def create_optimized_sub_trajectories(original_episode, all_dialog_turns, max_augmented=5):
    """Create optimal sub-trajectories based on trajectory length"""
    if len(all_dialog_turns) < 3:
        return []
    
    augmented_episodes = []
    first_turn = copy.deepcopy(all_dialog_turns[0])  # Deep copy to avoid modifying original
    
    # Number of turns in the original trajectory
    num_turns = len(all_dialog_turns)
    
    # Different strategies based on trajectory length
    unique_patterns = set()
    augmentation_patterns = []
    
    # For trajectories of length 3 (0,1,2)
    if num_turns == 3:
        # Only meaningful pattern is (0,2) - skipping the middle turn
        augmentation_patterns.append([0, 2])
        
    # For trajectories of length 4 (0,1,2,3)
    elif num_turns == 4:
        # Skip one turn - creating 3 unique patterns
        augmentation_patterns.append([0, 1, 3])  # Skip turn 2
        augmentation_patterns.append([0, 2, 3])  # Skip turn 1
        
        # First and last only
        if [0, 3] not in augmentation_patterns:
            augmentation_patterns.append([0, 3])
    
    # For longer trajectories (5+ turns)
    else:
        # Skip one turn in the middle
        middle_turns = list(range(1, num_turns-1))
        for skip_idx in middle_turns[:2]:  # Skip at most 2 different middle turns
            pattern = [0] + [j for j in range(1, num_turns) if j != skip_idx]
            augmentation_patterns.append(pattern)
        
        # Skip every other turn
        skip_alternating = [j for j in range(num_turns) if j == 0 or j % 2 == 0]
        if len(skip_alternating) >= 2:
            augmentation_patterns.append(skip_alternating)
        
        # First, middle, last turns only
        middle = num_turns // 2
        augmentation_patterns.append([0, middle, num_turns-1])
        
        # First and last only
        augmentation_patterns.append([0, num_turns-1])
    
    # Convert patterns to tuples and deduplicate
    unique_patterns = set(tuple(pattern) for pattern in augmentation_patterns)
    
    # Convert back to lists and limit to max_augmented
    augmentation_patterns = [list(pattern) for pattern in unique_patterns]
    augmentation_patterns = augmentation_patterns[:max_augmented]
    
    # Create an augmented episode for each pattern
    for pattern_idx, pattern in enumerate(augmentation_patterns):
        # Create new episode with base properties from original
        aug_episode = {
            "episode_id": original_episode["episode_id"] + f"aug_pattern_{pattern_idx}",
            "map_name": original_episode["map_name"],
            "gps_botm_left": original_episode["gps_botm_left"],
            "gps_top_right": original_episode["gps_top_right"],
            "lng_ratio": original_episode["lng_ratio"],
            "lat_ratio": original_episode["lat_ratio"],
            "first_instruction": original_episode["first_instruction"],
            "destination": original_episode["destination"],
            "dialogs": []
        }
        
        # Add turns according to pattern
        all_observations = []
        
        # Process each turn in the pattern
        for new_idx, orig_idx in enumerate(pattern):
            # Get the original turn
            turn = copy.deepcopy(all_dialog_turns[orig_idx])
            
            # First turn always stays as is
            if orig_idx == 0:
                aug_episode["dialogs"].append(turn)
                all_observations.append(turn["observation"]["view_area_coords"])
                continue
            
            # Update turn ID to be sequential
            turn["turn_id"] = new_idx
            
            # Add current observation to history
            current_observation = turn["observation"]["view_area_coords"]
            all_observations.append(current_observation)
            
            # Update previous observations based on the augmented trajectory
            turn["previous_observations"] = copy.deepcopy(all_observations[:-1])
            
            # Update dialog history based on turns in this sub-trajectory
            new_dialog_history = []
            
            # Add previous turns' QA pairs to history
            for prev_pattern_idx in range(len(pattern)):
                if pattern[prev_pattern_idx] < orig_idx:  # Only include turns that come before this one
                    prev_orig_idx = pattern[prev_pattern_idx]
                    prev_turn = all_dialog_turns[prev_orig_idx]
                    
                    if prev_orig_idx == 0:
                        # Include the first instruction (which has no question)
                        first_ins = original_episode["first_instruction"]
                        dialog_entry = f"First Instruction: {first_ins}"
                        new_dialog_history.append(dialog_entry)
                    elif prev_turn["question"] and prev_turn["answer"]:
                        # Normal Q&A case
                        dialog_entry = f"Question: {prev_turn['question']} Answer: {prev_turn['answer']}"
                        new_dialog_history.append(dialog_entry)
            
            turn["dialog_history"] = new_dialog_history
            aug_episode["dialogs"].append(turn)
        
        augmented_episodes.append(aug_episode)
    
    return augmented_episodes

# Process all datasets
train_data = load_data(
    '../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/train_data.json', 
    augment=True,
    max_augmented_per_episode=3
)
val_seen_data = load_data(
    '../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/val_seen_data.json',
    augment=True,
    max_augmented_per_episode=2
)
val_unseen_data = load_data(
    '../../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations/val_unseen_data.json',
    augment=True,
    max_augmented_per_episode=2
)

# Create a metadata file with counts and stats
metadata = {
    "dataset_counts": {
        "train": len(train_data),
        "val_seen": len(val_seen_data),
        "val_unseen": len(val_unseen_data)
    },
    "dialog_turn_counts": {
        "train": sum(len(ep['dialogs']) for ep in train_data),
        "val_seen": sum(len(ep['dialogs']) for ep in val_seen_data),
        "val_unseen": sum(len(ep['dialogs']) for ep in val_unseen_data)
    }
}

with open('processed_data/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Dataset processing complete. See processed_data/ directory for results.")