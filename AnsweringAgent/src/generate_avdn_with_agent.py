import torch
import os
import json
import argparse
import random
import time
from typing import Dict, List, Optional, Tuple
from transformers import T5Tokenizer
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Import your model and config
from models.answering_agent import AnsweringAgent
from config import Config
from data.dataset import AnsweringDataset
from utils.logger import setup_logger

# Import evaluation functions
import sys
sys.path.append('../scripts')
from run_eval_generation import composite_score

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

class AVDNGeneratorWithAgent:
    """Generate new AVDN dataset using Answering Agent while preserving AVDN structure."""
    
    def __init__(self, config: Config, tokenizer: T5Tokenizer, model: AnsweringAgent, 
                 output_dir: str, device: torch.device):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.output_dir = output_dir
        self.device = device
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generation parameters for instruction generation
        self.generation_params = {
            'task_type': 'default',
            'num_beams': 4,
            'do_sample': False,
            'repetition_penalty': 1.1,
            'length_penalty': 0.8,
            'min_new_tokens': 8,
            'max_new_tokens': 70,
            'early_stopping': True,
        }
        
        # No fallback needed - we'll use the Answering Agent directly
    
    def load_avdn_data(self, split: str) -> List[Dict]:
        """Load original AVDN dataset for a specific split."""
        # Use config path instead of hardcoded path
        data_file = os.path.join(self.config.data.avdn_annotations_dir, f'{split}_data.json')
        print(f"Loading AVDN data from: {data_file}")
        print(f"Config AVDN annotations dir: {self.config.data.avdn_annotations_dir}")
        print(f"File exists: {os.path.exists(data_file)}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples from {split} split")
        return data
    
    def load_formatted_dataset(self, split: str) -> AnsweringDataset:
        """Load formatted dataset for generation inputs."""
        print(f"Loading formatted dataset for split: {split}")
        dataset = AnsweringDataset(self.config, split=split, tokenizer=self.tokenizer)
        print(f"Loaded {len(dataset)} samples from formatted {split} split")
        return dataset
    
    def decode_tokenized_text(self, tokenized_data: Dict) -> str:
        """Decode tokenized text back to string."""
        return self.tokenizer.decode(tokenized_data['input_ids'], skip_special_tokens=True)
    
    def extract_dialog_components(self, sample: Dict) -> Tuple[str, str, str, List[str]]:
        """
        Extract dialog components from formatted dataset sample.
        Returns: (first_instruction, current_question, current_answer, dialog_history)
        """
        # Decode the tokenized components safely
        current_answer = self.decode_tokenized_text(sample['text_label'])
        dialog_context = self.decode_tokenized_text(sample['text_input'])
        
        # Conditionally decode other components if they exist
        first_instruction = ""
        current_question = ""
        
        if 'first_instruction_input' in sample:
            try:
                first_instruction = self.decode_tokenized_text(sample['first_instruction_input'])
            except Exception as e:
                print(f"Debug: first_instruction_input structure: {type(sample['first_instruction_input'])}")
                if isinstance(sample['first_instruction_input'], dict):
                    print(f"Debug: first_instruction_input keys: {sample['first_instruction_input'].keys()}")
                first_instruction = f"Error decoding: {e}"
        
        if 'current_question_input' in sample:
            current_question = self.decode_tokenized_text(sample['current_question_input'])
        
        # Parse dialog history from the context
        dialog_history = self.parse_dialog_history(dialog_context, first_instruction)
        
        return first_instruction, current_question, current_answer, dialog_history
    
    def parse_dialog_history(self, dialog_context: str, first_instruction: str) -> List[str]:
        """Parse dialog history from the unified context."""
        # Remove the first instruction from context since we have it separately
        context_without_first = dialog_context.replace(f"First Instruction: {first_instruction}", "").strip()
        
        # Split by "Question:" and "Answer:" to extract turns
        parts = context_without_first.split("Question:")
        dialog_history = []
        
        for part in parts[1:]:  # Skip first empty part
            if "Answer:" in part:
                qa_parts = part.split("Answer:", 1)
                if len(qa_parts) == 2:
                    question = qa_parts[0].strip()
                    answer = qa_parts[1].strip()
                    dialog_history.append(f"Question: {question}")
                    dialog_history.append(f"Answer: {answer}")
        
        return dialog_history
    
    def generate_new_answer(self, formatted_sample: Dict, 
                           current_view_image: torch.Tensor, previous_views_image: torch.Tensor,
                           destination_image: Optional[torch.Tensor] = None) -> str:
        """Generate a new answer using Answering Agent with actual inputs from formatted dataset."""
        
        # Construct text_input dictionary as expected by the AnsweringAgent model
        text_input = {
            'input_ids': formatted_sample['text_input']['input_ids'].unsqueeze(0).to(self.device),
            'attention_mask': formatted_sample['text_input']['attention_mask'].unsqueeze(0).to(self.device)
        }
        
        # Add separate components EXACTLY as in training (lines 687-691 in train.py)
        if 'first_instruction_input' in formatted_sample:
            text_input['first_instruction_input'] = {k: v.unsqueeze(0).to(self.device) for k, v in formatted_sample['first_instruction_input'].items()}
        if 'current_question_input' in formatted_sample:
            text_input['current_question_input'] = {k: v.unsqueeze(0).to(self.device) for k, v in formatted_sample['current_question_input'].items()}
        
        # Use actual visual features from the formatted dataset
        current_view = current_view_image.unsqueeze(0).to(self.device)  # Add batch dimension
        previous_views = previous_views_image.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Get the actual model (not DDP wrapper) for generation
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Generate new answer using the An[[swering Agent with exact same inputs as evaluation
        with torch.no_grad():
            generated_seq = model_to_use.generate_answer(
                text_input, current_view, previous_views,
                **self.generation_params
            )
        
        # Decode the generated sequence
        new_answer = self.tokenizer.decode(generated_seq[0], skip_special_tokens=True)
        
        # Clean up the answer
        new_answer = new_answer.strip()
        
        return new_answer
    

    
    def update_avdn_instruction(self, avdn_sample: Dict, new_answer: str, turn_index: int) -> Dict:
        """Update AVDN sample with new instruction and dialog history."""
        # Create new sample by copying the original
        new_sample = avdn_sample.copy()
        
        # Get the original instruction
        original_instruction = avdn_sample['instructions']
        
        # Handle different instruction formats
        if '[QUE]' in original_instruction and '[INS]' in original_instruction:
            # Complex format with question and answer
            parts = original_instruction.split('[INS]')
            question_part = parts[0].replace('[QUE]', '').strip()
            
            # Create new instruction with generated answer
            new_instruction = f"[QUE] {question_part} [INS] {new_answer}"
        else:
            # Simple instruction format - replace the instruction part
            if '[INS]' in original_instruction:
                # Keep the [INS] tag and replace the content
                new_instruction = f"[INS] {new_answer}"
            else:
                # No [INS] tag, just replace the whole instruction
                new_instruction = new_answer
        
        # Update the instruction
        new_sample['instructions'] = new_instruction
        
        # Dialog history will be updated at the episode level during processing
        # Individual sample updates only change the current instruction
        # The pre_dialogs will be updated when processing subsequent turns in the same episode
        
        return new_sample
    
    def load_preprocessed_json(self, split: str) -> List[Dict]:
        """Load the JSON format of preprocessed dataset with metadata."""
        # Use the config's JSON paths
        if split == 'train':
            json_file = self.config.data.train_json_path
        elif split == 'val_seen':
            json_file = self.config.data.val_seen_json_path
        elif split == 'val_unseen':
            json_file = self.config.data.val_unseen_json_path
        else:
            raise ValueError(f"Unknown split: {split}")
        
        print(f"Loading preprocessed JSON from: {json_file}")
        
        if not os.path.exists(json_file):
            # Try alternative paths
            alt_paths = [
                f"../datasets/{split}_data.json",
                f"./datasets/{split}_data.json", 
                f"./processed_data/{split}_data.json",
                f"/app/datasets/{split}_data.json",
                f"/app/UAV-Language-Guided-Navigation/AnsweringAgent/src/data/processed_data/{split}_data.json"
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    json_file = alt_path
                    print(f"Found JSON file at alternative path: {json_file}")
                    break
            else:
                print(f"❌ Could not find preprocessed JSON file for {split}")
                print(f"Looked in: {json_file}")
                for path in alt_paths:
                    print(f"  Also tried: {path}")
                return []
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Loaded {len(data)} preprocessed samples from JSON")
        return data
    
    def create_avdn_to_preprocessed_mapping(self, avdn_data: List[Dict], preprocessed_json: List[Dict], rank: int = 0) -> Dict[int, int]:
        """Create simple mapping based on exact map name and answer matching."""
        print("Creating AVDN to preprocessed dataset mapping using simple map+answer matching...")
        
        mapping = {}
        
        # Parse preprocessed episodes and create flattened turn index
        preprocessed_turns = []
        map_to_turns = {}  # map_name -> [turn_indices]
        
        for episode in preprocessed_json:
            episode_id = episode.get('episode_id', '')
            
            # Skip augmented episodes (contains "aug_pattern")
            if 'aug_pattern' in episode_id:
                continue
                
            map_name = episode.get('map_name', '')
            first_instruction = episode.get('first_instruction', '').strip()
            
            # Process each dialog turn in the episode
            for dialog in episode.get('dialogs', []):
                turn_id = dialog.get('turn_id', 0)
                
                # Skip turn 0 (usually just observation, no Q&A)
                if turn_id == 0:
                    continue
                    
                answer = dialog.get('answer', '').strip()
                question = dialog.get('question', '').strip()
                
                turn_data = {
                    'episode_id': episode_id,
                    'map_name': map_name,
                    'turn_id': turn_id,
                    'answer': answer,
                    'question': question,
                    'first_instruction': first_instruction,
                    'preprocessed_index': len(preprocessed_turns)
                }
                preprocessed_turns.append(turn_data)
                
                # Group by map name for easy lookup
                if map_name not in map_to_turns:
                    map_to_turns[map_name] = []
                map_to_turns[map_name].append(len(preprocessed_turns) - 1)
        
        print(f"Found {len(preprocessed_turns)} non-augmented dialog turns across {len(map_to_turns)} maps")
        
        matched_count = 0
        qa_samples_count = 0
        
        # Process each AVDN sample
        for i, avdn_sample in enumerate(avdn_data):
            instruction = avdn_sample['instructions']
            map_name = avdn_sample['map_name']
            
            # Only process entries with both question and answer (not first instructions)
            if '[QUE]' not in instruction or '[INS]' not in instruction:
                continue  # Skip first instructions
                
            qa_samples_count += 1
            
            # Extract question and answer from AVDN instruction
            que_start = instruction.find('[QUE]')
            ins_start = instruction.find('[INS]')
            avdn_question = instruction[que_start+5:ins_start].strip()
            avdn_answer = instruction[ins_start+5:].strip()
            
            # Get first instruction from pre_dialogs
            avdn_first_instruction = ""
            if avdn_sample.get('pre_dialogs') and len(avdn_sample['pre_dialogs']) > 0:
                first_dialog = avdn_sample['pre_dialogs'][0]
                if '[INS]' in first_dialog:
                    avdn_first_instruction = first_dialog.replace('[INS]', '').strip()
                else:
                    avdn_first_instruction = first_dialog.strip()
            
            # Find all processed turns for this map
            if map_name not in map_to_turns:
                if matched_count < 5:
                    print(f"❌ No processed data found for map {map_name}")
                continue
            
            # Look for exact answer match
            best_match = None
            fallback_match = None
            
            for turn_idx in map_to_turns[map_name]:
                turn_data = preprocessed_turns[turn_idx]
                
                # Check if answers match exactly (case-insensitive)
                if turn_data['answer'].lower().strip() == avdn_answer.lower().strip():
                    # Primary sanity check: verify question match
                    question_match = (turn_data['question'].lower().strip() == 
                                    avdn_question.lower().strip())
                    
                    if question_match:
                        # Store as fallback match (answer + question match)
                        if fallback_match is None:
                            fallback_match = turn_idx
                        
                        # More flexible first instruction matching - check for substantial overlap
                        def normalize_text(text):
                            # Remove extra spaces, punctuation, and normalize
                            import re
                            text = re.sub(r'[^\w\s]', ' ', text.lower())
                            return ' '.join(text.split())
                        
                        norm_proc_first = normalize_text(turn_data['first_instruction'])
                        norm_avdn_first = normalize_text(avdn_first_instruction)
                        
                        # Check if there's substantial overlap (at least 50% of words match - more lenient)
                        proc_words = set(norm_proc_first.split())
                        avdn_words = set(norm_avdn_first.split())
                        
                        if len(avdn_words) > 0:
                            overlap = len(proc_words & avdn_words) / len(avdn_words)
                            first_instruction_flexible_match = overlap >= 0.9  # Lowered threshold
                        else:
                            overlap = 1.0 if len(proc_words) == 0 else 0.0
                            first_instruction_flexible_match = len(proc_words) == 0
                        
                        if first_instruction_flexible_match:
                            best_match = turn_idx
                            break
                        elif matched_count < 5:
                            print(f"⚠️  Answer+Question match found but first instruction mismatch for AVDN[{i}]:")
                            print(f"   First instruction overlap: {overlap:.2f} (need ≥0.5)")
                            print(f"   AVDN first: {norm_avdn_first[:60]}...")
                            print(f"   Proc first: {norm_proc_first[:60]}...")
            
            # Use best match if found, otherwise use fallback (answer + question only)
            final_match = None
            match_type = ""
            
            if best_match is not None:
                final_match = best_match
                match_type = "full"
            elif fallback_match is not None:
                final_match = fallback_match
                match_type = "fallback"
            
            if final_match is not None:
                mapping[i] = final_match
                matched_count += 1
                
                if matched_count <= 5 and rank == 0:  # Debug first few matches (reduced from 10)
                    turn_data = preprocessed_turns[final_match]
                    match_symbol = "✅" if match_type == "full" else "📝"
                    print(f"{match_symbol} Match {matched_count} ({match_type}): AVDN[{i}] map:{map_name} -> Preprocessed[{final_match}] {turn_data['episode_id']}")
                    print(f"   Answer: {avdn_answer[:50]}...")
                    print(f"   Question: {avdn_question[:50]}...")
            else:
                if matched_count < 5:
                    print(f"❌ No exact answer match found for AVDN[{i}] map:{map_name}")
                    print(f"   Looking for answer: {avdn_answer[:50]}...")
        
        total_qa_samples = qa_samples_count
        skipped_samples = total_qa_samples - matched_count
        
        print(f"\n📊 Simple Mapping Summary:")
        print(f"🔍 Q&A samples to process: {total_qa_samples}/{len(avdn_data)} total samples")
        print(f"✅ Successfully matched: {matched_count}/{total_qa_samples} Q&A samples ({matched_count/total_qa_samples*100:.1f}%)")
        print(f"⚠️  No match found: {skipped_samples} Q&A samples ({skipped_samples/total_qa_samples*100:.1f}%)")
        
        # Store preprocessed_turns for later use
        self.preprocessed_turns = preprocessed_turns
        
        return mapping
    
    def process_avdn_sample(self, avdn_sample: Dict, formatted_dataset: AnsweringDataset, mapping: Dict[int, int], avdn_index: int, rank: int = 0) -> Dict:
        """Process a single AVDN sample and generate new instruction using mapping."""
        
        # Get AVDN sample metadata
        map_name = avdn_sample['map_name']
        route_index = avdn_sample['route_index']
        
        # Use the mapping to find corresponding formatted sample
        matching_sample = None
        if avdn_index in mapping:
            try:
                turn_index = mapping[avdn_index]
                # The turn_index corresponds to our flattened preprocessed_turns
                # But we need to map this to the formatted_dataset index
                # The formatted dataset should have the same indexing as our flattened turns
                matching_sample = formatted_dataset[turn_index]
                
                if avdn_index < 2 and rank == 0:  # Debug first few matches (reduced from 3)
                    turn_data = self.preprocessed_turns[turn_index]
                    dialog_context = self.decode_tokenized_text(matching_sample['text_input'])
                    formatted_answer = self.decode_tokenized_text(matching_sample['text_label'])
                    print(f"Mapped match {avdn_index}: AVDN({map_name}, {route_index}) -> Turn {turn_index}")
                    print(f"  Episode: {turn_data['episode_id']}, Turn: {turn_data['turn_id']}")
                    print(f"  Context: {dialog_context[:100]}...")
                    print(f"  Answer: {formatted_answer}")
                    
            except Exception as e:
                print(f"Error accessing mapped formatted sample {mapping[avdn_index]}: {e}")
        
        if matching_sample is None:
            # If no mapping found, skip this sample (return original unchanged)
            if avdn_index < 10 and rank == 0:  # Only log first few skipped samples to avoid spam
                print(f"⚠️  No mapping found for AVDN sample {avdn_index} ({map_name}_{route_index}), keeping original")
            return avdn_sample  # Return original sample unchanged
        
        # Get visual features from formatted sample
        current_view_image = matching_sample['current_view_image']
        previous_views_image = matching_sample['previous_views_image']
        destination_image = matching_sample.get('destination_image', None)
        
        # Generate new answer using Answering Agent with exact same inputs as evaluation
        new_answer = self.generate_new_answer(
            matching_sample, current_view_image, previous_views_image, destination_image
        )
        
        # Update AVDN sample with new instruction
        new_sample = self.update_avdn_instruction(avdn_sample, new_answer, 0)
        
        # Add debug information for verification
        if avdn_index in mapping:
            turn_index = mapping[avdn_index]
            if turn_index < len(self.preprocessed_turns):
                turn_data = self.preprocessed_turns[turn_index]
                
                # Add debug fields to the output for verification
                new_sample['_debug_info'] = {
                    'matched_episode_id': turn_data['episode_id'],
                    'matched_turn_id': turn_data['turn_id'],
                    'original_context_preview': self.decode_tokenized_text(matching_sample['text_input'])[:200] + "...",
                    'original_answer': avdn_sample['instructions'],
                    'generated_answer': new_answer,
                    'preprocessed_answer': turn_data['answer'],
                    'avdn_route_index': avdn_sample['route_index']
                }
                
                # Calculate generation score for this sample
                if '[INS]' in avdn_sample['instructions']:
                    original_answer = avdn_sample['instructions'].split('[INS]')[-1].strip()
                else:
                    original_answer = avdn_sample['instructions'].strip()
                
                try:
                    scores = composite_score(new_answer, original_answer, task_type="precision_short")
                    new_sample['_debug_info']['generation_scores'] = scores
                except Exception as e:
                    new_sample['_debug_info']['generation_scores'] = {'error': str(e)}
        
        return new_sample
    
    def broadcast_data(self, data, rank: int, world_size: int):
        """Broadcast data from rank 0 to all other ranks."""
        import torch.distributed as dist
        
        if world_size == 1:
            return data
        
        # Use all_gather_object to synchronize the data
        gathered_data = [None for _ in range(world_size)]
        
        # Only rank 0 has the correctly sampled data
        if rank == 0:
            local_data = data
        else:
            local_data = None
        
        # Gather data from all ranks (only rank 0 will have real data)
        dist.all_gather_object(gathered_data, local_data)
        
        # All ranks take the data from rank 0
        return gathered_data[0]
    
    def process_split(self, split: str, sample_ratio: float = 1.0, max_samples: Optional[int] = None, 
                     rank: int = 0, world_size: int = 1) -> List[Dict]:
        """Process an entire split of the AVDN dataset with distributed generation."""
        if rank == 0:
            print(f"\n🚀 Processing {split} split with {world_size} GPUs...")
        
        # Load original AVDN data (all ranks need this)
        avdn_data = self.load_avdn_data(split)
        
        # Load formatted dataset for generation inputs (all ranks need this)
        formatted_dataset = self.load_formatted_dataset(split)
        
        # Only rank 0 does the matching (computationally light)
        if rank == 0:
            print("🔍 Creating AVDN to preprocessed mapping (rank 0 only)...")
            # Load preprocessed JSON with metadata
            preprocessed_json = self.load_preprocessed_json(split)
            
            # Create mapping between AVDN and preprocessed datasets using metadata
            mapping = self.create_avdn_to_preprocessed_mapping(avdn_data, preprocessed_json, rank)
            
            # Store preprocessed_turns for all ranks
            preprocessed_turns = self.preprocessed_turns
        else:
            mapping = {}
            preprocessed_turns = []
        
        # Broadcast mapping and preprocessed_turns to all ranks
        if world_size > 1:
            if rank == 0:
                print("📡 Broadcasting mapping to all GPUs...")
            
            # Convert mapping to lists for broadcasting
            mapping_keys = list(mapping.keys()) if rank == 0 else []
            mapping_values = list(mapping.values()) if rank == 0 else []
            
            # Broadcast using torch tensors
            import torch.distributed as dist
            
            # Broadcast mapping size first
            mapping_size = torch.tensor(len(mapping_keys), device=self.device)
            dist.broadcast(mapping_size, src=0)
            
            if rank != 0:
                mapping_keys = [0] * mapping_size.item()
                mapping_values = [0] * mapping_size.item()
            
            # Broadcast mapping data
            if mapping_size.item() > 0:
                mapping_keys_tensor = torch.tensor(mapping_keys, device=self.device)
                mapping_values_tensor = torch.tensor(mapping_values, device=self.device)
                
                dist.broadcast(mapping_keys_tensor, src=0)
                dist.broadcast(mapping_values_tensor, src=0)
                
                if rank != 0:
                    mapping = dict(zip(mapping_keys_tensor.cpu().numpy(), mapping_values_tensor.cpu().numpy()))
            
            # Broadcast preprocessed_turns (simplified - just the count for now)
            preprocessed_turns_size = torch.tensor(len(preprocessed_turns), device=self.device)
            dist.broadcast(preprocessed_turns_size, src=0)
            
            if rank != 0:
                # For non-rank-0, we'll pass the mapping but won't need full preprocessed_turns for generation
                # The mapping is sufficient for finding the right formatted samples
                self.preprocessed_turns = [{'episode_id': 'distributed'}] * preprocessed_turns_size.item()
            else:
                self.preprocessed_turns = preprocessed_turns
            
            if rank == 0:
                print(f"✅ Broadcasted mapping of {len(mapping)} samples to all {world_size} GPUs")
        else:
            # Store for single GPU case
            self.preprocessed_turns = preprocessed_turns if rank == 0 else []
        
        # Apply sampling if requested (on rank 0, then broadcast)
        if rank == 0:
            if sample_ratio < 1.0:
                num_samples = int(len(avdn_data) * sample_ratio)
                avdn_data = avdn_data[:num_samples]
                print(f"Sampled {num_samples} samples from {split}")
            
            # Apply max_samples limit if specified
            if max_samples is not None:
                avdn_data = avdn_data[:max_samples]
                print(f"Limited to {max_samples} samples")
        
        # Synchronize actual data across all ranks (not just size)
        if world_size > 1:
            # Broadcast the exact sampled data to ensure all ranks have identical datasets
            avdn_data = self.broadcast_data(avdn_data, rank, world_size)
        
        # Group AVDN samples by episode for proper dialog history management
        avdn_episodes = {}  # episode_key -> [samples]
        for i, sample in enumerate(avdn_data):
            map_name = sample['map_name']
            route_index = sample['route_index']
            
            # Create episode key from map and trajectory
            route_parts = route_index.split('_')
            if len(route_parts) >= 2:
                trajectory_id = route_parts[0]
                turn_id = int(route_parts[1])
                episode_key = f"{map_name}_{trajectory_id}"
                
                if episode_key not in avdn_episodes:
                    avdn_episodes[episode_key] = []
                
                avdn_episodes[episode_key].append({
                    'original_index': i,
                    'turn_id': turn_id,
                    'sample': sample.copy()
                })
        
        # Sort turns within each episode
        for episode_key in avdn_episodes:
            avdn_episodes[episode_key].sort(key=lambda x: x['turn_id'])
        
        if rank == 0:
            print(f"Organized {len(avdn_data)} samples into {len(avdn_episodes)} episodes")
        
        # DISTRIBUTED PROCESSING: Each rank processes a subset of episodes
        episodes_list = list(avdn_episodes.items())
        episodes_per_rank = len(episodes_list) // world_size
        extra_episodes = len(episodes_list) % world_size
        
        # Calculate start and end indices for this rank
        if rank < extra_episodes:
            start_idx = rank * (episodes_per_rank + 1)
            end_idx = start_idx + episodes_per_rank + 1
        else:
            start_idx = rank * episodes_per_rank + extra_episodes  
            end_idx = start_idx + episodes_per_rank
        
        # Get this rank's episodes
        my_episodes = dict(episodes_list[start_idx:end_idx])
        
        # Show episode distribution for all ranks to debug the issue
        print(f"🔧 Rank {rank}: Processing {len(my_episodes)}/{len(episodes_list)} episodes (indices {start_idx}-{end_idx-1})")
        
        # Process samples episode by episode to maintain dialog history (distributed)
        local_processed_data = {}  # Will store this rank's processed samples by original_index
        local_scores = []
        local_successful_generations = 0
        
        # Process only this rank's episodes
        desc = f"Rank {rank} processing {split} episodes"
        for episode_key, episode_turns in tqdm(my_episodes.items(), desc=desc, disable=(rank != 0)):
            generated_instructions = {}  # turn_id -> generated_instruction
            
            for turn_data in episode_turns:
                original_index = turn_data['original_index']
                turn_id = turn_data['turn_id']
                sample = turn_data['sample']
                
                try:
                    # Update pre_dialogs with previously generated instructions in this episode
                    if turn_id > 1 and len(sample.get('pre_dialogs', [])) > 0:
                        updated_pre_dialogs = []
                        for j, prev_dialog in enumerate(sample['pre_dialogs']):
                            # Check if this dialog corresponds to a previously generated instruction
                            prev_turn_id = j + 1  # pre_dialogs[0] = turn 1, pre_dialogs[1] = turn 2, etc.
                            
                            if prev_turn_id in generated_instructions:
                                # Replace with generated instruction
                                updated_pre_dialogs.append(generated_instructions[prev_turn_id])
                            else:
                                # Keep original
                                updated_pre_dialogs.append(prev_dialog)
                        
                        sample['pre_dialogs'] = updated_pre_dialogs
                    
                    # Process the sample
                    processed_sample = self.process_avdn_sample(sample, formatted_dataset, mapping, original_index, rank)
                    local_processed_data[original_index] = processed_sample
                    
                    # Store generated instruction for next turns in this episode
                    if processed_sample['instructions'] != sample['instructions']:
                        generated_instructions[turn_id] = processed_sample['instructions']
                        local_successful_generations += 1
                        
                        # Calculate metrics
                        original_instruction = sample['instructions']
                        if '[INS]' in original_instruction:
                            original_answer = original_instruction.split('[INS]')[-1].strip()
                        else:
                            original_answer = original_instruction.strip()
                        
                        new_instruction = processed_sample['instructions']
                        if '[INS]' in new_instruction:
                            generated_answer = new_instruction.split('[INS]')[-1].strip()
                        else:
                            generated_answer = new_instruction.strip()
                        
                        # Calculate composite score (normalization now handled in evaluation script)
                        scores = composite_score(generated_answer, original_answer, task_type="precision_short")
                        local_scores.append(scores)
                        
                        # Debug scoring for first few samples (only rank 0)
                        if local_successful_generations <= 2 and rank == 0:
                            print(f"\n🔍 Scoring Debug for sample {original_index}:")
                            print(f"   Original: {original_answer}")
                            print(f"   Generated: {generated_answer}")
                            print(f"   Direction score: {scores['direction']}")
                            print(f"   Movement score: {scores['movement']}")
                            print(f"   Landmark score: {scores['landmark']}")
                    
                except Exception as e:
                    if rank == 0:
                        print(f"Rank {rank}: Error processing sample {original_index} in episode {episode_key}: {e}")
                    # Keep original sample if generation fails
                    local_processed_data[original_index] = sample
        
        # Only show completion stats on rank 0 for cleaner output
        if rank == 0:
            print(f"🔧 Rank {rank}: Completed processing. Generated {local_successful_generations} samples")
        
        # GATHER RESULTS FROM ALL RANKS
        if world_size > 1:
            if rank == 0:
                print("🔄 Gathering results from all ranks...")
            
            # Gather all processed data using all_gather
            gathered_data = [None for _ in range(world_size)]
            gathered_scores = [None for _ in range(world_size)]
            gathered_stats = [None for _ in range(world_size)]
            
            dist.all_gather_object(gathered_data, local_processed_data)
            dist.all_gather_object(gathered_scores, local_scores)
            dist.all_gather_object(gathered_stats, {
                'successful_generations': local_successful_generations,
                'rank': rank
            })
            
            # Combine results on all ranks
            processed_data = [None] * len(avdn_data)
            all_scores = []
            successful_generations = 0
            
            for rank_data in gathered_data:
                for idx, sample in rank_data.items():
                    processed_data[idx] = sample
            
            for rank_scores in gathered_scores:
                all_scores.extend(rank_scores)
            
            for rank_stats in gathered_stats:
                successful_generations += rank_stats['successful_generations']
            
            if rank == 0:
                print(f"✅ Gathered results: {successful_generations} successful generations across {world_size} GPUs")
                # Only show per-rank stats if there are issues
                if successful_generations < 10:  # If very few generations, show per-rank breakdown
                    print(f"📊 Per-rank stats:")
                    for i, stats in enumerate(gathered_stats):
                        print(f"   Rank {i}: {stats['successful_generations']} generations")
        else:
            # Single GPU case
            processed_data = [None] * len(avdn_data)
            for idx, sample in local_processed_data.items():
                processed_data[idx] = sample
            all_scores = local_scores
            successful_generations = local_successful_generations
        
        # Fill in any None values with original samples
        for i, sample in enumerate(processed_data):
            if sample is None:
                processed_data[i] = avdn_data[i]
        
        # Print some examples (only on rank 0)
        if rank == 0:
            examples_shown = 0
            for i, (original, processed) in enumerate(zip(avdn_data, processed_data)):
                if examples_shown >= 2:  # Reduced from 3
                    break
                if processed['instructions'] != original['instructions']:
                    examples_shown += 1
                    print(f"\nGenerated Example {examples_shown}:")
                    print(f"Map: {original['map_name']}, Route: {original['route_index']}")
                    print(f"Original: {original['instructions']}")
                    print(f"Generated: {processed['instructions']}")
                    print("-" * 80)
        
        # Report final metrics for this split (only on rank 0)
        if rank == 0:
            total_samples = len(avdn_data)
            if all_scores:
                avg_composite = sum(s['total'] for s in all_scores) / len(all_scores)
                avg_direction = sum(s['direction'] for s in all_scores) / len(all_scores)
                avg_movement = sum(s['movement'] for s in all_scores) / len(all_scores)
                avg_landmark = sum(s['landmark'] for s in all_scores) / len(all_scores)
                avg_attribute = sum(s['attribute'] for s in all_scores) / len(all_scores)
                
                print(f"\n📊 {split.upper()} GENERATION METRICS:")
                print(f"Total Samples: {total_samples}")
                print(f"Successful Generations: {successful_generations}")
                print(f"Success Rate: {successful_generations/total_samples*100:.1f}%")
                print(f"Average Composite Score: {avg_composite:.4f}")
                print(f"Average Direction Score: {avg_direction:.4f}")
                print(f"Average Movement Score: {avg_movement:.4f}")
                print(f"Average Landmark Score: {avg_landmark:.4f}")
                print(f"Average Attribute Score: {avg_attribute:.4f}")
            else:
                print(f"\n⚠️ No successful generations for {split} split")
        
        return processed_data, all_scores
    
    def save_processed_data(self, data: List[Dict], split: str, all_scores: List[Dict] = None):
        """Save processed data to output directory in AVDN format and save metrics."""
        # Save main AVDN dataset
        output_file = os.path.join(self.output_dir, f'{split}_data.json')
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} samples to {output_file}")
        
        # Save detailed metrics if available
        if all_scores:
            metrics_file = os.path.join(self.output_dir, f'{split}_generation_metrics.json')
            
            # Calculate aggregated metrics
            total_samples = len(data)
            successful_generations = len(all_scores)
            
            # Calculate averages
            if all_scores:
                avg_metrics = {}
                for metric in ['direction', 'yesno', 'attribute', 'landmark', 'movement', 'form', 'total', 'bleu4', 'rouge_l', 'bertscore']:
                    scores_for_metric = [s[metric] for s in all_scores if metric in s]
                    avg_metrics[metric] = sum(scores_for_metric) / len(scores_for_metric) if scores_for_metric else 0.0
                
                # Detailed metrics report
                detailed_metrics = {
                    'split': split,
                    'summary': {
                        'total_samples': total_samples,
                        'successful_generations': successful_generations,
                        'success_rate': successful_generations / total_samples if total_samples > 0 else 0.0,
                        'generation_coverage': f"{successful_generations}/{total_samples} samples ({successful_generations/total_samples*100:.1f}%)"
                    },
                    'average_scores': avg_metrics,
                    'individual_scores': []
                }
                
                # Add individual sample scores with context
                sample_idx = 0
                for i, sample in enumerate(data):
                    if '_debug_info' in sample and 'generation_scores' in sample['_debug_info']:
                        detailed_metrics['individual_scores'].append({
                            'sample_index': i,
                            'avdn_route_index': sample.get('route_index', 'unknown'),
                            'map_name': sample.get('map_name', 'unknown'),
                            'matched_episode_id': sample['_debug_info'].get('matched_episode_id', 'none'),
                            'original_instruction': sample['_debug_info'].get('original_answer', ''),
                            'generated_instruction': sample['_debug_info'].get('generated_answer', ''),
                            'scores': sample['_debug_info']['generation_scores']
                        })
                
                with open(metrics_file, 'w') as f:
                    json.dump(detailed_metrics, f, indent=2)
                
                print(f"Saved detailed metrics to {metrics_file}")
            else:
                print(f"No scores available to save metrics for {split}")
    
    def process_all_splits(self, splits: List[str], sample_ratio: float = 1.0, 
                          max_samples: Optional[int] = None, rank: int = 0, world_size: int = 1):
        """Process all specified splits with distributed processing."""
        overall_metrics = {}
        
        for split in splits:
            processed_data, scores = self.process_split(split, sample_ratio, max_samples, rank, world_size)
            
            # Only save on rank 0 to avoid conflicts
            if rank == 0:
                self.save_processed_data(processed_data, split, scores)
            
            # Store metrics for overall summary
            overall_metrics[split] = {
                'total_samples': len(processed_data),
                'successful_generations': sum(1 for p in processed_data if p['instructions'] != p.get('original_instructions', '')),
                'avg_composite_score': 0.0,  # Will be calculated if we track scores
                'avg_direction_score': 0.0,
                'avg_movement_score': 0.0,
                'avg_landmark_score': 0.0,
                'avg_attribute_score': 0.0
            }
        
        # Print overall summary (only on rank 0)
        if rank == 0:
            print(f"\n🎯 OVERALL GENERATION SUMMARY:")
            print("=" * 60)
            for split, metrics in overall_metrics.items():
                success_rate = metrics['successful_generations'] / metrics['total_samples'] * 100 if metrics['total_samples'] > 0 else 0
                print(f"{split.upper()}: {metrics['successful_generations']}/{metrics['total_samples']} ({success_rate:.1f}% success)")
            
            print(f"\n✅ Generation completed for all splits!")
            print(f"📁 Generated datasets saved to: {self.output_dir}")


# -------------------------
# Distributed setup functions
# -------------------------
def setup_distributed():
    """Set up distributed processing."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    torch.cuda.set_device(local_rank)
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
    
    return True, rank, world_size

def setup_environment():
    """Setup environment for distributed processing."""
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def main():
    """Main function for AVDN dataset generation."""
    parser = argparse.ArgumentParser(description="Generate AVDN dataset with Answering Agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to Answering Agent checkpoint")
    parser.add_argument("--output_dir", type=str, default="./generated_avdn_dataset",
                       help="Output directory for generated dataset")
    parser.add_argument("--splits", nargs="+", 
                       default=['train', 'val_seen', 'val_unseen'],
                       help="Dataset splits to process")
    parser.add_argument("--sample_ratio", type=float, default=1.0,
                       help="Ratio of dataset to sample (default: 1.0 = 100%)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to process per split")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    args = parser.parse_args()

    # Setup environment
    setup_environment()
    
    # Setup distributed processing
    is_distributed, rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"🚀 AVDN Dataset Generation with Answering Agent")
        print(f"World Size: {world_size}, Rank: {rank}")
        if torch.cuda.is_available():
            print(f"CUDA Devices: {torch.cuda.device_count()}")
        print(f"Output Dir: {args.output_dir}")
        print(f"Splits: {args.splits}")
        print(f"Sample Ratio: {args.sample_ratio}")

        print()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config and model
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name)

    # Initialize logger
    if rank == 0:
        logger = setup_logger('avdn_generation', log_dir=config.log_dir)
    else:
        class DummyLogger:
            def info(self, msg): pass
            def warning(self, msg): pass
            def error(self, msg): pass
            def debug(self, msg): pass
        logger = DummyLogger()

    # Load model
    if rank == 0:
        logger.info("🏗️ Loading Answering Agent model...")
    model = AnsweringAgent(config, tokenizer, logger)

    # Load checkpoint
    if rank == 0:
        logger.info(f"📂 Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if rank == 0:
            logger.info("✅ Model state loaded successfully")
    else:
        if rank == 0:
            logger.warning("⚠️ No model_state_dict found in checkpoint, trying direct load")
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create generator
    generator = AVDNGeneratorWithAgent(
        config=config,
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_dir,
        device=device
    )

    # Process all splits with distributed processing
    if rank == 0:
        print(f"\n🚀 Starting AVDN dataset generation...")
    
    generator.process_all_splits(
        splits=args.splits,
        sample_ratio=args.sample_ratio,
        max_samples=args.max_samples,
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        print(f"\n✅ AVDN dataset generation complete!")
        print(f"📁 Generated dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 