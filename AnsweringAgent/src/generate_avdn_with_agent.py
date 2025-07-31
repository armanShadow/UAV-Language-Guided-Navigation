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
    

    
    def create_avdn_to_preprocessed_mapping(self, avdn_data: List[Dict], formatted_dataset: AnsweringDataset, rank: int = 0) -> Dict[int, int]:
        """Create mapping between AVDN and formatted dataset using PKL files."""
        print("Creating AVDN to formatted dataset mapping...")
        
        mapping = {}
        
        # TODO: IMPLEMENT YOUR OWN MATCHING LOGIC HERE
        # 
        # You now have access to:
        # - avdn_data: Original AVDN dataset with 'map_name', 'route_index', 'instructions', 'pre_dialogs'
        # - formatted_dataset: AnsweringDataset object with PKL data
        # 
        # You can access formatted dataset samples like:
        # sample = formatted_dataset[i]  # Returns dict with 'text_input', 'text_label', 'current_view_image', etc.
        # 
        # Example matching criteria you might want to consider:
        # 1. Map name matching
        # 2. Answer text matching (decode text_label)
        # 3. Question text matching (parse from text_input)
        # 4. Route index to episode mapping
        # 5. Visual feature similarity
        # 
        # For now, this will find no matches until you implement your logic
        matched_count = 0
        qa_samples_count = 0
        
        # Process each AVDN sample
        print(f"🔍 Processing {len(avdn_data)} AVDN samples against {len(formatted_dataset)} formatted samples...")
        
        for i, avdn_sample in enumerate(avdn_data):
            if i % 100 == 0 and rank == 0:  # Progress update every 100 samples
                print(f"📊 Processing AVDN sample {i}/{len(avdn_data)}...")
                
            instruction = avdn_sample['instructions']
            map_name = avdn_sample['map_name']
            route_index = avdn_sample['route_index']
            pre_dialogs = avdn_sample['pre_dialogs']
            
            # Only process entries with both question and answer (not first instructions)
            if '[QUE]' not in instruction or '[INS]' not in instruction:
                continue  # Skip first instructions
                
            qa_samples_count += 1
            
            # Extract question and answer from AVDN instruction
            que_start = instruction.find('[QUE]')
            ins_start = instruction.find('[INS]')
            avdn_question = instruction[que_start+5:ins_start].strip().lower()
            avdn_answer = instruction[ins_start+5:].strip().lower()

            first_ins_start = pre_dialogs[0].find('[INS]')
            first_instruction = pre_dialogs[0][first_ins_start+5:].strip().lower()

            # Find matching sample in formatted dataset
            best_match = None
            
            # Pre-filter by map_name for efficiency
            matching_samples = []
            for j in range(len(formatted_dataset)):
                sample = formatted_dataset[j]
                turn_id = sample['turn_id']
                avdn_turn_id = route_index[route_index.find('_')+1:]
                if sample['map_name'] == map_name and 'aug_pattern' not in sample['episode_id'] and int(turn_id) + 1 == int(avdn_turn_id):
                    matching_samples.append((j, sample))
            
            # Now check fuzzy matches only on pre-filtered samples
            candidate_samples = []
            if len(matching_samples) > 0:
                for j, sample in matching_samples:
                    # Check Fuzzy text matches (case-insensitive)
                    from difflib import SequenceMatcher
                    if SequenceMatcher(None, sample['answer'].strip().lower(), avdn_answer).ratio() > 0.9:
                        candidate_samples.append((j, sample))

            if len(candidate_samples) > 0:
                for j, sample in candidate_samples:
                    if SequenceMatcher(None, sample['question'].strip().lower(), avdn_question).ratio() > 0.9 and \
                        SequenceMatcher(None, sample['first_instruction'].strip().lower(), first_instruction).ratio() > 0.9:
                        best_match = j
                        break
            else:
                # Debug: No samples found for this map_name
                if i < 10 and rank == 0:
                    print(f"⚠️  No formatted samples found for map_name: {map_name}, {avdn_turn_id}")
                

            if best_match is not None:
                mapping[i] = best_match
                matched_count += 1
                
                if matched_count <= 5 and rank == 0:
                    print(f"✅ Match {matched_count}: AVDN[{i}] map:{map_name} route:{route_index} -> Formatted[{best_match}]")
                    print(f"   Answer: {avdn_answer[:50]}...")
                    print(f"   Question: {avdn_question[:50]}...")
            else:
                if matched_count < 5:
                    print(f"❌ No match found for AVDN[{i}] map:{map_name} route:{route_index}")
                    print(f"   Answer: {avdn_answer[:50]}...")
        
        total_qa_samples = qa_samples_count
        skipped_samples = total_qa_samples - matched_count
        
        print(f"\n📊 Mapping Summary:")
        print(f"🔍 Q&A samples to process: {total_qa_samples}/{len(avdn_data)} total samples")
        print(f"✅ Successfully matched: {matched_count}/{total_qa_samples} Q&A samples ({matched_count/total_qa_samples*100:.1f}%)")
        print(f"⚠️  No match found: {skipped_samples} Q&A samples ({skipped_samples/total_qa_samples*100:.1f}%)")
        
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
                formatted_index = mapping[avdn_index]
                matching_sample = formatted_dataset[formatted_index]
                
                if avdn_index < 2 and rank == 0:  # Debug first few matches
                    dialog_context = self.decode_tokenized_text(matching_sample['text_input'])
                    formatted_answer = self.decode_tokenized_text(matching_sample['text_label'])
                    print(f"Mapped match {avdn_index}: AVDN({map_name}, {route_index}) -> Formatted[{formatted_index}]")
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
            formatted_index = mapping[avdn_index]
            
            # Add debug fields to the output for verification
            new_sample['_debug_info'] = {
                'matched_formatted_index': formatted_index,
                'original_context_preview': self.decode_tokenized_text(matching_sample['text_input'])[:200] + "...",
                'original_answer': avdn_sample['instructions'],
                'generated_answer': new_answer,
                'formatted_answer': self.decode_tokenized_text(matching_sample['text_label']),
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
    
    def broadcast_mapping(self, mapping: Dict[int, int], rank: int, world_size: int) -> Dict[int, int]:
        """Broadcast mapping from rank 0 to all other ranks."""
        import torch.distributed as dist
        
        if world_size == 1:
            return mapping
        
        # Use broadcast_object_list for proper broadcasting
        if rank == 0:
            print(f"📡 Broadcasting mapping of {len(mapping)} samples to all {world_size} GPUs")
            # Convert mapping to list for broadcasting
            mapping_list = [mapping]
        else:
            mapping_list = [None]
        
        # Broadcast the mapping from rank 0 to all ranks
        dist.broadcast_object_list(mapping_list, src=0)
        
        # Extract the mapping from the list
        broadcasted_mapping = mapping_list[0]
        
        if rank == 0:
            print(f"✅ Successfully broadcasted mapping to all ranks")
        
        return broadcasted_mapping
    
    def process_split(self, split: str, sample_ratio: float = 1.0, 
                     rank: int = 0, world_size: int = 1) -> List[Dict]:
        """Process an entire split with simplified distributed generation."""
        if rank == 0:
            print(f"\n🚀 Processing {split} split with {world_size} GPUs...")
        
        # STEP 1: Load all data on all ranks
        avdn_data = self.load_avdn_data(split)
        formatted_dataset = self.load_formatted_dataset(split)
        
        # STEP 2: Create mapping on FULL dataset (rank 0 only), then broadcast
        if rank == 0:
            print("🔍 Creating mapping on full dataset...")
            mapping = self.create_avdn_to_preprocessed_mapping(avdn_data, formatted_dataset, rank)
        else:
            mapping = {}
        
        # STEP 3: Broadcast mapping to all ranks
        if world_size > 1:
            mapping = self.broadcast_mapping(mapping, rank, world_size)
        
        # STEP 4: Apply sampling AFTER mapping (on rank 0, then broadcast)
        if rank == 0:
            if sample_ratio < 1.0:
                num_samples = int(len(avdn_data) * sample_ratio)
                avdn_data = avdn_data[:num_samples]
                print(f"📊 Sampled {num_samples}/{len(self.load_avdn_data(split))} samples ({sample_ratio*100:.1f}%)")
        
        # STEP 5: Broadcast sampled data to all ranks
        if world_size > 1:
            avdn_data = self.broadcast_data(avdn_data, rank, world_size)
        
        # STEP 6: Distribute work among ranks (simple sample-based distribution)
        total_samples = len(avdn_data)
        samples_per_rank = total_samples // world_size
        extra_samples = total_samples % world_size
        
        # Calculate this rank's sample indices
        if rank < extra_samples:
            start_idx = rank * (samples_per_rank + 1)
            end_idx = start_idx + samples_per_rank + 1
        else:
            start_idx = rank * samples_per_rank + extra_samples
            end_idx = start_idx + samples_per_rank
        
        my_samples = avdn_data[start_idx:end_idx]
        my_indices = list(range(start_idx, end_idx))
        
        print(f"🔧 Rank {rank}: Processing {len(my_samples)}/{total_samples} samples (indices {start_idx}-{end_idx-1})")
        
        # STEP 7: Process samples
        local_processed_data = {}
        local_scores = []
        local_successful_generations = 0
        
        desc = f"Rank {rank} processing {split}"
        for local_idx, (sample_idx, sample) in enumerate(zip(my_indices, tqdm(my_samples, desc=desc, disable=(rank != 0)))):
            try:
                processed_sample = self.process_avdn_sample(sample, formatted_dataset, mapping, sample_idx, rank)
                local_processed_data[sample_idx] = processed_sample
                
                # Check if generation was successful
                if processed_sample['instructions'] != sample['instructions']:
                    local_successful_generations += 1
                    
                    # Calculate scores
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
                    
                    scores = composite_score(generated_answer, original_answer, task_type="precision_short")
                    local_scores.append(scores)
                    
                    # Debug scoring for first few samples (only rank 0)
                    if local_successful_generations <= 2 and rank == 0:
                        print(f"\n🔍 Scoring Debug for sample {sample_idx}:")
                        print(f"   Original: {original_answer}")
                        print(f"   Generated: {generated_answer}")
                        print(f"   Direction score: {scores['direction']}")
                        print(f"   Movement score: {scores['movement']}")
                        print(f"   Landmark score: {scores['landmark']}")
                
            except Exception as e:
                print(f"⚠️ Rank {rank}: Error processing sample {sample_idx}: {e}")
                local_processed_data[sample_idx] = sample
        
        print(f"🔧 Rank {rank}: Completed processing. Generated {local_successful_generations} samples")
        
        # STEP 8: Gather results from all ranks
        if world_size > 1:
            if rank == 0:
                print("🔄 Gathering results from all ranks...")
            
            gathered_data = [None for _ in range(world_size)]
            gathered_scores = [None for _ in range(world_size)]
            gathered_stats = [None for _ in range(world_size)]
            
            dist.all_gather_object(gathered_data, local_processed_data)
            dist.all_gather_object(gathered_scores, local_scores)
            dist.all_gather_object(gathered_stats, {
                'successful_generations': local_successful_generations,
                'rank': rank
            })
            
            # Combine results
            processed_data = [None] * total_samples
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
                print(f"📊 Per-rank stats:")
                for i, stats in enumerate(gathered_stats):
                    print(f"   Rank {i}: {stats['successful_generations']} generations")
        else:
            # Single GPU case
            processed_data = [None] * total_samples
            for idx, sample in local_processed_data.items():
                processed_data[idx] = sample
            all_scores = local_scores
            successful_generations = local_successful_generations
        
        # Fill in any None values with original samples
        for i, sample in enumerate(processed_data):
            if sample is None:
                processed_data[i] = avdn_data[i]
        
        # Print examples and final metrics (only rank 0)
        if rank == 0:
            examples_shown = 0
            for i, (original, processed) in enumerate(zip(avdn_data, processed_data)):
                if examples_shown >= 2:
                    break
                if processed['instructions'] != original['instructions']:
                    examples_shown += 1
                    print(f"\nGenerated Example {examples_shown}:")
                    print(f"Map: {original['map_name']}, Route: {original['route_index']}")
                    print(f"Original: {original['instructions']}")
                    print(f"Generated: {processed['instructions']}")
                    print("-" * 80)
            
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
                          rank: int = 0, world_size: int = 1):
        """Process all specified splits with distributed processing."""
        overall_metrics = {}
        
        for split in splits:
            processed_data, scores = self.process_split(split, sample_ratio, rank, world_size)
            
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
        rank=rank,
        world_size=world_size
    )

    if rank == 0:
        print(f"\n✅ AVDN dataset generation complete!")
        print(f"📁 Generated dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 