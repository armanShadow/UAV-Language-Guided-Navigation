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
from run_eval_generation import composite_score, direction_score, yesno_score, attribute_score, landmark_score, movement_score

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
            'task_type': 'precision_short',
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
        
        # Generate new answer using the Answering Agent with exact same inputs as evaluation
        with torch.no_grad():
            generated_seq = self.model.generate_answer(
                text_input, current_view, previous_views,
                **self.generation_params
            )
        
        # Decode the generated sequence
        new_answer = self.tokenizer.decode(generated_seq[0], skip_special_tokens=True)
        
        # Clean up the answer
        new_answer = new_answer.strip()
        
        return new_answer
    

    
    def update_avdn_instruction(self, avdn_sample: Dict, new_answer: str, turn_index: int) -> Dict:
        """Update AVDN sample with new instruction."""
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
    
    def create_avdn_to_preprocessed_mapping(self, avdn_data: List[Dict], preprocessed_json: List[Dict]) -> Dict[int, int]:
        """Create mapping from AVDN indices to preprocessed dataset indices using metadata."""
        print("Creating AVDN to preprocessed dataset mapping using episode-based metadata...")
        
        mapping = {}
        
        # Parse preprocessed episodes and create flattened turn index
        preprocessed_turns = []
        for episode in preprocessed_json:
            episode_id = episode.get('episode_id', '')
            
            # Skip augmented episodes (contains "aug_pattern")
            if 'aug_pattern' in episode_id:
                continue
                
            map_name = episode.get('map_name', '')
            
            # Process each dialog turn in the episode
            for dialog in episode.get('dialogs', []):
                turn_id = dialog.get('turn_id', 0)
                
                # Skip turn 0 (usually just observation, no Q&A)
                if turn_id == 0:
                    continue
                    
                answer = dialog.get('answer', '').strip().lower()
                question = dialog.get('question', '').strip().lower()
                
                turn_data = {
                    'episode_id': episode_id,
                    'map_name': map_name,
                    'turn_id': turn_id,
                    'answer': answer,
                    'question': question,
                    'preprocessed_index': len(preprocessed_turns)
                }
                preprocessed_turns.append(turn_data)
        
        print(f"Found {len(preprocessed_turns)} non-augmented dialog turns")
        
        # Create index for matching
        preprocessed_index = {}
        for i, turn in enumerate(preprocessed_turns):
            map_name = turn['map_name']
            answer = turn['answer']
            
            # Create keys for matching
            keys = [
                f"{map_name}_{answer}",
                f"{map_name}_{turn['episode_id']}_{turn['turn_id']}"
            ]
            
            for key in keys:
                if key not in preprocessed_index:
                    preprocessed_index[key] = []
                preprocessed_index[key].append(i)
        
        matched_count = 0
        for i, avdn_sample in enumerate(avdn_data):
            # Parse AVDN instruction to determine turn type
            instruction = avdn_sample['instructions']
            
            # Determine if this is a first instruction or Q&A turn
            if '[QUE]' in instruction and '[INS]' in instruction:
                # This is a Q&A turn
                que_start = instruction.find('[QUE]')
                ins_start = instruction.find('[INS]')
                avdn_question = instruction[que_start+5:ins_start].strip().lower()
                avdn_answer = instruction[ins_start+5:].strip().lower()
                is_first_instruction = False
            elif '[INS]' in instruction:
                # This is a first instruction (no question)
                avdn_answer = instruction.replace('[INS]', '').strip().lower()
                avdn_question = None
                is_first_instruction = True
            else:
                # Unrecognized format, skip
                print(f"Warning: Unrecognized instruction format for sample {i}: {instruction}")
                continue
            
            map_name = avdn_sample['map_name']
            
            # Find matching preprocessed turn based on turn type
            match_found = False
            best_match_idx = None
            best_score = 0
            
            for turn_idx, turn_data in enumerate(preprocessed_turns):
                # Only consider turns from the same map
                if turn_data['map_name'] != map_name:
                    continue
                
                preprocessed_question = turn_data['question'].strip().lower()
                preprocessed_answer = turn_data['answer'].strip().lower()
                
                # Check turn type compatibility
                if is_first_instruction:
                    # AVDN first instruction should match preprocessed turns without questions
                    if preprocessed_question:  # Skip turns that have questions
                        continue
                else:
                    # AVDN Q&A turn should match preprocessed turns with questions
                    if not preprocessed_question:  # Skip turns without questions
                        continue
                    
                    # For Q&A turns, also check question similarity
                    question_words = set(avdn_question.split())
                    prep_question_words = set(preprocessed_question.split())
                    question_overlap = len(question_words & prep_question_words)
                    
                    # Require significant question overlap for Q&A turns
                    if question_overlap < 3:  # Require at least 3 common words
                        continue
                
                # Check answer similarity
                answer_words = set(avdn_answer.split())
                prep_answer_words = set(preprocessed_answer.split())
                answer_overlap = len(answer_words & prep_answer_words)
                
                # Calculate similarity score
                total_words = max(len(answer_words), len(prep_answer_words))
                if total_words == 0:
                    continue
                    
                similarity_score = answer_overlap / total_words
                
                # For Q&A turns, also factor in question similarity
                if not is_first_instruction and preprocessed_question:
                    question_total = max(len(question_words), len(prep_question_words))
                    if question_total > 0:
                        question_similarity = question_overlap / question_total
                        similarity_score = (similarity_score + question_similarity) / 2
                
                # Update best match if this is better
                if similarity_score > best_score and similarity_score > 0.3:  # Minimum threshold
                    best_score = similarity_score
                    best_match_idx = turn_idx
            
            # If we found a good match, use it
            if best_match_idx is not None:
                mapping[i] = best_match_idx
                matched_count += 1
                match_found = True
                
                if i < 3:  # Debug first few matches
                    turn_data = preprocessed_turns[best_match_idx]
                    print(f"Match {i}: AVDN({map_name}, {avdn_sample['route_index']}) -> Turn {best_match_idx}")
                    print(f"  Turn type: {'First Instruction' if is_first_instruction else 'Q&A Turn'}")
                    print(f"  AVDN instruction: {instruction}")
                    print(f"  Preprocessed answer: {turn_data['answer']}")
                    print(f"  Preprocessed question: {turn_data['question']}")
                    print(f"  Episode ID: {turn_data['episode_id']}")
                    print(f"  Turn ID: {turn_data['turn_id']}")
                    print(f"  Similarity score: {best_score:.3f}")
            else:
                if i < 3:  # Debug failed matches
                    turn_type = 'First Instruction' if is_first_instruction else 'Q&A Turn'
                    print(f"No match found for AVDN sample {i} ({turn_type}): {instruction}")
        
        print(f"Successfully mapped {matched_count}/{len(avdn_data)} AVDN samples to non-augmented preprocessed turns")
        return mapping
    
    def process_avdn_sample(self, avdn_sample: Dict, formatted_dataset: AnsweringDataset, mapping: Dict[int, int], avdn_index: int) -> Dict:
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
                
                if avdn_index < 3:  # Debug first few matches
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
            # If no mapping found, skip this sample
            print(f"No mapping found for AVDN sample {avdn_index} ({map_name}_{route_index}), skipping")
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
        
        return new_sample
    
    def process_split(self, split: str, sample_ratio: float = 1.0, max_samples: Optional[int] = None) -> List[Dict]:
        """Process an entire split of the AVDN dataset."""
        print(f"\nProcessing {split} split...")
        
        # Load original AVDN data
        avdn_data = self.load_avdn_data(split)
        
        # Load formatted dataset for generation inputs
        formatted_dataset = self.load_formatted_dataset(split)
        
        # Load preprocessed JSON with metadata
        preprocessed_json = self.load_preprocessed_json(split)
        
        # Create mapping between AVDN and preprocessed datasets using metadata
        mapping = self.create_avdn_to_preprocessed_mapping(avdn_data, preprocessed_json)
        
        # Store preprocessed turns for later access
        self.preprocessed_turns = []
        for episode in preprocessed_json:
            episode_id = episode.get('episode_id', '')
            if 'aug_pattern' in episode_id:
                continue
            map_name = episode.get('map_name', '')
            for dialog in episode.get('dialogs', []):
                turn_id = dialog.get('turn_id', 0)
                if turn_id == 0:
                    continue
                turn_data = {
                    'episode_id': episode_id,
                    'map_name': map_name,
                    'turn_id': turn_id,
                    'answer': dialog.get('answer', ''),
                    'question': dialog.get('question', '')
                }
                self.preprocessed_turns.append(turn_data)
        
        # Apply sampling if requested
        if sample_ratio < 1.0:
            num_samples = int(len(avdn_data) * sample_ratio)
            avdn_data = avdn_data[:num_samples]
            print(f"Sampled {num_samples} samples from {split}")
        
        # Apply max_samples limit if specified
        if max_samples is not None:
            avdn_data = avdn_data[:max_samples]
            print(f"Limited to {max_samples} samples")
        
        # Process samples
        processed_data = []
        all_scores = []
        successful_generations = 0
        total_samples = len(avdn_data)
        
        for i, avdn_sample in enumerate(tqdm(avdn_data, desc=f"Processing {split}")):
            try:
                processed_sample = self.process_avdn_sample(avdn_sample, formatted_dataset, mapping, i)
                
                # Calculate metrics if generation was successful (not skipped)
                if processed_sample['instructions'] != avdn_sample['instructions']:
                    # Extract original answer from AVDN instruction
                    original_instruction = avdn_sample['instructions']
                    if '[INS]' in original_instruction:
                        original_answer = original_instruction.replace('[INS]', '').strip()
                    else:
                        original_answer = original_instruction.strip()
                    
                    # Extract generated answer from new instruction
                    new_instruction = processed_sample['instructions']
                    if '[INS]' in new_instruction:
                        generated_answer = new_instruction.replace('[INS]', '').strip()
                    else:
                        generated_answer = new_instruction.strip()
                    
                    # Calculate composite score
                    scores = composite_score(generated_answer, original_answer, task_type="precision_short")
                    all_scores.append(scores)
                    successful_generations += 1
                
                processed_data.append(processed_sample)
                
                # Print some examples with scores
                if i < 3:
                    print(f"\nSample {i+1}:")
                    print(f"Map: {avdn_sample['map_name']}, Route: {avdn_sample['route_index']}")
                    print(f"Original: {avdn_sample['instructions']}")
                    print(f"Generated: {processed_sample['instructions']}")
                    
                    # Show scores if generation was successful
                    if processed_sample['instructions'] != avdn_sample['instructions']:
                        scores = composite_score(
                            processed_sample['instructions'].replace('[INS]', '').strip(),
                            avdn_sample['instructions'].replace('[INS]', '').strip(),
                            task_type="precision_short"
                        )
                        print(f"Composite Score: {scores['total']:.4f}")
                        print(f"Direction Score: {scores['direction']:.4f}")
                        print(f"Movement Score: {scores['movement']:.4f}")
                    
                    print("-" * 80)
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Keep original sample if generation fails
                processed_data.append(avdn_sample)
        
        # Report final metrics for this split
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
        
        return processed_data
    
    def save_processed_data(self, data: List[Dict], split: str):
        """Save processed data to output directory in AVDN format."""
        output_file = os.path.join(self.output_dir, f'{split}_data.json')
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} samples to {output_file}")
    
    def process_all_splits(self, splits: List[str], sample_ratio: float = 1.0, 
                          max_samples: Optional[int] = None):
        """Process all specified splits."""
        overall_metrics = {}
        
        for split in splits:
            processed_data = self.process_split(split, sample_ratio, max_samples)
            self.save_processed_data(processed_data, split)
            
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
        
        # Print overall summary
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
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                       help="Ratio of dataset to sample (default: 0.1 = 10%)")
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
        print(f"AVDN Data Dir: {args.avdn_data_dir}")
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

    # Process all splits
    if rank == 0:
        print(f"\n🚀 Starting AVDN dataset generation...")
    
    generator.process_all_splits(
        splits=args.splits,
        sample_ratio=args.sample_ratio,
        max_samples=args.max_samples
    )

    if rank == 0:
        print(f"\n✅ AVDN dataset generation complete!")
        print(f"📁 Generated dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 