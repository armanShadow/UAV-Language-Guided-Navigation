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
                 avdn_data_dir: str, output_dir: str, device: torch.device):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.avdn_data_dir = avdn_data_dir
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
        data_file = os.path.join(self.avdn_data_dir, f'{split}_data.json')
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
        # Decode the tokenized components
        first_instruction = self.decode_tokenized_text(sample['first_instruction_input'])
        current_question = self.decode_tokenized_text(sample['current_question_input'])
        current_answer = self.decode_tokenized_text(sample['text_label'])
        
        # Extract dialog history from the unified text input
        dialog_context = self.decode_tokenized_text(sample['text_input'])
        
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
        
        # Use the exact same text_input as the evaluation script (from formatted dataset)
        text_input = {
            'input_ids': formatted_sample['text_input']['input_ids'].unsqueeze(0).to(self.device),
            'attention_mask': formatted_sample['text_input']['attention_mask'].unsqueeze(0).to(self.device)
        }
        
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
    
    def create_dataset_index(self, formatted_dataset: AnsweringDataset) -> Dict[str, int]:
        """Create an index mapping (episode_id, turn_id, map_name) to dataset indices."""
        print("Creating dataset index for efficient matching...")
        
        index = {}
        for i in range(len(formatted_dataset)):
            try:
                sample = formatted_dataset[i]
                # Extract metadata from the preprocessed data
                # The normalizer stores this information in the dialog context
                dialog_context = self.decode_tokenized_text(sample['text_input'])
                
                # Extract episode and turn information from context
                # This is a heuristic - we'll need to find patterns in your data
                # For now, we'll use the index as a fallback
                key = f"sample_{i}"  # Fallback key
                index[key] = i
                
                if i < 5:  # Debug first few samples
                    print(f"Sample {i} context preview: {dialog_context[:100]}...")
                    
            except Exception as e:
                print(f"Error indexing sample {i}: {e}")
                continue
        
        print(f"Created index with {len(index)} entries")
        return index
    
    def process_avdn_sample(self, avdn_sample: Dict, formatted_dataset: AnsweringDataset, dataset_index: Dict[str, int]) -> Dict:
        """Process a single AVDN sample and generate new instruction."""
        # For now, we'll use a simple sequential matching as a starting point
        # This assumes both datasets are in the same order (which they likely are)
        
        # Get AVDN sample metadata
        map_name = avdn_sample['map_name']
        route_index = avdn_sample['route_index']
        
        # Try different matching strategies
        matching_sample = None
        sample_index = None
        
        # Strategy 1: Try to find in dataset index (if we had proper metadata)
        potential_key = f"{map_name}_{route_index}"
        if potential_key in dataset_index:
            sample_index = dataset_index[potential_key]
            matching_sample = formatted_dataset[sample_index]
        
        # Strategy 2: Linear search through formatted dataset (limited)
        if matching_sample is None:
            for i in range(min(100, len(formatted_dataset))):  # Limit search for efficiency
                try:
                    sample = formatted_dataset[i]
                    # Check if this sample could match based on text content
                    dialog_context = self.decode_tokenized_text(sample['text_input'])
                    
                    # Look for map name in the context
                    if map_name.lower() in dialog_context.lower():
                        matching_sample = sample
                        sample_index = i
                        break
                        
                except Exception as e:
                    continue
        
        # Strategy 3: Use index-based matching as fallback (assumes same order)
        if matching_sample is None and route_index < len(formatted_dataset):
            try:
                matching_sample = formatted_dataset[route_index % len(formatted_dataset)]
                sample_index = route_index % len(formatted_dataset)
                print(f"Using index-based matching: AVDN route {route_index} -> formatted sample {sample_index}")
            except:
                pass
        
        if matching_sample is None:
            # If no match found, skip this sample
            print(f"No matching formatted sample found for {map_name}_{route_index}, skipping")
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
        
        # Create dataset index for efficient matching
        dataset_index = self.create_dataset_index(formatted_dataset)
        
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
        
        for i, avdn_sample in enumerate(tqdm(avdn_data, desc=f"Processing {split}")):
            try:
                processed_sample = self.process_avdn_sample(avdn_sample, formatted_dataset, dataset_index)
                processed_data.append(processed_sample)
                
                # Print some examples
                if i < 3:
                    print(f"\nSample {i+1}:")
                    print(f"Map: {avdn_sample['map_name']}, Route: {avdn_sample['route_index']}")
                    print(f"Original: {avdn_sample['instructions']}")
                    print(f"Generated: {processed_sample['instructions']}")
                    print("-" * 80)
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                # Keep original sample if generation fails
                processed_data.append(avdn_sample)
        
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
        for split in splits:
            processed_data = self.process_split(split, sample_ratio, max_samples)
            self.save_processed_data(processed_data, split)


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
    parser.add_argument("--avdn_data_dir", type=str, 
                       default="../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/annotations",
                       help="Directory containing AVDN annotation files")
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
        avdn_data_dir=args.avdn_data_dir,
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