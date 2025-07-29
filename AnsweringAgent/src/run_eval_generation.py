import torch
import os
import random
import time
import json
import argparse
from transformers import T5Tokenizer
from typing import Dict, Iterable, Tuple, Optional, List
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Import your model class and the scoring utils you appended
from models.answering_agent import (
    AnsweringAgent,
    composite_score, direction_score, yesno_score, attribute_score,
)
from config import Config
from data.dataset import AnsweringDataset
from utils.logger import setup_logger

# -------------------------
# Lightweight detection helpers
# -------------------------
YES_TOKENS = {
    "yes", "yeah", "yep", "affirmative",
    "in your field of view", "in your field of vision",
}
NO_TOKENS = {
    "no", "nope", "negative",
    "not in your field of view", "not in your field of vision",
}

COLOR_SET = {"gray","grey","brown","red","blue","green","white","black","sand"}
SHAPE_SET = {"square","rectangular","rectangle","c-shaped","c shape","circle","round"}

def gold_yesno(gold: str) -> Optional[bool]:
    gl = gold.lower()
    if any(t in gl for t in YES_TOKENS): return True
    if any(t in gl for t in NO_TOKENS):  return False
    return None

# Split task-type detectors
def detect_task_type_question_only(question: str) -> str:
    q = question.lower()
    attr_kw = [
        "color","colour","shape","look like","appearance",
        "c shaped","c-shaped","rectangular","rectangle",
        "square","round","circle","basketball court","roof",
    ]
    if any(k in q for k in attr_kw):
        return "attribute_complete"
    dir_kw = ["which direction","go to","head to","turn to","o'clock","clock"]
    if any(k in q for k in dir_kw):
        return "precision_short"
    return "precision_short"

def detect_task_type_oracle(question: str, gold: str) -> str:
    # Uses gold only for analysis/upper bound — not for headline results
    q = question.lower(); g = gold.lower()
    attr_kw = [
        "color","colour","shape","look like","appearance",
        "c shaped","c-shaped","rectangular","rectangle",
        "square","round","circle","basketball court","roof",
    ]
    if any(k in q for k in attr_kw):
        return "attribute_complete"
    if any(w in g for w in (COLOR_SET | SHAPE_SET)):
        return "attribute_complete"
    dir_kw = ["which direction","go to","head to","turn to","o'clock","clock"]
    if any(k in q for k in dir_kw):
        return "precision_short"
    return "precision_short"

def detect_task_type(question: str, gold: str, mode: str) -> str:
    if mode == "oracle":
        return detect_task_type_oracle(question, gold)
    return detect_task_type_question_only(question)
# -------------------------
# Generation presets (text-mode sweep)
# -------------------------
BANNED = [
    "Yes, fate is in your field of vision",
    "Yes, your destination is in your field of view",
]

REQ_LEX = ["gray","grey","brown","red","blue","white","black",
           "square","rectangular","rectangle","c-shaped","circle","round"]

# Helper to move tensors to device
def to_device_text_input(d: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = to_device_text_input(v, device)
        elif torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out

# -------------------------
# Distributed setup functions
# -------------------------
def setup_distributed():
    """Set up distributed evaluation."""
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
    """Setup environment for distributed evaluation."""
    os.environ["NCCL_DEBUG"] = "WARN"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# -------------------------
# Hint tags for efficient instruction generation
# -------------------------
HINT_TAGS = {
    'spatial': "HINT: Use clock directions (1-12 o'clock) and landmark descriptions.",
    'movement': "HINT: Provide clear movement instructions (turn left/right, go straight).",
    'landmark': "HINT: Describe visible landmarks, buildings, and structures.",
    'navigation': "HINT: Give precise navigation with distance and direction."
}

def add_hint_to_text_input(tokenizer, text_input: Dict, hint_type: str = 'spatial') -> Dict:
    """Add hint tag to text input efficiently without token waste."""
    hint = HINT_TAGS.get(hint_type, HINT_TAGS['spatial'])
    
    # Create a new text input with hint prepended
    original_text = tokenizer.decode(text_input['input_ids'], skip_special_tokens=True)
    hinted_text = f"{hint} {original_text}"
    
    # Tokenize with efficient settings
    tokenized = tokenizer(
        hinted_text,
        max_length=512,  # Use reasonable max length
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    return {
        'input_ids': tokenized['input_ids'].squeeze(0),
        'attention_mask': tokenized['attention_mask'].squeeze(0)
    }

def sample_dataset_indices(dataset_size: int, sample_ratio: float = 0.1) -> List[int]:
    """Sample 10% of dataset indices efficiently."""
    num_samples = int(dataset_size * sample_ratio)
    return random.sample(range(dataset_size), num_samples)

# -------------------------
# Dataset iterator for distributed evaluation
# -------------------------
def iter_dataset_distributed(split: str, config: Config, tokenizer, 
                           sample_ratio: float = 0.1, rank: int = 0, world_size: int = 1) -> Iterable[Tuple[Dict, torch.Tensor, torch.Tensor, str, str]]:
    """
    Yield samples for the given split with 10% sampling and distributed processing.
    """
    # Create dataset
    dataset = AnsweringDataset(config, split=split, tokenizer=tokenizer)
    
    # Sample 10% of dataset indices
    dataset_size = len(dataset)
    sampled_indices = sample_dataset_indices(dataset_size, sample_ratio)
    
    # Create subset with sampled indices
    sampled_dataset = Subset(dataset, sampled_indices)
    
    # Use DistributedSampler for proper distribution
    sampler = DistributedSampler(
        sampled_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    dataloader = DataLoader(
        sampled_dataset,
        batch_size=1,  # Process one sample at a time for evaluation
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    for batch in dataloader:
        # Extract single sample from batch
        text_input = batch['text_input']
        current_view = batch['current_view_image']
        previous_views = batch['previous_views_image']
        
        # Get original question and answer text
        question = tokenizer.decode(text_input['input_ids'], skip_special_tokens=True)
        gold_answer = tokenizer.decode(batch['text_label']['input_ids'], skip_special_tokens=True)
        
        yield text_input, current_view, previous_views, question, gold_answer


# -------------------------
# Generation presets (text-mode sweep)
# -------------------------
BANNED = [
    "Yes, fate is in your field of vision",
    "Yes, your destination is in your field of view",
]

REQ_LEX = ["gray","grey","brown","red","blue","white","black",
           "square","rectangular","rectangle","c-shaped","circle","round"]

PRESETS: Dict[str, Dict] = {
    # Direction/Nearness focused
    "precision_short_default": dict(
        task_type="precision_short",
        num_beams=5,
        do_sample=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.25,
        length_penalty=0.4,
        min_new_tokens=3,
        max_new_tokens=18,
        early_stopping=True,
        banned_phrases=BANNED,
        bad_penalty=6.0,
    ),
    # Attribute + Direction
    "attribute_complete_default": dict(
        task_type="attribute_complete",
        num_beams=6,
        do_sample=False,
        no_repeat_ngram_size=4,
        repetition_penalty=1.15,
        length_penalty=1.15,
        min_new_tokens=12,
        max_new_tokens=40,
        early_stopping=True,
        banned_phrases=BANNED,
        required_lexicons=REQ_LEX,
        bad_penalty=5.0,
        req_boost=1.5,
    ),
    # A couple of simple variants for a tiny sweep
    "precision_short_rp135": dict(
        task_type="precision_short",
        num_beams=5,
        do_sample=False,
        no_repeat_ngram_size=3,
        repetition_penalty=1.35,
        length_penalty=0.4,
        min_new_tokens=3,
        max_new_tokens=18,
        early_stopping=True,
        banned_phrases=BANNED,
        bad_penalty=6.0,
    ),
    "attribute_complete_n4_rp135": dict(
        task_type="attribute_complete",
        num_beams=6,
        do_sample=False,
        no_repeat_ngram_size=4,
        repetition_penalty=1.35,
        length_penalty=1.10,
        min_new_tokens=12,
        max_new_tokens=40,
        early_stopping=True,
        banned_phrases=BANNED,
        required_lexicons=REQ_LEX,
        bad_penalty=5.0,
        req_boost=1.5,
    ),
}


# -------------------------
# Main evaluation loop (prints TEXT)
# -------------------------
def evaluate_split(
    model: AnsweringAgent,
    tokenizer: T5Tokenizer,
    split: str,
    preset_name: str,
    preset_kwargs: Dict,
    config: Config,
    rank: int = 0,
    world_size: int = 1,
    sample_ratio: float = 0.1,
    hint_types: List[str] = None,
    use_hints: bool = False,
    routing_mode: str = "question",
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate split with distributed processing and hint tags.
    Supports routing_mode in {"question", "oracle"} and use_hints for deterministic hinting.
    """
    if hint_types is None:
        hint_types = ['spatial', 'movement', 'landmark', 'navigation']
    if rank == 0:
        print("=" * 80)
        print(f"SPLIT: {split} | PRESET: {preset_name} | 10% SAMPLING")
        print("-" * 80)

    n = 0
    totals = {"direction": 0.0, "yesno": 0.0, "attribute": 0.0, "form": 0.0, "total": 0.0}
    hint_usage = {h: 0 for h in (hint_types or ['spatial','movement','landmark','navigation'])}
    hint_usage['none'] = 0

    # Get distributed dataset iterator
    dataset_iterator = iter_dataset_distributed(split, config, tokenizer, sample_ratio, rank, world_size)

    # DDP support for model
    gen_model = model.module if isinstance(model, DDP) else model

    for sample_idx, sample in enumerate(dataset_iterator):
        text_input, cur_view, prev_views, question, gold = sample
        task_type = detect_task_type(question, gold, routing_mode)

        # Choose preset; if its task_type mismatches detection, override softly
        kwargs = dict(preset_kwargs)
        kwargs.setdefault("task_type", task_type)

        hinted_text_input = text_input
        if use_hints:
            # Choose deterministic hint by task type
            hint_type = "landmark" if task_type == "attribute_complete" else "spatial"
            hinted_text_input = add_hint_to_text_input(tokenizer, text_input, hint_type)
            # Optional subfields
            if 'first_instruction_input' in text_input:
                hinted_text_input['first_instruction_input'] = add_hint_to_text_input(
                    tokenizer, text_input['first_instruction_input'], hint_type
                )
            if 'current_question_input' in text_input:
                hinted_text_input['current_question_input'] = add_hint_to_text_input(
                    tokenizer, text_input['current_question_input'], hint_type
                )
            hint_usage[hint_type] += 1
        else:
            hint_type = 'none'
            # hinted_text_input = text_input  # already set above
            hint_usage['none'] += 1

        # Ensure tensors are on the correct device
        hinted_text_input = to_device_text_input(hinted_text_input, device=torch.device(cur_view.device))
        cur_view = cur_view.to(cur_view.device)
        prev_views = prev_views.to(prev_views.device)

        with torch.no_grad():
            seq = gen_model.generate_answer(
                hinted_text_input, cur_view, prev_views,
                task_type=kwargs.pop("task_type"),
                **kwargs
            )
        pred = tokenizer.decode(seq[0], skip_special_tokens=True)

        # Clean generated text (remove hint if present)
        for hint in HINT_TAGS.values():
            if pred.startswith(hint.strip()):
                pred = pred[len(hint.strip()):].strip()
                break

        # Score
        sc = composite_score(
            pred, gold, task_type=task_type,
            banned_phrases=BANNED
        )

        # Accumulate
        for k in totals:
            totals[k] += sc[k]
        n += 1

        # Per-sample print (concise) - only on rank 0
        if rank == 0:
            print(f"[{n}] Task={task_type} | Hint={hint_type}")
            print(f"Q: {truncate(question)}")
            print(f"GOLD: {truncate(gold)}")
            print(f"PRED: {truncate(pred)}")
            print(f"Scores: dir={sc['direction']:.2f}  yn={sc['yesno']:.2f}  attr={sc['attribute']:.2f}  form={sc['form']:.2f}  total={sc['total']:.2f}")
            print("-" * 80)

        if max_samples and n >= max_samples:
            break

    if n == 0:
        if rank == 0:
            print("No samples found.")
        return {}

    # Calculate averages
    results = {k: v/n for k, v in totals.items()}
    results['hint_usage'] = hint_usage
    results['total_samples'] = n

    if rank == 0:
        print(f"SUMMARY  (n={n})")
        print(f"  Direction : {results['direction']:.3f}")
        print(f"  Yes/No    : {results['yesno']:.3f}")
        print(f"  Attribute : {results['attribute']:.3f}")
        print(f"  Form      : {results['form']:.3f}")
        print(f"  TOTAL     : {results['total']:.3f}")
        print(f"  Hint Usage: {hint_usage}")
        print("=" * 80)
        print()

    return results


def truncate(s: str, maxlen: int = 240) -> str:
    return s if len(s) <= maxlen else s[:maxlen-3] + "..."


def main():
    """Main distributed evaluation function."""
    parser = argparse.ArgumentParser(description="Distributed Evaluation with Hint Tags")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--sample_ratio", type=float, default=0.1,
                       help="Ratio of dataset to sample (default: 0.1 = 10%)")
    parser.add_argument("--splits", nargs="+",
                       default=['train', 'val_seen', 'val_unseen'],
                       help="Dataset splits to process")
    parser.add_argument("--hint_types", nargs="+",
                       default=['spatial', 'movement', 'landmark', 'navigation'],
                       help="Hint types to use for generation")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to process per split")
    parser.add_argument("--output_dir", type=str, default="./evaluation_outputs",
                       help="Output directory for results")
    # CLI flags for routing and hints
    parser.add_argument("--routing", choices=["question","oracle"], default="question",
                        help="Routing mode: question-only (fair) or oracle (uses gold; analysis only)")
    parser.add_argument("--use_hints", action="store_true",
                        help="Enable deterministic input hints")
    args = parser.parse_args()

    # Setup environment
    setup_environment()

    # Setup distributed training
    is_distributed, rank, world_size = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    if rank == 0:
        print(f"🚀 Distributed Evaluation with Hint Tags")
        print(f"🎯 Focus: 10% sampling with hint tags for optimal evaluation")
        print(f"World Size: {world_size}, Rank: {rank}")
        if torch.cuda.is_available():
            print(f"CUDA Devices: {torch.cuda.device_count()}")
        print(f"Routing mode: {args.routing}  |  Hints: {args.use_hints}")
        print()

    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    # Load config and model
    config = Config()
    tokenizer = T5Tokenizer.from_pretrained(config.model.t5_model_name)

    # Initialize logger only on rank 0
    if rank == 0:
        logger = setup_logger('distributed_evaluation', log_dir=config.log_dir)
    else:
        class DummyLogger:
            def info(self, msg): pass
            def warning(self, msg): pass
            def error(self, msg): pass
            def debug(self, msg): pass
        logger = DummyLogger()

    # Load model
    if rank == 0:
        logger.info("🏗️ Loading model for distributed evaluation...")
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

    # Run distributed evaluation
    run_distributed(model, tokenizer, config, args, rank, world_size)


def run_distributed(model: AnsweringAgent, tokenizer: T5Tokenizer, config: Config, args, rank: int, world_size: int):
    """Run distributed evaluation with hint tags."""
    all_results = {}

    # Evaluate on specified splits
    for split in args.splits:
        split_results = {}

        for name, kwargs in PRESETS.items():
            if rank == 0:
                print(f"\n🔍 Evaluating {split} with preset {name}")

            results = evaluate_split(
                model, tokenizer, split, name, kwargs, config,
                rank=rank, world_size=world_size,
                sample_ratio=args.sample_ratio,
                hint_types=args.hint_types,
                use_hints=args.use_hints,
                routing_mode=args.routing,
                max_samples=args.max_samples
            )

            if results:  # Only add if we got results
                split_results[name] = results

        all_results[split] = split_results

    # Save results (only on rank 0)
    if rank == 0:
        save_evaluation_results(all_results, args.output_dir)


def save_evaluation_results(results: Dict, output_dir: str):
    """Save evaluation results efficiently."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save comprehensive results in a single file
    output_file = os.path.join(output_dir, f"distributed_evaluation_results_{int(time.time())}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    
    # Print summary
    print(f"\n📊 DISTRIBUTED EVALUATION SUMMARY")
    for split, split_results in results.items():
        print(f"\n{split.upper()}:")
        for preset, preset_results in split_results.items():
            print(f"  {preset}:")
            print(f"    Total: {preset_results['total']:.3f}")
            print(f"    Direction: {preset_results['direction']:.3f}")
            print(f"    Yes/No: {preset_results['yesno']:.3f}")
            print(f"    Attribute: {preset_results['attribute']:.3f}")
            print(f"    Form: {preset_results['form']:.3f}")
            print(f"    Samples: {preset_results['total_samples']}")
            print(f"    Hint Usage: {preset_results['hint_usage']}")


def run(model: AnsweringAgent, tokenizer: T5Tokenizer):
    """Legacy run function for backward compatibility."""
    print("⚠️ Using legacy run function. Consider using main() for distributed evaluation.")
    # Small sweep: evaluate both presets on val_seen and val_unseen
    for split in ["val_seen", "val_unseen"]:
        for name, kwargs in PRESETS.items():
            evaluate_split(model, tokenizer, split, name, kwargs, max_samples=None)


if __name__ == "__main__":
    # If you prefer, replace with your actual initialization and then call run(model, tokenizer)
    main()