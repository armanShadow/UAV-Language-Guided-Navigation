import torch
import os
import re
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

# NLP evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Import your model class
from models.answering_agent import AnsweringAgent
from config import Config
from data.dataset import AnsweringDataset
from utils.logger import setup_logger

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Suppress transformers logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# Suppress BERTScore logging
logging.getLogger("bert_score").setLevel(logging.ERROR)

# -------------------------
# Lightweight detection helpers
# -------------------------
YES_TOKENS = {
    # basic affirmatives
    "yes", "yeah", "yep", "affirmative",
    # camera‐visibility affirmatives
    "in your field of view", "in your field of vision",
    "you can see it", "you can see the destination", "visible", "in sight",
}

NO_TOKENS = {
    # basic negatives
    "no", "nope", "negative",
    # camera‐visibility negatives
    "not in your field of view", "not in your field of vision",
    "cannot see", "can't see", "not visible", "out of sight",
}

# Enhanced spatial feature definitions (inspired by validation_pipeline.py)
SPATIAL_FEATURES = {
    'directions': {
        'regex_patterns': [
            r'\d+\s*o\'?clock', r'one\s+o\'?clock', r'two\s+o\'?clock', r'three\s+o\'?clock', 
            r'four\s+o\'?clock', r'five\s+o\'?clock', r'six\s+o\'?clock', r'seven\s+o\'?clock',
            r'eight\s+o\'?clock', r'nine\s+o\'?clock', r'ten\s+o\'?clock', r'eleven\s+o\'?clock',
            r'twelve\s+o\'?clock'
        ],
        'string_patterns': [
            'north', 'south', 'east', 'west', 
            'northwest', 'northeast', 'southwest', 'southeast',
            'northern', 'southern', 'eastern', 'western',
            'northeastern', 'northwestern', 'southeastern', 'southwestern',
            'left', 'right', 'forward', 'ahead', 'straight', 'backwards', 'backward', 'reverse'
        ],
        'synonyms': {
            'forward': ['forward', 'ahead', 'straight', 'front'],
            'backward': ['backward', 'backwards', 'reverse', 'back', 'behind'],
            'left': ['left', 'port'],
            'right': ['right', 'starboard'],
            'north': ['north', 'northern', 'northward'],
            'south': ['south', 'southern', 'southward'],
            'east': ['east', 'eastern', 'eastward'],
            'west': ['west', 'western', 'westward'],
            'northeast': ['northeast', 'northeastern', 'north-east'],
            'northwest': ['northwest', 'northwestern', 'north-west'],
            'southeast': ['southeast', 'southeastern', 'south-east'],
            'southwest': ['southwest', 'southwestern', 'south-west']
        }
    },
    'landmarks': {
        'string_patterns': [
            'building', 'structure', 'road', 'street', 'highway', 'house',
            'parking', 'lot', 'area', 'destination', 'target', 'goal', 'construction', 'edifice'
        ],
        'synonyms': {
            'building': ['building', 'structure', 'house', 'edifice', 'construction'],
            'road': ['road', 'street', 'highway', 'path'],
            'destination': ['destination', 'target', 'goal', 'endpoint']
        }
    },
    'colors': {
        'string_patterns': ["gray", "grey", "brown", "red", "blue", "green", "white", "black", "sand", "yellow", "orange", "pink", "purple", "beige"],
        'synonyms': {
            'gray': ['gray', 'grey'],
            'grey': ['grey', 'gray'],
        }
    },
    'shapes': {
        'string_patterns': ["square", "rectangular", "rectangle", "c-shaped", "c shape", "circle", "round", "triangular", "hexagonal", "octagonal", "l-shaped", "u-shaped", "cylindrical", "dome", "arch", "irregular"],
        'synonyms': {
            'rectangular': ['rectangular', 'rectangle'],
            'rectangle': ['rectangle', 'rectangular'],
            'square': ['square'],
            'round': ['round', 'circle'],
            'circle': ['circle', 'round'],
            'c-shaped': ['c-shaped', 'c shape'],
            'c shape': ['c shape', 'c-shaped'],
            'l-shaped': ['l-shaped', 'l shape'],
            'u-shaped': ['u-shaped', 'u shape'],            
        }
    },
    'movement_verbs': {
        'string_patterns': ['move', 'go', 'turn', 'head', 'fly', 'navigate', 'reverse', 'pivot', 'proceed', 'advance'],
        'synonyms': {
            'move': ['move', 'go', 'head', 'proceed', 'travel', 'navigate', 'advance'],
            'turn': ['turn', 'rotate', 'pivot', 'swing', 'veer'],
            'reverse': ['reverse', 'back', 'backwards', 'backward'],
            'fly': ['fly', 'soar', 'hover', 'pilot']
        }
    }
}

# Clock number mappings for equivalence checking
CLOCK_MAPPINGS = {
    '1': ['1', 'one'], '2': ['2', 'two'], '3': ['3', 'three'], '4': ['4', 'four'],
    '5': ['5', 'five'], '6': ['6', 'six'], '7': ['7', 'seven'], '8': ['8', 'eight'],
    '9': ['9', 'nine'], '10': ['10', 'ten'], '11': ['11', 'eleven'], '12': ['12', 'twelve']
}

def gold_yesno(gold: str) -> Optional[bool]:
    gl = gold.lower()
    if any(t in gl for t in YES_TOKENS): return True
    if any(t in gl for t in NO_TOKENS):  return False
    return None

def normalize_features(features: List[str], category: str) -> List[str]:
    """Normalize features using synonym mappings."""
    if category not in SPATIAL_FEATURES or 'synonyms' not in SPATIAL_FEATURES[category]:
        return features
    
    synonyms = SPATIAL_FEATURES[category]['synonyms']
    normalized = []
    
    for feature in features:
        # Find the canonical form (first synonym in each group)
        canonical = None
        for canonical_form, synonym_list in synonyms.items():
            if feature.lower() in [syn.lower() for syn in synonym_list]:
                canonical = canonical_form
                break
        
        if canonical:
            normalized.append(canonical)
        else:
            # If no synonym found, keep original
            normalized.append(feature)
    
    return list(set(normalized))  # Remove duplicates

def extract_spatial_features(text: str) -> Dict[str, List[str]]:
    """Extract spatial features from text using enhanced parsing."""
    text_lower = text.lower()
    features = {}
    
    for category, category_data in SPATIAL_FEATURES.items():
        found_features = []
        
        # Process regex patterns
        if 'regex_patterns' in category_data:
            for pattern in category_data['regex_patterns']:
                matches = re.findall(pattern, text_lower)
                found_features.extend(matches)
        
        # Process string patterns
        if 'string_patterns' in category_data:
            for pattern in category_data['string_patterns']:
                if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                    found_features.append(pattern)
        
        if found_features:
            # Normalize features using synonyms
            normalized_features = normalize_features(found_features, category)
            features[category] = normalized_features
    
    return features

def normalize_avdn_text(text: str) -> str:
    """Normalize AVDN text to fix common formatting issues."""
    import re
    
    # Fix clock direction patterns (most important fix)
    # Handle: "6'o clock", "11o'clock", "1 o clock", etc. -> "6 o'clock"
    text = re.sub(r"(\d+)'o\s+clock", r"\1 o'clock", text)  # "6'o clock" -> "6 o'clock"
    text = re.sub(r"(\d+)o'clock", r"\1 o'clock", text)      # "11o'clock" -> "11 o'clock"
    text = re.sub(r"(\d+)\s+o\s+clock", r"\1 o'clock", text) # "1 o clock" -> "1 o'clock"
    text = re.sub(r"(\d+)\s*o'clock'", r"\1 o'clock", text)  # "6 o'clock'" -> "6 o'clock"
    
    # Fix invalid clock times (30 o'clock should be ignored, but normalize format)
    text = re.sub(r"(\d+)\s*o'clock", lambda m: f"{m.group(1)} o'clock" if int(m.group(1)) <= 12 else m.group(0), text)
    
    # Normalize color variations
    text = re.sub(r'\bgrey\b', 'gray', text, flags=re.IGNORECASE)
    
    # Fix common typos
    text = re.sub(r'\bsorrounded\b', 'surrounded', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdestinaton\b', 'destination', text, flags=re.IGNORECASE)
    text = re.sub(r'\bappartment\b', 'apartment', text, flags=re.IGNORECASE)
    
    # Normalize whitespace issues
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
    text = text.strip()  # Remove leading/trailing spaces
    
    return text

def extract_clock_hour(direction_text: str) -> Optional[str]:
    """Extract clock hour from direction text with AVDN normalization."""
    # First normalize the text to handle AVDN formatting issues
    normalized_text = normalize_avdn_text(direction_text)
    
    # Match numeric clock (e.g., "5 o'clock") - now works on normalized text
    numeric_match = re.search(r'(\d+)\s*o\'?clock', normalized_text.lower())
    if numeric_match:
        hour = int(numeric_match.group(1))
        # Only return valid clock hours (1-12)
        if 1 <= hour <= 12:
            return str(hour)
    
    # Match word form clock (e.g., "five o'clock")
    for hour, variants in CLOCK_MAPPINGS.items():
        for variant in variants:
            if re.search(rf'\b{variant}\s+o\'?clock', normalized_text.lower()):
                return hour
    return None

def find_direction_synonym_match(orig_dir: str, para_dirs: List[str]) -> bool:
    """Find if original direction has synonym match in paraphrase directions."""
    synonyms = SPATIAL_FEATURES['directions']['synonyms']
    
    # Normalize the original direction
    orig_dir_lower = orig_dir.lower()
    
    # Check direct match first
    for para_dir in para_dirs:
        if orig_dir_lower == para_dir.lower():
            return True
    
    # Check synonym groups
    for base_dir, synonym_list in synonyms.items():
        if orig_dir_lower in [syn.lower() for syn in synonym_list]:
            for para_dir in para_dirs:
                para_dir_lower = para_dir.lower()
                # Check if paraphrase direction is in the same synonym group
                if para_dir_lower in [syn.lower() for syn in synonym_list]:
                    return True
                # Check if paraphrase direction contains the synonym (for "northeastern direction" cases)
                if any(syn.lower() in para_dir_lower for syn in synonym_list):
                    return True
    
    return False

def find_landmark_synonym_match(orig_landmark: str, para_landmarks: List[str]) -> bool:
    """Find if original landmark has synonym match in paraphrase landmarks."""
    synonyms = SPATIAL_FEATURES['landmarks']['synonyms']
    
    for base_landmark, synonym_list in synonyms.items():
        if orig_landmark.lower() in synonym_list:
            for para_landmark in para_landmarks:
                if any(syn in para_landmark.lower() for syn in synonym_list):
                    return True
    return False

# -------------------------
# Scoring functions
# -------------------------
def direction_score(pred: str, gold: str) -> float:
    """Score direction accuracy between prediction and gold using enhanced spatial parsing."""
    # Normalize texts to handle AVDN formatting issues
    pred_normalized = normalize_avdn_text(pred)
    gold_normalized = normalize_avdn_text(gold)
    
    # Extract spatial features using enhanced parsing on normalized text
    pred_features = extract_spatial_features(pred_normalized)
    gold_features = extract_spatial_features(gold_normalized)
    
    pred_dirs = pred_features.get('directions', [])
    gold_dirs = gold_features.get('directions', [])
    
    if not gold_dirs and not pred_dirs:
        return 1.0  # Both don't have directions
    if not gold_dirs or not pred_dirs:
        return 0.0
    
    # Extract clock hours for comparison
    pred_clock_hours = set()
    gold_clock_hours = set()
    
    for direction in pred_dirs:
        clock_hour = extract_clock_hour(direction)
        if clock_hour:
            pred_clock_hours.add(clock_hour)
    
    for direction in gold_dirs:
        clock_hour = extract_clock_hour(direction)
        if clock_hour:
            gold_clock_hours.add(clock_hour)
    
    # Clock direction similarity
    if gold_clock_hours and pred_clock_hours:
        # Both have clock directions - compare them
        clock_similarity = 1.0 if pred_clock_hours == gold_clock_hours else 0.0
    elif not gold_clock_hours and not pred_clock_hours:
        # Neither has clock directions - no clock information to compare
        clock_similarity = 0.0
    else:
        # One has clock directions, other doesn't - different
        clock_similarity = 0.0
    
    # Synonym-based similarity for other directions
    synonym_matches = 0
    total_directions = len(gold_dirs)
    
    for gold_dir in gold_dirs:
        if find_direction_synonym_match(gold_dir, pred_dirs):
            synonym_matches += 1
    
    synonym_similarity = synonym_matches / total_directions if total_directions > 0 else 0.0
    
    # Combined similarity
    return max(clock_similarity, synonym_similarity)

def yesno_score(pred: str, gold: str) -> float:
    """Score yes/no accuracy between prediction and gold."""
    pred_lower = pred.lower()
    gold_lower = gold.lower()
    
    # Determine gold answer type
    gold_ans = gold_yesno(gold)
    if gold_ans is None:
        return 0.0  # Return 0 for non-yes/no questions
    
    # Determine prediction answer type
    pred_ans = None
    if any(t in pred_lower for t in YES_TOKENS):
        pred_ans = True
    elif any(t in pred_lower for t in NO_TOKENS):
        pred_ans = False
    
    if pred_ans is None:
        return 0.0  # No clear yes/no in prediction
    
    return 1.0 if pred_ans == gold_ans else 0.0

def attribute_score(pred: str, gold: str) -> float:
    """Score attribute accuracy between prediction and gold using enhanced spatial parsing."""
    # Normalize texts to handle AVDN formatting issues
    pred_normalized = normalize_avdn_text(pred)
    gold_normalized = normalize_avdn_text(gold)
    
    # Extract spatial features using enhanced parsing on normalized text
    pred_features = extract_spatial_features(pred_normalized)
    gold_features = extract_spatial_features(gold_normalized)
    
    # Extract colors and shapes (already normalized by extract_spatial_features)
    pred_colors = pred_features.get('colors', [])
    gold_colors = gold_features.get('colors', [])
    
    pred_shapes = pred_features.get('shapes', [])
    gold_shapes = gold_features.get('shapes', [])
    
    scores = []
    # colours
    if gold_colors:
        if pred_colors:
            # Use set intersection since features are already normalized
            intersection = set(pred_colors) & set(gold_colors)
            scores.append(len(intersection) / len(set(gold_colors)))
        else:
            scores.append(0.0)
    # shapes
    if gold_shapes:
        if pred_shapes:
            # Use set intersection since features are already normalized
            intersection = set(pred_shapes) & set(gold_shapes)
            scores.append(len(intersection) / len(set(gold_shapes)))
        else:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0

def landmark_score(pred: str, gold: str) -> float:
    """Score landmark accuracy between prediction and gold using enhanced spatial parsing."""
    # Normalize texts to handle AVDN formatting issues
    pred_normalized = normalize_avdn_text(pred)
    gold_normalized = normalize_avdn_text(gold)
    
    # Extract spatial features using enhanced parsing on normalized text
    pred_features = extract_spatial_features(pred_normalized)
    gold_features = extract_spatial_features(gold_normalized)
    
    pred_landmarks = pred_features.get('landmarks', [])
    gold_landmarks = gold_features.get('landmarks', [])
    
    # Return 0 if gold has no landmarks
    if not gold_landmarks:
        return 0.0
    
    if not pred_landmarks:
        return 0.0
    
    # Create combined strings for multi-word landmark detection
    pred_combined = ' '.join(sorted(pred_landmarks)).lower()
    gold_combined = ' '.join(sorted(gold_landmarks)).lower()
    
    # Check for exact match first (handles "parking lot" cases)
    if pred_combined == gold_combined:
        return 1.0
    
    # Check for multi-word landmark combinations
    pred_compound = pred_combined.replace(' ', '')
    gold_compound = gold_combined.replace(' ', '')
    if pred_compound == gold_compound:
        return 1.0
    
    # Check if one is subset of other (e.g., "lot" in "parking lot")
    if pred_combined in gold_combined or gold_combined in pred_combined:
        return 0.8  # High similarity for subset matches
    
    # Traditional synonym-based matching
    synonym_matches = 0
    total_landmarks = len(gold_landmarks)
    
    for gold_landmark in gold_landmarks:
        if find_landmark_synonym_match(gold_landmark, pred_landmarks):
            synonym_matches += 1
    
    return synonym_matches / total_landmarks if total_landmarks > 0 else 0.0

def movement_score(pred: str, gold: str) -> float:
    """Score movement verb accuracy between prediction and gold using enhanced spatial parsing."""
    # Normalize texts to handle AVDN formatting issues
    pred_normalized = normalize_avdn_text(pred)
    gold_normalized = normalize_avdn_text(gold)
    
    # Extract spatial features using enhanced parsing on normalized text
    pred_features = extract_spatial_features(pred_normalized)
    gold_features = extract_spatial_features(gold_normalized)
    
    pred_movements = pred_features.get('movement_verbs', [])
    gold_movements = gold_features.get('movement_verbs', [])
    
    # If gold has no movement verbs, score is 0 by definition
    if not gold_movements:
        return 0.0

    if not pred_movements:
        return 0.0

    match_cnt = 0
    for g in gold_movements:
        if g.lower() in [p.lower() for p in pred_movements]:
            match_cnt += 1
        else:
            # synonym check
            for syn_list in SPATIAL_FEATURES['movement_verbs']['synonyms'].values():
                if g.lower() in [s.lower() for s in syn_list]:
                    if any(p.lower() in [s.lower() for s in syn_list] for p in pred_movements):
                        match_cnt += 1
                    break

    return match_cnt / len(gold_movements)

def calculate_bleu4(pred: str, gold: str) -> float:
    """Calculate BLEU-4 score between prediction and gold."""
    try:
        # Tokenize into words
        pred_tokens = pred.lower().split()
        gold_tokens = gold.lower().split()
        
        # Calculate BLEU-4 with smoothing
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu([gold_tokens], pred_tokens, smoothing_function=smoothing)
        return bleu_score
    except Exception as e:
        print(f"BLEU calculation error: {e}")
        return 0.0

def calculate_rouge_l(pred: str, gold: str) -> float:
    """Calculate ROUGE-L F1 score between prediction and gold."""
    try:
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        # Calculate ROUGE-L
        scores = scorer.score(gold, pred)
        return scores['rougeL'].fmeasure
    except Exception as e:
        print(f"ROUGE calculation error: {e}")
        return 0.0

def calculate_bertscore(pred: str, gold: str) -> float:
    """Calculate BERTScore F1 between prediction and gold."""
    try:
        # Skip BERTScore if environment variable is set to disable it
        if os.environ.get('DISABLE_BERTSCORE', 'false').lower() == 'true':
            return 0.0
            
        # Use GPU if available, otherwise CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Calculate BERTScore with verbose=False
        P, R, F1 = bert_score([pred], [gold], lang='en', verbose=False, device=device)
        return F1.mean().item()
    except Exception as e:
        # If BERTScore fails, return 0 instead of printing error
        return 0.0

def composite_score(pred: str, gold: str, task_type: str = "precision_short", 
                   banned_phrases: List[str] = None) -> Dict[str, float]:
    """Compute composite score with unified metrics across all tasks."""
    if banned_phrases is None:
        banned_phrases = []
    
    # Check for banned phrases - only apply penalty if gold doesn't contain them
    form_penalty = 0.0
    for phrase in banned_phrases:
        if phrase.lower() in pred.lower():
            # Only penalize if the gold answer doesn't contain this phrase
            if phrase.lower() not in gold.lower():
                form_penalty = 1.0
                break
    
    # Calculate individual scores (unified across all tasks)
    dir_score = direction_score(pred, gold)
    yn_score = yesno_score(pred, gold)
    attr_score = attribute_score(pred, gold)
    landmark_score_val = landmark_score(pred, gold)
    movement_score_val = movement_score(pred, gold)
    
    # Form score (inverse of penalty)
    form_score = 1.0 - form_penalty

    # Define weights for different metrics (higher = more important)
    weights = {
        'direction': 1.0,    # Most important for navigation
        'attribute': 0.8,    # Very important for landmark identification
        'landmark': 0.6,     # Important for navigation
        'movement': 0.6,     # Important for instructions
        'yesno': 0.4,        # Standard importance
        'form': 0.2,         # Least important (just avoids banned phrases)
    }
    
    # Build weighted score components
    weighted_scores = []
    total_weight = 0.0
    
    # Always include direction and form
    weighted_scores.append(weights['direction'] * dir_score)
    weighted_scores.append(weights['form'] * form_score)
    total_weight += weights['direction'] + weights['form']

    # Include yes/no only if the gold question is yes/no
    if gold_yesno(gold) is not None:
        weighted_scores.append(weights['yesno'] * yn_score)
        total_weight += weights['yesno']

    # include attribute / landmark / movement only if gold contains them
    if extract_spatial_features(gold).get('colors') or extract_spatial_features(gold).get('shapes'):
        weighted_scores.append(weights['attribute'] * attr_score)
        total_weight += weights['attribute']

    if extract_spatial_features(gold).get('landmarks'):
        weighted_scores.append(weights['landmark'] * landmark_score_val)
        total_weight += weights['landmark']

    if extract_spatial_features(gold).get('movement_verbs'):
        weighted_scores.append(weights['movement'] * movement_score_val)
        total_weight += weights['movement']

    total_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
    
    # Calculate NLP metrics
    bleu4_score = calculate_bleu4(pred, gold)
    rouge_l_score = calculate_rouge_l(pred, gold)
    bertscore_f1 = calculate_bertscore(pred, gold)
    
    return {
        "direction": dir_score,
        "yesno": yn_score,
        "attribute": attr_score,
        "landmark": landmark_score_val,
        "movement": movement_score_val,
        "form": form_score,
        "total": total_score,
        "bleu4": bleu4_score,
        "rouge_l": rouge_l_score,
        "bertscore": bertscore_f1
    }

# Split task-type detectors
def detect_task_type_question_only(question: str) -> str:
    """Decide task type using *only* the question text (fair routing)."""
    q = question.lower()

    # ── Attribute cues ───────────────────────────────────────────
    if re.search(r"\b(what|which)\s+(does|do|is)\s+(the\s+)?destination.*(look|colour|color|shape)", q):
        return "attribute_complete"
    if re.search(r"\b(color|colour|shape|look like|appearance|c\s*-?shaped|rectangular|rectangle|square|round|circle|basketball court|roof)\b", q):
        return "attribute_complete"

    # ── Direction cues ───────────────────────────────────────────
    if re.search(r"\b(which|what)\s+direction\b", q):
        return "precision_short"
    if re.search(r"\b\d+\s*o'?clock\b", q):
        return "precision_short"
    if re.search(r"\b(north|south|east|west|left|right|forward|backward)\b", q):
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
                           sample_ratio: float = 0.1, rank: int = 0, world_size: int = 1, 
                           device: torch.device = None, gen_model = None, use_hints: bool = False,
                           hint_types: List[str] = None, dataset: AnsweringDataset = None, **kwargs) -> Iterable[Tuple[str, str, str]]:
    """
    Yield processed samples for the given split with distributed processing.
    Now processes entire batches like the working distributed_generation_pipeline.py.
    """
    # Use pre-loaded dataset if provided, otherwise create new one and sample
    if dataset is None:
        dataset = AnsweringDataset(config, split=split, tokenizer=tokenizer)
        # Sample 10% of dataset indices
        dataset_size = len(dataset)
        sampled_indices = sample_dataset_indices(dataset_size, sample_ratio)
        # Create subset with sampled indices
        dataset = Subset(dataset, sampled_indices)
    else:
        # Dataset is already pre-sampled, use as is
        pass
    
    # Use DistributedSampler for proper distribution
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,  # Increased from 4 to 8 for better efficiency
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    for batch in dataloader:
        # Process the entire batch at once (like distributed_generation_pipeline.py)
        batch_size = batch['text_input']['input_ids'].size(0)
        
        # Move entire batch to device
        text_input = {k: v.to(device, non_blocking=True) for k, v in batch['text_input'].items()}
        
        # Handle separate encoding components
        if 'first_instruction_input' in batch:
            text_input['first_instruction_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['first_instruction_input'].items()}
        if 'current_question_input' in batch:
            text_input['current_question_input'] = {k: v.to(device, non_blocking=True) for k, v in batch['current_question_input'].items()}
        
        # Add hints if requested
        if use_hints:
            # Process each sample in the batch to add hints
            for i in range(batch_size):
                # Get question for task type detection
                question = tokenizer.decode(text_input['input_ids'][i], skip_special_tokens=True)
                current_question = tokenizer.decode(batch['current_question_input']['input_ids'][i], skip_special_tokens=True)
                gold = tokenizer.decode(batch['text_label']['input_ids'][i], skip_special_tokens=True)
                task_type = detect_task_type(current_question, gold, "question")
                
                # Choose hint type based on task
                hint_type = "landmark" if task_type == "attribute_complete" else "spatial"
                
                # Add hint to the main text input
                single_text_input = {
                    'input_ids': text_input['input_ids'][i],
                    'attention_mask': text_input['attention_mask'][i]
                }
                hinted = add_hint_to_text_input(tokenizer, single_text_input, hint_type)
                text_input['input_ids'][i] = hinted['input_ids'].to(device)
                text_input['attention_mask'][i] = hinted['attention_mask'].to(device)
        
        current_view = batch['current_view_image'].to(device, non_blocking=True)
        previous_views = batch['previous_views_image'].to(device, non_blocking=True)
        
        # Generate for the entire batch
        with torch.no_grad():
            # Get task_type safely from kwargs
            task_type = kwargs.get("task_type", "precision_short")
            # Remove task_type from kwargs to avoid passing it twice
            generation_kwargs = {k: v for k, v in kwargs.items() if k != "task_type"}
            
            seq = gen_model.generate_answer(
                text_input, current_view, previous_views,
                task_type=task_type,
                **generation_kwargs
            )
        
        # Process each result in the batch
        for i in range(batch_size):
            # Decode individual results
            unified_context = tokenizer.decode(text_input['input_ids'][i], skip_special_tokens=True)
            gold_answer = tokenizer.decode(batch['text_label']['input_ids'][i], skip_special_tokens=True)
            pred = tokenizer.decode(seq[i], skip_special_tokens=True)
            
            # Extract current question from the batch
            current_question = tokenizer.decode(batch['current_question_input']['input_ids'][i], skip_special_tokens=True)
            
            yield unified_context, gold_answer, pred, current_question


# -------------------------
# Generation presets (text-mode sweep)
# -------------------------
BANNED = [
    "Yes, fate is in your field of vision",
    "Yes, your destination is in your field of view",
    'You can see it',
]

# 1. add near the top
# ──────────────────────────────────────────────────────────────
#  Expanded lexical feature sets (quick manual expansion + sparse
#  sampling of training data revealed frequent extra tokens)      
# ──────────────────────────────────────────────────────────────

COLOR_SET = {
    # basic
    "gray","grey","brown","red","blue","green","white","black",
    # extra from data
    "sand","yellow","orange","pink","purple","beige"
}

SHAPE_SET = {
    "square","rectangular","rectangle","round","circle",
    "triangular","hexagonal","octagonal",
    "l-shaped","u-shaped","c-shaped","cylindrical",
    "dome","arch","irregular"
}

MATERIAL_SET = {
    "metal","concrete","glass","brick","wooden",
    "steel","asphalt","stone","tile"
}

SPATIAL_SET = {
    "north","south","east","west",
    "northwest","northeast","southwest","southeast",
    "left","right","forward","backward","ahead","behind"
}

LANDMARK_SET = {
    "building","structure","road","street","highway","house",
    "parking","lot","bridge","tower","river","runway"
}

MOVEMENT_SET = {
    "turn","move","go","head","fly","navigate",
    "proceed","advance","reverse","pivot"
}

# unified required lexicon capped to 35 tokens (sorted for reproducibility)
_FULL_LEX = sorted(list(COLOR_SET | MATERIAL_SET | SHAPE_SET | SPATIAL_SET | LANDMARK_SET | MOVEMENT_SET))
REQ_LEX = _FULL_LEX[:35]

PRESETS: Dict[str, Dict] = {
    # Conservative beam search
    "conservative": dict(
        task_type="precision_short",
        num_beams=4,
        do_sample=False,
        repetition_penalty=1.1,
        length_penalty=0.8,
        min_new_tokens=8,
        max_new_tokens=70,
        early_stopping=True,
    ),
    # Aggressive beam search 
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
    dataset: AnsweringDataset = None, # Added dataset parameter
    is_distributed: bool = False, # Added is_distributed parameter
) -> Dict[str, float]:
    """
    Evaluate split with distributed processing and hint tags.
    Supports routing_mode in {"question", "oracle"} and use_hints for deterministic hinting.
    """
    if hint_types is None:
        hint_types = ['spatial', 'movement', 'landmark', 'navigation']
    if rank == 0:
        print("=" * 80)
        print(f"SPLIT: {split} | PRESET: {preset_name} | {sample_ratio*100}% SAMPLING")
        print("-" * 80)

    # Counters
    n = 0  # total samples processed (for reporting)
    totals = {"direction": 0.0, "yesno": 0.0, "attribute": 0.0, "landmark": 0.0, "movement": 0.0, "form": 0.0, "total": 0.0, "bleu4": 0.0, "rouge_l": 0.0, "bertscore": 0.0}
    counts = {"direction": 0, "yesno": 0, "attribute": 0, "landmark": 0, "movement": 0, "form": 0, "bleu4": 0, "rouge_l": 0, "bertscore": 0}  # per-metric denom
    hint_usage = {h: 0 for h in (hint_types or ['spatial','movement','landmark','navigation'])}
    hint_usage['none'] = 0
    
    # Length tracking
    total_length = 0  # Total character length of all generated answers
    
    # Counter for showing examples (only show 2 per dataset)
    examples_shown = 0
    max_examples_to_show = 2

    # Get distributed dataset iterator with model and generation parameters
    gen_model = model.module if isinstance(model, DDP) else model
    
    # Use pre-loaded dataset if provided, otherwise create new one
    if dataset is None:
        dataset = AnsweringDataset(config, split=split, tokenizer=tokenizer)
    
    dataset_iterator = iter_dataset_distributed(
        split, config, tokenizer, sample_ratio, rank, world_size,
        device=next(gen_model.parameters()).device, gen_model=gen_model,
        use_hints=use_hints, hint_types=hint_types,
        dataset=dataset,  # Pass the dataset to the iterator
        **preset_kwargs
    )

    # DDP support for model
    # gen_model = model.module if isinstance(model, DDP) else model  # Already set above

    for sample_idx, sample in enumerate(dataset_iterator):
        unified_context, gold, pred, current_question = sample
        task_type = detect_task_type(current_question, gold, routing_mode)

        # Determine hint type and usage
        hint_type = "landmark" if task_type == "attribute_complete" else "spatial"
        if use_hints:
            hint_usage[hint_type] += 1
            display_hint = hint_type  # Show hint type when actually using hints
        else:
            hint_usage['none'] += 1
            display_hint = 'none'  # Show 'none' when not using hints

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

        # Accumulate totals and counts – only count relevant metrics
        totals["direction"] += sc["direction"]; counts["direction"] += 1  # always relevant
        totals["form"] += sc["form"]; counts["form"] += 1               # always relevant

        # yes/no relevance
        if gold_yesno(gold) is not None:
            totals["yesno"] += sc["yesno"]
            counts["yesno"] += 1

        # attribute relevance
        gold_feats = extract_spatial_features(gold)
        if gold_feats.get('colors') or gold_feats.get('shapes'):
            totals["attribute"] += sc["attribute"]
            counts["attribute"] += 1

        # landmark relevance
        if gold_feats.get('landmarks'):
            totals["landmark"] += sc["landmark"]
            counts["landmark"] += 1

        # movement relevance
        if gold_feats.get('movement_verbs'):
            totals["movement"] += sc["movement"]
            counts["movement"] += 1

        # NLP metrics (always relevant)
        totals["bleu4"] += sc["bleu4"]
        counts["bleu4"] += 1
        totals["rouge_l"] += sc["rouge_l"]
        counts["rouge_l"] += 1
        totals["bertscore"] += sc["bertscore"]
        counts["bertscore"] += 1

        # total composite always present
        totals["total"] += sc["total"]
        
        # Track answer length
        total_length += len(pred)
        n += 1

        # Per-sample print (concise) - only on rank 0 and only first 2 examples
        if rank == 0 and examples_shown < max_examples_to_show and random.random() < 0.2:
            print(f"[{n}] Task={task_type} | Hint={display_hint}")
            print(f"Q: {truncate(current_question)}")
            print(f"GOLD: {truncate(gold)}")
            print(f"PRED: {truncate(pred)}")
            print(f"Scores: dir={sc['direction']:.2f}  yn={sc['yesno']:.2f}  attr={sc['attribute']:.2f}  land={sc['landmark']:.2f}  mov={sc['movement']:.2f}  form={sc['form']:.2f}  total={sc['total']:.2f}")
            print(f"NLP: BLEU4={sc['bleu4']:.3f}  ROUGE-L={sc['rouge_l']:.3f}  BERTScore={sc['bertscore']:.3f}")
            print("-" * 80)
            examples_shown += 1

        if max_samples and n >= max_samples:
            break

    if n == 0:
        if rank == 0:
            print("No samples found.")
        return {}

    # Helper to compute safe average
    def _avg(metric):
        return totals[metric] / counts[metric] if counts[metric] > 0 else 0.0

    # Calculate averages using relevant denominators
    results = {
        "direction": _avg("direction"),
        "yesno": _avg("yesno"),
        "attribute": _avg("attribute"),
        "landmark": _avg("landmark"),
        "movement": _avg("movement"),
        "form": _avg("form"),
        "total": totals["total"] / n if n > 0 else 0.0,
        "avg_length": total_length / n if n > 0 else 0.0,  # Average character length
    }
    results['hint_usage'] = hint_usage
    results['total_samples'] = n

    # Wait for all GPUs to finish and aggregate results
    if is_distributed:
        # Gather results from all ranks
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        # Create tensors for aggregation (totals and counts)
        total_samples_tensor = torch.tensor(n, dtype=torch.long, device=next(model.parameters()).device)
        totals_tensor = torch.tensor([
            totals['direction'], totals['yesno'], totals['attribute'], totals['landmark'], totals['movement'], totals['form'], totals['total'],
            totals['bleu4'], totals['rouge_l'], totals['bertscore'], total_length
        ], dtype=torch.float32, device=next(model.parameters()).device)
        counts_tensor = torch.tensor([
            counts['direction'], counts['yesno'], counts['attribute'], counts['landmark'], counts['movement'], counts['form'],
            n,  # dummy count for total (not used, just for alignment)
            counts['bleu4'], counts['rouge_l'], counts['bertscore'], n  # count for length
        ], dtype=torch.long, device=next(model.parameters()).device)
        
        # Gather from all ranks
        gathered_samples = [torch.zeros_like(total_samples_tensor) for _ in range(world_size)]
        gathered_totals = [torch.zeros_like(totals_tensor) for _ in range(world_size)]
        gathered_counts = [torch.zeros_like(counts_tensor) for _ in range(world_size)]
        
        dist.all_gather(gathered_samples, total_samples_tensor)
        dist.all_gather(gathered_totals, totals_tensor)
        dist.all_gather(gathered_counts, counts_tensor)
        
        # Gather hint usage from all ranks
        hint_usage_tensor = torch.tensor([
            hint_usage.get('spatial', 0), hint_usage.get('movement', 0), 
            hint_usage.get('landmark', 0), hint_usage.get('navigation', 0), 
            hint_usage.get('none', 0)
        ], dtype=torch.long, device=next(model.parameters()).device)
        gathered_hint_usage = [torch.zeros_like(hint_usage_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_hint_usage, hint_usage_tensor)
        
        # Aggregate results from all GPUs
        total_samples_all_gpus = sum([s.item() for s in gathered_samples])
        totals_all_gpus = torch.stack(gathered_totals).sum(dim=0)
        counts_all_gpus = torch.stack(gathered_counts).sum(dim=0)
        
        # Aggregate hint usage from all GPUs
        hint_usage_all_gpus = torch.stack(gathered_hint_usage).sum(dim=0)
        aggregated_hint_usage = {
            'spatial': hint_usage_all_gpus[0].item(),
            'movement': hint_usage_all_gpus[1].item(),
            'landmark': hint_usage_all_gpus[2].item(),
            'navigation': hint_usage_all_gpus[3].item(),
            'none': hint_usage_all_gpus[4].item()
        }
        
        # Calculate final averages across all GPUs
        if total_samples_all_gpus > 0:
            # Safe averages with per-metric counts
            # totals_tensor: direction, yesno, attribute, landmark, movement, form, total, bleu4, rouge_l, bertscore, total_length (11 elements)
            # counts_tensor: direction, yesno, attribute, landmark, movement, form, total_dummy, bleu4, rouge_l, bertscore, length_count (11 elements)
            results = {
                'direction': (totals_all_gpus[0] / counts_all_gpus[0] if counts_all_gpus[0] > 0 else 0.0).item(),
                'yesno': (totals_all_gpus[1] / counts_all_gpus[1] if counts_all_gpus[1] > 0 else 0.0).item(),
                'attribute': (totals_all_gpus[2] / counts_all_gpus[2] if counts_all_gpus[2] > 0 else 0.0).item(),
                'landmark': (totals_all_gpus[3] / counts_all_gpus[3] if counts_all_gpus[3] > 0 else 0.0).item(),
                'movement': (totals_all_gpus[4] / counts_all_gpus[4] if counts_all_gpus[4] > 0 else 0.0).item(),
                'form': (totals_all_gpus[5] / counts_all_gpus[5] if counts_all_gpus[5] > 0 else 0.0).item(),
                'total': (totals_all_gpus[6] / total_samples_all_gpus).item() if total_samples_all_gpus>0 else 0.0,
                'bleu4': (totals_all_gpus[7] / counts_all_gpus[7] if counts_all_gpus[7] > 0 else 0.0).item(),
                'rouge_l': (totals_all_gpus[8] / counts_all_gpus[8] if counts_all_gpus[8] > 0 else 0.0).item(),
                'bertscore': (totals_all_gpus[9] / counts_all_gpus[9] if counts_all_gpus[9] > 0 else 0.0).item(),
                'avg_length': (totals_all_gpus[10] / counts_all_gpus[10] if counts_all_gpus[10] > 0 else 0.0).item(),
                'total_samples': total_samples_all_gpus,
                'hint_usage': aggregated_hint_usage  # Include aggregated hint_usage in distributed results
            }
        else:
            results = {}
    else:
        # Single GPU case - use local results
        results = {
            'direction': _avg('direction'),
            'yesno': _avg('yesno'),
            'attribute': _avg('attribute'),
            'landmark': _avg('landmark'),
            'movement': _avg('movement'),
            'form': _avg('form'),
            'total': totals['total']/n if n>0 else 0.0,
            'bleu4': _avg('bleu4'),
            'rouge_l': _avg('rouge_l'),
            'bertscore': _avg('bertscore'),
            'avg_length': total_length/n if n>0 else 0.0,
            'hint_usage': hint_usage,
            'total_samples': n,
        }

    if rank == 0:
        print(f"SUMMARY  (n={results['total_samples']})")
        print(f"  Direction : {results['direction']:.3f}")
        print(f"  Yes/No    : {results['yesno']:.3f}")
        print(f"  Attribute : {results['attribute']:.3f}")
        print(f"  Landmark  : {results['landmark']:.3f}")
        print(f"  Movement  : {results['movement']:.3f}")
        print(f"  Form      : {results['form']:.3f}")
        print(f"  TOTAL     : {results['total']:.3f}")
        print(f"  BLEU-4    : {results['bleu4']:.3f}")
        print(f"  ROUGE-L   : {results['rouge_l']:.3f}")
        print(f"  BERTScore : {results['bertscore']:.3f}")
        print(f"  Avg Length: {results['avg_length']:.1f} chars")
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
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for consistent sampling (default: 42)")
    parser.add_argument("--disable_bertscore", action="store_true",
                        help="Disable BERTScore calculation to avoid warnings")
    args = parser.parse_args()

    # Setup environment
    setup_environment()
    
    # Disable BERTScore if requested
    if args.disable_bertscore:
        os.environ['DISABLE_BERTSCORE'] = 'true'

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
        print(f"Random seed: {args.seed}")
        print()

    # Set consistent random seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    """Run distributed evaluation with hint tags - all splits and presets in one execution."""
    all_results = {}
    
    # Determine if we're in distributed mode
    is_distributed = dist.is_initialized() if 'dist' in globals() else False
    
    if rank == 0:
        print(f"\n🚀 Running ALL evaluations in single execution")
        print(f"📊 Splits: {args.splits}")
        print(f"🎯 Presets: {list(PRESETS.keys())}")
        print(f"🔧 Sample ratio: {args.sample_ratio}")
        print(f"💡 Hints enabled: {args.use_hints}")
        print(f"🔄 Routing mode: {args.routing}")
        print(f"🌐 Distributed: {is_distributed} (World Size: {world_size})")
        print("=" * 80)

    # Load ALL datasets once at the beginning
    if rank == 0:
        print(f"\n📂 Loading datasets once for all evaluations...")
    
    datasets = {}
    sampled_datasets = {}  # Store pre-sampled datasets for each split
    
    for split in args.splits:
        if rank == 0:
            print(f"  Loading {split} dataset...")
        full_dataset = AnsweringDataset(config, split=split, tokenizer=tokenizer)
        
        # Sample 10% of dataset indices ONCE for this split
        dataset_size = len(full_dataset)
        sampled_indices = sample_dataset_indices(dataset_size, args.sample_ratio)
        sampled_dataset = Subset(full_dataset, sampled_indices)
        
        datasets[split] = full_dataset
        sampled_datasets[split] = sampled_dataset
        
        if rank == 0:
            print(f"    Sampled {len(sampled_indices)} indices from {dataset_size} total samples")
    
    if rank == 0:
        print(f"✅ All datasets loaded and sampled successfully!")

    # Evaluate on ALL specified splits with ALL presets
    for split in args.splits:
        if rank == 0:
            print(f"\n📈 Processing split: {split}")
        
        split_results = {}
        
        for preset_name, preset_kwargs in PRESETS.items():
            if rank == 0:
                print(f"  🎯 Evaluating preset: {preset_name}")
            
            try:
                results = evaluate_split(
                    model, tokenizer, split, preset_name, preset_kwargs, config,
                    rank=rank, world_size=world_size,
                    sample_ratio=args.sample_ratio,
                    hint_types=args.hint_types,
                    use_hints=args.use_hints,
                    routing_mode=args.routing,
                    max_samples=args.max_samples,
                    dataset=sampled_datasets[split],  # Pass the pre-sampled dataset
                    is_distributed=is_distributed # Pass the is_distributed flag
                )
                
                if results:  # Only add if we got results
                    split_results[preset_name] = results
                    if rank == 0:
                        print(f"    ✅ {preset_name}: {results['total']:.3f} total score")
                else:
                    if rank == 0:
                        print(f"    ❌ {preset_name}: No results")
                        
            except Exception as e:
                if rank == 0:
                    print(f"    ❌ {preset_name}: Error - {str(e)}")
                continue
        
        all_results[split] = split_results
        
        if rank == 0:
            print(f"  📊 {split} completed with {len(split_results)} presets")

    # Save results (only on rank 0)
    if rank == 0:
        print(f"\n💾 Saving comprehensive results...")
        save_evaluation_results(all_results, args.output_dir)
        
        # Print final summary
        print(f"\n🎯 COMPREHENSIVE EVALUATION COMPLETE")
        print("=" * 80)
        for split, split_results in all_results.items():
            print(f"\n{split.upper()}:")
            for preset, results in split_results.items():
                print(f"  {preset}: {results['total']:.3f} total, {results['total_samples']} samples")


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
            print(f"    Landmark: {preset_results['landmark']:.3f}")
            print(f"    Movement: {preset_results['movement']:.3f}")
            print(f"    Form: {preset_results['form']:.3f}")
            print(f"    BLEU-4: {preset_results['bleu4']:.3f}")
            print(f"    ROUGE-L: {preset_results['rouge_l']:.3f}")
            print(f"    BERTScore: {preset_results['bertscore']:.3f}")
            print(f"    Avg Length: {preset_results['avg_length']:.1f} chars")
            print(f"    Samples: {preset_results['total_samples']}")
            if 'hint_usage' in preset_results:
                print(f"    Hint Usage: {preset_results['hint_usage']}")
            else:
                print(f"    Hint Usage: Not available")


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