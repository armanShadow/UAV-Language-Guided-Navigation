from dataclasses import dataclass, field
from pathlib import Path
import os
import torch

# Update paths for Docker container structure
PROJECT_ROOT = Path("/app/UAV-Language-Guided-Navigation")
DATASET_ROOT = Path("/app/datasets")

@dataclass
class ModelConfig:
    """Configuration for the AnsweringAgent model."""
    bert_model_name: str = 'bert-base-uncased'  # Legacy setting
    t5_model_name: str = 't5-base'  # New setting for T5
    hidden_size: int = 768  # Match T5-base hidden size (d_model)
    dropout: float = 0.3
    feat_dropout: float = 0.4
    num_decoder_layers: int = 4  # Not used when using pretrained T5 decoder
    num_attention_heads: int = 8  # Match T5-base (8 heads)
    num_visual_tokens: int = 32  # Number of visual tokens
    feedforward_dim: int = 2048  # Match T5-base feed forward dimension
    max_answer_length: int = 128
    vocab_size: int = 32128  # T5 vocabulary size for t5-base
    img_size: int = 224  # Image size for Darknet/YOLO model
    use_t5: bool = True  # Flag to control which model type to use
    use_pretrained_decoder: bool = True  # Use T5's pretrained decoder instead of custom

@dataclass
class TrainingConfig:      
    num_epochs: int = 10000
    learning_rate: float = 5e-6  # Reduced from 5e-5 for second-stage fine-tuning
    weight_decay: float = 0.02
    gradient_clip: float = 0.5
    warmup_steps: int = 1000
    log_freq: int = 2
    eval_freq: int = 50  #(validate every ~66 minutes)
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False
    device: str = 'cuda'
    seed: int = 42
    checkpoint_frequency: int = 200
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_verbose: bool = True
    gradient_accumulation_steps: int = 3
    # Early stopping parameters
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.003
    # Validation parameters
    per_gpu_batch_size_val: int = 8  # Smaller validation batch size to save VRAM
    train_chunk_size: int = 1000
    # Curriculum learning parameters
    curriculum_epochs: int = 30  # Number of epochs for curriculum learning phase
    destination_loss_weight_start: float = 1.0
    destination_loss_weight_end: float = 0.2
    
    ce_loss_weight_start: float = 0.02  # Lower start weight so CE does not dominate early
    ce_loss_weight_end: float = 0.5   # Final weight still substantial but lower than before
    
    # Contrastive Learning Parameters - FIXED WEIGHTS FOR BETTER BALANCE
    use_contrastive_learning: bool = True
    contrastive_loss_type: str = "infonce"
    contrastive_margin: float = 0.1
    contrastive_temperature: float = 0.02  # Lower temperature for sharper InfoNCE
    # FIXED: Increased contrastive weights to match CE loss scale
    contrastive_weight_start: float = 10.0  # Increased from 0.1 to 10.0
    contrastive_weight_end: float = 10.0    # Increased from 0.5 to 25.0
    # New triplet loss options
    use_cosine_distance: bool = True  # Use cosine distance instead of L2 for triplet loss - Better for normalized embeddings
    contrastive_mean_all: bool = True  # Use mean over all elements instead of non-zero for triplet loss - More stable for large batches
    
    # Add per-epoch weight logging for debugging
    log_loss_weights: bool = True  # Log weight values each epoch
    
    # Knowledge-distillation (KD) parameters
    use_kd: bool = True  # Enable teacher-student KD
    kd_teacher_model_name: str = "sentence-transformers/all-mpnet-base-v2"  # Teacher model for KD
    kd_weight_start: float = 5.0  # KD weight at epoch 0
    kd_weight_end: float = 0.5    # KD weight after kd_epochs
    kd_epochs: int = 30  # Epochs over which KD weight is annealed to kd_weight_end

    def __post_init__(self):
        """Initialize GPU settings and scale batch size/workers."""
        if not torch.cuda.is_available():
            print("CUDA is not available. Please ensure a compatible GPU is installed and drivers are set up correctly.")

        self.num_gpus = torch.cuda.device_count()
        self.device = 'cuda'

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_processed_path_dir: str = str(DATASET_ROOT / "train/")
    val_seen_processed_path: str = str(DATASET_ROOT / "val_seen_processed_dataset.pkl")
    val_unseen_processed_path: str = str(DATASET_ROOT / "val_unseen_processed_dataset.pkl")
    train_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/processed_data/train_data.json")
    val_seen_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/processed_data/val_seen_data.json")
    val_unseen_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/processed_data/val_unseen_data.json")
    
    # Augmented dataset paths with paraphrases (NEW - for contrastive learning)
    use_augmented_data: bool = True  # Toggle to use augmented data with paraphrases
    train_augmented_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/augmented_data/train_data_with_paraphrases.json")
    val_seen_augmented_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/augmented_data/val_seen_data_with_paraphrases.json")
    val_unseen_augmented_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/augmented_data/val_unseen_data_with_paraphrases.json")
    
    # Data preprocessing settings
    use_augmentation: bool = False  # Enable/disable visual augmentation during preprocessing
    
    avdn_image_dir: str = str(DATASET_ROOT / "AVDN/train_images")
    darknet_config_path: str = str(PROJECT_ROOT / "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/yolo_v3.cfg")
    darknet_weights_path: str = str(PROJECT_ROOT / "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/best.pt")
    max_previous_views: int = 3
    train_val_split: float = 0.90  # Updated: 90% for training
    val_test_split: float = 0.5    # New: 50% of remaining data for validation, 50% for testing
    max_seq_length: int = 512

    def __post_init__(self):
        """Verify paths exist."""
        if self.use_augmented_data:
            # Check augmented data paths
            paths = [self.train_augmented_json_path, self.val_seen_augmented_json_path, 
                    self.val_unseen_augmented_json_path, self.avdn_image_dir,
                self.darknet_config_path, self.darknet_weights_path]
        else:
            # Check original data paths
            paths = [self.train_json_path, self.val_seen_json_path, self.val_unseen_json_path,
                    self.avdn_image_dir, self.darknet_config_path, self.darknet_weights_path]
        
        for path in paths:
            if not os.path.exists(path):
                print(f"Warning: Path does not exist: {path}")
                if self.use_augmented_data and "augmented_data" in path:
                    print(f"  Hint: Run comprehensive_avdn_pipeline.py to generate augmented data")
    
    def get_json_path(self, split: str) -> str:
        """Get the appropriate JSON path for a dataset split."""
        if self.use_augmented_data:
            if split == 'train':
                return self.train_augmented_json_path
            elif split == 'val_seen':
                return self.val_seen_augmented_json_path
            elif split == 'val_unseen':
                return self.val_unseen_augmented_json_path
        else:
            if split == 'train':
                return self.train_json_path
            elif split == 'val_seen':
                return self.val_seen_json_path
            elif split == 'val_unseen':
                return self.val_unseen_json_path
        
        raise ValueError(f"Unknown split: {split}")

@dataclass
class Config:
    """Main configuration class combining all settings."""
    checkpoint_dir: str = str(PROJECT_ROOT/ 'AnsweringAgent/outputs/checkpoints')
    log_dir: str = str(PROJECT_ROOT/ 'AnsweringAgent/outputs/logs')

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)