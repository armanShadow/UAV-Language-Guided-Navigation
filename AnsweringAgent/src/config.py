from dataclasses import dataclass
from pathlib import Path
import os
import torch

# Update paths for Docker container structure
PROJECT_ROOT = Path("/app/UAV-Language-Guided-Navigation")
DATASET_ROOT = Path("/app/datasets")

@dataclass
class ModelConfig:
    """Configuration for the AnsweringAgent model."""
    bert_model_name: str = 'bert-base-uncased'
    hidden_size: int = 768
    dropout: float = 0.2
    feat_dropout: float = 0.4
    num_decoder_layers: int = 4
    num_attention_heads: int = 4
    feedforward_dim: int = 3072
    max_answer_length: int = 128
    vocab_size: int = 30522  # BERT vocabulary size for bert-base-uncased
    img_size: int = 224  # Image size for Darknet/YOLO model

@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    per_gpu_batch_size: int = 7
    num_epochs: int = 200000
    learning_rate: float = 5e-5
    weight_decay: float = 0.02
    gradient_clip: float = 0.5
    warmup_steps: int = 1000
    log_freq: int = 2
    eval_freq: int = 100  #(validate every ~66 minutes)
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    device: str = 'cuda'
    seed: int = 42
    checkpoint_frequency: int = 400
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_verbose: bool = True
    gradient_accumulation_steps: int = 3
    # Early stopping parameters
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    use_augmentation: bool = False

    def __post_init__(self):
        """Initialize GPU settings and scale batch size/workers."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure a compatible GPU is installed and drivers are set up correctly.")

        self.num_gpus = torch.cuda.device_count()
        self.device = 'cuda'

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_processed_path: str = str(DATASET_ROOT / "train_processed_dataset.pkl")
    val_seen_processed_path: str = str(DATASET_ROOT / "val_seen_processed_dataset.pkl")
    val_unseen_processed_path: str = str(DATASET_ROOT / "val_unseen_processed_dataset.pkl")
    train_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/processed_data/train_data.json")
    val_seen_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/processed_data/val_seen_data.json")
    val_unseen_json_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/processed_data/val_unseen_data.json")
    avdn_image_dir: str = str(DATASET_ROOT / "AVDN/train_images")
    darknet_config_path: str = str(PROJECT_ROOT / "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/yolo_v3.cfg")
    darknet_weights_path: str = str(PROJECT_ROOT / "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/best.pt")
    max_previous_views: int = 3
    train_val_split: float = 0.90  # Updated: 90% for training
    val_test_split: float = 0.5    # New: 50% of remaining data for validation, 50% for testing
    max_seq_length: int = 512

    def __post_init__(self):
        """Verify paths exist."""
        paths = [self.train_csv_path, self.avdn_image_dir,
                self.darknet_config_path, self.darknet_weights_path]
        for path in paths:
            if not os.path.exists(path):
                raise RuntimeError(f"Error: Path does not exist: {path}")

@dataclass
class Config:
    """Main configuration class combining all settings."""
    checkpoint_dir: str = str(PROJECT_ROOT/ 'AnsweringAgent/outputs/checkpoints')
    log_dir: str = str(PROJECT_ROOT/ 'AnsweringAgent/outputs/logs')

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()

    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)