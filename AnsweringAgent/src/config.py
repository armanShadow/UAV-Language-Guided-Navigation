from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os
import torch
import subprocess


# Update paths for Docker container structure
PROJECT_ROOT = Path("/app/UAV-Language-Guided-Navigation")
DATASET_ROOT = Path("/app/datasets")

@dataclass
class ModelConfig:
    """Configuration for the AnsweringAgent model."""
    bert_model_name: str = 'bert-base-uncased'
    hidden_size: int = 768
    dropout: float = 0.5
    feat_dropout: float = 0.4
    num_decoder_layers: int = 6
    num_attention_heads: int = 8
    feedforward_dim: int = 3072  # 4 * hidden_size
    max_seq_length: int = 512
    max_answer_length: int = 128
    vocab_size: int = 30522  # BERT vocabulary size for bert-base-uncased
    img_size: int = 224  # Image size for Darknet/YOLO model

def get_nvidia_smi_output():
    """Get GPU information directly from nvidia-smi."""
    try:
        output = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
        return output
    except Exception as e:
        return f"Error running nvidia-smi: {str(e)}"

@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    base_batch_size: int = 4  # Base batch size per GPU
    num_epochs: int = 200000
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    log_freq: int = 2
    save_freq: int = 1000
    eval_freq: int = 1000
    base_num_workers: int = 4  # Base workers per GPU
    pin_memory: bool = True
    mixed_precision: bool = True
    device: str = 'cuda'
    primary_gpu: int = 0
    seed: int = 42
    checkpoint_frequency: int = 10000
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_verbose: bool = True
    gradient_accumulation_steps: int = 1

    def __post_init__(self):
        """Initialize GPU settings and scale batch size/workers."""
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = 'cpu'
            self.num_gpus = 0
            self.batch_size = self.base_batch_size
            self.num_workers = self.base_num_workers
        else:
            # Force single GPU usage
            self.num_gpus = 1
            self.device = f'cuda:{self.primary_gpu}'
            # Use base values without scaling
            self.batch_size = self.base_batch_size
            self.num_workers = self.base_num_workers
            
            print(f"Using single GPU (device {self.primary_gpu})")
            print(f"Batch size: {self.batch_size}")
            print(f"Number of workers: {self.num_workers}")
            
            # Log GPU information
            gpu_name = torch.cuda.get_device_name(self.primary_gpu)
            memory_total = torch.cuda.get_device_properties(self.primary_gpu).total_memory / 1024**2
            print(f"GPU {self.primary_gpu}: {gpu_name} - Total Memory: {memory_total:.1f}MB")

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_csv_path: str = str(PROJECT_ROOT / "AnsweringAgent/src/data/train_data.csv")
    avdn_image_dir: str = str(DATASET_ROOT / "AVDN/train_images")
    darknet_config_path: str = str(PROJECT_ROOT / "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/yolo_v3.cfg")
    darknet_weights_path: str = str(PROJECT_ROOT / "Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/pretrain_weights/best.pt")
    max_previous_views: int = 4
    train_val_split: float = 0.95
    max_length: int = 512

    def __post_init__(self):
        """Verify paths exist."""
        paths = [self.train_csv_path, self.avdn_image_dir, 
                self.darknet_config_path, self.darknet_weights_path]
        for path in paths:
            if not os.path.exists(path):
                print(f"Warning: Path does not exist: {path}")

@dataclass
class Config:
    """Main configuration class combining all settings."""
    checkpoint_dir: str = str(PROJECT_ROOT/ 'AnsweringAgent/outputs/checkpoints')
    log_dir: str = str(PROJECT_ROOT/ 'AnsweringAgent/outputs/logs')

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    

   

    def __post_init__(self):
        """Create necessary directories and setup full logger."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        