from dataclasses import dataclass
from typing import Optional

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

@dataclass
class TrainingConfig:
    """Configuration for training settings."""
    batch_size: int = 4
    num_epochs: int = 200000
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    warmup_steps: int = 1000
    log_freq: int = 2
    save_freq: int = 1000
    eval_freq: int = 1000
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    device: str = 'cuda'
    seed: int = 42

@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    train_csv_path: str = 'data/train_data.csv'
    avdn_image_dir: str = '../../Aerial-Vision-and-Dialog-Navigation/datasets/AVDN/train_images'
    max_previous_views: int = 4
    train_val_split: float = 0.95
    max_length: int = 512

@dataclass
class Config:
    """Main configuration class combining all settings."""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    def __post_init__(self):
        """Create necessary directories."""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True) 