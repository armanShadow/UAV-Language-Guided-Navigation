import torch
import torch.nn as nn
import torch.nn.functional as F
from models.darknet import Darknet
from typing import Optional
from utils.logger import get_logger
from config import Config

# Get the logger instance
logger = get_logger()

class FeatureExtractor(nn.Module):
    """Feature extractor using Darknet backbone with attention-based feature aggregation."""
    
    def __init__(self, config: Config):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        # Use config for device
        self.device = torch.device(config.training.device)
        
        # Initialize Darknet backbone
        self._init_darknet(
            config.data.darknet_config_path,
            config.data.darknet_weights_path,
            img_size=416
        )
        
        # Initialize feature processing layers
        self._init_feature_layers()
        
        # Move model to GPU
        self.to(self.device)
        
        
    def _init_darknet(self, config_path: str, weights_path: str, img_size: int):
        """Initialize and load Darknet backbone."""
        self.darknet = Darknet(config_path, img_size)
        
        # Load weights with proper device mapping
        new_state = torch.load(weights_path, map_location=self.device)
        state = self.darknet.state_dict()
        
        # Update state dict with new weights
        for k, v in new_state.items():
            if k in state:
                state[k] = v
        self.darknet.load_state_dict(state)
        
        # Freeze Darknet weights
        for param in self.darknet.parameters():
            param.requires_grad = False
            
    def _init_feature_layers(self):
        """Initialize feature processing layers."""
        # Feature flattening layers
        self.flatten = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 768)
        )
        
        # Initialize weights
        for m in self.flatten.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Attention for weighted aggregation of views
        self.view_attention = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1)
        )
        
        # Initialize attention weights
        for m in self.view_attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, current_view: torch.Tensor, previous_views: Optional[list] = None) -> torch.Tensor:
        """
        Forward pass with weighted aggregation of current and previous views.
        
        Args:
            current_view: Current view tensor [batch_size, channels, height, width]
            previous_views: List of previous view tensors, each [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Aggregated visual features [batch_size, hidden_size]
        """
        batch_size = current_view.size(0)
        
        # Extract features from current view
        current_features = self._extract_features(current_view)  # [batch_size, 768]
        
        if not previous_views:  # Handle empty list or None
            return current_features
        
        # Stack previous views into a single tensor
        prev_views_tensor = torch.stack(previous_views, dim=1)  # [batch_size, num_views, channels, height, width]
        
        # Extract features from previous views
        num_prev_views = prev_views_tensor.size(1)
        prev_views_reshaped = prev_views_tensor.view(-1, *prev_views_tensor.shape[2:])  # [batch_size * num_views, C, H, W]
        prev_features = self._extract_features(prev_views_reshaped)  # [batch_size * num_views, 768]
        prev_features = prev_features.view(batch_size, num_prev_views, -1)  # [batch_size, num_views, 768]
        
        # Combine current and previous features for attention
        all_features = torch.cat([
            current_features.unsqueeze(1),  # [batch_size, 1, 768]
            prev_features  # [batch_size, num_views, 768]
        ], dim=1)  # [batch_size, num_views + 1, 768]
        
        # Calculate attention weights
        attention_weights = self.view_attention(all_features)  # [batch_size, num_views + 1, 1]
        
        # Apply weighted aggregation
        aggregated_features = torch.sum(all_features * attention_weights, dim=1)  # [batch_size, 768]
        
        return aggregated_features
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor using Darknet and flatten layers."""
        # Keep input on current device
        features = self.darknet(x)
        
        # Reshape and process features
        features = features.view(features.size(0), -1, 7, 7)
        
        # Process through flatten layers
        output = self.flatten(features)
        
        return output 