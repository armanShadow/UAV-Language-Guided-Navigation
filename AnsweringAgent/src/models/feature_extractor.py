import torch
import torch.nn as nn
import torch.nn.functional as F
from models.darknet import Darknet
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor(nn.Module):
    """Feature extractor using Darknet backbone with attention-based feature aggregation."""
    
    def __init__(self, darknet_config_path: str, darknet_weights_path: str, img_size: int = 224, device: str = 'cpu'):
        """
        Initialize the feature extractor.
        
        Args:
            darknet_config_path: Path to Darknet configuration file
            darknet_weights_path: Path to pre-trained Darknet weights
            img_size: Input image size (default: 224)
            device: Device to run the model on (default: 'cpu')
        """
        super().__init__()
        # Force CPU usage
        self.device = torch.device('cpu')
        
        # Initialize Darknet backbone
        self._init_darknet(darknet_config_path, darknet_weights_path, img_size)
        
        # Initialize feature processing layers
        self._init_feature_layers()
        
        # Initialize weights
        self._init_weights()
        
        # Ensure everything is on CPU
        self.to('cpu')
        
    def _init_darknet(self, config_path: str, weights_path: str, img_size: int):
        """Initialize and load Darknet backbone."""
        # Force CPU for Darknet
        self.darknet = Darknet(config_path, img_size, device='cpu')
        
        # Load weights to CPU with weights_only=True for security
        new_state = torch.load(weights_path, map_location='cpu', weights_only=True)
        state = self.darknet.state_dict()
        
        # Update state dict with new weights
        for k, v in new_state.items():
            if k in state:
                state[k] = v
        self.darknet.load_state_dict(state)
        
        # Ensure Darknet is on CPU and freeze weights
        self.darknet.to('cpu')
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
        
        # Attention for weighted aggregation
        self.attention = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize attention weights
        for m in self.attention.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # For linear and conv layers
                    nn.init.xavier_normal_(param, gain=1.0)
                elif len(param.shape) == 1:  # For batch norm layers
                    nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                    
    def forward(self, current_view: torch.Tensor, previous_views: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract and aggregate features from current and previous views.
        
        Args:
            current_view: Current view image tensor (B, C, H, W)
            previous_views: Optional previous views tensor (B, N, C, H, W)
            
        Returns:
            torch.Tensor: Combined features (B, 768)
        """
        # Process current view
        current_features = self._extract_features(current_view)
        
        if previous_views is None:
            return current_features
            
        # Process previous views
        B, N, C, H, W = previous_views.shape
        previous_views = previous_views.view(-1, C, H, W)
        previous_features = self._extract_features(previous_views)
        previous_features = previous_features.view(B, N, -1)
        
        # Compute attention weights and aggregate features
        attention_weights = self.attention(previous_features)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Aggregate features with attention
        aggregated_features = torch.sum(attention_weights * previous_features, dim=1)
        
        # Combine features with residual connection
        combined_features = current_features + aggregated_features
        
        return combined_features
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input tensor using Darknet and flatten layers.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Extracted features (B, 768)
        """
        # Ensure input is on CPU
        x = x.to('cpu')
        
        # Extract features from Darknet
        features = self.darknet(x)
        
        # Reshape and process features
        features = features.view(features.size(0), -1, 7, 7)
        
        # Process through flatten layers
        output = self.flatten(features)
        
        return output 