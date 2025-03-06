import torch
import torch.nn as nn
import torch.nn.functional as F
from models.darknet import Darknet
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor(nn.Module):
    """Feature extractor using Darknet backbone with attention-based feature aggregation."""
    
    def __init__(self, config_path: str, weights_path: str, img_size: int = 416):
        """
        Initialize the feature extractor.
        
        Args:
            config_path: Path to Darknet configuration file
            weights_path: Path to pre-trained Darknet weights
            img_size: Input image size (default: 416)
        """
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Darknet backbone
        self._init_darknet(config_path, weights_path, img_size)
        
        # Initialize feature processing layers
        self._init_feature_layers()
        
        # Move model to GPU
        self.to(self.device)
        
    def _init_darknet(self, config_path: str, weights_path: str, img_size: int):
        """Initialize and load Darknet backbone."""
        self.darknet = Darknet(config_path, img_size)
        
        # Load weights
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
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        
        Args:
            x: Input image tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Extracted features [batch_size, hidden_size]
        """
        # Move input to device
        x = x.to(self.device)
        
        # Extract features using Darknet
        features = self.darknet(x)
        
        # Process features through flattening layers
        features = self.flatten(features)
        
        return features
        
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