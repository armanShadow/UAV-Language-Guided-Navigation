import torch
import torch.nn as nn
import torch.nn.functional as F
from models.darknet import Darknet
from typing import Optional
from utils.logger import get_logger
from config import Config

# Get the logger instance
logger = get_logger()

class SoftDotAttention(nn.Module):
    '''Soft Dot Attention from AVDN codebase.'''
    def __init__(self, dim):
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax(dim=1)
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, h, context, mask=None):
        '''
        h: batch x dim
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        '''
        target = self.linear_in(h).unsqueeze(2)  # batch x dim x 1
        attn = torch.bmm(context, target).squeeze(2)  # batch x seq_len
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len
        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, h), 1)
        h_tilde = self.tanh(self.linear_out(h_tilde))
        return h_tilde, attn

class FeatureExtractor(nn.Module):
    """Feature extractor using Darknet backbone with attention-based feature aggregation."""
    
    def __init__(self, config: Config):
        """
        Initialize the feature extractor.
        
        Args:
            config: Configuration object containing model settings
        """
        super().__init__()
        self.config = config
        self.device = torch.device(config.training.device)
        self.hidden_size = config.model.hidden_size
        self.visual_feature_size = 384  # AVDN's visual feature dimension
        self.input_size = config.model.img_size  # Size from AVDN Normalizer
        
        # Initialize Darknet backbone
        self._init_darknet(config)
        
        # Initialize feature processing layers
        self._init_feature_layers()
        
        # Initialize projection layer to match BERT dimension if needed
        self.project_to_bert = nn.Sequential(
            nn.Linear(self.visual_feature_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(config.model.dropout)
        )
        
        # Move model to device
        self.to(self.device)
        
        # Verify output dimensions
        self._verify_output_dimensions()
    
    def _init_weights(self):
        """Initialize weights with AVDN's initialization scheme."""
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
    
    def forward(self, current_view: torch.Tensor, previous_views: Optional[list] = None) -> torch.Tensor:
        """
        Forward pass with weighted aggregation of current and previous views.
        
        Args:
            current_view: Current view tensor [batch_size, channels, height, width]
            previous_views: List of previous view tensors, each [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Aggregated visual features [batch_size, hidden_size]
        """
        # Extract and aggregate features in AVDN dimension space
        features = super().forward(current_view, previous_views)
        
        # Project to BERT dimension space
        features = self.project_to_bert(features)
        
        return features
        
    def _init_feature_layers(self):
        """Initialize feature processing layers."""
        # Feature flattening layers with explicit output size
        self.flatten = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 384),  # Match AVDN dimension
            nn.ReLU(),
            nn.LayerNorm(384)  # Add normalization for stability
        )
        
        # View attention for weighted aggregation (using AVDN dimension)
        self.view_attention = SoftDotAttention(384)  # Match AVDN dimension
        
        # Initialize weights
        self._init_weights()
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor using Darknet and flatten layers.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Extracted features [batch_size, hidden_size]
        """
        # Log input shape
        logger.info(f"Input shape to _extract_features: {x.shape}")
        
        # Resize input to Darknet size if necessary
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=True)
            logger.info(f"Resized input shape: {x.shape}")
        
        # Extract features through Darknet
        features = self.darknet(x)  # [batch_size, channels, height, width]
        logger.info(f"Darknet output shape: {features.shape}")
        
        # Verify feature dimensions
        if features.dim() == 3:
            features = features.unsqueeze(0)  # Add batch dimension if missing
            logger.info(f"Added batch dimension, new shape: {features.shape}")
        
        # Reshape to expected dimensions
        features = features.view(features.size(0), 512, 7, 7)  # Ensure correct channel dimension
        logger.info(f"Reshaped features shape: {features.shape}")
        
        # Process through flatten layers
        output = self.flatten(features)  # [batch_size, 384]
        logger.info(f"Final output shape: {output.shape}")
        
        return output

    def _verify_output_dimensions(self):
        """Verify that the output dimensions match the expected hidden size."""
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size, device=self.device)
        with torch.no_grad():
            # Verify AVDN feature dimension
            features = self._extract_features(dummy_input)
            assert features.size(-1) == self.visual_feature_size, \
                f"Visual feature size {features.size(-1)} != expected size {self.visual_feature_size}"
            
            # Verify final output dimension matches BERT
            output = self.project_to_bert(features)
            assert output.size(-1) == self.hidden_size, \
                f"Final output size {output.size(-1)} != BERT hidden size {self.hidden_size}"
        
    def _init_darknet(self, config: Config):
        """Initialize and load Darknet backbone.
        
        Args:
            config: Configuration object containing model settings
        """
        self.darknet = Darknet(config)
        
        # Load weights with proper device mapping
        new_state = torch.load(config.data.darknet_weights_path, map_location=self.device)
        state = self.darknet.state_dict()
        
        # Update state dict with new weights
        for k, v in new_state.items():
            if k in state:
                state[k] = v
        self.darknet.load_state_dict(state)
        
        # Freeze Darknet weights
        for param in self.darknet.parameters():
            param.requires_grad = False
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor using Darknet and flatten layers."""
        # Resize input to Darknet size if necessary
        if x.size(-1) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=True)
        
        # Extract features through Darknet
        features = self.darknet(x)
        
        # Reshape and process features
        features = features.view(features.size(0), -1, 7, 7)
        
        # Process through flatten layers
        output = self.flatten(features)
        
        return output 