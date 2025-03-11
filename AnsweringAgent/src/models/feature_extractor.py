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
        self.input_size = config.model.img_size  # Size from AVDN Normalizer
        
        # Initialize Darknet backbone
        self._init_darknet(config)
        
        # Initialize feature processing layers
        self._init_feature_layers()
        
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
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input tensor using Darknet and flatten layers.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Extracted features [batch_size, 384] (AVDN dimension)
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
        
        # Process through flatten layers to get AVDN features
        output = self.flatten(features)  # [batch_size, 384]
        logger.info(f"AVDN features shape: {output.shape}")
        
        return output

    def forward(self, current_view: torch.Tensor, previous_views: Optional[list] = None) -> torch.Tensor:
        """
        Forward pass with weighted aggregation of current and previous views.
        
        Args:
            current_view: Current view tensor [batch_size, channels, height, width]
            previous_views: List of previous view tensors, each [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Aggregated visual features [batch_size, hidden_size]
        """
        # Ensure inputs are on the correct device
        current_view = current_view.to(self.device)
        batch_size = current_view.size(0)
        logger.info(f"Current view batch size: {batch_size}")
        
        # Extract features from current view (in AVDN dimension)
        logger.info("Processing current view...")
        current_features = self._extract_features(current_view)  # [batch_size, 384]
        logger.info(f"Current features shape after extraction: {current_features.shape}")
        
        if not previous_views:  # Handle empty list or None
            # Project to BERT dimension before returning
            return self.projection(current_features)
        
        # Move previous views to device and stack
        if isinstance(previous_views, list):
            logger.info(f"Previous views is a list of length {len(previous_views)}")
            previous_views = [view.to(self.device) for view in previous_views]
            prev_views_tensor = torch.stack(previous_views, dim=1)  # [batch_size, num_views, channels, height, width]
        else:
            # If it's already a tensor
            logger.info("Previous views is already a tensor")
            prev_views_tensor = previous_views.to(self.device)
            
        # Log shapes for debugging
        logger.info(f"Previous views tensor shape before reshape: {prev_views_tensor.shape}")
        
        # Get the actual batch size from the previous views tensor
        actual_batch_size = prev_views_tensor.size(0)
        num_prev_views = prev_views_tensor.size(1)
        logger.info(f"Previous views batch size: {actual_batch_size}")
        
        # Ensure current features match the batch size of previous views
        if batch_size > actual_batch_size:
            logger.info(f"Adjusting current features batch size from {batch_size} to {actual_batch_size}")
            # Take only the first actual_batch_size samples from current_features
            current_features = current_features[:actual_batch_size]
            logger.info(f"Adjusted current features shape: {current_features.shape}")
        elif batch_size < actual_batch_size:
            logger.info(f"Adjusting previous views batch size from {actual_batch_size} to {batch_size}")
            # Take only the first batch_size samples from previous views
            prev_views_tensor = prev_views_tensor[:batch_size]
            actual_batch_size = batch_size
            logger.info(f"Adjusted previous views shape: {prev_views_tensor.shape}")
        
        # Extract features from previous views (in AVDN dimension)
        prev_views_reshaped = prev_views_tensor.view(-1, *prev_views_tensor.shape[2:])  # [batch_size * num_views, C, H, W]
        logger.info(f"Previous views shape after reshape: {prev_views_reshaped.shape}")
        
        # Process each previous view
        logger.info("Processing previous views...")
        prev_features = self._extract_features(prev_views_reshaped)  # [batch_size * num_views, 384]
        logger.info(f"Previous features shape after extraction: {prev_features.shape}")
        
        # Verify AVDN dimension before reshape
        assert prev_features.size(-1) == 384, \
            f"Previous features dimension {prev_features.size(-1)} != 384 before reshape"
        
        # Reshape using the actual batch size from previous views
        prev_features = prev_features.view(actual_batch_size, num_prev_views, -1)  # [batch_size, num_views, 384]
        logger.info(f"Previous features final shape: {prev_features.shape}")
        
        # Verify dimensions match (should both be 384)
        logger.info(f"Comparing dimensions - current: {current_features.size(-1)}, previous: {prev_features.size(-1)}")
        assert current_features.size(-1) == prev_features.size(-1), \
            f"Feature dimensions mismatch: current {current_features.size(-1)} vs previous {prev_features.size(-1)}"
        
        # Verify batch sizes match
        assert current_features.size(0) == prev_features.size(0), \
            f"Batch size mismatch: current {current_features.size(0)} vs previous {prev_features.size(0)}"
        
        # Combine current and previous features using attention (in AVDN dimension)
        all_features = torch.cat([
            current_features.unsqueeze(1),  # [actual_batch_size, 1, 384]
            prev_features  # [actual_batch_size, num_views, 384]
        ], dim=1)  # [actual_batch_size, num_views + 1, 384]
        logger.info(f"Combined features shape: {all_features.shape}")
        
        # Apply attention mechanism (still in AVDN dimension)
        aggregated_features, _ = self.view_attention(
            current_features,  # Use current view as query [actual_batch_size, 384]
            all_features,      # Use all features as context [actual_batch_size, num_views + 1, 384]
        )
        logger.info(f"Aggregated features shape: {aggregated_features.shape}")
        
        # Finally project to BERT dimension
        output = self.projection(aggregated_features)  # [actual_batch_size, 768]
        logger.info(f"Final output shape: {output.shape}")
        
        # Ensure output batch size matches input batch size if needed
        if output.size(0) < batch_size:
            # Repeat the last sample to match the expected batch size
            num_repeats = batch_size - output.size(0)
            output = torch.cat([output, output[-1:].repeat(num_repeats, 1)], dim=0)
            logger.info(f"Adjusted final output shape to match input batch size: {output.shape}")
        
        return output

    def _init_feature_layers(self):
        """Initialize feature processing layers."""
        # Feature flattening layers with explicit output size (AVDN architecture)
        self.flatten = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 384),  # AVDN dimension
            nn.ReLU(),
            nn.LayerNorm(384)  # Add normalization for stability
        )
        
        # Projection layer from AVDN dimension to BERT dimension
        self.projection = nn.Sequential(
            nn.Linear(384, self.hidden_size),  # Project from 384 to 768
            nn.LayerNorm(self.hidden_size)
        )
        
        # View attention for weighted aggregation (using AVDN dimension)
        self.view_attention = SoftDotAttention(384)  # Keep attention in AVDN dimension
        
        # Initialize weights
        self._init_weights()
        
    def _verify_output_dimensions(self):
        """Verify that the output dimensions match the expected hidden size."""
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size, device=self.device)
        with torch.no_grad():
            # Get AVDN features
            avdn_features = self._extract_features(dummy_input)
            # Project to BERT dimension
            output = self.projection(avdn_features)
            
        # Verify AVDN features dimension
        assert avdn_features.size(-1) == 384, \
            f"AVDN feature size {avdn_features.size(-1)} != 384"
            
        # Verify final output dimension
        assert output.size(-1) == self.hidden_size, \
            f"Final output size {output.size(-1)} != hidden size {self.hidden_size}"
        
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