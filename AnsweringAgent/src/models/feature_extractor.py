import torch
import torch.nn as nn
import torch.nn.functional as F
from models.darknet import Darknet
from typing import Optional
from config import Config

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
        self.hidden_size = config.model.hidden_size
        self.input_size = config.model.img_size
        
        # Initialize Darknet backbone
        self._init_darknet(config)
        
        # Initialize feature processing layers
        self._init_feature_layers()
        
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
        
        # Resize input to Darknet size if necessary
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size), 
                            mode='bilinear', align_corners=True)
        
        # Extract features through Darknet
        features = self.darknet(x)  # [batch_size, channels, height, width]
        
        # Verify feature dimensions
        if features.dim() == 3:
            features = features.unsqueeze(0)  # Add batch dimension if missing
        
        # Reshape to expected dimensions
        features = features.view(features.size(0), 512, 7, 7)  # Ensure correct channel dimension
        
        # Process through flatten layers to get AVDN features
        output = self.flatten(features)  # [batch_size, 384]
        
        return output

    def forward(self, current_view: torch.Tensor, previous_views: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with weighted aggregation of current and previous views.
        
        Args:
            current_view: Current view tensor [batch_size, channels, height, width]
            previous_views: Tensor of previous views [batch_size, num_views, channels, height, width]
            
        Returns:
            torch.Tensor: Aggregated visual features [batch_size, hidden_size]
        """
        batch_size = current_view.size(0)
        
        # Use mixed precision for feature extraction to save memory
        with torch.cuda.amp.autocast(enabled=True):
            # Extract features from current view (in AVDN dimension)
            current_features = self._extract_features(current_view)  # [batch_size, 384]
            
            if previous_views is None or (isinstance(previous_views, torch.Tensor) and previous_views.numel() == 0):
                # Project to BERT dimension before returning
                return self.projection(current_features)
            
            # Process the tensor of previous views [batch_size, num_views, C, H, W]
            prev_features = self._process_prev_views_chunk(previous_views, current_features)
            
            # Combine current and previous features using attention
            all_features = torch.cat([
                current_features.unsqueeze(1),
                prev_features
            ], dim=1)
            
            # Apply attention mechanism
            aggregated_features, _ = self.view_attention(
                current_features,
                all_features
            )
            
            # Project to BERT dimension
            output = self.projection(aggregated_features)
            
            # Ensure output batch size matches input batch size if needed
            if output.size(0) < batch_size:
                num_repeats = batch_size - output.size(0)
                output = torch.cat([output, output[-1:].repeat(num_repeats, 1)], dim=0)
        
        return output
        
    def _process_prev_views_chunk(self, chunk_tensor, current_features):
        """
        Process a chunk of previous views to extract features.
        
        Args:
            chunk_tensor: Tensor of previous views [batch_size, chunk_size, C, H, W]
            current_features: Features from current view for size reference
            
        Returns:
            torch.Tensor: Features from previous views [batch_size, chunk_size, feature_dim]
        """
        batch_size = chunk_tensor.size(0)
        num_views = chunk_tensor.size(1)
        
        # Process in smaller sub-chunks if needed to save memory
        max_chunk_size = 4
        if num_views > max_chunk_size:
            # Process in chunks and concatenate
            chunk_features_list = []
            for i in range(0, num_views, max_chunk_size):
                end_idx = min(i + max_chunk_size, num_views)
                sub_chunk = chunk_tensor[:, i:end_idx]
                # Process this sub-chunk
                with torch.cuda.amp.autocast(enabled=True):
                    # Reshape to [batch_size*sub_chunk_size, C, H, W]
                    sub_chunk_reshaped = sub_chunk.reshape(-1, *sub_chunk.shape[2:])
                    # Extract features
                    sub_features = self._extract_features(sub_chunk_reshaped)
                    # Reshape back
                    sub_features = sub_features.view(batch_size, end_idx-i, -1)
                    chunk_features_list.append(sub_features)
            
            # Concatenate all chunks
            chunk_features = torch.cat(chunk_features_list, dim=1)
        else:
            # Process the entire tensor at once
            with torch.cuda.amp.autocast(enabled=True):
                # Reshape to [batch_size*num_views, C, H, W]
                reshaped_tensor = chunk_tensor.reshape(-1, *chunk_tensor.shape[2:])
                # Extract features
                features = self._extract_features(reshaped_tensor)
                # Reshape back to [batch_size, num_views, feature_dim]
                chunk_features = features.view(batch_size, num_views, -1)
        
        # Ensure batch dimension matches the current features
        if batch_size != current_features.size(0):
            if batch_size > current_features.size(0):
                chunk_features = chunk_features[:current_features.size(0)]
            else:
                # Pad with repeats of the last batch item
                padding = current_features.size(0) - batch_size
                padding_shape = (padding, num_views, chunk_features.size(-1))
                padding_tensor = chunk_features[-1:].expand(*padding_shape)
                chunk_features = torch.cat([chunk_features, padding_tensor], dim=0)
        
        return chunk_features

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
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size)
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
        
        # Load weights
        new_state = torch.load(config.data.darknet_weights_path)
        state = self.darknet.state_dict()
        model_keys = set(state.keys())
        state_dict = {k: v for k, v in new_state['model'].items() if k in model_keys}
        state.update(state_dict)
        self.darknet.load_state_dict(state)
        
        # Freeze Darknet weights
        for param in self.darknet.parameters():
            param.requires_grad = False 