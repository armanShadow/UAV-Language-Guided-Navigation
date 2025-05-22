import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from models.feature_extractor import FeatureExtractor
from typing import Dict, Tuple, Optional, List
import math
from config import Config
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import BaseModelOutput

class TemporalObservationEncoder(nn.Module):
    """Encodes temporal observations with attention mechanism."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Attention mechanism for temporal observations
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network for processing attended features
        self.ff_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, current_features: torch.Tensor, prev_features: torch.Tensor) -> torch.Tensor:
        """
        Process current and previous features with attention.
        
        Args:
            current_features: Current view features [batch_size, hidden_size]
            prev_features: Previous views features [batch_size, num_prev, hidden_size]
            
        Returns:
            Temporally contextualized features [batch_size, hidden_size]
        """
        batch_size = current_features.size(0)
        
        # Add current features to previous features for self-attention
        # Shape: [batch_size, 1 + num_prev, hidden_size]
        combined_features = torch.cat([
            current_features.unsqueeze(1),
            prev_features
        ], dim=1)
        
        # Apply attention with current features as query, all features as key/value
        attn_output, _ = self.temporal_attention(
            query=current_features.unsqueeze(1),
            key=combined_features,
            value=combined_features
        )
        
        # Shape: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        attn_output = attn_output.squeeze(1)
            
        features = self.norm1(current_features + attn_output)
        
        # Feed-forward network
        ff_output = self.ff_network(features)
        
        # Final residual connection and normalization
        output = self.norm2(features + ff_output)
        
        return output


class CrossModalFusion(nn.Module):
    """Fuses text and visual features using cross-attention."""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Visual -> Text attention
        self.visual_to_text_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Text -> Visual attention
        self.text_to_visual_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-5)
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, text_features: torch.Tensor, visual_features: torch.Tensor, 
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse text and visual features with bidirectional attention.
        
        Args:
            text_features: Text encoded features [batch_size, seq_len, hidden_size]
            visual_features: Visual features [batch_size, hidden_size]
            text_mask: Attention mask for text [batch_size, seq_len]
            
        Returns:
            Fused features with the same shape as text_features
        """
        batch_size, seq_len, _ = text_features.size()
        
        # Create attention mask from padding mask if provided
        attn_mask = None
        if text_mask is not None:
            # Convert from [batch_size, seq_len] to attention mask
            attn_mask = ~text_mask.bool()
        
        # Visual conditioning on text
        attended_text, _ = self.text_to_visual_attention(
            query=visual_features,
            key=text_features,
            value=text_features,
            key_padding_mask=attn_mask
        )
        
        # Text conditioning on visual
        attended_visual, _ = self.visual_to_text_attention(
            query=text_features,
            key=visual_features,
            value=visual_features
        )


        # Interpolate attended_text to the sequence length
        attended_text = F.interpolate(attended_text.transpose(1, 2), size=seq_len, mode='linear', align_corners=False).transpose(1, 2)

        attended_text = self.norm1(attended_text)
        attended_visual = self.norm2(attended_visual)
        
        # Compute fusion gate
        # Determine how much of each modality to use at each position
        gate = self.fusion_gate(torch.cat([attended_text, attended_visual], dim=-1))
        
        # Weighted combination of the two streams
        fused_features = gate * attended_visual + (1 - gate) * attended_text

        fused_features = self.norm3(fused_features)
        
        # Concatenate and project for rich feature representation
        output = text_features + self.output_projection(torch.cat([fused_features, text_features], dim=-1))
        
        return output


class AnsweringAgent(nn.Module):
    """
    Answering Agent for aerial navigation.
    
    This model integrates:
    1. Pretrained T5 language model for text processing
    2. Visual feature extraction for individual images
    3. Temporal observation encoding (current + previous views)
    4. Cross-modal fusion between text and visual features
    
    Architecture follows cognitive science principles:
    - Explicit memory for past observations via the temporal encoder
    - Input formatting that highlights the first instruction naturally
    - Cross-modal alignment of vision and language
    - Fine-tuning only necessary parts of the pretrained model
    """
    
    def __init__(self, config: Config, tokenizer=None, logger=None):
        super().__init__()
        self.config = config
        
        # Set up logger for this instance
        self.logger = logger
        
        # Store T5 model name for loading correct tokenizer
        self.model_name = config.model.t5_model_name
        
        # Use provided tokenizer or create one
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, model_max_length=self.config.data.max_seq_length, add_special_tokens=True)
        
        # Load T5 base model (encoder-decoder)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.t5_config = self.t5_model.config
        
        # Feature extractor for visual processing
        self.feature_extractor = FeatureExtractor(config)
        
        # Temporal observation encoder for processing previous views
        self.temporal_encoder = TemporalObservationEncoder(
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.dropout
        )

        # Project visual context to 32 times the hidden size
        self.visual_context_projection = nn.Linear(config.model.hidden_size, config.model.num_visual_tokens * config.model.hidden_size)
        
        # Cross-modal fusion for text and visual features
        self.fusion_module = CrossModalFusion(
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.dropout
        )
        
        # T5 Adapter layer - bridges the gap between our fused features and what T5 decoder expects
        self.t5_adapter = nn.Sequential(
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.LayerNorm(config.model.hidden_size),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.LayerNorm(config.model.hidden_size)
        )

        self.destination_reconstruction_head = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        self.visual_reconstruction_head = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        
        # Initialize adapter weights
        self._init_adapter_weights()
        
        # Freeze the entire T5 model
        self._freeze_t5_parameters()
        
    def _init_adapter_weights(self):
        """Initialize adapter weights carefully to ensure good initial performance"""
        for module in self.t5_adapter.modules():
            if isinstance(module, nn.Linear):
                # Use small initialization for stability
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def _freeze_t5_parameters(self):
        """Freeze ALL T5 parameters for efficiency."""
        # Count parameters
        total_params = 0
        
        # Freeze all T5 parameters
        for name, param in self.t5_model.named_parameters():
            total_params += param.numel()
            param.requires_grad = False
                
        self.logger.info(f"T5 model: 0.00% of parameters are trainable (all frozen)")
        self.logger.info(f"Total T5 parameters: {total_params:,}")
        
        # Count our trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.logger.info(f"Total trainable parameters: {trainable_params:,}")
    
    def forward(self, text_input: dict, current_view: torch.Tensor, 
                previous_views: torch.Tensor, labels: torch.Tensor = None, generate: bool = False,
                destination_view: Optional[torch.Tensor] = None, curriculum_ratio: float = 0.0,
                positive_input: Optional[dict] = None, negative_input: Optional[dict] = None) -> Dict:
        """
        Forward pass of the model.
        
        Args:
            text_input (dict): Tokenized input with keys 'input_ids' and 'attention_mask'
            current_view (torch.Tensor): Current view image tensor [batch_size, 3, H, W]
            previous_views (torch.Tensor): Previous views tensor [batch_size, max_prev, 3, H, W]
            labels (torch.Tensor, optional): Target labels for generation/loss calculation
            generate (bool): Whether to generate text instead of calculating loss
            destination_view (torch.Tensor, optional): Destination view for curriculum learning
            curriculum_ratio (float): Ratio for curriculum learning (0-1)
            positive_input (dict, optional): Positive example for contrastive learning
            negative_input (dict, optional): Negative example for contrastive learning
            
        Returns:
            Dict containing model outputs, including:
                - logits: Output logits
                - encoder_last_hidden_state: Encoder hidden states
                - visual_context: Visual context
                - adapted_features: Adapted features for contrastive learning
        """
        batch_size = current_view.size(0)
        device = current_view.device
        
        # --- Visual Processing ---
        # Extract visual features from current view
        # Extract visual features
        if hasattr(self, 'feature_extractor'):
            current_features = self.feature_extractor(current_view)
        else:
            # Handle cases where we might load a checkpoint with different architecture
            self.logger.warning("Feature extractor not found, returning zero features")
            current_features = torch.zeros(batch_size, self.config.model.hidden_size, 
                                        device=device)
        
        # Process previous views if available
        if previous_views.size(0) > 0:
            # Extract features for each previous view
            num_prev = min(previous_views.size(1), self.config.data.max_previous_views)
            prev_features_list = []
            
            # Reshape to process each view separately
            views_to_process = previous_views[:, :num_prev].contiguous()
            views_flat = views_to_process.view(-1, *views_to_process.shape[2:])
            
            # Process all views at once for efficiency
            all_prev_features = self.feature_extractor(views_flat)
            
            # Reshape back to [batch, num_prev, hidden]
            prev_features = all_prev_features.view(batch_size, num_prev, -1)
                else:
            # Default to empty tensor if no previous views
            prev_features = torch.zeros(batch_size, 1, self.config.model.hidden_size,
                                      device=device)
        
        # Apply temporal encoding to incorporate previous views
        visual_context = self.temporal_encoder(current_features, prev_features)

        # Process destination image if provided (for curriculum learning)
        dest_features = None
        if destination_view is not None and curriculum_ratio > 0:
            dest_features = self.feature_extractor(destination_view)
            
            # Use linear interpolation for curriculum learning
            # As training progresses, curriculum_ratio decreases
            # - Early training: rely more on destination (oracle)
            # - Later training: rely more on visual context (learned)
            visual_context = (
                curriculum_ratio * dest_features + 
                (1 - curriculum_ratio) * visual_context
            )
        
        # --- Text Processing ---
        # Get T5 encoder outputs for the input text
        encoder_outputs = self.t5_model.encoder(
            input_ids=text_input["input_ids"],
            attention_mask=text_input["attention_mask"],
            return_dict=True
        )

        # --- Cross-Modal Fusion ---
        # Visual tokens need to be the same dimension as T5's hidden states
        # Project visual context to create multiple visual tokens
        visual_ctx_expanded = self.visual_context_projection(visual_context)
        visual_ctx_expanded = visual_ctx_expanded.view(
            batch_size, 
            self.config.model.num_visual_tokens, 
            self.config.model.hidden_size
        )
        
        # Get text features from encoder
        text_features = encoder_outputs.last_hidden_state

        # Apply cross-modal fusion between text and visual features
        fused_features = self.fusion_module(
            text_features=text_features,
            visual_features=visual_ctx_expanded,
            text_mask=text_input["attention_mask"]
        )
            
        # Adapt the fused features to work with T5 decoder
        encoder_hidden_states = self.t5_adapter(fused_features)

        # --- Create reconstruction targets for additional training signal ---
        visual_context_target = visual_context.detach()  # Stop gradients
        reconstructed_visual_features = self.visual_reconstruction_head(encoder_hidden_states.mean(dim=1))
        
        # Create reconstruction target for destination if available
        reconstructed_destination_features = None
        if dest_features is not None:
            dest_features_detached = dest_features.detach()  # Stop gradients
            reconstructed_destination_features = self.destination_reconstruction_head(
                encoder_hidden_states.mean(dim=1)
            )
        
        # --- Decoder Processing ---
        # Calculate logits or generate text
        if not generate:
            # Training or validation mode
            
            # Get decoder outputs
            decoder_outputs = self.t5_model.decoder(
                input_ids=labels,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=text_input["attention_mask"],
                return_dict=True
            )
            
            # Get logits for token prediction
            logits = self.t5_model.lm_head(decoder_outputs.last_hidden_state)
                
            # Create output dictionary
            outputs = {
                "logits": logits,
                "encoder_last_hidden_state": encoder_hidden_states,
                "visual_context": visual_context,
                "visual_context_target": visual_context_target,
                "reconstructed_visual_features": reconstructed_visual_features,
                "adapted_features": encoder_hidden_states.mean(dim=1),
                "feature_norm": visual_context.norm(p=2, dim=1).mean()
            }
            
            if dest_features is not None:
                outputs["reconstructed_destination_features"] = reconstructed_destination_features
                
            # --- Process positive examples for contrastive learning ---
            if positive_input is not None:
                positive_encoder_outputs = self.t5_model.encoder(
                    input_ids=positive_input["input_ids"],
                    attention_mask=positive_input["attention_mask"],
                    return_dict=True
                )
                
                # Apply fusion with the same visual context
                positive_fused = self.fusion_module(
                    text_features=positive_encoder_outputs.last_hidden_state,
                    visual_features=visual_ctx_expanded,
                    text_mask=positive_input["attention_mask"]
                )
                
                # Adapt the fused features
                positive_adapted = self.t5_adapter(positive_fused)
                
                # Add to outputs
                outputs["positive_encoder_hidden_state"] = positive_adapted
                outputs["positive_adapted_features"] = positive_adapted.mean(dim=1)
                
            # --- Process negative examples for contrastive learning ---
            if negative_input is not None:
                negative_encoder_outputs = self.t5_model.encoder(
                    input_ids=negative_input["input_ids"],
                    attention_mask=negative_input["attention_mask"],
                    return_dict=True
                )
                
                # Apply fusion with the same visual context
                negative_fused = self.fusion_module(
                    text_features=negative_encoder_outputs.last_hidden_state,
                    visual_features=visual_ctx_expanded,
                    text_mask=negative_input["attention_mask"]
                )
                
                # Adapt the fused features
                negative_adapted = self.t5_adapter(negative_fused)
                
                # Add to outputs
                outputs["negative_encoder_hidden_state"] = negative_adapted
                outputs["negative_adapted_features"] = negative_adapted.mean(dim=1)
                
            return outputs
        else:
            # Generation mode
            # Return encoder outputs for use with generate_answer
            return {
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": text_input["attention_mask"]
            }
    
            
    def generate_answer(self, text_input: dict, current_view: torch.Tensor, 
                       previous_views: torch.Tensor, max_length: int = 128) -> torch.Tensor:
        """
        Generate answer text.
        
        Args:
            text_input: Dictionary with input_ids and attention_mask
            current_view: Current visual input [batch_size, channels, height, width] 
            previous_views: Previous visual inputs [batch_size, num_prev, channels, height, width]
            max_length: Maximum generated sequence length
            
        Returns:
            Generated token IDs [batch_size, seq_len]
        """
        # Forward pass in evaluation mode
        with torch.no_grad():
            outputs = self.forward(text_input, current_view, previous_views)
            
        return outputs["sequences"]