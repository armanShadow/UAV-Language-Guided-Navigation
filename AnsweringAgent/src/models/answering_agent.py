import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer
from models.feature_extractor import FeatureExtractor
from typing import Dict, Tuple, Optional, List
import math
from config import Config
import torch.nn.functional as F


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
        
        # Residual connection and normalization
        if torch.isnan(attn_output).any() or torch.isnan(current_features).any():
            print("NaN detected before norm1 in TemporalObservationEncoder!")
            attn_output = torch.nan_to_num(attn_output, nan=0.0)
            current_features = torch.nan_to_num(current_features, nan=0.0)
            
        features = self.norm1(current_features + attn_output)
        
        # Feed-forward network
        ff_output = self.ff_network(features)
        
        # Check for NaNs in ff_output
        if torch.isnan(ff_output).any():
            print("NaN detected in ff_output in TemporalObservationEncoder!")
            ff_output = torch.nan_to_num(ff_output, nan=0.0)
        
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
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
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
            if (text_mask.sum(dim=1) == 0).any():
                print("Warning: A sample in the batch has all tokens masked!")
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
        
        # Normalize attended features
        if torch.isnan(attended_text).any():
            print("NaN detected in attended_text in CrossModalFusion!")
            attended_text = torch.nan_to_num(attended_text, nan=0.0)
            
        if torch.isnan(attended_visual).any():
            print("NaN detected in attended_visual in CrossModalFusion!")
            attended_visual = torch.nan_to_num(attended_visual, nan=0.0)
            
        attended_text = self.norm1(attended_text)
        attended_visual = self.norm2(attended_visual)
        
        # Compute fusion gate
        # Determine how much of each modality to use at each position
        gate = self.fusion_gate(torch.cat([attended_text, attended_visual], dim=-1))
        
        # Check for NaNs in gate
        if torch.isnan(gate).any():
            print("NaN detected in fusion gate in CrossModalFusion!")
            gate = torch.nan_to_num(gate, nan=0.5)  # Default to equal weighting
        
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
    
    def __init__(self, config: Config, tokenizer=None):
        super().__init__()
        self.config = config
        
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
                
        print(f"T5 model: 0.00% of parameters are trainable (all frozen)")
        print(f"Total T5 parameters: {total_params:,}")
        
        # Count our trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {trainable_params:,}")
    
    def forward(self, text_input: dict, current_view: torch.Tensor, 
                previous_views: torch.Tensor, labels: torch.Tensor = None) -> Dict:
        """
        Forward pass of the answering agent.
        
        Args:
            text_input: Dictionary with input_ids and attention_mask
            current_view: Current visual input [batch_size, channels, height, width]
            previous_views: Previous visual inputs [batch_size, num_prev, channels, height, width]
            labels: Target token IDs for training [batch_size, seq_len]
            
        Returns:
            Dictionary with logits and feature_norm for training or sequences for inference
        """
        # Extract needed inputs
        input_ids = text_input['input_ids']
        attention_mask = text_input.get('attention_mask', None)
        
        # Extract current view features using the dedicated method
        current_features = self.feature_extractor.extract_single_view_features(current_view)
        if torch.isnan(current_features).any():
            print(f"NaN detected in current_features! - shape: {current_features.shape} - mean: {current_features.mean().item()} - std: {current_features.std().item()}")
            current_features = torch.nan_to_num(current_features, nan=0.0)
        
        # Extract previous view features
        prev_features = []
        prev_views_count = previous_views.size(1)
        
        # Process each previous view separately
        for i in range(prev_views_count):
            prev_view = previous_views[:, i]
            prev_feat = self.feature_extractor.extract_single_view_features(prev_view)
            prev_features.append(prev_feat)
        
        # Stack previous features [batch_size, num_prev, hidden_size]
        prev_features = torch.stack(prev_features, dim=1)

        if torch.isnan(prev_features).any():
            print(f"NaN detected in prev_features! - shape: {prev_features.shape} - mean: {prev_features.mean().item()} - std: {prev_features.std().item()}")
            prev_features = torch.nan_to_num(prev_features, nan=0.0)
        
        # Now apply our specialized temporal observation encoder
        visual_context = self.temporal_encoder(current_features, prev_features)

        if torch.isnan(visual_context).any():
            print(f"NaN detected in visual_context! - shape: {visual_context.shape} - mean: {visual_context.mean().item()} - std: {visual_context.std().item()}")
            visual_context = torch.nan_to_num(visual_context, nan=0.0)

        visual_context = self.visual_context_projection(visual_context)
        visual_context = visual_context.view(batch_size, self.config.model.num_visual_tokens, self.config.model.hidden_size)

        if torch.isnan(visual_context).any():
            print(f"NaN detected in visual_context! - shape: {visual_context.shape} - mean: {visual_context.mean().item()} - std: {visual_context.std().item()}")
            visual_context = torch.nan_to_num(visual_context, nan=0.0)
        
        # Encode text with T5 encoder
        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_features = encoder_outputs.last_hidden_state
        
        # Apply cross-modal fusion
        fused_features = self.fusion_module(
            text_features=text_features,
            visual_features=visual_context,
            text_mask=attention_mask
        )

        if torch.isnan(fused_features).any():
            print(f"NaN detected in fused_features! - shape: {fused_features.shape} - mean: {fused_features.mean().item()} - std: {fused_features.std().item()}")
            fused_features = torch.nan_to_num(fused_features, nan=0.0)
        
        # Apply the adapter to bridge the gap between our features and what T5 expects
        adapted_features = self.t5_adapter(fused_features)
        
        # Check for NaNs in adapted features
        if torch.isnan(adapted_features).any():
            print(f"NaN detected in adapted_features - mean: {fused_features.mean().item()}, std: {fused_features.std().item()}")
            # Replace NaNs with zeros for stability
            adapted_features = torch.nan_to_num(adapted_features, nan=0.0)
        
        # Calculate feature norm for regularization (detach to prevent memory leak)
        feature_norm = adapted_features.norm(2).detach()
        
        # During training mode, return logits for loss calculation in training loop
        if self.training:
            # Note: Don't use torch.no_grad() here as we need gradients for backward
            # T5 parameters are already frozen in _freeze_t5_parameters method
            
            # Check if labels contain NaN values
            if labels is not None and torch.isnan(labels).any():
                print("NaN detected in labels!")
                # Replace NaNs with pad token id
                labels = torch.nan_to_num(labels, nan=self.tokenizer.pad_token_id)
                
            outputs = self.t5_model(
                input_ids=None,
                attention_mask=attention_mask,
                encoder_outputs=(adapted_features,),
                labels=labels,
                return_dict=True
            )
                
            # Return logits for external loss calculation
            return {
                "logits": outputs.logits,
                "feature_norm": feature_norm
            }
        
        # Inference mode
        else:
            # Generate output sequence
            with torch.no_grad():  # Don't train the T5 model during generation
                outputs = self.t5_model.generate(
                    encoder_outputs=(adapted_features,),
                    attention_mask=attention_mask,
                    max_length=self.config.model.max_answer_length,
                    num_beams=4,  # Use beam search for better quality
                    early_stopping=True,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            return {
                "sequences": outputs.sequences,
                "scores": outputs.scores if hasattr(outputs, "scores") else None
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