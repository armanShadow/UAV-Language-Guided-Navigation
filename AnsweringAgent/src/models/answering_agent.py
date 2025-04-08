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
            if (text_mask.sum(dim=1) == 0).any():
                logger.warning("A sample in the batch has all tokens masked!")
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
        
        # Check for NaNs in gate
        if torch.isnan(gate).any():
            logger.warning("NaN detected in fusion gate in CrossModalFusion!")
            # For gates, 0.5 is a balanced choice between the two modalities
            # We use the original inputs rather than potentially corrupted attended features
            return text_features  # Return original text features as a fallback
        
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
                destination_view: Optional[torch.Tensor] = None, curriculum_ratio: float = 0.0) -> Dict:
        """
        Forward pass of the answering agent.
        
        Args:
            text_input: Dictionary with input_ids and attention_mask
            current_view: Current visual input [batch_size, channels, height, width]
            previous_views: Previous visual inputs [batch_size, num_prev, channels, height, width]
            labels: Target token IDs for training/validation [batch_size, seq_len]
            generate: Whether to generate output text (inference mode)
            destination_view: Optional destination view for curriculum learning [batch_size, channels, height, width]
            curriculum_ratio: The ratio (0-1) of destination view information to use (0=none, 1=full)
            
        Returns:
            Dictionary with logits, feature_norm and destination_loss (when applicable) for training or validation
        """
        # Extract needed inputs
        input_ids = text_input['input_ids']
        attention_mask = text_input.get('attention_mask', None)

        batch_size = current_view.size(0)
        
        # Extract current view features using the dedicated method
        current_features = self.feature_extractor.extract_single_view_features(current_view)
        if torch.isnan(current_features).any():
            nan_percentage = torch.isnan(current_features).float().mean().item() * 100
            self.logger.warning(f"NaN detected in current_features! - {nan_percentage:.2f}% of values are NaN")
            
            # If more than 50% of values are NaN, this is a serious issue
            if nan_percentage > 50:
                self.logger.error("More than 50% of current_features are NaN - training may be unstable")
            
            # Instead of simply replacing with zeros, which can lead to gradient issues,
            # we can use a small random value within the feature's existing range
            # This might help prevent the network from getting stuck in bad local minima
            with torch.no_grad():
                # Calculate valid statistics
                valid_features = current_features[~torch.isnan(current_features)]
                if len(valid_features) > 0:
                    mean_val = valid_features.mean().item()
                    std_val = valid_features.std().item()
                    # Replace NaNs with small random noise around the mean
                    mask = torch.isnan(current_features)
                    current_features[mask] = mean_val + torch.randn_like(current_features[mask]) * std_val * 0.1
                else:
                    # If all values are NaN, use a small random initialization
                    current_features = torch.randn_like(current_features) * 0.01
        
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
            nan_percentage = torch.isnan(prev_features).float().mean().item() * 100
            self.logger.warning(f"NaN detected in prev_features! - {nan_percentage:.2f}% of values are NaN")
            
            # Similar approach as with current_features
            with torch.no_grad():
                valid_features = prev_features[~torch.isnan(prev_features)]
                if len(valid_features) > 0:
                    mean_val = valid_features.mean().item()
                    std_val = valid_features.std().item()
                    mask = torch.isnan(prev_features)
                    prev_features[mask] = mean_val + torch.randn_like(prev_features[mask]) * std_val * 0.1
                else:
                    prev_features = torch.randn_like(prev_features) * 0.01
        
        # Now apply our specialized temporal observation encoder
        visual_context = self.temporal_encoder(current_features, prev_features)

        # Process destination features if available for curriculum learning
        destination_features = None
        if destination_view is not None and curriculum_ratio > 0:
            # Extract destination features
            destination_features = self.feature_extractor.extract_single_view_features(destination_view)
            # Apply curriculum learning - mix current visual context with destination features
            visual_context = (1 - curriculum_ratio) * visual_context + curriculum_ratio * destination_features

        if torch.isnan(visual_context).any():
            nan_percentage = torch.isnan(visual_context).float().mean().item() * 100
            self.logger.warning(f"NaN detected in visual_context! - {nan_percentage:.2f}% of values are NaN")
            
            # If NaNs appear after the temporal encoder, we should fall back to the input
            # This maintains the computational graph while avoiding NaN propagation
            if nan_percentage > 50:
                self.logger.error("Severe NaN issue in visual_context - using current_features as fallback")
                visual_context = current_features  # Use the input as fallback
            else:
                # Replace only the NaN values with corresponding values from current_features
                mask = torch.isnan(visual_context)
                visual_context[mask] = current_features[mask]

        visual_context = self.visual_context_projection(visual_context)
        visual_context = visual_context.view(batch_size, self.config.model.num_visual_tokens, self.config.model.hidden_size)
        
        # Normalize feature vectors to control their magnitude
        # Apply normalization along the feature dimension with a fixed scale factor
        scale_factor = 1.0
        visual_context = F.normalize(visual_context, p=2, dim=-1) * scale_factor

        if torch.isnan(visual_context).any():
            self.logger.warning(f"NaN detected in projected visual_context!")
            # After projection, we should handle NaNs carefully to maintain gradient flow
            with torch.no_grad():
                mask = torch.isnan(visual_context)
                # Replace with small random values to maintain gradient flow
                visual_context[mask] = torch.randn_like(visual_context[mask]) * 0.01
        
        # Check if any sequence has all masks set to 0 (completely masked)
        if attention_mask is not None and (attention_mask.sum(dim=1) == 0).any():
            self.logger.warning("Some sequences have all positions masked!")

        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        text_features = encoder_outputs.last_hidden_state

        if torch.isnan(text_features).any():
            nan_percentage = torch.isnan(text_features).float().mean().item() * 100
            self.logger.warning(f"NaN detected in T5 text_features! - {nan_percentage:.2f}% of values are NaN")
            
            # T5 encoder outputs should not have NaNs; this indicates a serious issue
            if nan_percentage > 10:
                self.logger.error("Significant NaNs in T5 encoder output - model stability is compromised")
                
            # For T5 outputs, since we're not training the T5 model, we can safely replace with zeros
            # without affecting the training dynamics of our trainable components
            mask = torch.isnan(text_features)
            text_features[mask] = 0.0
        
        # Apply cross-modal fusion
        fused_features = self.fusion_module(
            text_features=text_features,
            visual_features=visual_context,
            text_mask=attention_mask
        )

        if torch.isnan(fused_features).any():
            nan_percentage = torch.isnan(fused_features).float().mean().item() * 100
            self.logger.warning(f"NaN detected in fused_features! - {nan_percentage:.2f}% of values are NaN")
            
            # If fusion fails, fall back to text features
            if nan_percentage > 30:
                self.logger.error("Severe NaN issue in fusion - using text_features as fallback")
                fused_features = text_features  # Use text features as fallback
            else:
                # Replace NaN values with corresponding values from text_features
                mask = torch.isnan(fused_features)
                fused_features[mask] = text_features[mask]
        
        # Apply the adapter to bridge the gap between our features and what T5 expects
        adapted_features = self.t5_adapter(fused_features)
        
        # Check for NaNs in adapted features
        if torch.isnan(adapted_features).any():
            nan_percentage = torch.isnan(adapted_features).float().mean().item() * 100
            self.logger.warning(f"NaN detected in adapted_features - {nan_percentage:.2f}% of values are NaN")
            
            # For the final output to T5, we need to ensure no NaNs
            if nan_percentage > 20:
                self.logger.error("Severe NaN issue in adapter output - using fused_features as fallback")
                # Use fused features as fallback for stability
                adapted_features = fused_features
            else:
                # Replace only NaN values
                mask = torch.isnan(adapted_features)
                adapted_features[mask] = fused_features[mask]
        
        # Calculate feature norm for regularization (detach to prevent memory leak)
        feature_norm = adapted_features.norm(2).detach()
        
        # During training mode, return logits for loss calculation in training loop
        if not generate:
            # Note: Don't use torch.no_grad() here as we need gradients for backward
            # T5 parameters are already frozen in _freeze_t5_parameters method
            
            # Check if labels contain NaN values
            if labels is not None and torch.isnan(labels).any():
                self.logger.warning("NaN detected in labels!")
                # NaN in labels is invalid - must replace with a valid token ID
                # Use pad token for missing values as it's semantically appropriate
                labels = torch.nan_to_num(labels, nan=self.tokenizer.pad_token_id)
                
            outputs = self.t5_model(
                input_ids=None,
                attention_mask=attention_mask,
                encoder_outputs=(adapted_features,),
                labels=labels,
                return_dict=True,
                output_hidden_states=True
            )
                
            # Return logits for external loss calculation
            result = {
                "logits": outputs.logits,
                "feature_norm": feature_norm,
                "adapted_features": adapted_features.mean(dim=1),
                "hidden_states": outputs.decoder_hidden_states[-1]
            }
            
            # Include destination features if available for external loss calculation
            if destination_features is not None:
                result["destination_features"] = destination_features
            
            return result
        
        # Inference mode
        else:
            # Generate output sequence
            with torch.no_grad():  # Don't train the T5 model during generation
                outputs = self.t5_model.generate(
                    encoder_outputs=BaseModelOutput(
                        last_hidden_state=adapted_features,
                    ),
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