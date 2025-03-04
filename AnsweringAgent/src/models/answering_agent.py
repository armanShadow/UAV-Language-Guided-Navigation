import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from models.feature_extractor import FeatureExtractor
from typing import Dict, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [seq_len, batch_size, d_model]
        # pe shape: [max_len, 1, d_model]
        return x + self.pe[:x.size(0)]

class MultiModalAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Project queries, keys, and values
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        if key_padding_mask is not None:
            # Convert attention mask to boolean (0 -> True for padding, 1 -> False for non-padding)
            key_padding_mask = (key_padding_mask == 0)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        return self.out_proj(attn_output)

class AnsweringAgent(nn.Module):
    def __init__(self, 
                 bert_model_name: str = 'bert-base-uncased',
                 hidden_size: int = 768,
                 dropout: float = 0.5,
                 feat_dropout: float = 0.4,
                 darknet_config_path: str = None,
                 darknet_weights_path: str = None,
                 device: str = 'cpu'):
        super().__init__()
        
        # Set device
        self.device = torch.device(device)
        
        # Initialize BERT model and tokenizer
        self.bert = BertModel.from_pretrained(bert_model_name).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            darknet_config_path=darknet_config_path,
            darknet_weights_path=darknet_weights_path,
            device=self.device
        )
        
        # Multi-modal attention for feature fusion
        self.feature_attention = MultiModalAttention(hidden_size, dropout=feat_dropout)
        
        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(feat_dropout),
            nn.Linear(hidden_size, hidden_size)
        ).to(self.device)
        
        # Decoder layers
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='relu'
            ),
            num_layers=6
        ).to(self.device)
        
        # Output layer for answer generation
        self.output_layer = nn.Linear(hidden_size, self.tokenizer.vocab_size).to(self.device)
        
        # Positional encoding for decoder
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_size).to(self.device)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:  # For linear and conv layers
                    nn.init.xavier_uniform_(param)
                elif len(param.shape) == 1:  # For batch norm layers
                    nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def to_device(self, device):
        """Move the model and all its parameters to the specified device"""
        self.device = torch.device(device)
        self.to(self.device)
        
        # Move BERT model to device
        self.bert = self.bert.to(self.device)
        
        # Move feature extractor and its components to device
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.feature_extractor.darknet = self.feature_extractor.darknet.to(self.device)
        self.feature_extractor.flatten = self.feature_extractor.flatten.to(self.device)
        self.feature_extractor.attention = self.feature_extractor.attention.to(self.device)
        
        # Move all darknet modules to device
        for module in self.feature_extractor.darknet.modules():
            if isinstance(module, nn.Conv2d):
                module.to(self.device)
            elif isinstance(module, nn.BatchNorm2d):
                module.to(self.device)
            elif isinstance(module, nn.LeakyReLU):
                module.to(self.device)
            elif isinstance(module, nn.Sequential):
                module.to(self.device)
        
        # Move other components to device
        self.feature_attention = self.feature_attention.to(self.device)
        self.feature_fusion = self.feature_fusion.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.output_layer = self.output_layer.to(self.device)
        self.pos_encoder = self.pos_encoder.to(self.device)
        
        # Ensure all parameters are on the correct device
        for param in self.parameters():
            param.data = param.data.to(self.device)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
    def forward(self, 
                text_input: Dict[str, torch.Tensor],
                current_view: torch.Tensor,
                previous_views: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            text_input: Dictionary containing BERT input tensors
            current_view: Current view image tensor
            previous_views: Optional tensor of previous views
            
        Returns:
            torch.Tensor: Output logits for answer generation
        """
        # Ensure inputs are on CPU
        text_input = {k: v.to('cpu') for k, v in text_input.items()}
        current_view = current_view.to('cpu')
        if previous_views is not None:
            previous_views = previous_views.to('cpu')
            
        # Extract text features using BERT
        text_outputs = self.bert(**text_input)
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Extract visual features
        visual_features = self.feature_extractor(current_view)  # [batch_size, hidden_size]
        
        # Expand visual features to match sequence length
        visual_features = visual_features.unsqueeze(1).expand(-1, text_features.size(1), -1)
        
        # Apply multi-modal attention
        fused_features = self.feature_attention(
            text_features,
            visual_features,
            visual_features,
            key_padding_mask=text_input['attention_mask']
        )
        
        # Additional feature fusion
        combined_features = torch.cat([fused_features, visual_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)  # [batch_size, seq_len, hidden_size]
        
        # Create decoder input with learned embeddings
        batch_size = fused_features.size(0)
        seq_len = fused_features.size(1)
        decoder_input = fused_features.clone()  # Use fused features as initial input
        
        # Add positional encoding
        decoder_input = self.pos_encoder(decoder_input.transpose(0, 1)).transpose(0, 1)
        
        # Generate sequence using decoder
        # Reshape to [seq_len, batch_size, hidden_size] for transformer
        decoder_input = decoder_input.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        fused_features = fused_features.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        
        # Generate sequence using decoder
        decoder_output = self.decoder(
            decoder_input,  # [seq_len, batch_size, hidden_size]
            fused_features  # [seq_len, batch_size, hidden_size]
        ).transpose(0, 1)  # Back to [batch_size, seq_len, hidden_size]
        
        # Generate logits for each position
        logits = self.output_layer(decoder_output)  # [batch_size, seq_len, vocab_size]
        
        # Ensure output sequence length matches label sequence length (128)
        if logits.size(1) > 128:
            logits = logits[:, :128, :]
        elif logits.size(1) < 128:
            # Pad with zeros if needed
            pad_size = 128 - logits.size(1)
            logits = torch.nn.functional.pad(logits, (0, 0, 0, pad_size))
        
        return logits
    
    def generate_answer(self, 
                       text_input: Dict[str, torch.Tensor],
                       current_view: torch.Tensor,
                       previous_views: torch.Tensor = None,
                       max_length: int = 128,
                       num_beams: int = 4) -> str:
        """
        Generate an answer using beam search.
        
        Args:
            text_input (Dict[str, torch.Tensor]): Dictionary containing tokenized text inputs
            current_view (torch.Tensor): Current view image tensor
            previous_views (torch.Tensor, optional): Previous views tensor
            max_length (int): Maximum length of generated answer
            num_beams (int): Number of beams for beam search
            
        Returns:
            str: Generated answer
        """
        self.eval()
        with torch.no_grad():
            # Get initial logits
            logits = self(text_input, current_view, previous_views)
            
            # Initialize beam search
            beam_outputs = self.tokenizer.beam_search(
                logits,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
            # Decode the best sequence
            answer = self.tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
            
        self.train()
        return answer 