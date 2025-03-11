import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from models.feature_extractor import FeatureExtractor
from typing import Dict, Tuple, Optional
import math
from utils.logger import get_logger
from config import Config

# Get the logger instance
logger = get_logger()

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])  # handle odd d_model case
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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
    def __init__(self, config: Config):
        super().__init__()
        self.device = torch.device(config.training.device)
        
        # Initialize BERT
        self.bert = BertModel.from_pretrained(config.model.bert_model_name)
        self.bert_dropout = nn.Dropout(config.model.dropout)
        
        # Initialize feature extractor with device info
        self.feature_extractor = FeatureExtractor(config)
        
        # Initialize feature attention
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=config.model.hidden_size,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.feat_dropout
        )
        
        # Initialize feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.model.hidden_size * 2, config.model.hidden_size),
            nn.LayerNorm(config.model.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout)
        )
        
        # Initialize positional encoding
        self.pos_encoder = PositionalEncoding(config.model.hidden_size)
        
        # Initialize decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model.hidden_size,
            nhead=config.model.num_attention_heads,
            dim_feedforward=config.model.feedforward_dim,
            dropout=config.model.dropout,
            activation='relu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.model.num_decoder_layers)
        
        # Add output projection layer
        self.output_projection = nn.Linear(config.model.hidden_size, config.model.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        # Move model to device
        self.to(self.device)
        
        # Verify all components are on correct devices
        self._verify_device_placement()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize feature fusion layers
        for m in self.feature_fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
        # Initialize output layer
        nn.init.xavier_normal_(self.output_projection.weight, gain=1.0)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
            
        # Initialize decoder
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=1.0)
                
        # Ensure all parameters are on the correct device
        for param in self.parameters():
            param.data = param.data.to(self.device)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from a checkpoint file."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
    def forward(self, text_input, current_view, previous_views):
        """
        Forward pass of the model.
        Args:
            text_input (dict): Dictionary containing BERT inputs
            current_view (torch.Tensor): Current view image tensor
            previous_views (list): List of previous view image tensors
        """
        # Ensure all text inputs are on the correct device
        text_input = {k: v.to(text_input['input_ids'].device) for k, v in text_input.items()}
        
        # Reshape text inputs to remove extra dimension
        text_input = {k: v.squeeze(1) for k, v in text_input.items()}
        
        # Process text input with BERT
        text_outputs = self.bert(
            **text_input
        )
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Get visual features
        visual_features = self.feature_extractor(current_view, previous_views)  # [batch_size, hidden_size]
        
        # Expand visual features to match sequence length
        visual_features = visual_features.unsqueeze(1).expand(-1, text_features.size(1), -1)
        
        # Combine features (using text features as target and visual features as memory)
        decoder_output = self.decoder(
            tgt=text_features.transpose(0, 1),  # [seq_len, batch_size, hidden_size]
            memory=visual_features.transpose(0, 1)  # [seq_len, batch_size, hidden_size]
        )
        
        # Transpose back to [batch_size, seq_len, hidden_size]
        decoder_output = decoder_output.transpose(0, 1)
        
        # Project to vocabulary size
        output = self.output_projection(decoder_output)  # [batch_size, seq_len, vocab_size]
        
        return output
    
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

    def combine_features(self, text_features, visual_features):
        """
        Combines text and visual features using attention and fusion.
        
        Args:
            text_features (torch.Tensor): Text features from BERT [batch_size, seq_len, hidden_size]
            visual_features (torch.Tensor): Visual features [batch_size, hidden_size]
            
        Returns:
            torch.Tensor: Combined features [batch_size, seq_len, hidden_size]
        """
        batch_size = text_features.size(0)
        
        # Expand visual features to match sequence length
        visual_features = visual_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # Apply positional encoding to visual features
        visual_features = self.pos_encoder(visual_features)
        
        # Apply feature attention between text and visual features
        attended_features, _ = self.feature_attention(
            text_features.transpose(0, 1),    # [seq_len, batch_size, hidden_size]
            visual_features.transpose(0, 1),   # [1, batch_size, hidden_size]
            visual_features.transpose(0, 1)    # [1, batch_size, hidden_size]
        )
        
        # Transpose back to [batch_size, seq_len, hidden_size]
        attended_features = attended_features.transpose(0, 1)
        
        # Concatenate and fuse text and attended visual features
        combined = torch.cat([text_features, attended_features], dim=-1)  # [batch_size, seq_len, hidden_size*2]
        fused_features = self.feature_fusion(combined)  # [batch_size, seq_len, hidden_size]
        
        return fused_features 

    def _verify_device_placement(self):
        """Verify all components are on the correct device."""
        for name, param in self.named_parameters():
            if param.data.device != self.device:
                raise RuntimeError(f"Parameter {name} is not on the correct device. Expected {self.device}, found {param.data.device}")
        for buffer in self.buffers():
            if buffer.device != self.device:
                raise RuntimeError(f"Buffer {buffer} is not on the correct device. Expected {self.device}, found {buffer.device}")
        logger.info("All components are on the correct device.") 