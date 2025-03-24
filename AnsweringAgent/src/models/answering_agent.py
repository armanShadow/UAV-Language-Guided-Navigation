import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from models.feature_extractor import FeatureExtractor
from typing import Dict, Tuple, Optional
import math
from config import Config
import torch.nn.functional as F
from torch.profiler import record_function


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
        assert hidden_size % num_heads == 0, f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
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
      
        # Get sizes
        tgt_len, batch_size, embed_dim = query.size()
        src_len = key.size(0)

        scaling = float(self.head_dim) ** -0.5

        # Project and reshape
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        q = q.contiguous().view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # Compute attention scores
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * scaling

        if key_padding_mask is not None:
            # Convert attention mask to boolean (0 -> True for padding, 1 -> False for non-padding)
            key_padding_mask = (key_padding_mask == 0)
            attn_weights = attn_weights.view(batch_size, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            attn_weights = attn_weights.view(batch_size * self.num_heads, tgt_len, src_len)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.bmm(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output

class AnsweringAgent(nn.Module):
    def __init__(self, config: Config, tokenizer=None):
        super().__init__()

        self.config = config
        
        # Use provided tokenizer or create one if not provided
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(config.model.bert_model_name)
            
        self.eos_token_id = self.tokenizer.sep_token_id  # Using [SEP] as EOS token
        
        # Initialize BERT and tokenizer
        self.bert = BertModel.from_pretrained(config.model.bert_model_name)
        self.bert_dropout = nn.Dropout(config.model.dropout)

        # Verify hidden sizes match
        assert config.model.hidden_size == self.bert.config.hidden_size, \
            f"Config hidden size {config.model.hidden_size} != BERT hidden size {self.bert.config.hidden_size}"

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(config)

        # Initialize positional encoding
        self.pos_encoder = PositionalEncoding(config.model.hidden_size)

        # Initialize feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.model.hidden_size * 2, config.model.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_size, config.model.hidden_size)
        )

        # Initialize feature attention
        self.feature_attention = MultiModalAttention(
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.dropout
        )

        # Initialize decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model.hidden_size,
            nhead=config.model.num_attention_heads,
            dim_feedforward=config.model.feedforward_dim,
            dropout=config.model.dropout,
            activation='relu'
        )

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.model.num_decoder_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize feature fusion layers
        for m in self.feature_fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize decoder
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p, gain=1.0)


    def forward(self, text_input, current_view, previous_views):
        """
        Forward pass of the model.
        Args:
            text_input (dict): Dictionary containing BERT inputs
            current_view (torch.Tensor): Current view image tensor [batch_size, C, H, W]
            previous_views (list): List of previous view image tensors
        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size]
        """
        # Ensure input tensors are properly shaped
        input_ids = text_input['input_ids']
        attention_mask = text_input.get('attention_mask', None)
        token_type_ids = text_input.get('token_type_ids', None)
        
        # Ensure tensors are 2D [batch_size, seq_len]
        if input_ids.dim() > 2:
            input_ids = input_ids.view(-1, input_ids.size(-1))
        if attention_mask is not None and attention_mask.dim() > 2:
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if token_type_ids is not None and token_type_ids.dim() > 2:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            
        # Get dimensions
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        
        # Verify sequence length doesn't exceed model's maximum
        assert seq_len <= self.config.data.max_seq_length, \
            f"Input sequence length {seq_len} exceeds maximum allowed length {self.config.data.max_seq_length}"
        
        # Update text_input with properly shaped tensors
        text_input = {k: v for k, v in {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }.items() if v is not None}
        
        # Process text input with BERT
        with record_function("bert_forward"):
            text_outputs = self.bert(**text_input)
        text_features = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        text_features = self.bert_dropout(text_features)
        
        # Add positional encoding to text features
        text_features = self.pos_encoder(text_features)
        
        # Get visual features
        with record_function("feature_extractor_forward"):
            visual_features = self.feature_extractor(current_view, previous_views)  # [batch_size, hidden_size]
        
        # Verify feature dimensions
        assert visual_features.size(-1) == self.config.model.hidden_size, \
            f"Visual features dim {visual_features.size(-1)} != hidden size {self.config.model.hidden_size}"
        
        # Expand visual features to match sequence length
        # TODO: #9: You could use other approaches other than this. it copies the data 512 times
        visual_features = visual_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]
        
        # Combine text and visual features
        with record_function("combine_features"):
            combined_features = self.combine_features(text_features, visual_features)  # [batch_size, seq_len, hidden_size]
        
        # Create target mask to prevent attention to future tokens
        target_mask = self.generate_square_subsequent_mask(seq_len, input_ids.device)
        
        # Process through decoder
        with record_function("decoder_forward"):
            decoder_output = self.decoder(
                tgt=combined_features.transpose(0, 1),
                memory=visual_features.transpose(0, 1),
                tgt_mask=target_mask
            )
        
        # Transpose back to [batch_size, seq_len, hidden_size]
        decoder_output = decoder_output.transpose(0, 1)
        
        # Project to vocabulary size using weight tying with BERT embeddings
        with record_function("output_projection"):
            output = F.linear(decoder_output, self.bert.embeddings.word_embeddings.weight)
        
        # Dynamic length handling instead of truncation
        if self.training:
            # During training, still use max_answer_length as a safety cap
            output = output[:, :self.config.model.max_answer_length, :]
        else:
            # During inference, process until EOS token or max length
            batch_size = output.size(0)
            device = output.device
            max_len = min(output.size(1), self.config.model.max_answer_length)
            
            # Initialize output tensor with padded length
            dynamic_output = torch.zeros(batch_size, self.config.model.max_answer_length, 
                                         output.size(2), device=device)
            
            for b in range(batch_size):
                # Get the predicted token ids for this batch
                pred_tokens = output[b, :max_len].argmax(dim=-1)
                
                # Find the first occurrence of EOS token
                eos_positions = (pred_tokens == self.eos_token_id).nonzero(as_tuple=True)[0]
                
                # If EOS found, only keep tokens until EOS (inclusive)
                if len(eos_positions) > 0:
                    end_pos = eos_positions[0].item() + 1  # +1 to include the EOS token
                    dynamic_output[b, :end_pos] = output[b, :end_pos]
                else:
                    # No EOS found, keep all tokens up to max_len
                    dynamic_output[b, :max_len] = output[b, :max_len]
            
            output = dynamic_output
        
        return output
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate square mask for transformer decoder.
        
        Args:
            sz (int): Size of the square mask (sequence length)
            device (torch.device): Device to create the mask on
            
        Returns:
            torch.Tensor: Mask tensor of shape [sz, sz] with -inf for masked positions
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, float(0.0))
        return mask


    def combine_features(self, text_features, visual_features):
        """
        Combines text and visual features using attention and fusion.
        
        Args:
            text_features (torch.Tensor): Text features from BERT [batch_size, seq_len, hidden_size]
            visual_features (torch.Tensor): Visual features [batch_size, seq_len, hidden_size]
            
        Returns:
            torch.Tensor: Combined features [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = text_features.size()
        
        # Verify dimensions
        assert hidden_size == self.config.model.hidden_size, \
            f"Text features dimension {hidden_size} != hidden size {self.config.model.hidden_size}"
        assert visual_features.size(-1) == hidden_size, \
            f"Visual features dimension {visual_features.size(-1)} != hidden size {hidden_size}"
        assert visual_features.size(1) == seq_len, \
            f"Visual features sequence length {visual_features.size(1)} != text sequence length {seq_len}"
        
        # Transpose inputs for attention (from [batch_size, seq_len, hidden_size] to [seq_len, batch_size, hidden_size])
        text_features = text_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)
        
        # Apply feature attention between text and visual features
        with record_function("feature_attention"):
            attended_features = self.feature_attention(
                query=text_features,              # [seq_len, batch_size, hidden_size]
                key=visual_features,              # [seq_len, batch_size, hidden_size]
                value=visual_features             # [seq_len, batch_size, hidden_size]
            )
        
        # Transpose back to [batch_size, seq_len, hidden_size]
        attended_features = attended_features.transpose(0, 1)
        text_features = text_features.transpose(0, 1)
        
        # Concatenate and fuse text and attended visual features
        with record_function("feature_fusion"):
            combined = torch.cat([text_features, attended_features], dim=-1)  # [batch_size, seq_len, hidden_size*2]
            fused_features = self.feature_fusion(combined)  # [batch_size, seq_len, hidden_size]
        
        return fused_features