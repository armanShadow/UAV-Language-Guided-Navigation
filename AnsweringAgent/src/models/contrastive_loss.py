import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

class ContrastiveLoss:
    """
    Contrastive learning loss implementations for language-guided navigation.
    
    Supports:
    - Triplet loss (with L2 or cosine distance)
    - InfoNCE/NT-Xent loss 
    - Supervised contrastive loss
    """
    
    def __init__(self, margin: float = 0.5, temperature: float = 0.07, loss_type: str = "triplet", 
                 use_cosine_distance: bool = False, mean_all: bool = False):
        """
        Initialize contrastive loss function.
        
        Args:
            margin: Margin for triplet loss
            temperature: Temperature for InfoNCE loss
            loss_type: Type of contrastive loss ("triplet", "infonce", or "supcon")
            use_cosine_distance: Use cosine distance instead of L2 for triplet loss
            mean_all: Use mean over all elements instead of mean over non-zero for triplet loss
        """
        self.margin = margin
        self.temperature = temperature
        self.loss_type = loss_type
        self.use_cosine_distance = use_cosine_distance
        self.mean_all = mean_all
        
    def __call__(self, 
                anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute contrastive loss based on specified type.
        
        Args:
            anchor_embeddings: Anchor embeddings [batch_size, hidden_size]
            positive_embeddings: Positive embeddings [batch_size, hidden_size]
            negative_embeddings: Negative embeddings [batch_size, hidden_size] or None for in-batch negatives
            mask: Optional mask to apply [batch_size]
            
        Returns:
            Calculated contrastive loss
        """
        if self.loss_type == "triplet":
            return self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        elif self.loss_type == "infonce":
            return self.infonce_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        elif self.loss_type == "supcon":
            return self.supervised_contrastive_loss(anchor_embeddings, positive_embeddings, mask)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
            
    def triplet_loss(self, 
                   anchor_embeddings: torch.Tensor,
                   positive_embeddings: torch.Tensor,
                   negative_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss with a margin.
        
        Args:
            anchor_embeddings: Anchor embeddings [batch_size, hidden_size]
            positive_embeddings: Positive embeddings [batch_size, hidden_size]
            negative_embeddings: Negative embeddings [batch_size, hidden_size]
            
        Returns:
            Triplet loss
        """
        # Validate input shapes
        if anchor_embeddings.shape != positive_embeddings.shape or anchor_embeddings.shape != negative_embeddings.shape:
            raise ValueError(f"Shape mismatch in triplet_loss: anchor={anchor_embeddings.shape}, "
                           f"positive={positive_embeddings.shape}, negative={negative_embeddings.shape}")
        
        # Normalize embeddings
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=-1)
        negative_norm = F.normalize(negative_embeddings, p=2, dim=-1)
        
        if self.use_cosine_distance:
            # Use cosine distance (1 - cosine similarity)
            pos_distance = 1 - F.cosine_similarity(anchor_norm, positive_norm, dim=-1)
            neg_distance = 1 - F.cosine_similarity(anchor_norm, negative_norm, dim=-1)
        else:
            # Use L2 distance
            pos_distance = torch.sum((anchor_norm - positive_norm) ** 2, dim=-1)
            neg_distance = torch.sum((anchor_norm - negative_norm) ** 2, dim=-1)
        
        # Compute triplet loss with margin
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
        
        # Return mean over non-zero elements or all elements based on flag
        if self.mean_all:
            return loss.mean()
        else:
            # Return mean over non-zero elements
            num_non_zero = torch.sum(loss > 0).float()
            if num_non_zero > 0:
                return torch.sum(loss) / num_non_zero
            else:
                return torch.sum(loss) * 0.0  # Return 0 if all elements are 0
            
    def infonce_loss(self,
                    anchor_embeddings: torch.Tensor,
                    positive_embeddings: torch.Tensor,
                    negative_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute InfoNCE/NT-Xent loss with temperature scaling.
        
        Args:
            anchor_embeddings: Anchor embeddings [batch_size, hidden_size]
            positive_embeddings: Positive embeddings [batch_size, hidden_size]
            negative_embeddings: Optional explicit negative embeddings [batch_size or k, hidden_size]
                If None, uses in-batch negatives approach
                
        Returns:
            InfoNCE loss
        """
        batch_size = anchor_embeddings.size(0)
        device = anchor_embeddings.device
        
        # Normalize embeddings
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=-1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=-1)
        
        # Compute similarity between anchors and positives
        pos_sim = torch.exp(torch.sum(anchor_norm * positive_norm, dim=-1) / self.temperature)
        
        if negative_embeddings is not None:
            # Explicit negative examples provided
            negative_norm = F.normalize(negative_embeddings, p=2, dim=-1)
            
            # Compute similarity between anchors and all negatives
            neg_sim = torch.exp(torch.matmul(anchor_norm, negative_norm.T) / self.temperature)
            
            # Sum over all negatives for each anchor
            neg_sum = neg_sim.sum(dim=1)
            
            # Compute InfoNCE loss
            loss = -torch.log(pos_sim / (pos_sim + neg_sum + 1e-12))
        else:
            # Use in-batch negatives approach
            # Compute similarity between all pairs in the batch
            sim_matrix = torch.matmul(anchor_norm, anchor_norm.T) / self.temperature
            
            # Create a mask to exclude self-similarity
            mask = torch.eye(batch_size, device=device)
            
            # Compute exponential of similarities
            exp_sim = torch.exp(sim_matrix) * (1 - mask)
            
            # Compute InfoNCE loss
            pos_sim = torch.exp(torch.sum(anchor_norm * positive_norm, dim=-1) / self.temperature)
            loss = -torch.log(pos_sim / (exp_sim.sum(dim=1) + pos_sim))
            
        return loss.mean()
        
    def supervised_contrastive_loss(self,
                                  embeddings: torch.Tensor,
                                  labels: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute supervised contrastive loss.
        
        Args:
            embeddings: Embeddings for all examples [batch_size, hidden_size]
            labels: Class labels for supervised grouping [batch_size]
            mask: Optional mask to apply [batch_size]
            
        Returns:
            Supervised contrastive loss
        """
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.contiguous().view(-1, 1)
        mask_pos = torch.eq(labels, labels.T).float()
        
        # Exclude self-contrast
        mask_self = torch.eye(embeddings.size(0), device=embeddings.device)
        mask_pos = mask_pos - mask_self
        
        # Apply optional mask if provided
        if mask is not None:
            mask = mask.float().unsqueeze(1)
            mask_pos = mask_pos * mask * mask.T
            
        # Compute log-sum-exp for numerical stability
        # First, remove self-similarity by setting it to a large negative value
        sim_matrix = sim_matrix - mask_self * 1e9
        
        # Compute log sum exp
        logsumexp = torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        
        # Compute loss only over positive pairs
        loss = (mask_pos * (logsumexp - sim_matrix)).sum(1) / mask_pos.sum(1).clamp(min=1.0)
        
        return loss.mean()

    def get_embeddings(self, model_outputs: Dict[str, torch.Tensor], 
                     positive_outputs: Optional[Dict[str, torch.Tensor]] = None,
                     negative_outputs: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, ...]:
        """
        Extract embeddings from model outputs.
        
        Args:
            model_outputs: Model outputs containing embeddings
            positive_outputs: Optional positive example outputs
            negative_outputs: Optional negative example outputs
            
        Returns:
            Tuple of extracted embeddings
        """
        # Get anchor embeddings (mean pooling of encoder hidden states)
        if "encoder_last_hidden_state" in model_outputs:
            # Use encoder hidden states
            hidden_states = model_outputs["encoder_last_hidden_state"]
            attention_mask = model_outputs.get("encoder_attention_mask")
            
            if attention_mask is not None:
                # Mean pooling with attention mask
                expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * expanded_mask.float(), dim=1)
                sum_mask = attention_mask.sum(1, keepdim=True).clamp(min=1e-9)
                anchor_embeddings = sum_embeddings / sum_mask
            else:
                # Simple mean pooling
                anchor_embeddings = hidden_states.mean(dim=1)
        else:
            # Fallback to other embedding types
            embedding_options = [
                "adapted_features", 
                "sentence_embedding",
                "pooled_output"
            ]
            
            for option in embedding_options:
                if option in model_outputs:
                    anchor_embeddings = model_outputs[option]
                    break
            else:
                raise ValueError("No suitable embeddings found in model outputs")
        
        # If no positive/negative embeddings provided, return just anchor embeddings
        if positive_outputs is None and negative_outputs is None:
            return anchor_embeddings
            
        # Get positive embeddings (using same approach as anchor)
        positive_embeddings = None
        if positive_outputs is not None:
            if "encoder_last_hidden_state" in positive_outputs:
                hidden_states = positive_outputs["encoder_last_hidden_state"]
                attention_mask = positive_outputs.get("encoder_attention_mask")
                
                if attention_mask is not None:
                    expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    sum_embeddings = torch.sum(hidden_states * expanded_mask.float(), dim=1)
                    sum_mask = attention_mask.sum(1, keepdim=True).clamp(min=1e-9)
                    positive_embeddings = sum_embeddings / sum_mask
                else:
                    positive_embeddings = hidden_states.mean(dim=1)
            else:
                for option in embedding_options:
                    if option in positive_outputs:
                        positive_embeddings = positive_outputs[option]
                        break
        
        # Get negative embeddings
        negative_embeddings = None
        if negative_outputs is not None:
            if "encoder_last_hidden_state" in negative_outputs:
                hidden_states = negative_outputs["encoder_last_hidden_state"]
                attention_mask = negative_outputs.get("encoder_attention_mask")
                
                if attention_mask is not None:
                    expanded_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    sum_embeddings = torch.sum(hidden_states * expanded_mask.float(), dim=1)
                    sum_mask = attention_mask.sum(1, keepdim=True).clamp(min=1e-9)
                    negative_embeddings = sum_embeddings / sum_mask
                else:
                    negative_embeddings = hidden_states.mean(dim=1)
            else:
                for option in embedding_options:
                    if option in negative_outputs:
                        negative_embeddings = negative_outputs[option]
                        break
        
        return anchor_embeddings, positive_embeddings, negative_embeddings 