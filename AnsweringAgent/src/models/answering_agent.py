import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5.modeling_t5 import BaseModelOutput
from models.feature_extractor import FeatureExtractor
from typing import Dict, Optional
from config import Config


class TemporalObservationEncoder(nn.Module):
    """Per-spatial-position temporal attention over current + previous views.

    Inputs and outputs preserve the spatial token grid (``[B, S, H]`` where
    ``S`` is the number of spatial tokens, e.g. 49 = 7x7). For each spatial
    position, the module attends across the temporal axis (current frame
    plus previous frames). A learnable per-timestep embedding distinguishes
    the current observation from its history.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_timesteps: int = 8,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_timesteps = max_timesteps

        # Index 0 is reserved for the current frame; 1..T-1 for history.
        self.time_embed = nn.Parameter(torch.zeros(max_timesteps, hidden_size))
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        current_tokens: torch.Tensor,
        prev_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_tokens: ``[B, S, H]`` spatial tokens for the current view.
            prev_tokens:    ``[B, T_prev, S, H]`` spatial tokens for previous
                            views. ``T_prev`` may be 0.

        Returns:
            Temporally contextualized current tokens ``[B, S, H]``.
        """
        B, S, H = current_tokens.shape
        T_prev = prev_tokens.shape[1] if prev_tokens.dim() == 4 else 0
        T = 1 + T_prev
        if T > self.max_timesteps:
            raise ValueError(
                f"Got {T} timesteps but only {self.max_timesteps} time embeddings."
            )

        if T_prev > 0:
            seq = torch.cat([current_tokens.unsqueeze(1), prev_tokens], dim=1)
        else:
            seq = current_tokens.unsqueeze(1)

        seq = seq + self.time_embed[:T].view(1, T, 1, H)
        # Treat each spatial position as its own short temporal sequence:
        # [B, T, S, H] -> [B, S, T, H] -> [B*S, T, H].
        seq = seq.transpose(1, 2).reshape(B * S, T, H)

        attn_out, _ = self.temporal_attn(seq, seq, seq)
        # Index 0 is the current-frame slot.
        attn_out = attn_out[:, 0].view(B, S, H)

        x = self.norm1(current_tokens + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class AnsweringAgent(nn.Module):
    """Vision-language answering agent for aerial UAV navigation.

    Pipeline:
      1. Visual: Darknet (frozen) -> 1x1 conv projection -> 49 spatial tokens
         ``[B, 49, H]`` (preserves the 7x7 spatial grid).
      2. Temporal: per-position temporal attention over current + previous views.
      3. Adapter: small visual adapter aligns visual tokens with the T5
         embedding distribution. Applied **separately** to the current view and
         the destination view so each produces its own post-adapter spatial
         tokens.
      4. Curriculum: optional convex mix between the *post-adapter* current and
         destination tokens to form the visual prefix consumed by T5.
      5. Joint encoding: visual tokens are concatenated with T5 text embeddings
         and passed through the T5 encoder, so vision and language mix in
         self-attention from the bottom up.
      6. Generation: the T5 decoder cross-attends to the joint encoder output.

    Auxiliary losses (``vl_align_*``, ``destination_*``) operate on the
    *pre-T5* post-adapter pooled representation of the **current view only**.
    This ensures their gradient flows directly into ``feature_extractor`` and
    ``t5_visual_adapter`` and cannot be satisfied by T5 self-attention
    routing text content through visual positions (a failure mode observed
    when these losses were sourced from post-T5 visual states).
    """

    def __init__(self, config: Config, tokenizer=None, logger=None):
        super().__init__()
        self.config = config
        self.logger = logger

        self.model_name = config.model.t5_model_name
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                model_max_length=self.config.data.max_seq_length,
                add_special_tokens=True,
            )

        self.t5_model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.t5_config = self.t5_model.config

        self.feature_extractor = FeatureExtractor(config)
        self.num_visual_tokens = self.feature_extractor.NUM_TOKENS  # 49

        self.temporal_encoder = TemporalObservationEncoder(
            hidden_size=config.model.hidden_size,
            num_heads=config.model.num_attention_heads,
            dropout=config.model.dropout,
            max_timesteps=max(8, config.data.max_previous_views + 2),
        )

        # Visual adapter: aligns visual tokens with T5's embedding distribution
        # before they are fed into the (mostly frozen) T5 encoder alongside text.
        self.t5_visual_adapter = nn.Sequential(
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.LayerNorm(config.model.hidden_size),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.LayerNorm(config.model.hidden_size),
        )

        # Paraphrase / contrastive heads (semantics-only, applied on text-portion mean).
        self.paraphrase_proj = nn.Linear(
            config.model.hidden_size, config.model.hidden_size
        )
        self.paraphrase_weight = nn.Parameter(torch.tensor(0.0))

        self.contrastive_proj = nn.Sequential(
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.ReLU(),
            nn.Linear(config.model.hidden_size, config.model.hidden_size),
            nn.LayerNorm(config.model.hidden_size),
        )

        # Vision-language alignment heads (CLIP-style starter signal).
        # ``vl_align_proj_v`` consumes the **pre-T5** pooled current-view
        # adapter output, and ``vl_align_proj_t`` consumes the **post-T5**
        # pooled text states. train.py contrasts them in-batch so the visual
        # pathway receives a direct, vision-required gradient that cannot be
        # satisfied by T5 routing text content through visual positions.
        self.vl_align_dim = 256
        self.vl_align_proj_v = nn.Linear(config.model.hidden_size, self.vl_align_dim)
        self.vl_align_proj_t = nn.Linear(config.model.hidden_size, self.vl_align_dim)

        self._init_adapter_weights()
        self._init_vl_align_weights()
        self._freeze_t5_parameters()

    def _init_adapter_weights(self):
        for module in self.t5_visual_adapter.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _init_vl_align_weights(self):
        # Slightly larger init than the adapter so the alignment signal has
        # non-trivial magnitude from the very first step.
        for proj in (self.vl_align_proj_v, self.vl_align_proj_t):
            nn.init.normal_(proj.weight, mean=0.0, std=0.05)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _freeze_t5_parameters(self):
        """Freeze T5 except the last few encoder/decoder blocks."""
        total_params = 0
        for _, param in self.t5_model.named_parameters():
            total_params += param.numel()
            param.requires_grad = False

        # Unfreeze the last 3 encoder blocks so they can learn to mix visual + text.
        for idx in [-1, -2, -3]:
            for _, param in self.t5_model.encoder.block[idx].named_parameters():
                param.requires_grad = True

        # Unfreeze the last 2 decoder blocks + final layer norm to adapt generation.
        for idx in [-1, -2]:
            for _, param in self.t5_model.decoder.block[idx].named_parameters():
                param.requires_grad = True
        for _, param in self.t5_model.decoder.final_layer_norm.named_parameters():
            param.requires_grad = True

        if self.logger is not None:
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            self.logger.info(f"Total trainable parameters: {trainable_params:,}")
            self.logger.info(f"Total T5 parameters:        {total_params:,}")
            self.logger.info(
                f"T5 model: {trainable_params/total_params*100:.2f}% of parameters are trainable"
            )
            encoder_trainable = sum(
                p.numel()
                for _, p in self.t5_model.encoder.named_parameters()
                if p.requires_grad
            )
            decoder_trainable = sum(
                p.numel()
                for _, p in self.t5_model.decoder.named_parameters()
                if p.requires_grad
            )
            other_trainable = trainable_params - encoder_trainable - decoder_trainable
            self.logger.info("Trainable breakdown:")
            self.logger.info(
                f"  - T5 encoder (last 3 blocks):              {encoder_trainable:,}"
            )
            self.logger.info(
                f"  - T5 decoder (last 2 blocks + final norm): {decoder_trainable:,}"
            )
            self.logger.info(
                f"  - Other modules (extractor, adapter, etc): {other_trainable:,}"
            )

    # ------------------------------------------------------------------
    # Visual processing
    # ------------------------------------------------------------------

    def _extract_visual_tokens(
        self,
        current_view: torch.Tensor,
        previous_views: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Extract temporally contextualized spatial tokens.

        Returns: ``[B, 49, hidden_size]``.
        """
        batch_size = current_view.size(0)
        S = self.num_visual_tokens
        H = self.config.model.hidden_size

        current_tokens = self.feature_extractor(current_view)  # [B, S, H]

        if (
            previous_views is not None
            and previous_views.dim() == 5
            and previous_views.size(1) > 0
        ):
            num_prev = min(
                previous_views.size(1), self.config.data.max_previous_views
            )
        else:
            num_prev = 0

        if num_prev > 0:
            prev = previous_views[:, :num_prev].contiguous()
            prev_flat = prev.view(batch_size * num_prev, *prev.shape[2:])
            prev_tokens = self.feature_extractor(prev_flat)  # [B*num_prev, S, H]
            prev_tokens = prev_tokens.view(batch_size, num_prev, S, H)
        else:
            prev_tokens = current_tokens.new_zeros(batch_size, 0, S, H)

        return self.temporal_encoder(current_tokens, prev_tokens)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        text_input: dict,
        current_view: torch.Tensor,
        previous_views: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        generate: bool = False,
        destination_view: Optional[torch.Tensor] = None,
        curriculum_ratio: float = 0.0,
        positive_input: Optional[dict] = None,
        positive_input_2: Optional[dict] = None,
        negative_input: Optional[dict] = None,
        negative_input_2: Optional[dict] = None,
        **generation_kwargs,
    ) -> Dict:
        """Forward pass.

        Args:
            text_input: dict with at least ``input_ids`` and ``attention_mask``
                (the unified dialog context). Extra keys such as
                ``first_instruction_input`` / ``current_question_input`` are
                accepted but ignored by this model.
            current_view: ``[B, 3, H, W]`` current observation.
            previous_views: ``[B, T_prev, 3, H, W]`` previous observations.
            labels: target answer token ids (training/eval).
            generate: if True, run T5 ``generate`` and return ``sequences``.
            destination_view: optional ``[B, 3, H, W]`` for curriculum learning.
            curriculum_ratio: in [0, 1]; weight on destination tokens.
            positive_input / negative_input(_2): paraphrase hints for
                contrastive learning.

        Returns:
            Dict with model outputs. See in-line documentation below.
        """
        device = current_view.device
        batch_size = current_view.size(0)

        # ---- Visual ----
        # Run the feature extractor + temporal encoder on the current view
        # (with history) and (optionally) the destination view, then apply
        # ``t5_visual_adapter`` to each *separately*. Pooling the current-view
        # adapter output yields ``current_pool`` (and the destination-view
        # output yields ``dest_pool``); these are the pre-T5 representations
        # used by the vl_align and destination cosine losses. Curriculum
        # mixing happens in *post-adapter* space, so the visual prefix
        # consumed by T5 is a convex combination of two fully-adapted views.
        current_raw = self._extract_visual_tokens(
            current_view, previous_views
        )                                                      # [B, S, H]
        current_adapted = self.t5_visual_adapter(current_raw)  # [B, S, H]
        current_pool = current_adapted.mean(dim=1)             # [B, H]

        dest_adapted = None
        dest_pool = None
        if destination_view is not None:
            dest_raw = self.feature_extractor(destination_view)  # [B, S, H]
            dest_adapted = self.t5_visual_adapter(dest_raw)      # [B, S, H]
            dest_pool = dest_adapted.mean(dim=1)                 # [B, H]

        if dest_adapted is not None and curriculum_ratio > 0:
            visual_tokens = (
                curriculum_ratio * dest_adapted
                + (1.0 - curriculum_ratio) * current_adapted
            )
        else:
            visual_tokens = current_adapted
        S = visual_tokens.size(1)

        # ---- Joint encoding (T5 encoder over visual + text) ----
        text_ids = text_input["input_ids"]
        text_mask = text_input["attention_mask"]
        text_embeds = self.t5_model.encoder.embed_tokens(text_ids)  # [B, T, H]

        visual_mask = torch.ones(
            batch_size, S, dtype=text_mask.dtype, device=device
        )
        joint_embeds = torch.cat([visual_tokens, text_embeds], dim=1)  # [B, S+T, H]
        joint_mask = torch.cat([visual_mask, text_mask], dim=1)        # [B, S+T]

        encoder_out = self.t5_model.encoder(
            inputs_embeds=joint_embeds,
            attention_mask=joint_mask,
            return_dict=True,
        )
        encoder_states = encoder_out.last_hidden_state  # [B, S+T, H]

        # ---- Generation mode ----
        if generate:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            default_gen_args = {
                "max_new_tokens": 32,
                "min_length": 5,
                "num_beams": 3,
                "do_sample": False,
                "repetition_penalty": 1.1,
                "length_penalty": 1.0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "early_stopping": True,
            }
            default_gen_args.update(generation_kwargs)

            generated_ids = self.t5_model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=encoder_states),
                attention_mask=joint_mask,
                **default_gen_args,
            )
            return {
                "sequences": generated_ids,
                "encoder_last_hidden_state": encoder_states,
                "encoder_attention_mask": joint_mask,
            }

        # ---- Training / validation mode ----
        t5_outputs = self.t5_model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_states),
            attention_mask=joint_mask,
            labels=labels,
            return_dict=True,
        )

        # Pool the entire joint sequence (visual + text) with the joint mask
        # for the shared anchor used by contrastive / KD losses. Pooling over
        # visual+text gives those losses a gradient path through the visual
        # pathway via T5 self-attention.
        joint_mask_f = joint_mask.float().unsqueeze(-1)          # [B, S+T, 1]
        joint_mean = (encoder_states * joint_mask_f).sum(dim=1) / joint_mask_f.sum(
            dim=1
        ).clamp(min=1.0)

        # Post-T5 pooled text (used as the text side of the vl_align loss).
        text_states = encoder_states[:, S:]                      # [B, T, H]
        text_mask_f = text_mask.float().unsqueeze(-1)            # [B, T, 1]
        text_mean_post = (text_states * text_mask_f).sum(dim=1) / text_mask_f.sum(
            dim=1
        ).clamp(min=1.0)

        outputs: Dict[str, torch.Tensor] = {
            "logits": t5_outputs.logits,
            "loss": t5_outputs.loss,
            "encoder_last_hidden_state": encoder_states,
            "encoder_attention_mask": joint_mask,
            "raw_adapted_features": joint_mean,
            "adapted_features": self.contrastive_proj(joint_mean),
            "feature_norm": encoder_states.norm(p=2, dim=-1).mean(),
            # Pre-T5, current-view-only pooled adapter output. Both the
            # destination cosine and the vl_align visual side read from this
            # tensor, so their gradient lands directly on
            # ``feature_extractor`` + ``t5_visual_adapter`` and cannot be
            # absorbed by T5's encoder self-attention.
            "destination_anchor": current_pool,
            "vl_align_visual": F.normalize(
                self.vl_align_proj_v(current_pool), p=2, dim=-1
            ),
            "vl_align_text": F.normalize(
                self.vl_align_proj_t(text_mean_post), p=2, dim=-1
            ),
        }

        if dest_pool is not None:
            # Pre-T5 destination pool: matches ``destination_anchor`` in space
            # (both are post-adapter, mean-over-spatial-positions), so the
            # cosine loss measures a single coherent quantity.
            outputs["destination_features"] = dest_pool

        # Contrastive paraphrase combinations: shared anchor (joint_mean)
        # shifted by a (sigmoid-gated) paraphrase-hint encoding.
        p_weight = torch.sigmoid(self.paraphrase_weight)
        for hint_input, key_name in [
            (positive_input, "positive_adapted_features"),
            (positive_input_2, "positive_adapted_features_2"),
            (negative_input, "negative_adapted_features"),
            (negative_input_2, "negative_adapted_features_2"),
        ]:
            if hint_input is None:
                continue
            hint_out = self.t5_model.encoder(
                input_ids=hint_input["input_ids"].to(device),
                attention_mask=hint_input["attention_mask"].to(device),
                return_dict=True,
            )
            hint_features = self.paraphrase_proj(
                hint_out.last_hidden_state.mean(dim=1)
            )
            combined = joint_mean + p_weight * hint_features
            outputs[key_name] = self.contrastive_proj(combined)

        return outputs

    def generate_answer(
        self,
        text_input: dict,
        current_view: torch.Tensor,
        previous_views: torch.Tensor,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Convenience wrapper for inference."""
        with torch.no_grad():
            outputs = self.forward(
                text_input,
                current_view,
                previous_views,
                generate=True,
                **generation_kwargs,
            )
        return outputs["sequences"]
