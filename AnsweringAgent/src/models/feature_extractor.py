import torch
import torch.nn as nn
import torch.nn.functional as F
from models.darknet import Darknet
from config import Config


class FeatureExtractor(nn.Module):
    """Spatial visual feature extractor.

    Uses a frozen Darknet backbone and projects its ``[B, 512, 7, 7]`` feature
    map into a sequence of spatial tokens ``[B, 49, hidden_size]`` suitable for
    cross-modal attention with language tokens.

    Unlike the original AVDN-style extractor (which globally pooled the feature
    map into a single vector), this module preserves the 7x7 spatial grid so
    that downstream modules can perform fine-grained vision-language grounding.
    """

    # Darknet at img_size=224 produces a 7x7 feature map with 512 channels.
    SPATIAL_SIZE = 7
    NUM_TOKENS = SPATIAL_SIZE * SPATIAL_SIZE  # 49
    DARKNET_CHANNELS = 512

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.model.hidden_size
        self.input_size = config.model.img_size

        self._init_darknet(config)
        self._init_projection()
        self._init_position_embedding()
        self._verify_output_dimensions()

    def _init_darknet(self, config: Config):
        """Initialize the (frozen) Darknet backbone with pretrained weights."""
        self.darknet = Darknet(config)

        # Load weights on CPU first to avoid OOM in distributed init.
        new_state = torch.load(config.data.darknet_weights_path, map_location='cpu')
        state = self.darknet.state_dict()
        model_keys = set(state.keys())
        state_dict = {k: v for k, v in new_state['model'].items() if k in model_keys}
        state.update(state_dict)
        self.darknet.load_state_dict(state)

        for param in self.darknet.parameters():
            param.requires_grad = False

    def _init_projection(self):
        """1x1 conv projection from Darknet channels to hidden_size.

        We keep the 7x7 spatial grid intact: each of the 49 output tokens
        corresponds to a distinct ~32x32 image region (224/7).
        """
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(self.DARKNET_CHANNELS, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.hidden_size, kernel_size=1),
            nn.BatchNorm2d(self.hidden_size),
            nn.ReLU(inplace=True),
        )
        self.token_norm = nn.LayerNorm(self.hidden_size)

        for m in self.spatial_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_position_embedding(self):
        """Learned 2D positional embedding for the 49 spatial tokens."""
        self.pos_embed = nn.Parameter(torch.zeros(1, self.NUM_TOKENS, self.hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _verify_output_dimensions(self):
        dummy = torch.randn(1, 3, self.input_size, self.input_size)
        with torch.no_grad():
            out = self._forward_impl(dummy)
        assert out.shape == (1, self.NUM_TOKENS, self.hidden_size), (
            f"Expected (1, {self.NUM_TOKENS}, {self.hidden_size}), "
            f"got {tuple(out.shape)}"
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_size or x.size(-2) != self.input_size:
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=True,
            )

        feats = self.darknet(x)  # expected: [B, 512, 7, 7]

        # Defensive: some darknet configurations drop the batch dim for B==1.
        if feats.dim() == 3:
            feats = feats.unsqueeze(0)
        feats = feats.view(
            feats.size(0),
            self.DARKNET_CHANNELS,
            self.SPATIAL_SIZE,
            self.SPATIAL_SIZE,
        )

        feats = self.spatial_proj(feats)               # [B, hidden, 7, 7]
        tokens = feats.flatten(2).transpose(1, 2)      # [B, 49, hidden]
        tokens = tokens + self.pos_embed
        tokens = self.token_norm(tokens)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial tokens from a batch of images.

        Args:
            x: image tensor of shape ``[B, 3, H, W]``. Callers handling
               temporal stacks (e.g. ``[B, T, 3, H, W]``) must flatten the
               leading dims before calling and reshape the result afterwards.

        Returns:
            Spatial tokens of shape ``[B, 49, hidden_size]``.
        """
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        tokens = self._forward_impl(x)

        if torch.isnan(tokens).any():
            tokens = torch.nan_to_num(tokens, nan=0.0)

        return tokens
