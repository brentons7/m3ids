import torch
import torch.nn as nn

from .sequence import SequenceDetector


class CausalTransformer(nn.Module):
    """Pre-norm Transformer encoder with learned positions and a causal mask."""

    def __init__(self, seq_len, d_model, n_layers, n_heads, ff_mult, dropout):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=ff_mult * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(seq_len), persistent=False)

    def forward(self, h):
        return self.layers(h + self.pos, mask=self.mask, is_causal=True)


class TransformerDetector(SequenceDetector):
    default_hparams = {
        **SequenceDetector.default_hparams,
        "n_heads": 4,
        "ff_mult": 2,   # feed-forward width = ff_mult * d_model
    }

    def build_backbone(self) -> nn.Module:
        hp = self.hp
        return CausalTransformer(hp["seq_len"], hp["d_model"], hp["n_layers"], hp["n_heads"], hp["ff_mult"], hp["dropout"])
