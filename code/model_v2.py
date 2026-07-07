"""
model_v2.py -- Extension of model.py for Fase II (init scheme parameter).

Identical to model.py except:
  - Adds init_scheme parameter ('xavier' default | 'kaiming')
  - 'xavier' preserves backward-compat with all prior phases.

Used only by phase8.py (Fase II). Does not affect Phases 1-7.
"""

import torch
import torch.nn as nn
import math


class GrokkingTransformerV2(nn.Module):
    """Same architecture as GrokkingTransformer; configurable init scheme."""

    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=8, dropout=0.0, init_scheme='xavier'):
        super().__init__()
        self.d_model = d_model
        self.init_scheme = init_scheme

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            activation='gelu',
            dropout=dropout,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size)

        self._init_weights(init_scheme)

    def _init_weights(self, scheme):
        if scheme == 'xavier':
            nn.init.xavier_uniform_(self.embed.weight)
            nn.init.xavier_uniform_(self.pos_embed.weight)
            nn.init.xavier_uniform_(self.unembed.weight)
            nn.init.zeros_(self.unembed.bias)
        elif scheme == 'kaiming':
            # Kaiming uniform with fan_in mode and ReLU nonlinearity (gelu approximated as relu).
            nn.init.kaiming_uniform_(self.embed.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.pos_embed.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.unembed.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.unembed.bias)
        else:
            raise ValueError(f"Unknown init_scheme: {scheme!r}. Use 'xavier' or 'kaiming'.")

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.embed(x) * math.sqrt(self.d_model) + self.pos_embed(pos)
        h = self.encoder(h)
        h = self.ln_final(h)
        return self.unembed(h[:, -1, :])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
