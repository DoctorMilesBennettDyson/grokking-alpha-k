"""
Grokking Transformer -- Model definition.
Phase-Transition Framework Falsification Test (Paper 04, S6.3).

Small transformer for modular arithmetic tasks with configurable depth
(composition of operations) and noise injection (dropout, label smoothing).
"""

import torch
import torch.nn as nn
import math


class GrokkingTransformer(nn.Module):
    """
    Minimal transformer for modular arithmetic grokking experiments.
    
    Architecture: token embedding + positional embedding → TransformerEncoder → linear unembed.
    Predicts the result token from the last position of the input sequence.
    """
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2,
                 d_ff=512, max_seq_len=8, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            activation='gelu',
            dropout=dropout,
            norm_first=True  # Pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization for embeddings and linear layers."""
        nn.init.xavier_uniform_(self.embed.weight)
        nn.init.xavier_uniform_(self.pos_embed.weight)
        nn.init.xavier_uniform_(self.unembed.weight)
        nn.init.zeros_(self.unembed.bias)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) tensor of token indices
        Returns:
            logits: (batch, vocab_size) predictions from last position
        """
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.embed(x) * math.sqrt(self.d_model) + self.pos_embed(pos)
        h = self.encoder(h)
        h = self.ln_final(h)
        return self.unembed(h[:, -1, :])  # predict from last position


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
