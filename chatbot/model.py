"""
model.py — Arsitektur LSTM untuk prediksi harga CPO
"""

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM hidden states."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attn(lstm_out)              # (batch, seq_len, 1)
        weights = torch.softmax(scores, dim=1)    # normalized weights
        context = (weights * lstm_out).sum(dim=1) # (batch, hidden)
        return context, weights


class CPO_LSTM(nn.Module):
    """
    Stacked LSTM with optional bidirectional mode and attention mechanism.

    Architecture:
        Input → LSTM Stack → [Attention] → FC Head → Scalar prediction
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * self.num_directions
        self.attention = AttentionLayer(lstm_out_size) if use_attention else None

        # Fully-connected regression head
        self.fc = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Linear(lstm_out_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, seq_len, input_size)
        Returns:
            out: (batch_size, 1)  — predicted Close price (scaled)
            attn_weights: (batch_size, seq_len, 1) or None
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*dir)

        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
        else:
            context = lstm_out[:, -1, :]  # Last timestep
            attn_weights = None

        out = self.fc(context)
        return out, attn_weights
