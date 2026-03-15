"""GRU controller for stateful routing state.

Maintains a small recurrent state h_t updated every W tokens (window);
group router takes [pool(x), h] as input so routing is temporally aware.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GRUController(nn.Module):
    """Small GRU that summarizes history over a token window for routing.

    State h has shape (batch, d_h). Updated every window_size tokens using
    pooled token features. d_h << d_model (e.g. 128--256).
    """

    def __init__(
        self,
        d_model: int,
        d_h: int,
        window_size: int = 32,
    ) -> None:
        super().__init__()
        self.d_h = d_h
        self.window_size = window_size
        self.gru = nn.GRUCell(d_model, d_h)

    def forward(
        self,
        pooled: Tensor,
        h_prev: Tensor | None = None,
    ) -> Tensor:
        """Update controller state from pooled token features.

        Args:
            pooled: (batch, d_model) pooled features (e.g. mean over window).
            h_prev: (batch, d_h) previous state; if None, zeros.

        Returns:
            h_new: (batch, d_h) new state.
        """
        batch = pooled.size(0)
        if h_prev is None:
            h_prev = pooled.new_zeros(batch, self.d_h)
        return self.gru(pooled, h_prev)

    def reset_state_for_batch(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Zero initial state for a given batch size (e.g. start of sequence)."""
        if dtype is None:
            dtype = next(self.gru.parameters()).dtype
        return torch.zeros(batch_size, self.d_h, device=device, dtype=dtype)
