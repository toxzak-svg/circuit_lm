"""Hierarchical group router: pooled token + controller state -> group logits, top-k groups."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HierarchicalGroupRouter(nn.Module):
    """Level 0: route to 1--2 groups from [pool(x), h_controller].

    Input: pooled token features (mean over window or routing token), controller state h.
    Output: group logits -> top-k groups, normalized.
    """

    def __init__(
        self,
        d_model: int,
        d_h: int,
        num_groups: int,
        top_k_groups: int = 2,
    ) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.top_k_groups = top_k_groups
        self.proj = nn.Linear(d_model + d_h, num_groups)

    def forward(
        self,
        pooled: Tensor,
        h: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute group logits and top-k group weights.

        Args:
            pooled: (batch, d_model) or (batch, seq_pool, d_model); we mean over seq_pool if present.
            h: (batch, d_h) controller state.

        Returns:
            group_logits: (batch, num_groups).
            group_weights: (batch, num_groups) softmax over top-k, zeros elsewhere.
        """
        if pooled.dim() == 3:
            pooled = pooled.mean(dim=1)
        inp = torch.cat([pooled, h], dim=-1)  # (B, d_model + d_h)
        group_logits = self.proj(inp)  # (B, num_groups)
        top_probs, top_idx = torch.topk(F.softmax(group_logits, dim=-1), self.top_k_groups, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
        group_weights = torch.zeros_like(group_logits)
        group_weights.scatter_(-1, top_idx, top_probs)
        return group_logits, group_weights
