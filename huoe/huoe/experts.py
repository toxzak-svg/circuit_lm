"""Intra-group experts: small sparse MLPs with top-k routing within a group."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from huoe.sparse_linear import SparseLinear


class SparseMLP(nn.Module):
    """Two-layer MLP with sparse linear layers (SET-style)."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        density: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = SparseLinear(d_model, d_ff, density=density)
        self.fc2 = SparseLinear(d_ff, d_model, density=density)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GroupExperts(nn.Module):
    """One group of experts (small MoE): top-k over E_g experts per token."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int = 2,
        density: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            SparseMLP(d_model, d_ff, density=density)
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(
        self,
        x: Tensor,
        group_weights: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward through top-k experts; optional group_weights for load balancing.

        Args:
            x: (batch, seq, d_model).
            group_weights: (batch, seq, num_experts) pre-softmax logits from router;
                          if None, compute gate from x.

        Returns:
            out: (batch, seq, d_model).
            aux: dict or tensor for load-balancing loss (e.g. gate probs).
        """
        batch, seq, d = x.shape
        if group_weights is None:
            gate_logits = self.gate(x)  # (B, S, E)
        else:
            gate_logits = group_weights

        gate_probs = F.softmax(gate_logits, dim=-1)
        top_probs, top_idx = torch.topk(gate_probs, self.top_k, dim=-1)  # (B, S, k)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        flat_x = x.view(-1, d)
        flat_top_idx = top_idx.view(-1, self.top_k)  # (B*S, k)
        flat_top_probs = top_probs.view(-1, self.top_k)

        out_flat = flat_x.new_zeros(flat_x.size(0), d)
        for k in range(self.top_k):
            expert_idx = flat_top_idx[:, k]  # (B*S,)
            prob_k = flat_top_probs[:, k : k + 1]  # (B*S, 1)
            for e in range(self.num_experts):
                sel = expert_idx == e
                if sel.any():
                    out_flat[sel] = out_flat[sel] + prob_k[sel] * self.experts[e](flat_x[sel])
        out = out_flat.view(batch, seq, d)
        return out, gate_probs
