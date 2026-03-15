"""Macro-block: self-attention (dense) + hierarchical MoE MLP with shared expert and controller."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from huoe.controller import GRUController
from huoe.router import HierarchicalGroupRouter
from huoe.experts import GroupExperts
from huoe.sparse_linear import SparseLinear


class MacroBlock(nn.Module):
    """One macro-block: dense self-attn + hierarchical experts (group router -> groups) + shared expert.

    Controller state h is updated every window_size tokens from pooled hidden states;
    group router uses [pool(x), h]. Each group has its own GroupExperts (sparse MLPs);
    one shared expert is always applied.
    """

    def __init__(
        self,
        d_model: int,
        d_h: int,
        num_heads: int,
        num_groups: int,
        num_experts_per_group: int,
        d_ff: int,
        expert_top_k: int = 2,
        group_top_k: int = 2,
        window_size: int = 32,
        expert_density: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.num_groups = num_groups

        # Self-attention (dense)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        # Controller for routing state
        self.controller = GRUController(d_model, d_h, window_size)

        # Hierarchical router: [pool(x), h] -> group logits, top-k groups
        self.router = HierarchicalGroupRouter(d_model, d_h, num_groups, group_top_k)

        # One expert group per group type
        self.group_experts = nn.ModuleList([
            GroupExperts(d_model, d_ff, num_experts_per_group, expert_top_k, expert_density)
            for _ in range(num_groups)
        ])

        # Shared expert (always active, small)
        shared_ff = max(d_model, d_ff // 2)
        self.shared_expert = nn.Sequential(
            nn.Linear(d_model, shared_ff),
            nn.GELU(),
            nn.Linear(shared_ff, d_model),
        )
        self.shared_scale = 0.5  # mix with routed output

        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def _pool_over_windows(self, x: Tensor) -> Tensor:
        """Pool (batch, seq, d_model) to (batch, num_windows, d_model) by mean over window_size."""
        B, S, D = x.shape
        if S <= self.window_size:
            return x.mean(dim=1, keepdim=True)
        pad = (self.window_size - S % self.window_size) % self.window_size
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        S = x.size(1)
        x = x.view(B, S // self.window_size, self.window_size, D)
        return x.mean(dim=2)  # (B, num_windows, D)

    def forward(
        self,
        x: Tensor,
        h_controller: Tensor | None = None,
        need_controller_update: bool = True,
    ) -> tuple[Tensor, Tensor, dict]:
        """One macro-block forward.

        Args:
            x: (batch, seq, d_model).
            h_controller: (batch, d_h) or None (use zeros).
            need_controller_update: if True, return updated h for next step.

        Returns:
            out: (batch, seq, d_model).
            h_new: (batch, d_h) updated controller state.
            aux: dict with group_weights, gate_probs per group, for load-balance / stability loss.
        """
        B, S, D = x.shape
        # Self-attention
        residual = x
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.attn_drop(attn_out)

        # Pool for router (and controller): use current x
        pooled = self._pool_over_windows(x)  # (B, num_windows, D)
        pooled_flat = pooled.mean(dim=1)  # (B, D) for router

        if h_controller is None:
            h_controller = self.controller.reset_state_for_batch(B, x.device, x.dtype)

        # Update controller
        if need_controller_update:
            h_new = self.controller(pooled_flat, h_controller)
        else:
            h_new = h_controller

        # Group router
        group_logits, group_weights = self.router(pooled_flat, h_new)  # (B, num_groups)
        group_weights_exp = group_weights.unsqueeze(1)  # (B, 1, num_groups)

        # Run each group's experts; combine by group_weights (per-batch, same for all tokens in seq)
        expert_out = x.new_zeros(B, S, D)
        gate_probs_list = []
        for g in range(self.num_groups):
            out_g, gate_probs_g = self.group_experts[g](x)
            expert_out = expert_out + group_weights_exp[:, :, g : g + 1] * out_g
            gate_probs_list.append(gate_probs_g)

        # Shared expert
        shared_out = self.shared_expert(x)
        expert_out = expert_out + self.shared_scale * shared_out

        residual = x
        out = residual + self.drop(self.ln2(expert_out))

        aux = {
            "group_logits": group_logits,
            "group_weights": group_weights,
            "gate_probs_per_group": gate_probs_list,
        }
        return out, h_new, aux
