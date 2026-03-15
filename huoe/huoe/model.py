"""H-UoE model: decoder-only with one or more macro-blocks (minimal prototype = single block)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from huoe.macro_block import MacroBlock


class HUoEModel(nn.Module):
    """Hierarchical Union-of-Experts: embedding + macro-block(s) + LM head.

    Minimal prototype: one macro-block, 4 groups, 4 experts per group,
    10% sparse experts, one GRU controller.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_h: int,
        num_heads: int,
        num_groups: int = 4,
        num_experts_per_group: int = 4,
        d_ff: int = 1024,
        expert_top_k: int = 2,
        group_top_k: int = 2,
        window_size: int = 32,
        expert_density: float = 0.1,
        num_layers: int = 1,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_h = d_h
        self.num_layers = num_layers
        self.window_size = window_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.embed_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MacroBlock(
                d_model=d_model,
                d_h=d_h,
                num_heads=num_heads,
                num_groups=num_groups,
                num_experts_per_group=num_experts_per_group,
                d_ff=d_ff,
                expert_top_k=expert_top_k,
                group_top_k=group_top_k,
                window_size=window_size,
                expert_density=expert_density,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Tie embedding and lm_head if same dim
        if d_model == vocab_size:
            self.lm_head.weight = self.embed.weight
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: Tensor,
        h_controllers: list[Tensor] | None = None,
        labels: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], dict]:
        """Forward pass.

        Args:
            input_ids: (batch, seq) token ids.
            h_controllers: list of (batch, d_h) per layer, or None (zero init).
            labels: (batch, seq) for loss; if None, no CE loss.

        Returns:
            logits: (batch, seq, vocab_size).
            h_list: list of updated controller states per layer.
            aux: dict with routing aux from last block (for load balance / stability).
        """
        B, S = input_ids.shape
        x = self.embed(input_ids) + self.pos_embed[:, :S]
        x = self.embed_drop(x)

        if h_controllers is None:
            h_controllers = [
                self.blocks[0].controller.reset_state_for_batch(B, input_ids.device, x.dtype)
                for _ in range(self.num_layers)
            ]
        h_list = []
        aux_all = {}
        for i, block in enumerate(self.blocks):
            x, h_new, aux = block(x, h_controllers[i], need_controller_update=True)
            h_list.append(h_new)
            aux_all = aux
        x = self.ln_f(x)
        logits = self.lm_head(self.drop(x))

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = nn.functional.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
            )
        return logits, h_list, {"loss": loss, **aux_all}

    @staticmethod
    def minimal_prototype(
        vocab_size: int = 50257,
        d_model: int = 512,
        d_h: int = 128,
        num_heads: int = 8,
        num_groups: int = 4,
        num_experts_per_group: int = 4,
        d_ff: int = 1024,
        expert_density: float = 0.1,
        window_size: int = 32,
        dropout: float = 0.0,
    ) -> "HUoEModel":
        """Single macro-block prototype: 4 groups, 4 experts/group, 10% sparse."""
        return HUoEModel(
            vocab_size=vocab_size,
            d_model=d_model,
            d_h=d_h,
            num_heads=num_heads,
            num_groups=num_groups,
            num_experts_per_group=num_experts_per_group,
            d_ff=d_ff,
            expert_top_k=2,
            group_top_k=2,
            window_size=window_size,
            expert_density=expert_density,
            num_layers=1,
            dropout=dropout,
        )
