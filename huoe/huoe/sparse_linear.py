"""SET-style sparse linear layer with evolutionary rewire.

Fixed density (e.g. 5--10%); Erdős--Rényi init; periodically prune
smallest-magnitude edges and add new edges biased by activations.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


def _erdos_renyi_mask(
    in_features: int,
    out_features: int,
    density: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Erdős--Rényi random mask: each edge present with probability density."""
    num_edges = max(1, int(in_features * out_features * density))
    # Sample num_edges distinct (i, j) indices.
    total = in_features * out_features
    if num_edges >= total:
        return torch.ones(out_features, in_features, device=device, dtype=torch.bool)
    perm = torch.randperm(total, device=device, generator=generator)[:num_edges]
    i = perm % in_features
    j = perm // in_features
    mask = torch.zeros(out_features, in_features, device=device, dtype=torch.bool)
    mask[j, i] = True
    return mask


class SparseLinear(nn.Module):
    """Linear layer with a fixed sparsity mask; weights only where mask is True."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.density = density
        self.register_buffer("mask", _erdos_renyi_mask(in_features, out_features, density, torch.device("cpu")))
        # Initialize only non-masked weights (mask will be moved with module).
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Fan-in for non-zero entries
        fan_in = max(1, int(self.in_features * self.density))
        bound = 1 / math.sqrt(fan_in)
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)
            self.weight.mul_(self.mask.float())
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        # (batch, in) @ (out, in).T -> (batch, out); only masked weights used.
        weight = self.weight * self.mask.float()
        return nn.functional.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, density={self.density}"


def rewire_sparse_layer(
    layer: SparseLinear,
    prune_frac: float = 0.2,
    activation_scores: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> None:
    """SET-style rewire: prune smallest |weight|, add new edges.

    Prune a fraction ζ of smallest-magnitude edges; add the same number
    in currently-zero positions. If activation_scores is provided (e.g.
    (in_features,) or (out_features,)), bias new in-positions by activity.

    Modifies layer.mask and layer.weight in-place.
    """
    mask = layer.mask
    weight = layer.weight
    device = weight.device
    in_f, out_f = layer.in_features, layer.out_features

    # Weights only on masked positions
    w_flat = weight[mask].detach()
    k_prune = max(1, int(w_flat.numel() * prune_frac))
    # Indices of smallest magnitude among active
    _, idx_small = torch.topk(w_flat.abs(), k_prune, largest=False)
    flat_indices = torch.where(mask.view(-1))[0]
    to_prune_flat = flat_indices[idx_small]

    # Deactivate pruned
    new_mask = mask.clone()
    for idx in to_prune_flat:
        new_mask.view(-1)[idx] = False
    # Zero the pruned weights so they stay small if mask is later re-added
    weight.view(-1)[to_prune_flat] = 0.0

    # Choose k_prune new positions among currently zero
    zero_flat = (new_mask.view(-1) == False).nonzero(as_tuple=True)[0]
    if zero_flat.numel() < k_prune:
        layer.mask.copy_(new_mask)
        return
    if activation_scores is not None:
        # activation_scores: prefer (in_f,) for in-dim or (out_f,) for out-dim
        if activation_scores.dim() == 1 and activation_scores.size(0) == in_f:
            # Score by in-feature activity: map flat index -> in_idx
            in_idx = zero_flat % in_f
            probs = (activation_scores[in_idx].float() + 1e-8).to(device)
        elif activation_scores.dim() == 1 and activation_scores.size(0) == out_f:
            out_idx = zero_flat // in_f
            probs = (activation_scores[out_idx].float() + 1e-8).to(device)
        else:
            probs = None
        if probs is not None:
            probs = probs / probs.sum()
            add_flat = torch.multinomial(probs, k_prune, replacement=False, generator=generator)
            to_add_flat = zero_flat[add_flat]
        else:
            perm = torch.randperm(zero_flat.numel(), device=device, generator=generator)[:k_prune]
            to_add_flat = zero_flat[perm]
    else:
        perm = torch.randperm(zero_flat.numel(), device=device, generator=generator)[:k_prune]
        to_add_flat = zero_flat[perm]

    new_mask.view(-1)[to_add_flat] = True
    # Init new weights small
    with torch.no_grad():
        fan_in = max(1, int(in_f * layer.density))
        bound = 1 / math.sqrt(fan_in)
        weight.view(-1)[to_add_flat] = torch.empty(
            k_prune, device=device, dtype=weight.dtype
        ).uniform_(-bound, bound)
    layer.mask.copy_(new_mask)
