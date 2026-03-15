"""Training loop for H-UoE: stages, load balancing, stability loss, SET rewire."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from huoe.model import HUoEModel
from huoe.sparse_linear import SparseLinear, rewire_sparse_layer


# ---------------------------------------------------------------------------
# Routing regularizers
# ---------------------------------------------------------------------------


def load_balance_loss_group(
    group_weights: Tensor,
    num_groups: int,
) -> Tensor:
    """Encourage uniform usage of groups over batch (entropy bonus / auxiliary loss).

    group_weights: (batch, num_groups). Mean over batch gives (num_groups,);
    we want high entropy -> uniform.
    """
    usage = group_weights.mean(dim=0)  # (num_groups,)
    # Auxiliary: encourage usage to be uniform. Loss = negative entropy.
    entropy = -(usage * (usage + 1e-8).log()).sum()
    return -entropy  # minimize -entropy = maximize entropy


def load_balance_loss_experts(
    gate_probs_per_group: list[Tensor],
    num_experts_per_group: int,
) -> Tensor:
    """Per-group expert usage: encourage uniform over experts."""
    total = 0.0
    for gate_probs in gate_probs_per_group:
        # gate_probs: (B, S, E)
        usage = gate_probs.mean(dim=(0, 1))  # (E,)
        entropy = -(usage * (usage + 1e-8).log()).sum()
        total = total - entropy
    return total / max(1, len(gate_probs_per_group))


def group_sparsity_loss(group_logits: Tensor, l0_approx_scale: float = 0.01) -> Tensor:
    """L0 approximation: L1 on gate logits to penalize many active groups."""
    return l0_approx_scale * group_logits.abs().mean()


def stability_loss(
    group_weights_prev: Tensor | None,
    group_weights_curr: Tensor,
) -> Tensor:
    """Temporal consistency: discourage rapid oscillation of group assignments.

    KL(prev || curr) or symmetric; if prev is None, return 0.
    """
    if group_weights_prev is None:
        return group_weights_curr.new_tensor(0.0)
    # (B, G) each; add small eps for log
    eps = 1e-8
    p = group_weights_prev + eps
    q = group_weights_curr + eps
    kl = (p * (p.log() - q.log())).sum(dim=-1).mean()
    return kl


# ---------------------------------------------------------------------------
# SET rewire (call every T steps)
# ---------------------------------------------------------------------------


def collect_sparse_layers(module: nn.Module) -> list[SparseLinear]:
    out: list[SparseLinear] = []
    for m in module.modules():
        if isinstance(m, SparseLinear):
            out.append(m)
    return out


def run_rewire_step(
    model: HUoEModel,
    prune_frac: float = 0.2,
    activation_scores: dict[str, Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> None:
    """One evolutionary rewire step on all SparseLinear layers in model."""
    layers = collect_sparse_layers(model)
    for layer in layers:
        rewire_sparse_layer(layer, prune_frac=prune_frac, activation_scores=None, generator=generator)


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------


class StageConfig:
    """Per-stage config: rewire on/off, router temperature, load-balance weight, etc."""

    def __init__(
        self,
        rewire_every_steps: int = 0,
        rewire_prune_frac: float = 0.2,
        load_balance_group_weight: float = 0.01,
        load_balance_expert_weight: float = 0.01,
        group_sparsity_weight: float = 0.001,
        stability_weight: float = 0.01,
        router_temperature: float = 1.0,
    ) -> None:
        self.rewire_every_steps = rewire_every_steps
        self.rewire_prune_frac = rewire_prune_frac
        self.load_balance_group_weight = load_balance_group_weight
        self.load_balance_expert_weight = load_balance_expert_weight
        self.group_sparsity_weight = group_sparsity_weight
        self.stability_weight = stability_weight
        self.router_temperature = router_temperature


def train_step(
    model: HUoEModel,
    batch: dict[str, Tensor],
    stage: StageConfig,
    step: int,
    h_controllers: list[Tensor] | None,
    group_weights_prev: Tensor | None,
    device: torch.device,
) -> tuple[Tensor, list[Tensor], Tensor | None, dict[str, float]]:
    """One training step; returns loss, new h_controllers, group_weights for next stability, metrics."""
    model.train()
    input_ids = batch["input_ids"].to(device)
    labels = batch.get("labels", input_ids.clone())
    labels = labels.to(device)
    if h_controllers is not None:
        h_controllers = [h.to(device) for h in h_controllers]

    logits, h_list, aux = model(input_ids, h_controllers, labels)
    loss = aux["loss"]
    if loss is None:
        loss = model(input_ids, h_controllers, labels)[2]["loss"]

    # Routing regularizers
    group_weights = aux.get("group_weights")
    group_logits = aux.get("group_logits")
    gate_probs_per_group = aux.get("gate_probs_per_group", [])

    if group_weights is not None and stage.load_balance_group_weight != 0:
        lb_g = load_balance_loss_group(group_weights, group_weights.size(-1))
        loss = loss + stage.load_balance_group_weight * lb_g
    if gate_probs_per_group and stage.load_balance_expert_weight != 0:
        num_e = gate_probs_per_group[0].size(-1)
        lb_e = load_balance_loss_experts(gate_probs_per_group, num_e)
        loss = loss + stage.load_balance_expert_weight * lb_e
    if group_logits is not None and stage.group_sparsity_weight != 0:
        loss = loss + group_sparsity_loss(group_logits) * stage.group_sparsity_weight
    if group_weights_prev is not None and group_weights is not None and stage.stability_weight != 0:
        loss = loss + stage.stability_weight * stability_loss(group_weights_prev, group_weights)

    metrics = {"loss": loss.item()}
    if aux.get("loss") is not None:
        metrics["ce_loss"] = aux["loss"].item()
    return loss, h_list, group_weights.detach() if group_weights is not None else None, metrics


def train_epoch(
    model: HUoEModel,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    stage: StageConfig,
    device: torch.device,
    step_offset: int = 0,
    rewire_callback: Callable[[HUoEModel, float], None] | None = None,
) -> dict[str, float]:
    """One epoch; optionally run rewire every N steps via rewire_callback (e.g. run_rewire_step)."""
    model.train()
    agg: dict[str, list[float]] = {}
    h_controllers: list[Tensor] | None = None
    group_weights_prev: Tensor | None = None
    global_step = step_offset
    for batch in dataloader:
        loss, h_controllers, group_weights_prev, metrics = train_step(
            model, batch, stage, global_step, h_controllers, group_weights_prev, device
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # Detach so next batch does not backprop through this batch's controller state
        if h_controllers is not None:
            h_controllers = [h.detach() for h in h_controllers]
        for k, v in metrics.items():
            agg.setdefault(k, []).append(v)
        global_step += 1
        if (
            stage.rewire_every_steps > 0
            and global_step % stage.rewire_every_steps == 0
            and rewire_callback is not None
        ):
            rewire_callback(model, stage.rewire_prune_frac)
    return {k: sum(v) / len(v) for k, v in agg.items()}
