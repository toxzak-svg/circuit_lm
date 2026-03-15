# Design: Hierarchical Union-of-Experts (H-UoE) – Refactor Direction

**Date:** 2026-03-08  
**Status:** Draft / Implementation in progress

## Summary

Refactor direction: a **multi-level Mixture-of-Experts transformer** where the hierarchy is over **state** and **time**, not just tokens, with learned structural sparsity in both routing and weights. This is implemented as a **separate neural stack** (PyTorch) in this repo, leaving the existing `circuit_lm` package unchanged (integer-only FSM/PDA/PPM + CP-SAT).

---

## Why a Separate Package

- **circuit_lm** is strictly zero-floats, no tensors, no torch/jax (enforced by CI). The H-UoE design is explicitly neural (GRU, sparse linear layers, SET-style rewire, GPU training).
- **Placement:** New top-level package `huoe/` (Hierarchical Union-of-Experts). Same repo, separate install: `pip install -e ./huoe` with `torch` dependency.
- **No refactor of circuit_lm:** FSM/PDA/PPM and CP-SAT training remain as-is. H-UoE is an alternative research direction for “sparse / hierarchical / trainable-at-scale” that can later inform or complement circuit models (e.g. routing semantics, stateful control).

---

## High-Level Concept: H-UoE

- **Base:** Decoder-only transformer; every N layers is a **macro-block** with:
  - Token stream (standard hidden states).
  - **Controller state:** small GRU (or low-dim latent) summarizing history over 128–512 tokens.
- **Inside a macro-block:**
  - Self-attention: dense (for now).
  - **MLP replaced by hierarchical experts:**
    - **Level 0 – Group router:** pooled token features + controller state → logits over 4–6 groups (e.g. general, reasoning, memory, code, dialog, fast-path); top-k groups (k=1–2).
    - **Level 1 – Intra-group experts:** each group has E_g experts (e.g. 8–32); standard top-k MoE within group.
    - **Shared path:** one small shared expert per macro-block always active.
- **Sparsity (threefold):**
  1. **Routing:** few groups, few experts per token.
  2. **Structural weight sparsity:** sparse linear layers inside experts with evolutionary rewire (SET-style).
  3. **State sparsity:** small recurrent controller state modulating routing, not full hidden state.

---

## Core Architecture (Concise)

| Component | Description |
|-----------|-------------|
| Macro-block | Self-attn (dense) + hierarchical MoE MLP; exposes token stream + controller state. |
| Group router | Input: pooled token + controller state. Output: group logits → top-k groups. |
| Intra-group experts | Per-group MoE (E_g experts, top-k); each expert = small MLP with **sparse linear** layers. |
| Shared expert | One small MLP per macro-block, always active. |
| GRU controller | State h_t, dim d_h ≪ d_model; updated every W tokens (e.g. 32–64); fed into group router. |
| Sparse linear (SET) | Fixed density (e.g. 5–10%); Erdős–Rényi init; periodically prune smallest \|w\|, add new edges biased by activations. |

---

## Training

- **Loss:** Next-token cross-entropy (+ optional multi-task heads).
- **Regularization:** Load balancing (groups + experts), group sparsity (L0 approx), temporal routing stability (KL between adjacent windows).
- **Stages:** (1) Warmup with fixed masks, soft routing; (2) Enable SET rewire every T steps, tighten routing; (3) Optional freeze connectivity, fine-tune.

---

## Minimal Prototype (2–3 Weeks)

1. **Single macro-block:** 4 groups, 4 experts per group; 10% sparse linear in experts with SET rewire; one GRU controller.
2. **Train** on small mixture (e.g. Pile subset + code + dialog); compare to dense baseline (same active params); visualize group/expert usage.
3. **Benchmarks:** Perplexity vs dense at equal FLOPs; task-switch (alternating tasks every 256 tokens); ablations (no controller, no SET, flat MoE).

Scale target after prototype: multi-block 300–700M active model for 2×RTX 5000 (2×32 GB).

---

## File Layout (Implementation)

- **Design / direction:** `docs/plans/2026-03-08-huoe-design-and-refactor-direction.md` (this file).
- **Neural package:** `huoe/` at repo root.
  - `huoe/huoe/controller.py` – GRU controller for routing state.
  - `huoe/huoe/sparse_linear.py` – SET-style sparse linear + rewire.
  - `huoe/huoe/experts.py` – Intra-group experts (sparse MLPs) + top-k routing.
  - `huoe/huoe/router.py` – Hierarchical group router (pooled x + h → groups).
  - `huoe/huoe/macro_block.py` – Self-attn + hierarchical MoE MLP + shared path.
  - `huoe/huoe/model.py` – Single macro-block H-UoE model (minimal prototype).
  - `huoe/huoe/train.py` – Training loop, stages, load balancing, stability loss.
  - `huoe/README.md` – Install, 2×RTX 5000 recipe, benchmarks.

---

## References (from proposal)

- Hierarchical MoE: [arxiv 2503.02495](https://arxiv.org/abs/2503.02495); [mlfrontiers.substack](https://mlfrontiers.substack.com/p/hierarchical-mixtures-of-experts).
- SET / evolutionary sparsity: [nature s41467-018-04316-3](https://www.nature.com/articles/s41467-018-04316-3); [discovery.ucl.ac.uk 10184037](https://discovery.ucl.ac.uk/10184037/1/wang23a.pdf).
- Shared expert, load balancing: [friendli MoE](https://friendli.ai/blog/moe-models-comparison); [arxiv 2507.11181](https://arxiv.org/html/2507.11181v1).
