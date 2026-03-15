# H-UoE: Hierarchical Union-of-Experts

Stateful, sparse Mixture-of-Experts transformer with **hierarchy over state and time**: group routing conditioned on a GRU controller, SET-style evolutionary sparse weights in experts, and a shared expert path per macro-block.

Lives in the same repo as **circuit_lm** (integer FSM/PDA); this package is PyTorch-based and separate so `circuit_lm` stays zero-floats / no tensors.

## Install

From repo root:

```bash
pip install -e ./huoe
```

Requires `torch>=2.0`.

## Minimal prototype (single macro-block)

- **4 groups**, **4 experts per group**, **10% sparse** linear layers in experts (SET-style rewire).
- One **GRU controller** (d_h=128) for stateful group routing; updated every 32 tokens.
- **Shared expert** always active per block.

```python
from huoe import HUoEModel

model = HUoEModel.minimal_prototype(
    vocab_size=50257,
    d_model=512,
    d_h=128,
    num_groups=4,
    num_experts_per_group=4,
    expert_density=0.1,
    window_size=32,
)
```

## Training (2×RTX 5000 recipe)

- **Parameter budget:** Dense-equivalent ~1–2B, active per token ~5–10% (routing + sparse experts). Start with minimal prototype (~50–100M) to validate.
- **Parallelism:** Data-parallel across 2 GPUs (single-node); experts can share GPUs via token dispatch (extend with all-to-all later).
- **Stages:**
  1. **Stage 1:** Warmup without evolutionary rewiring (fixed masks), softer routing (higher temperature).
  2. **Stage 2:** SET rewire every T steps (e.g. 100); tighten routing; load balancing + stability losses.
  3. **Stage 3 (optional):** Freeze connectivity, fine-tune on target domains.

Run minimal training:

```bash
# Dummy data (default)
python -m huoe.scripts.train_minimal --epochs 3 --output-dir ./huoe_out

# Real data: pre-tokenized tokens in --data-dir (see Data format below)
python -m huoe.scripts.train_minimal --data-dir ./data --seq-len 1024 --batch-size 16 --epochs 3 --output-dir ./huoe_out

# Multi-GPU (2×RTX 5000)
torchrun --nproc_per_node=2 -m huoe.scripts.train_minimal --data-dir ./data --seq-len 1024 --batch-size 16 --output-dir ./huoe_out
```

Suggested for 2 GPUs: seq len 1–2k, batch size so both GPUs are utilized.

### Data format (real data)

- **Single file:** `data_dir/tokens.npy` with shape `(num_seqs, seq_len)` or `(num_seqs, L)` (sliced to `--seq-len`). Or 1D array of tokens, reshaped into `(N, seq_len)`.
- **Multiple files:** `data_dir/*.npy` or `data_dir/*.bin` — one file per sequence; each sequence truncated/padded to `--seq-len`. Use vocab tokenizer that matches your data (e.g. GPT-2 50257).

## Benchmarks

Run with a trained checkpoint:

```bash
# Perplexity on pre-tokenized data
python -m huoe.scripts.run_benchmark perplexity --checkpoint ./huoe_out/model.pt --data-dir ./data --seq-len 512 --output ./huoe_out/perplexity.json

# Task-switch: synthetic alternating segments (loss per segment)
python -m huoe.scripts.run_benchmark task_switch --checkpoint ./huoe_out/model.pt --segment-len 256 --num-segments 8 --output ./huoe_out/task_switch.json

# Ablation: no controller (zero state throughout)
python -m huoe.scripts.run_benchmark perplexity --checkpoint ./huoe_out/model.pt --data-dir ./data --ablation no_controller
```

- **Perplexity:** Compare to dense baseline at equal FLOPs (same `d_model`, no MoE).
- **Task-switch:** Alternating segments every 256 tokens; see `segment_losses` and even/odd means in output JSON.
- **Ablations:** `no_controller` wired; future: no SET (dense experts), flat MoE (no hierarchy) via model variants.

## Design doc

Full architecture and refactor direction: [docs/plans/2026-03-08-huoe-design-and-refactor-direction.md](../docs/plans/2026-03-08-huoe-design-and-refactor-direction.md) (in repo root).
