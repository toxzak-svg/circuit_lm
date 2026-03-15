# HGSEL: Hash-based Gradient-guided Sparse Expert Layer

A deterministic, production-grade Sparse Mixture of Experts (MoE) architecture for dense Transformers.

## Overview

HGSEL replaces each Transformer MLP block with a bank of tiny experts routed deterministically via hashing:

```
Input Token
  |
  +---> [Quantize → Sign + Magnitude Bucketing]
         |
         +---> [Multi-Hash (H=4) → Routing Keys]
                |
                +---> [Route to k=2 Active Experts from N=64 total]
                      |
                      +---> [Expert FFN Forward]
                            |
                            +---> [Combine Outputs via Learned Weights]
                                  |
                                  +---> Output
```

## Design Philosophy

**Deterministic Routing:**
- No learned router network (eliminates training instability, simplifies compilation)
- Multi-hash function provides load balancing without all-to-all communication
- Salt-based tuning enables predictable expert specialization

**Scalable:**
- 300M baseline: Proof-of-concept on single GPU
- 1T+ production: Two-tier experts (local + global) with shared residual expert for stability

**Production-Ready:**
- Routing compiler enables cache-hot inference
- Block-level packing for GPU kernel fusion
- Comprehensive monitoring and introspection

## Project Phases

| Phase | Duration | Focus | Deliverables |
|-------|----------|-------|--------------|
| 1 | Days 1-2 | Foundation | Project structure, routing engine |
| 2 | Days 2-3 | Single-GPU Core | Expert bank, layer integration, training loop |
| 3 | Days 4-5 | Training | Load balancing, utilization loss, salt tuning |
| 4 | Days 5-6 | Multi-GPU | All-reduce, expert sharding |
| 5 | Days 6-7 | Benchmarking | 300M inference, throughput, quality |
| 6–9 | Days 7–14 | 1T Scale | Two-tier experts, router state, hierarchical dispatch, advanced inference |

For detailed plan, see [HGSEL_BUILD_PLAN.md](./HGSEL_BUILD_PLAN.md).

## Project Status & Documentation

**Current Phase:** Phase 3 ✓ Complete | Phase 4 Ready

Phase completion summaries:
- [Phase 1 Completion](./PHASE1_COMPLETION.md): Routing engine & architecture
- [Phase 2 Completion](./PHASE2_COMPLETION.md): Single-GPU training infrastructure
- [Phase 3 Completion](./PHASE3_COMPLETION.md): Load balancing & salt tuning
- [Phase 4 Planning](./PHASE4_PLANNING.md): Multi-GPU distribution roadmap
- [Project Status](./PROJECT_STATUS.md): Overall progress & next actions

## Quick Start

```python
import torch
from hgsel.layer import HGSELLayer

# Create a HGSEL layer to replace MLP in Transformer
layer = HGSELLayer(
    d_ff=2048,
    n_experts=64,
    k_active=2,
    hidden_dim=512
)

# Forward pass
tokens = torch.randn(batch_size=32, seq_len=128, d_model=512)
output = layer(tokens)
```

## Directory Structure

```
hgsel-moe/
├── hgsel/                          # Main package
│   ├── routing/                    # Hash-based routing engine
│   │   ├── hash_functions.py       # Multi-hash router
│   │   ├── router_state.py         # 1T: EMA + quantization (Phase 6)
│   │   └── novelty_detection.py    # 1T: Gate Tier B (Phase 6)
│   ├── expert/                     # Expert bank
│   │   ├── expert_bank.py          # Sparse dispatch
│   │   └── two_tier_experts.py     # 1T: Local + Global tiers (Phase 6)
│   ├── layer/                      # HGSEL layer
│   │   ├── hgsel_layer.py          # Main MLP replacement
│   │   ├── combine_weights.py      # Output combination strategies
│   │   ├── shared_expert.py        # 1T: Always-on residual (Phase 6)
│   │   └── load_balancer.py        # Salt-based load tuning
│   ├── training/                   # Training utilities
│   │   ├── losses.py               # Auxiliary loss functions
│   │   └── salt_optimizer.py       # Hill-climb salt tuning
│   ├── distributed/                # Multi-GPU support
│   │   ├── sharding.py             # Expert placement
│   │   └── hierarchical_dispatch.py # 1T: Local→Global routing (Phase 7)
│   └── inference/                  # Production inference
│       ├── routing_cache.py        # 1T: Compiled route cache (Phase 8)
│       ├── block_packing.py        # Token block grouping (Phase 8)
│       └── prefetch_scheduler.py   # Expert prefetching (Phase 8)
├── experiments/                    # Run configs and baselines
│   ├── configs/                    # YAML experiment configs
│   └── baselines/                  # Dense Transformer baseline, vanilla MoE
├── tests/                          # Unit and integration tests
├── notebooks/                      # Analysis and visualization
├── results/                        # Experiment outputs
└── HGSEL_BUILD_PLAN.md            # Detailed implementation roadmap
```

## Key Papers & Related Work

- **Mixture of Experts (MoE):** Shazeer et al. (2017), GShard, Switch Transformers
- **Deterministic Routing:** Hash-based and power-of-two choices
- **Load Balancing:** EMA-based expert utilization (avoiding communication overhead)
- **Production Inference:** Ansor, TVM compiler for expert dispatch

## Roadmap

### Phase 1–5 (300M Baseline)
- [x] Build plan and architecture
- [ ] Implement routing engine
- [ ] Implement expert bank
- [ ] Implement HGSEL layer
- [ ] Training loop and evaluation

### Phase 6–9 (1T Scale)
- [ ] Two-tier expert system
- [ ] Router state quantization
- [ ] Hierarchical multi-GPU dispatch
- [ ] Advanced inference optimizations
- [ ] Benchmarking and systems optimization

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Train 300M baseline
python experiments/train_300m.py --config experiments/configs/hgsel_tiny.yaml

# Benchmark inference
python experiments/benchmark_300m.py

# Phase 3 quick validation
python experiments/validate_phase3.py
```

## Advanced Benchmarks (Phase 3+)

### 1. Trace-Driven Expert Working-Set Modeling

**File:** `experiments/trace_driven_workset.py`

Builds a "working set size vs context length" curve for expert IDs, similar to OS page working sets.

```bash
python experiments/trace_driven_workset.py
```

**Tests:**
- Is expert working set predictable across varying context lengths?
- Does working set grow sublinearly? (cache-friendly)
- Can we compile expert dispatch patterns?

**Falsifies:**
- Working sets are random (unpredictable routing)
- Full expert cache needed (no optimization possible)

**Output:** `results/workset_curve.png`

---

### 2. Tail-Latency Decomposition

**File:** `experiments/tail_latency_decomposition.py`

Breaks p99 latency into components: routing, dispatch, kernel, combine, synchronization.

```bash
python experiments/tail_latency_decomposition.py
```

**Analyzes:**
- Which operation dominates tail latency?
- Is tail predictable (CV < 0.2) or random?
- Can we optimize via batching or pipelining?

**Falsifies:**
- All components have equal latency (no clear optimization path)
- Tail is unpredictable (hardware randomness dominates)

**Output:** `results/tail_latency_decomp.png`

---

### 3. Expert Interference Benchmarks

**File:** `experiments/expert_interference_benchmark.py`

Simulates two concurrent workloads (coding + math tokens) to measure cache interference, quality cross-talk, and latency blowup.

```bash
python experiments/expert_interference_benchmark.py
```

**Scenarios:**
- **Isolated:** Each workload runs alone (baseline)
- **Alternating:** A-B-A-B interleaving (L1/L2 cache pressure)
- **Random:** Worst-case mixing

**Tests:**
- Is interference negligible (< 5%) → multi-tenancy safe?
- Is interference moderate (5-20%) → soft partitioning acceptable?
- Is interference severe (> 20%) → need strict isolation?

**Falsifies:**
- Zero interference (then isolation is unnecessary)
- Catastrophic interference (then system is not viable)

**Output:** `results/expert_interference.png`

---

## Contributing

This is an active research project. Pull requests and issues welcome!

## License

Apache 2.0

---

**Status:** Phase 3 complete. Load balancing and salt tuning validated on CPU quick tests.
