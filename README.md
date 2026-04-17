# circuit_lm

[![CI](https://github.com/toxzak-svg/circuit_lm/actions/workflows/ci.yml/badge.svg)](https://github.com/toxzak-svg/circuit_lm/actions/workflows/ci.yml)

A **finite-state circuit language model** trained with [OR-Tools CP-SAT](https://developers.google.com/optimization/reference/python/sat/python/cp_model).

The core idea: use integer-only automata (FSM, PDA, PPM) to capture structural patterns in text, then layer a small neural corrector on top to handle the residual. The circuit handles what structure handles well; the neural net handles what compression handles well.

---

## What Was Built (2026-04-13 to 2026-04-17)

### Benchmark: PPM vs PDA vs FSM

Ran 5 variants of the multi-bracket depth generalization benchmark to test whether explicit stack structure (PDA) outperforms compression (PPM) on out-of-distribution generalization.

**Result: PPM wins every time — but PDA beats FSM consistently.**

| Version | Config | PDA OOD | FSM OOD | PPM OOD |
|---------|--------|---------|---------|---------|
| v1 | 200 seqs, 8 tokens | 23.9% | 22.0% | **31.8%** |
| v2 | 400 seqs, 4-12 pairs | 25.8% | 22.0% | **31.8%** |
| v3 | bracket-only vocab (7 tokens) | 25.8% | 22.0% | **31.8%** |
| v4 | hard split (train depth≤3) | 22.3% | 22.1% | **29.6%** |
| v5 | mismatched bracket types | 26.5% | 20.4% | **32.7%** |

**Key findings:**
- PDA consistently beats FSM by +2 to +6pp — the stack provides real structural benefit over finite context
- PPM beats both because bracket prediction is fundamentally local — the last open bracket predicts the next token regardless of nesting depth
- Next-token prediction is a local task; structural recursion doesn't help as much as n-gram compression
- The hybrid approach (circuit for structure + neural corrector for residuals) is still the right architecture

See [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md) for full results and CSVs.

### Streaming Data Loader

`iter_hybrid_examples()` and `build_dataset_streaming()` in `src/hybrid.py` — processes corpora larger than RAM without OOM. Maintains persistent circuit state across file boundaries (FSM state and PDA stack survive between lines).

```python
from src.hybrid import iter_hybrid_examples, build_dataset_streaming

# Chunked streaming (memory-efficient)
for chunk in iter_hybrid_examples(circuit, tokenizer, "large_corpus.txt", chunk_size=2000):
    # process chunk
    pass

# Convenience wrapper (collects everything)
examples = build_dataset_streaming(circuit, tokenizer, "large_corpus.txt")
```

### Larger Corrector + Streaming Training

`train_hybrid_streaming()` in `src/hybrid.py` — trains the neural corrector using streaming data, with configurable size:

| Parameter | Old | New Default | Notes |
|-----------|-----|-------------|-------|
| embed_dim | 64 | 128 | corrector embedding |
| hidden_dim | 128 | 256 | corrector hidden |
| num_layers | 2 | 3 | corrector depth |

Full training script with all options: `scripts/train_bpe_hybrid.py`

### Reproduce Script

`scripts/reproduce.ps1` — one script that runs the entire pipeline:

```powershell
.\scripts\reproduce.ps1 -DataFile "training_data.txt"
```

Trains circuit → trains corrector → runs trace → evaluates. No manual steps.

### Kaggle/Colab Notebook

`kaggle_run.ipynb` — 7-cell Colab notebook with GPU support:

- Cell 1: Clone repo + install deps + wandb login from Kaggle secrets
- Cell 2: Upload training data
- Cell 3: Train circuit (CPU, ~1 min)
- Cell 4: Train corrector (GPU, ~5 min)
- Cell 5: Trace — see why each token was predicted
- Cell 6: Evaluate accuracy
- Cell 7: Download trained models

Auto-detects `WANDB_API_KEY` and Kaggle credentials from Colab secrets. WandB logs metrics for both circuit and corrector training.

### Trace CLI

Step-by-step interpretability for any trained model:

```bash
py -3.12 -m circuit_lm.cli trace --model circuit.json --prompt "User: hello" --top_k 5
```

Output shows state + stack_top + top-k ranked predictions at each token position. Works for FSM, PDA, and PPM.

---

## Hard Constraints

| Constraint | Enforcement |
|---|---|
| No floating-point arithmetic | `tests/test_no_floats.py` — static regex scan |
| No numpy / torch / jax / scipy | `tests/test_forbidden_imports.py` — runtime import check |
| No tensor / matmul | No such imports in the package |
| Solver: OR-Tools CP-SAT only | `pyproject.toml` dependency, integer variables only |

---

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Train a circuit (CPU, ~30 sec on 1MB text)
py -3.12 -m circuit_lm.cli train \
  --data training_data.txt \
  --out circuit.json \
  --automaton pda \
  --tokenizer bpe \
  --bpe_merges 512 \
  --vocab_size 1024 \
  --state_bits 5 \
  --stack_depth 6

# Train the neural corrector (GPU recommended)
py -3.12 scripts/train_bpe_hybrid.py \
  --data training_data.txt \
  --circuit-out circuit.json \
  --corrector-out corrector.pt \
  --streaming \
  --max-examples 100000 \
  --embed-dim 256 --hidden-dim 512 --num-layers 3

# Chat
py -3.12 -m circuit_lm.cli chat --model circuit.json --corrector corrector.pt

# Trace (interpretability)
py -3.12 -m circuit_lm.cli trace --model circuit.json --prompt "User: hello" --top_k 5

# Full pipeline (Windows PowerShell)
.\scripts\reproduce.ps1 -DataFile "training_data.txt"
```

---

## Architecture

```
Text  ──▶  Tokenizer              char → int ID or BPE → int ID
           │
           ▼
        Data loader               list[list[int]]
           │
           ├─▶ CircuitLM (FSM)    state ∈ {0 … 2^state_bits − 1}
           │     transitions:  (state, token) → next_state
           │     state_counts: state → [count_per_token]
           │
           ├─▶ PDACircuitLM (PDA) config = (state, stack_top)
           │     push_configs / pop_configs: frozenset[tuple[int,int,int]]
           │     config_counts: (state, stack_top) → [count_per_token]
           │
           └─▶ PPMModel           pure n-gram compression (order=N)
           
           ▼
        HybridModel = CircuitLM + NeuralCorrector
           circuit_state + stack_top + context → residual correction
           blended_logit = w * circuit_logit + (1-w) * corrector_logit
```

### Circuit trainers

| Module | Automaton | Stack | Objective |
|--------|-----------|-------|-----------|
| `train_cpsat` | FSM | — | emission argmax |
| `train_pda_cpsat` | PDA | 2-phase CP-SAT | emission argmax |
| `train_ppm` | PPM | n-gram | prediction accuracy |
| `train_joint_pda_cpsat` | PDA | joint CP-SAT | prediction accuracy |

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## CLI Reference

```bash
# Train circuit
py -3.12 -m circuit_lm.cli train --data data.txt --out model.json \
  --automaton pda --tokenizer bpe --bpe_merges 512 \
  --vocab_size 1024 --state_bits 5 --stack_depth 6 --steps 60

# Evaluate
py -3.12 -m circuit_lm.cli eval --data data.txt --model model.json --per_token

# Sample
py -3.12 -m circuit_lm.cli sample --prompt "Hello" --model model.json --max_tokens 64

# Trace (interpretability)
py -3.12 -m circuit_lm.cli trace --model model.json --prompt "User: hello" --top_k 5 --json-out trace.json

# Chat
py -3.12 -m circuit_lm.cli chat --model model.json --corrector corrector.pt
```

---

## Training Data

The repo ships with several training datasets:

| File | Size | Format |
|------|------|--------|
| `all_personal_training.txt` | 7.4 MB | user:/assistant: chat |
| `starfire_training_data.txt` | 5.0 MB | user:/assistant: chat |
| `chat_data.txt` | 1.5 MB | user:/assistant: chat |
| `topical_chat_data.txt` | 22.7 MB | conversation text |
| `training_data.txt` | 10.4 MB | general text |

For chat formatting:
```
user: hello
assistant: hi there, how are you?
user: good, you?
assistant: doing well
```

---

## Colab / Kaggle

Open `kaggle_run.ipynb` in Google Colab:

1. File → Open Notebook → Import `kaggle_run.ipynb`
2. Set Kaggle secrets: `KAGGLE_USERNAME`, `KAGGLE_API_KEY`, `WANDB_API_KEY`
3. Upload training data in cell 2
4. Runtime → Run all

GPU (T4 free tier) recommended for corrector training. Circuit training is CPU-fast.

---

## Model Format

Circuits are saved as JSON:

```json
{
  "model_type": "pda",
  "vocab_size": 1024,
  "num_states": 32,
  "stack_depth": 6,
  "transitions": { "3,1021": 7 },
  "config_counts": { "3,-1": [0, 2, 9, ...] },
  "push_configs": [[0, 40, -1], [1, 40, -1]],
  "pop_configs": [[0, 41, 40]],
  "tokenizer": { "mode": "bpe", "pieces": ["<PAD>", "th", "he", ...] }
}
```

Corrector is a PyTorch model saved as `.pt`:

```python
from src.hybrid import HybridModel
hybrid = HybridModel.load("circuit.json", "corrector.pt")
reply = generate_reply_hybrid(hybrid, tokenizer, prompt_ids, max_tokens=128)
```

---

## Tests

```bash
pytest
```

Key tests:
- `test_no_floats` — static source scan, no float literals
- `test_forbidden_imports` — no numpy/torch/jax/scipy
- `test_circuit_eval` — integration tests

---

## Files

```
.
├── README.md                          # This file
├── kaggle_run.ipynb                   # Colab notebook (GPU training + wandb)
├── scripts/
│   ├── reproduce.ps1                  # Full pipeline (Windows PowerShell)
│   ├── train_bpe_hybrid.py            # Circuit + corrector training
│   ├── benchmark_code.py              # v1: baseline benchmark
│   ├── benchmark_code_v2.py           # v2: longer sequences
│   ├── benchmark_code_v3.py           # v3: bracket-only vocab
│   ├── benchmark_code_v4.py           # v4: hard split (train shallow)
│   ├── benchmark_code_v5.py           # v5: mismatched bracket types
│   └── results/                       # CSV results from all benchmarks
├── src/
│   ├── hybrid.py                      # HybridModel + NeuralCorrector + streaming
│   ├── circuits.py                    # CircuitLM (FSM)
│   ├── pda.py                         # PDACircuitLM
│   ├── ppm.py                         # PPMModel
│   └── ...
├── docs/
│   ├── BENCHMARK_RESULTS.md           # Full benchmark results and analysis
│   ├── ARCHITECTURE.md                # Architecture deep-dive
│   └── QUICKSTART.md                  # Detailed getting started
└── tests/
```
