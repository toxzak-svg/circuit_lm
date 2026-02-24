# circuit_lm

A **finite-state circuit language model** trained with [OR-Tools CP-SAT](https://developers.google.com/optimization/reference/python/sat/python/cp_model).

## Hard constraints

| Constraint | Enforcement |
|---|---|
| No floating-point arithmetic | `tests/test_no_floats.py` — static regex scan of `circuit_lm/` and `scripts/` |
| No numpy / torch / jax / scipy / tensorflow | `tests/test_forbidden_imports.py` — runtime `sys.modules` check |
| No tensor / matmul | No such imports anywhere in the package |
| Solver: OR-Tools CP-SAT only | `pyproject.toml` dependency, integer variables only |

## Installation

```bash
pip install -e ".[dev]"
```

## CLI

### Train

```bash
circuit-lm train \
  --data   data.txt   \
  --out    model.json \
  --vocab_size 128    \
  --state_bits 4      \
  --steps  30
```

`--steps` is an integer (CP-SAT wall-clock limit in seconds).
`--state_bits S` gives `2^S` FSM states.

### Evaluate

```bash
circuit-lm eval --data data.txt --model model.json
```

Prints integer counts (correct, total) and accuracy as `XX.YY%` (basis-point arithmetic, no floats).

### Sample

```bash
circuit-lm sample \
  --prompt    "Hello"   \
  --model     model.json \
  --max_tokens 64        \
  --seed       42
```

Sampling uses integer-weighted random choice — no softmax, no temperature float.

## Running tests

```bash
pytest
```

All three test modules must pass on a clean install:

- `test_forbidden_imports` — runtime import check
- `test_no_floats` — static source scan
- `test_circuit_eval` — integration / unit tests

## Benchmark

```bash
python scripts/benchmark_small.py
```

## Architecture

```
Text  ──▶  Tokenizer         char → int ID
           │
           ▼
        Data loader          list[list[int]]
           │
           ▼
      CircuitLM (FSM)        state ∈ {0 … 2^state_bits − 1}
        transitions:  (state, token) → next_state    (int × int → int)
        state_counts: state → [count_per_token]       (int → list[int])
           │
           ├─▶ train_cpsat   OR-Tools CP-SAT emission optimiser
           ├─▶ infer         integer-weighted sampling / greedy
           ├─▶ eval          integer accuracy
           └─▶ io            JSON save / load (all integers)
```

## Model format (JSON)

```json
{
  "vocab_size":   128,
  "num_states":   16,
  "state_bits":   4,
  "transitions":  { "0,65": 3, "3,66": 7, "...": "..." },
  "state_counts": { "0": [0, 5, 3, "..."], "...": "..." },
  "tokenizer":    { "chars": ["<PAD>", "<UNK>", "e", "t", "..."] }
}
```

All numeric values are JSON integers.

## TODOs

- [ ] Full transition-function learning via CP-SAT (currently transitions use a fixed rolling hash)
- [ ] Iterative state-assignment refinement (re-derive state hashes after each CP-SAT pass)
- [ ] BPE / subword tokenisation
- [ ] Top-k integer sampling and integer repetition penalty
- [ ] Compressed binary model format (MessagePack or similar)
- [ ] Multi-pass CP-SAT for larger state spaces
- [ ] Per-token accuracy breakdown in `eval`
- [ ] Streaming data loading for large corpora
