# circuit_lm

A **finite-state circuit language model** trained with [OR-Tools CP-SAT](https://developers.google.com/optimization/reference/python/sat/python/cp_model).

Current project snapshot and next milestones: see [`STATUS.md`](STATUS.md).

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
  --tokenizer bpe     \
  --bpe_merges 256    \
  --state_bits 4      \
  --transition_steps 12 \
  --emission_steps   18 \
  --refinement_rounds 1
```

`--transition_steps` and `--emission_steps` are integer CP-SAT wall-clock budgets (seconds) for the transition and emission optimisers.
`--refinement_rounds` controls additional EM-like state-assignment re-estimation passes (default `1`).
`--steps` is still available as a legacy fallback and will be split automatically when explicit budgets are not provided.
`--state_bits S` gives `2^S` FSM states.
`--tokenizer` supports `char` (default) and `bpe` (simple integer BPE over the raw text stream).

Advanced CP-SAT knobs are also exposed:

- `--context_len`
- `--top_k_coverage`
- (PDA only) `--stack_steps`, `--max_push`, `--max_pop`, `--top_k_pairs`

### Evaluate

```bash
circuit-lm eval --data data.txt --model model.json
```

Prints integer counts (correct, total) and accuracy as `XX.YY%` (basis-point arithmetic, no floats).

Optional per-token (gold-token) breakdown:

```bash
circuit-lm eval --data data.txt --model model.json --per_token --per_token_limit 20
```

### Sample

```bash
circuit-lm sample \
  --prompt    "Hello"   \
  --model     model.json  \
  --max_tokens 64         \
  --seed       42         \
  --top_k      16         \
  --repeat_penalty_div 2  \
  --repeat_window 64
```

Sampling uses integer-weighted random choice — no softmax, no temperature float.
`--top_k` and repetition penalty controls are all integer-only.

## Running tests

```bash
pytest
```

Key test modules:

- `test_forbidden_imports` — runtime import check
- `test_no_floats` — static source scan
- `test_circuit_eval` — integration / unit tests

## Benchmark

```bash
python scripts/benchmark_small.py
```

## Architecture

```
Text  ──▶  Tokenizer              char → int ID
           │
           ▼
        Data loader               list[list[int]]
           │
           ├─▶ CircuitLM (FSM)    state ∈ {0 … 2^state_bits − 1}
           │     transitions:  (state, token) → next_state    (int × int → int)
           │     state_counts: state → [count_per_token]       (int → list[int])
           │     ├─▶ train_cpsat           hash-bootstrap + CP-SAT emission
           │     └─▶ train_joint_cpsat     TRUE JOINT: states as CP-SAT vars
           │
           └─▶ PDACircuitLM (PDA) config = (state, stack_top) ∈ int × int
                 push_tokens / pop_tokens: frozenset[int]
                 config_counts: (state, stack_top) → [count_per_token]
                 ├─▶ train_pda_cpsat       two-phase: stack-policy then emission
                 └─▶ train_joint_pda_cpsat TRUE JOINT: states + stack ops + emissions
```

### Trainers at a glance

| Module | Solver | States | Stack policy | Objective |
|--------|--------|--------|--------------|-----------|
| `train_cpsat` | CP-SAT | hash-fixed | — | emission argmax |
| `train_joint_cpsat` | CP-SAT | **free vars** | — | **prediction accuracy** |
| `train_pda_cpsat` | CP-SAT (2-phase) | hash-fixed | co-occurrence score | emission argmax |
| `train_joint_pda_cpsat` | CP-SAT (joint) | **free vars** | **accuracy-driven** | **prediction accuracy** |

## Model format (JSON)

FSM example (character tokenizer + learned emission table):

```json
{
  "model_type": "fsm",
  "vocab_size": 128,
  "num_states": 16,
  "state_bits": 4,
  "transitions": { "0,65": 3, "3,66": 7, "...": "..." },
  "state_counts": { "0": [0, 5, 3, "..."], "...": "..." },
  "pred_tokens": { "0": 2, "3": 17, "...": "..." },
  "tokenizer": { "mode": "char", "chars": ["<PAD>", "<UNK>", "e", "t", "..."] }
}
```

Tokenizer payloads are mode-specific:

```json
{ "mode": "char", "chars": ["<PAD>", "<UNK>", "e", "t", "..."] }
```

```json
{ "mode": "bpe", "pieces": ["<PAD>", "<UNK>", "th", "e", "he", "..."] }
```

PDA models extend the format with stack settings and per-config learned emissions:

```json
{
  "model_type": "pda",
  "stack_depth": 4,
  "push_tokens": [40, 41],
  "pop_tokens": [42, 43],
  "config_counts": { "3,-1": [0, 2, 9, "..."], "7,40": [1, 0, 4, "..."] },
  "config_pred_tokens": { "3,-1": 2, "7,40": 9 }
}
```

All numeric values are JSON integers.

## TODOs

- [x] Joint FSM learning via CP-SAT — `train_joint_cpsat.train_joint`: states, transitions, and emissions as free decision variables; objective = prediction accuracy
- [x] Joint PDA learning via CP-SAT — `train_joint_pda_cpsat.train_joint_pda`: states, push/pop policy, and config emissions jointly in a single model
- [x] Iterative state-assignment refinement (EM-like re-estimation loop over transition / emission counts)
- [ ] Compressed binary model format (MessagePack or similar)
- [ ] Multi-pass CP-SAT for larger state spaces
- [ ] Streaming data loading for large corpora
- [ ] Per-(state, token, stack_top) stack operations (currently token-only)
