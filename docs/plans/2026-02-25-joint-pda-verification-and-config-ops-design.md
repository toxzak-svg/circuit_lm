# Design: Joint-PDA Verification + Config-Conditioned Stack Ops

**Date:** 2026-02-25
**Status:** Approved

## Goals

1. **Joint-PDA small-scale verification** — confirm that `train_joint_pda` discovers the push/pop
   structure when the corpus is kept within the T_total ≤ 2000 tractability limit (50 seqs, vocab=3).

2. **Config-conditioned stack operations** — replace the token-only `push_tokens`/`pop_tokens`
   fields in `PDACircuitLM` with `push_configs`/`pop_configs` (`frozenset[tuple[int,int,int]]`)
   so that the stack operation for a given token can vary by FSM state and current stack top.

---

## Task 1: Verification Script

### File
`scripts/verify_joint_pda_small.py`

### Purpose
Answer one research question: *does the joint CP-SAT PDA solver discover the balanced-parentheses
stack structure when T_total is held within the 2000-token tractability limit?*  Produces a
concise result row suitable for copying directly into STATUS.md.

### Parameters
| Parameter | Default | Notes |
|-----------|---------|-------|
| `--seed` | 42 | Integer RNG seed |
| `--train-seqs` | 50 | Training sequences; keep ≤ 80 to stay under T_total 2000 |
| `--steps` | 20 | CP-SAT wall-clock budget (seconds) |

Vocabulary: `OPEN=0`, `CLOSE=1`, `EOS=2`, `vocab_size=3`.
Model: `num_states=4`, `max_push=1`, `max_pop=1`, `max_depth=3`.

### Script Logic
1. Generate `train_seqs` balanced sequences (max_depth=3).
2. Compute and print `T_total = sum(len(seq) for seq in train_data)`.
   Hard-assert `T_total ≤ 2000`.
3. Train `train_joint_pda` with the above parameters.
4. Print discovered `push_configs` and `pop_configs`.
5. Print `PASS: stack discovered` if any push and pop config exist; else `WARN: no stack`.
6. Generate 100 test sequences each at depths 3, 5, 8 and evaluate with `evaluate_pda`.
7. Print a 3-row table (depth / accuracy / seqs).

### Correctness Notes
- Sequence generation reuses `gen_train_seqs` / `gen_test_seqs_at_depth` from
  `reproduce_depth_generalization.py` — same integer-only, deterministic-seed logic.
- T_total calculation uses raw sequence lengths (same counting as the CP-SAT model).
- No FSM or PPM baselines — this script has exactly one purpose.

---

## Task 2: Config-Conditioned Stack Operations

### 2a. `circuit_lm/pda.py` — Data Model

**Remove:**
```python
push_tokens: frozenset[int]
pop_tokens:  frozenset[int]
```

**Add:**
```python
push_configs: frozenset[tuple[int, int, int]]
pop_configs:  frozenset[tuple[int, int, int]]
```

Each triple is `(src_state, token, stack_top_before_op)` where:
- `src_state ∈ [0, num_states)` — FSM state **before** consuming the token
- `token ∈ [0, vocab_size)` — current token ID
- `stack_top_before_op ∈ {STACK_EMPTY, 0, …, vocab_size-1}` — stack top **before** the operation
  (`STACK_EMPTY = -1` is the universal sentinel for an empty stack)

**`stack_op` signature:**
```python
# Before:
def stack_op(self, token: int) -> int

# After:
def stack_op(self, state: int, token: int, stack_top: int) -> int
```

Set membership lookup remains O(1). All callers (`step`, evaluators, serialization) pass all three
arguments. `step` already computes `stack_top = stack[-1] if stack else STACK_EMPTY` before
calling `stack_op`, so the extra arguments require no new computation.

**`stack_op` body:**
```python
def stack_op(self, state: int, token: int, stack_top: int) -> int:
    key = (state, token, stack_top)
    if key in self.push_configs:
        return OP_PUSH
    if key in self.pop_configs:
        return OP_POP
    return OP_NOP
```

### 2b. JSON Serialization

Field names change: `"push_tokens"` → `"push_configs"`, `"pop_tokens"` → `"pop_configs"`.

New wire format (list of `[state, token, stack_top]` integer triples):
```json
"push_configs": [[0, 0, -1], [1, 0, -1], [0, 0, 1], ...],
"pop_configs":  [[0, 1, 0],  [1, 1, 0],  ...]
```

**Migration shim in loader:** if the JSON contains `"push_tokens"` (old format), expand to
degenerate configs: `{(s, tok, st) for tok in push_tokens for s in range(num_states) for st in
[STACK_EMPTY] + list(range(vocab_size))}`.

### 2c. `train_joint_pda_cpsat.py` — New CP-SAT Formulation

**Variable counts (replacing V + V booleans):**

```
PUSH_SIZE = S × V × ST_RANGE
is_push_flat[i]  ∈ {0,1}    i = s * V * ST_RANGE + tok * ST_RANGE + st_enc
is_pop_flat[i]   ∈ {0,1}    (same indexing)
```

For the verification experiment (S=4, V=3, ST_RANGE=4): 48 bool vars each vs. 3 previously.

**Mutual exclusivity:** `is_push_flat[i] + is_pop_flat[i] ≤ 1` for each `i`.

**Per-occurrence lookup (non-start positions):**

For occurrence `k` with constant `tok`, variables `src[k]` and `st_prev = st_vars[prev_occ_idx]`:

```
push_idx[k] = src[k] * (V * ST_RANGE)  +  tok * ST_RANGE  +  st_prev
              ^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^     ^^^^^^^
              coefficient × IntVar         constant offset      IntVar
```

This is a valid CP-SAT linear expression. A single `IntVar` in `[0, PUSH_SIZE-1]` holds it.

```
push_at_k = BoolVar
add_element(push_idx[k], is_push_flat, push_at_k)

pop_at_k = BoolVar
add_element(push_idx[k], is_pop_flat, pop_at_k)
```

`push_at_k` and `pop_at_k` replace `is_push[tok]` / `is_pop[tok]` in the stack-update
constraints. Adds 2 `IntVar`s + 4 constraints per non-start occurrence.

**Sequence-start positions:** `src[k] = 0` (constant) and `st_prev = EMPTY_ENC` (constant), so
`push_idx = 0 * V * ST_RANGE + tok * ST_RANGE + EMPTY_ENC` is a compile-time constant.
Reference `is_push_flat[constant]` directly — no extra variable needed.

**max_push / max_pop budget (preserves "distinct token ID" semantics):**

Introduce `tok_ever_pushes[tok]` auxiliary booleans (V of them):
```
is_push_flat[s * V * ST_RANGE + tok * ST_RANGE + st_enc]  ≤  tok_ever_pushes[tok]
    ∀ s, st_enc       (tok_ever_pushes dominates every per-config var for that token)
sum(tok_ever_pushes) ≤ max_push
```
Similarly for pop. This means `max_push=1` still means "at most 1 distinct token ID ever
triggers a push, across all state/stack-top combinations."

**Output extraction:**
```python
push_configs = frozenset(
    (s, tok, STACK_EMPTY if st_enc == EMPTY_ENC else st_enc)
    for s in range(S)
    for tok in range(V)
    for st_enc in range(ST_RANGE)
    if solver.value(is_push_flat[s * V * ST_RANGE + tok * ST_RANGE + st_enc])
)
```

### 2d. `train_pda_cpsat.py` — Degenerate Expansion (B1)

Phase 1 logic is unchanged. At output, expand token-level sets to config triples:

```python
all_stack_tops = [STACK_EMPTY] + list(range(vocab_size))
push_configs = frozenset(
    (s, tok, st)
    for tok in push_tokens_found
    for s in range(num_states)
    for st in all_stack_tops
)
pop_configs = frozenset(
    (s, tok, st)
    for tok in pop_tokens_found
    for s in range(num_states)
    for st in all_stack_tops
)
```

Size: `|push_tokens| × S × (V+1)`. For depth-generalization experiment: 1 × 4 × 4 = 16 triples.

### 2e. `eval.py` — Simulation Updates

Both `_simulate_and_collect` and `_simulate_and_collect_runtime` update their stack step to:
```python
stack_top = stack[-1] if stack else STACK_EMPTY
op = model.stack_op(state, token, stack_top)
```

`predict_token` and `config_histogram` in `PDACircuitLM` are unchanged (they still key on
`(state, stack_top)` — the stack op resolution is upstream of them).

---

## Test Plan

| Test | File | What it checks |
|------|------|----------------|
| `push_configs` / `pop_configs` are frozensets of int-triples | `test_pda.py` | data model |
| `stack_op(s, tok, st)` returns correct op | `test_pda.py` | dispatch logic |
| `step()` passes correct stack_top to `stack_op` | `test_pda.py` | caller correctness |
| Degenerate expansion produces correct triple count | `test_pda_cpsat.py` | B1 expansion |
| Degenerate expansion is token-id-invariant (all states/stack-tops covered) | `test_pda_cpsat.py` | B1 completeness |
| Joint trainer push_configs disjoint from pop_configs | `test_joint_pda.py` | mutual exclusivity |
| `max_push=1` → at most 1 distinct token ID pushes | `test_joint_pda.py` | budget semantics |
| Config-conditioned op varies across (state, stack_top) for same token | `test_joint_pda.py` | expressiveness |
| JSON round-trip preserves push_configs / pop_configs | serialization tests | wire format |
| Old JSON with `push_tokens` migrates to degenerate configs | serialization tests | migration shim |
| No floats in any new code | `test_no_floats.py` (static scan) | integer discipline |

---

## Scalability Notes

- Joint solver: adding `S × V × ST_RANGE` bool vars grows the model. Recommended limits stay at
  T_total ≤ 2000, S ≤ 8, V ≤ 32 as before (config vars add ≤ 8×32×33 = 8448 bools, still
  tractable for CP-SAT).
- Two-phase solver: degenerate expansion is pure Python set construction — negligible runtime
  overhead. Frozenset size is O(S × V) per push/pop token.
- The O(1) frozenset membership check in `stack_op` scales to S=16, V=128 with no concern
  (max frozenset size ≈ 16 × 128 × 129 = 264 192 entries in the degenerate case — well within
  Python set performance range).

---

## Files Touched

| File | Change |
|------|--------|
| `scripts/verify_joint_pda_small.py` | **new** |
| `circuit_lm/pda.py` | replace push/pop fields + stack_op signature |
| `circuit_lm/train_joint_pda_cpsat.py` | new CP-SAT config-op variables + extraction |
| `circuit_lm/train_pda_cpsat.py` | degenerate expansion at output |
| `circuit_lm/eval.py` | pass (state, token, stack_top) to stack_op |
| serialization code | field name change + migration shim |
| `tests/test_pda.py` | update for new API |
| `tests/test_joint_pda.py` | update + new config-op tests |
| `tests/test_pda_cpsat.py` (new or existing) | degenerate expansion tests |
| serialization test file | round-trip + migration shim tests |
| `STATUS.md` | record verification result |
