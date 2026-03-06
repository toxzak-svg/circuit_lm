# Architecture Overview

CircuitLM is a finite-state circuit language model system that uses multiple automaton architectures for sequence modeling. All computations are performed using integer arithmetic only—no floating-point operations.

## System Architecture

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
                 push_configs / pop_configs: frozenset[tuple[int,int,int]]
                 config_counts: (state, stack_top) → [count_per_token]
                 ├─▶ train_pda_cpsat       two-phase: stack-policy then emission
                 └─▶ train_joint_pda_cpsat TRUE JOINT: states + stack ops + emissions
```

## Model Types

### FSM (Finite-State Machine)

A Mealy-style finite-state machine with:
- **State**: Integer in `[0, num_states)`
- **Transition**: `next_state = delta(state, token)`
- **Emission**: `prediction = argmax(emit[state])`

Memory complexity: O(log num_states)

### PDA (Pushdown Automaton)

Extends FSM with a bounded integer stack:
- **Configuration**: `(state, stack_top)` where `stack_top` is an integer token ID or STACK_EMPTY (-1)
- **Stack Operations**: PUSH, POP, NOP (determined by config triples)
- **Emission**: `prediction = argmax(emit[state, stack_top])`

Memory complexity: O(D × log V) where D = stack depth, V = vocabulary size

### PPM (Prediction by Partial Matching)

Variable-order n-gram model with longest-match backoff:
- **Context**: Sliding window of up to `order` tokens
- **Prediction**: Argmax over longest available context
- **Blended histogram**: Integer-weighted combination across all context levels

Memory complexity: O(K × V) where K = distinct contexts

## Training Approaches

| Module | Solver | States | Stack policy | Objective |
|--------|--------|--------|--------------|-----------|
| `train_cpsat` | CP-SAT | hash-fixed | — | emission argmax |
| `train_joint_cpsat` | CP-SAT | **free vars** | — | **prediction accuracy** |
| `train_pda_cpsat` | CP-SAT (2-phase) | hash-fixed | co-occurrence score | emission argmax |
| `train_joint_pda_cpsat` | CP-SAT (joint) | **free vars** | **accuracy-driven** | **prediction accuracy** |

### FSM Training

1. **Hash-based bootstrap**: States computed from context via rolling polynomial hash
2. **Transition optimization**: CP-SAT learns `delta(s, t)` to maximize successor agreement
3. **Emission optimization**: CP-SAT learns `pred[s]` to maximize correct predictions
4. **Refinement**: EM-like re-estimation passes

### PDA Training (Two-Phase)

1. **Phase 1 - Stack Policy**: CP-SAT learns which tokens trigger PUSH/POP
2. **Phase 2 - Config Emissions**: CP-SAT learns emissions per `(state, stack_top)`
3. **Refinement**: EM-like passes over transition and emission counts

### Joint Training

True joint optimization where:
- States are CP-SAT decision variables (not hash-derived)
- Stack operations are learned alongside emissions
- Objective is direct prediction accuracy

## Integer Arithmetic

All operations use integer arithmetic:

- **Counts**: Integer histograms over tokens
- **Accuracy**: Basis-points (1/100 of percent)
- **Sampling**: Integer-weighted random choice
- **Transitions**: Deterministic hash fallback

### Accuracy Representation

```python
# Accuracy in basis-points (no floats)
accuracy_bp = (correct * 10000) // total

# Format as percentage string
whole = accuracy_bp // 100
frac = accuracy_bp % 100
formatted = f"{whole}.{frac:02d}%"  # e.g., "25.00%"
```

### Sampling

Integer-weighted sampling without softmax:
1. Get integer count histogram `h[t]`
2. Draw uniform `r` in `[0, sum(h))`
3. Return index where cumulative sum exceeds `r`

## Model Format

Models are serialized as JSON with integer-only values:

**FSM**:
```json
{
  "model_type": "fsm",
  "vocab_size": 128,
  "num_states": 16,
  "state_bits": 4,
  "transitions": { "0,65": 3, "3,66": 7 },
  "state_counts": { "0": [0, 5, 3], "...": "..." },
  "pred_tokens": { "0": 2, "3": 17 },
  "tokenizer": { "mode": "char", "chars": ["<PAD>", "<UNK>", "e", "t"] }
}
```

**PDA**:
```json
{
  "model_type": "pda",
  "stack_depth": 4,
  "push_configs": [[0, 40, -1], [1, 40, -1]],
  "pop_configs": [[0, 42, 40], [1, 42, 40]],
  "config_counts": { "3,-1": [0, 2, 9], "7,40": [1, 0, 4] },
  "config_pred_tokens": { "3,-1": 2, "7,40": 9 }
}
```

**PPM**:
```json
{
  "model_type": "ppm",
  "vocab_size": 128,
  "order": 4,
  "counts": { "": [0, 5, 3], "101,116": [1, 2, 0] }
}
```
