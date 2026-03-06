# PDA Circuit API

The `PDACircuitLM` class implements a Pushdown Automaton language model that extends the FSM with a bounded integer stack.

## Overview

```python
from circuit_lm.pda import PDACircuitLM
```

## Class Definition

```python
@dataclass
class PDACircuitLM:
    """Pushdown automaton language model."""
    
    vocab_size: int
    num_states: int
    state_bits: int
    stack_depth: int
    push_configs: frozenset[tuple[int, int, int]]
    pop_configs: frozenset[tuple[int, int, int]]
    transitions: dict[tuple[int, int], int]
    config_counts: dict[tuple[int, int], list[int]]
    config_pred_tokens: dict[tuple[int, int], int]
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `vocab_size` | int | Number of distinct token IDs |
| `num_states` | int | Total FSM states (= 2^state_bits) |
| `state_bits` | int | Bit-width of state representation |
| `stack_depth` | int | Maximum stack depth |
| `push_configs` | frozenset[tuple[int, int, int]] | (src_state, token, stack_top_before_op) triples that trigger PUSH |
| `pop_configs` | frozenset[tuple[int, int, int]] | (src_state, token, stack_top_before_op) triples that trigger POP |
| `transitions` | dict[tuple[int, int], int] | FSM transition mapping |
| `config_counts` | dict[tuple[int, int], list[int]] | Histograms per (state, stack_top) |
| `config_pred_tokens` | dict[tuple[int, int], int] | Learned emissions per config |

## Constants

```python
STACK_EMPTY: int = -1   # Sentinel for empty stack
OP_NOP:      int = 0    # No stack change
OP_PUSH:     int = 1    # Push current token
OP_POP:      int = 2    # Pop top element
```

## Methods

### stack_op

```python
def stack_op(self, state: int, token: int, stack_top: int) -> int
```

Return the stack operation for the given (state, token, stack_top) triple.

**Parameters:**
- `state` (int): Current FSM state
- `token` (int): Current input token ID
- `stack_top` (int): Stack top before operation (STACK_EMPTY for empty)

**Returns:**
- int: One of OP_NOP, OP_PUSH, or OP_POP

### next_state

```python
def next_state(self, state: int, token: int) -> int
```

Return the next FSM state (with integer hash fallback).

**Parameters:**
- `state` (int): Current state
- `token` (int): Current token

**Returns:**
- int: Next state

### step

```python
def step(self, state: int, stack: list[int], token: int) -> tuple[int, list[int]]
```

Execute one PDA step and return (next_state, new_stack).

**Parameters:**
- `state` (int): Current FSM state
- `stack` (int): Current stack contents
- `token` (int): Current input token

**Returns:**
- tuple[int, list[int]]: (next_state, new_stack)

**Notes:**
- Stack operation is resolved using (state, token, stack_top) before the FSM transition
- Stack is bounded by stack_depth; PUSH silently ignored when full
- POP on empty stack is a no-op

### predict_token

```python
def predict_token(self, state: int, stack: list[int]) -> int
```

Return the argmax next token for the current configuration.

**Parameters:**
- `state` (int): Current FSM state
- `stack` (list[int]): Current stack contents

**Returns:**
- int: Predicted next token ID

**Notes:**
- Uses config_pred_tokens if available
- Falls back to argmax of config_counts[(state, stack_top)]
- Returns 0 (PAD) if no observations exist

### config_histogram

```python
def config_histogram(self, state: int, stack: list[int]) -> list[int]
```

Return the integer count histogram for the current configuration.

**Parameters:**
- `state` (int): Current FSM state
- `stack` (list[int]): Current stack contents

**Returns:**
- list[int]: Integer histogram of length vocab_size

### run

```python
def run(self, tokens: list[int], initial_state: int = 0, initial_stack: list[int] | None = None) -> list[tuple[int, list[int]]]
```

Run the PDA over tokens, returning configuration sequence.

**Parameters:**
- `tokens` (list[int]): Token sequence to process
- `initial_state` (int): Starting FSM state (default 0)
- `initial_stack` (list[int] | None): Starting stack (default empty)

**Returns:**
- list[tuple[int, list[int]]]: Configuration sequence, same length as tokens

## Configuration Format

The PDA uses config-conditioned stack operations. Each config triple is:

```
(src_state, token, stack_top_before_op)
```

Where:
- `src_state`: FSM state before consuming the token
- `token`: The input token
- `stack_top_before_op`: Stack top before the stack operation (STACK_EMPTY = -1 for empty stack)

## Usage Example

```python
from circuit_lm.pda import PDACircuitLM, STACK_EMPTY, OP_PUSH, OP_POP

# Create a simple PDA model
model = PDACircuitLM(
    vocab_size=128,
    num_states=16,
    state_bits=4,
    stack_depth=4,
    push_configs=frozenset({(0, 40, -1), (1, 40, -1)}),  # push '('
    pop_configs=frozenset({(0, 41, 40), (1, 41, 40)}),   # pop ')'
    transitions={},
    config_counts={(3, -1): [0, 2, 9], (7, 40): [1, 0, 4]},
    config_pred_tokens={}
)

# Execute a step
state = 0
stack = []
token = 40  # '('
next_state, new_stack = model.step(state, stack, token)
# next_state = hash(0, 40), new_stack = [40]

# Predict next token
predicted = model.predict_token(next_state, new_stack)
```

## Stack Operations

The PDA maintains a bounded integer stack that enables modeling of nested structures:

- **PUSH**: Adds current token to stack top (when stack not full)
- **POP**: Removes stack top (when stack not empty)
- **NOP**: No change

This allows the PDA to learn bracket-matching and scope-tracking patterns that pure FSMs cannot capture.
