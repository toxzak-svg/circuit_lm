# CircuitLM (FSM) API

The `CircuitLM` class implements a Mealy-style finite-state machine for language modeling.

## Overview

```python
from circuit_lm.circuits import CircuitLM
```

## Class Definition

```python
@dataclass
class CircuitLM:
    """Finite-state circuit language model."""
    
    vocab_size: int
    num_states: int
    state_bits: int
    transitions: dict[tuple[int, int], int]
    state_counts: dict[int, list[int]]
    pred_tokens: dict[int, int]
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `vocab_size` | int | Number of distinct token IDs |
| `num_states` | int | Total number of FSM states (= 2^state_bits) |
| `state_bits` | int | Bit-width of the state representation |
| `transitions` | dict[tuple[int, int], int] | Mapping (state, token) → next_state |
| `state_counts` | dict[int, list[int]] | Integer histograms per state |
| `pred_tokens` | dict[int, int] | Learned emission predictions per state |

## Methods

### next_state

```python
def next_state(self, state: int, token: int) -> int
```

Return the next state given the current state and observed token.

**Parameters:**
- `state` (int): Current state in [0, num_states)
- `token` (int): Observed token ID in [0, vocab_size)

**Returns:**
- int: Next state in [0, num_states)

**Notes:**
- Falls back to a deterministic rolling-hash formula if the (state, token) pair is not in `transitions`.

### run

```python
def run(self, tokens: list[int], initial_state: int = 0) -> list[int]
```

Run the FSM over tokens and return the resulting state sequence.

**Parameters:**
- `tokens` (list[int]): List of token IDs to process
- `initial_state` (int): Starting state (default: 0)

**Returns:**
- list[int]: State sequence, same length as input tokens. Element i is the state *before* consuming tokens[i].

### predict_token

```python
def predict_token(self, state: int) -> int
```

Return the argmax (most frequent) next token from the given state.

**Parameters:**
- `state` (int): Current state in [0, num_states)

**Returns:**
- int: Predicted next token ID. Returns 0 (PAD) if no observations exist.

**Notes:**
- Uses `pred_tokens` if available (learned emission)
- Otherwise falls back to argmax of `state_counts[state]`

### state_histogram

```python
def state_histogram(self, state: int) -> list[int]
```

Return the integer count histogram for a state.

**Parameters:**
- `state` (int): Current state

**Returns:**
- list[int]: Integer histogram of length vocab_size

## Constants

```python
HASH_PRIME: int = 31
```

The prime multiplier used in the default hash-based fallback transition.

## Usage Example

```python
from circuit_lm.circuits import CircuitLM

# Create a simple FSM model
model = CircuitLM(
    vocab_size=128,
    num_states=16,
    state_bits=4,
    transitions={(0, 65): 3, (3, 66): 7},
    state_counts={0: [0, 5, 3], 3: [1, 2, 4]},
    pred_tokens={0: 2, 3: 17}
)

# Get next state
next_state = model.next_state(0, 65)  # Returns 3

# Predict next token
predicted = model.predict_token(0)  # Returns 2 (or argmax of state_counts[0])

# Get count histogram
hist = model.state_histogram(0)  # Returns [0, 5, 3]
```
