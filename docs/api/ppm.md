# PPM Model API

The `PPMModel` class implements Prediction by Partial Matching, a variable-order n-gram model with longest-match backoff.

## Overview

```python
from circuit_lm.ppm import PPMModel
```

## Class Definition

```python
@dataclass
class PPMModel:
    """Variable-order n-gram model with longest-match backoff."""
    
    vocab_size: int
    order: int
    counts: dict[tuple[int, ...], list[int]]
```

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `vocab_size` | int | Number of distinct token IDs |
| `order` | int | Maximum context length in tokens |
| `counts` | dict[tuple[int, ...], list[int]] | Trie mapping context tuples to integer histograms |

## Methods

### predict_token

```python
def predict_token(self, context: tuple[int, ...]) -> int
```

Return argmax next token using longest-match backoff.

**Parameters:**
- `context` (tuple[int, ...]): Tuple of up to `order` preceding token IDs

**Returns:**
- int: Token ID with highest count at deepest available context

**Notes:**
- Walks back through shorter context suffixes until a node with nonzero counts is found
- Falls back to token 0 if no context has observations

### context_histogram

```python
def context_histogram(self, context: tuple[int, ...]) -> list[int]
```

Return integer-blended count histogram across all context levels.

**Parameters:**
- `context` (tuple[int, ...]): Tuple of up to `order` preceding token IDs

**Returns:**
- list[int]: Blended integer histogram of length vocab_size

**Notes:**
- Each level l (0 = empty context, 1 = last-token, ...) is weighted by `l + 1`
- Longer contexts are favoured in the blend

### step

```python
def step(self, context: tuple[int, ...], token: int) -> tuple[int, ...]
```

Advance the context window by consuming a token.

**Parameters:**
- `context` (tuple[int, ...]): Current context tuple
- `token` (int): Newly observed token ID

**Returns:**
- tuple[int, ...]: Updated context tuple (length <= order)

### run

```python
def run(self, tokens: list[int], initial_context: tuple[int, ...] | None = None) -> list[tuple[int, ...]]
```

Run the context window over tokens.

**Parameters:**
- `tokens` (list[int]): Token sequence to process
- `initial_context` (tuple[int, ...] | None): Starting context (default: empty)

**Returns:**
- list[tuple[int, ...]]: Context sequence, one per position

## Algorithm

PPM uses a variable-order n-gram model with:

1. **Longest-match backoff**: Start with full context, fall back to shorter contexts
2. **Blended histograms**: Combine counts from all context levels with integer weights

### Prediction Strategy

1. Start with current context of length `min(len(seen), order)`
2. Walk back through shorter context suffixes until a node with nonzero counts is found
3. If no node found at any depth, fall back to token 0

### Blended Histogram

For sampling, PPM blends histograms from all context levels:

```
weight(level) = level + 1
blended[t] = Σ context_hist[l][t] * (l + 1)
```

This gives more weight to longer contexts while still considering shorter ones.

## Complexity

- **Prediction**: O(order × vocab_size)
- **Memory**: O(K × vocab_size) where K = number of distinct contexts

## Usage Example

```python
from circuit_lm.ppm import PPMModel

# Create a simple PPM model
model = PPMModel(
    vocab_size=128,
    order=4,
    counts={
        (): [0, 5, 3, 2],           # Unigram counts
        (65,): [1, 2, 0],           # "A" context
        (65, 66): [0, 1, 2],       # "AB" context
    }
)

# Predict with full context
predicted = model.predict_token((65, 66, 67))  # Uses "ABC" context

# Predict with shorter context  
predicted = model.predict_token((65,))         # Falls back to "A" context

# Get blended histogram for sampling
hist = model.context_histogram((65, 66))

# Step the context
new_ctx = model.step((65,), 66)  # Returns (65, 66)
```
