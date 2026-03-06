# Tokenizer API

The `Tokenizer` class converts between text and integer token IDs.

## Overview

```python
from circuit_lm.tokenizer import Tokenizer
```

## Class Definition

```python
class Tokenizer:
    PAD_ID: int = 0
    UNK_ID: int = 1
    
    vocab_size: int
    mode: str  # "char" or "bpe"
```

## Reserved IDs

| ID | Symbol | Description |
|----|--------|-------------|
| 0 | `<PAD>` | Padding / unknown |
| 1 | `<UNK>` | Out-of-vocabulary |

User tokens start at ID 2.

## Construction

### from_text

Build a tokenizer from raw text.

```python
Tokenizer.from_text(
    text: str,
    vocab_size: int | None = None,
    mode: str = "char",
    bpe_merges: int | None = None,
) -> Tokenizer
```

**Parameters:**
- `text` (str): Input text to derive vocabulary from
- `vocab_size` (int | None): Maximum vocabulary size (including PAD + UNK). None = include all characters.
- `mode` (str): "char" (default) or "bpe"
- `bpe_merges` (int | None): Maximum BPE merges (for bpe mode)

**Example:**
```python
# Character tokenizer
tokenizer = Tokenizer.from_text(text, vocab_size=128, mode="char")

# BPE tokenizer
tokenizer = Tokenizer.from_text(text, vocab_size=256, mode="bpe", bpe_merges=256)
```

### from_dict

Restore a tokenizer from a dict.

```python
Tokenizer.from_dict(d: dict) -> Tokenizer
```

## Methods

### encode

Encode a string into token IDs.

```python
def encode(self, text: str) -> list[int]
```

**Parameters:**
- `text` (str): Input text

**Returns:**
- list[int]: List of token IDs

### decode

Decode token IDs back to a string.

```python
def decode(self, ids: list[int]) -> str
```

**Parameters:**
- `ids` (list[int]): List of token IDs

**Returns:**
- str: Decoded string

**Notes:**
- PAD and UNK tokens are rendered as replacement character U+FFFD

### to_dict

Serialize tokenizer to a dict.

```python
def to_dict(self) -> dict
```

**Returns:**
- dict: JSON-serializable dict

## Modes

### Character Mode

Character-level tokenizer:
- Each character becomes a token
- Most frequent characters get lowest IDs (after PAD/UNK)
- Simple and deterministic

### BPE Mode

Byte Pair Encoding tokenizer:
- Learns merge operations on the character stream
- Greedy longest-piece encoding for text→IDs
- More compact representations for frequent patterns

## Usage Example

```python
from circuit_lm.tokenizer import Tokenizer

# Create tokenizer
tokenizer = Tokenizer.from_text("hello world hello", vocab_size=32, mode="char")

# Encode
ids = tokenizer.encode("hello")
print(ids)  # e.g., [2, 3, 4, 4, 5]

# Decode
text = tokenizer.decode(ids)
print(text)  # "hello"

# Check vocabulary size
print(tokenizer.vocab_size)  # e.g., 10

# Check mode
print(tokenizer.mode)  # "char"
```
