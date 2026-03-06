# I/O API

CircuitLM provides functions for saving and loading models.

## Functions

### save_model

Save a model to JSON.

```python
from circuit_lm.io import save_model
```

```python
def save_model(
    model: CircuitLM | PDACircuitLM | PPMModel,
    tokenizer: Tokenizer,
    path: str | pathlib.Path,
) -> None
```

**Parameters:**
- `model` (CircuitLM | PDACircuitLM | PPMModel): Trained model
- `tokenizer` (Tokenizer): Tokenizer
- `path` (str | pathlib.Path): Output path

### load_model

Load a model from JSON.

```python
from circuit_lm.io import load_model
```

```python
def load_model(path: str | pathlib.Path) -> tuple[AnyModel, Tokenizer]
```

**Parameters:**
- `path` (str | pathlib.Path): Path to model JSON

**Returns:**
- tuple[AnyModel, Tokenizer]: (model, tokenizer)

### save_msgpack

Save a model to MessagePack (compressed binary format).

```python
from circuit_lm.io import save_msgpack
```

```python
def save_msgpack(
    model: CircuitLM | PDACircuitLM | PPMModel,
    tokenizer: Tokenizer,
    path: str | pathlib.Path,
) -> None
```

**Note:** Requires `msgpack` package.

### load_msgpack

Load a model from MessagePack.

```python
from circuit_lm.io import load_msgpack
```

```python
def load_msgpack(path: str | pathlib.Path) -> tuple[AnyModel, Tokenizer]
```

**Parameters:**
- `path` (str | pathlib.Path): Path to MessagePack file

**Returns:**
- tuple[AnyModel, Tokenizer]: (model, tokenizer)

### has_msgpack

Check if MessagePack support is available.

```python
from circuit_lm.io import has_msgpack
```

```python
def has_msgpack() -> bool
```

## File Formats

### JSON Format

**FSM:**
```json
{
  "model_type": "fsm",
  "vocab_size": 128,
  "num_states": 16,
  "state_bits": 4,
  "transitions": { "0,65": 3, "3,66": 7 },
  "state_counts": { "0": [0, 5, 3], "3": [1, 2, 4] },
  "pred_tokens": { "0": 2, "3": 17 },
  "tokenizer": { "mode": "char", "chars": ["<PAD>", "<UNK>", "e", "t"] }
}
```

**PDA:**
```json
{
  "model_type": "pda",
  "vocab_size": 128,
  "num_states": 16,
  "state_bits": 4,
  "stack_depth": 4,
  "push_configs": [[0, 40, -1], [1, 40, -1]],
  "pop_configs": [[0, 41, 40], [1, 41, 40]],
  "transitions": {},
  "config_counts": { "3,-1": [0, 2, 9], "7,40": [1, 0, 4] },
  "config_pred_tokens": { "3,-1": 2, "7,40": 9 },
  "tokenizer": { "mode": "char", "chars": ["<PAD>", "<UNK>", "e", "t"] }
}
```

**PPM:**
```json
{
  "model_type": "ppm",
  "vocab_size": 128,
  "order": 4,
  "counts": { "": [0, 5, 3], "101,116": [1, 2, 0] },
  "tokenizer": { "mode": "char", "chars": ["<PAD>", "<UNK>", "e", "t"] }
}
```

### MessagePack Format

Binary format with integer-key encoding for compact storage.

## Usage Example

```python
from circuit_lm.io import save_model, load_model

# Save model
save_model(model, tokenizer, "model.json")

# Load model
model, tokenizer = load_model("model.json")

# Check model type
if hasattr(model, 'stack_depth'):
    print("PDA model with stack depth", model.stack_depth)
elif hasattr(model, 'order'):
    print("PPM model with order", model.order)
else:
    print("FSM model with", model.num_states, "states")
```

## Migration

Old model files without `"model_type"` field are automatically treated as FSM.
Files with `"push_tokens"`/`"pop_tokens"` (integer lists) are automatically migrated to the new `"push_configs"`/`"pop_configs"` format.
