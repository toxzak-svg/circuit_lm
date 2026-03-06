# Evaluation API

CircuitLM provides evaluation functions for measuring next-token prediction accuracy.

## Functions

### evaluate

Evaluate next-token prediction accuracy for an FSM model.

```python
from circuit_lm.eval import evaluate
```

```python
def evaluate(
    model: CircuitLM,
    sequences: list[list[int]],
    per_token: bool = False,
) -> EvalResult
```

**Parameters:**
- `model` (CircuitLM): Trained FSM model
- `sequences` (list[list[int]]): List of integer token-ID sequences
- `per_token` (bool): Include per-gold-token breakdown

**Returns:**
- `EvalResult`: `{"correct": int, "total": int}` or with `"per_token"` if requested

### evaluate_pda

Evaluate next-token prediction accuracy for a PDA model.

```python
from circuit_lm.eval import evaluate_pda
```

```python
def evaluate_pda(
    model: PDACircuitLM,
    sequences: list[list[int]],
    per_token: bool = False,
) -> EvalResult
```

**Parameters:**
- `model` (PDACircuitLM): Trained PDA model
- `sequences` (list[list[int]]): List of integer token-ID sequences
- `per_token` (bool): Include per-gold-token breakdown

**Returns:**
- `EvalResult`: `{"correct": int, "total": int}` or with `"per_token"` if requested

### evaluate_ppm

Evaluate next-token prediction accuracy for a PPM model.

```python
from circuit_lm.eval import evaluate_ppm
```

```python
def evaluate_ppm(
    model: PPMModel,
    sequences: list[list[int]],
    per_token: bool = False,
) -> EvalResult
```

**Parameters:**
- `model` (PPMModel): Trained PPM model
- `sequences` (list[list[int]]): List of integer token-ID sequences
- `per_token` (bool): Include per-gold-token breakdown

**Returns:**
- `EvalResult`: `{"correct": int, "total": int}` or with `"per_token"` if requested

### evaluate_any

Evaluate any model type (FSM, PDA, or PPM).

```python
from circuit_lm.eval import evaluate_any
```

```python
def evaluate_any(
    model: CircuitLM | PDACircuitLM | PPMModel,
    sequences: list[list[int]],
    per_token: bool = False,
) -> EvalResult
```

**Parameters:**
- `model` (CircuitLM | PDACircuitLM | PPMModel): Trained model
- `sequences` (list[list[int]]): List of integer token-ID sequences
- `per_token` (bool): Include per-gold-token breakdown

**Returns:**
- `EvalResult`: `{"correct": int, "total": int}` or with `"per_token"` if requested

## Type Definitions

```python
PerTokenBreakdown = dict[int, dict[str, int]]
EvalResult = dict[str, int | PerTokenBreakdown]
```

## Return Format

### Basic Result
```python
{
    "correct": int,   # Number of correct predictions
    "total": int      # Total number of predictions
}
```

### With Per-Token Breakdown
```python
{
    "correct": int,
    "total": int,
    "per_token": {
        token_id: {
            "correct": int,
            "total": int
        }
    }
}
```

## Usage Example

```python
from circuit_lm.eval import evaluate_any
from circuit_lm.io import load_model
from circuit_lm.data import load_sequences
from circuit_lm.metrics import format_accuracy

# Load model and data
model, tokenizer = load_model("model.json")
sequences = load_sequences("test.txt", tokenizer)

# Evaluate
result = evaluate_any(model, sequences, per_token=True)

# Print results
correct = result["correct"]
total = result["total"]
print(f"Accuracy: {format_accuracy(correct, total)}")

# Per-token breakdown
if "per_token" in result:
    for token_id, stats in result["per_token"].items():
        print(f"Token {token_id}: {stats['correct']}/{stats['total']}")
```
