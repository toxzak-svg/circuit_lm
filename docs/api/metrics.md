# Metrics API

CircuitLM provides integer-only metrics for evaluating model performance.

## Functions

### accuracy_fraction

Return accuracy as a reduced integer fraction.

```python
from circuit_lm.metrics import accuracy_fraction
```

```python
def accuracy_fraction(correct: int, total: int) -> tuple[int, int]
```

**Parameters:**
- `correct` (int): Number of correct predictions
- `total` (int): Total number of predictions

**Returns:**
- tuple[int, int]: Reduced fraction (numerator, denominator)

**Example:**
```python
accuracy_fraction(1, 4)  # Returns (1, 4)
accuracy_fraction(2, 4)  # Returns (1, 2)
accuracy_fraction(0, 0)  # Returns (0, 1)
```

### accuracy_pct_times100

Return accuracy in integer basis-points (hundredths of a percent).

```python
from circuit_lm.metrics import accuracy_pct_times100
```

```python
def accuracy_pct_times100(correct: int, total: int) -> int
```

**Parameters:**
- `correct` (int): Number of correct predictions
- `total` (int): Total number of predictions

**Returns:**
- int: Accuracy in basis-points (1/100 of a percent)

**Example:**
```python
accuracy_pct_times100(1, 4)   # Returns 2500 (25%)
accuracy_pct_times100(3, 3)   # Returns 10000 (100%)
accuracy_pct_times100(0, 0)   # Returns 0
```

### format_accuracy

Format accuracy as a percentage string.

```python
from circuit_lm.metrics import format_accuracy
```

```python
def format_accuracy(correct: int, total: int) -> str
```

**Parameters:**
- `correct` (int): Number of correct predictions
- `total` (int): Total number of predictions

**Returns:**
- str: Formatted as "XX.YY%"

**Example:**
```python
format_accuracy(1, 4)   # Returns "25.00%"
format_accuracy(3, 4)   # Returns "75.00%"
format_accuracy(1, 3)   # Returns "33.33%"
format_accuracy(0, 0)  # Returns "N/A (0 samples)"
```

## Basis-Point Encoding

CircuitLM uses basis-points (bp) to represent accuracy without floating-point:

| Accuracy | Basis Points |
|----------|--------------|
| 100%     | 10000 bp     |
| 50%      | 5000 bp      |
| 25%      | 2500 bp      |
| 1%       | 100 bp       |
| 0%       | 0 bp         |

Conversion:
```python
bps = (correct * 10000) // total
```

## Integer Arithmetic

All metrics use only integer arithmetic:

- **No floats**: Division is integer floor division (`//`)
- **GCD reduction**: Fractions are reduced using Euclidean GCD
- **No logarithms**: Perplexity not directly supported (TODO)

## Usage Example

```python
from circuit_lm.metrics import (
    accuracy_fraction,
    accuracy_pct_times100,
    format_accuracy
)

# Suppose we have results
correct = 75
total = 100

# Get fraction
num, den = accuracy_fraction(correct, total)
print(f"{num}/{den}")  # "3/4"

# Get basis-points
bps = accuracy_pct_times100(correct, total)
print(f"{bps} bp")  # "7500 bp"

# Format as percentage
print(format_accuracy(correct, total))  # "75.00%"
```
