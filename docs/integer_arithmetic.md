# Integer Arithmetic

CircuitLM is designed to work entirely with integer arithmetic—no floating-point operations anywhere in the codebase.

## Why Integer-Only?

There are several motivations for avoiding floating-point:

1. **Determinism**: Integer operations are exactly reproducible
2. **No precision issues**: No floating-point rounding errors
3. **Hardware simplicity**: Integer operations are faster on many platforms
4. **Verification**: Easier to verify correctness mathematically

## Constraints Enforced

CircuitLM enforces strict constraints:

| Constraint | Enforcement |
|------------|--------------|
| No floats | `tests/test_no_floats.py` — static regex scan |
| No numpy/torch/jax | `tests/test_forbidden_imports.py` — runtime check |
| No matmul | No such imports in package |

## Integer Representations

### Accuracy

Accuracy is represented as basis-points (1/100 of a percent):

```python
# 25% = 2500 basis-points
bps = (correct * 10000) // total
```

### Sampling

Integer-weighted random sampling without softmax:

```python
def _weighted_choice(weights: list[int], rng: random.Random) -> int:
    total = sum(weights)
    if total == 0:
        return rng.randrange(len(weights))
    r = rng.randrange(total)
    cumsum = 0
    for i, w in enumerate(weights):
        cumsum += w
        if r < cumsum:
            return i
    return len(weights) - 1
```

### Transitions

State transitions use deterministic hash fallback:

```python
def next_state(self, state: int, token: int) -> int:
    key = (state, token)
    if key in self.transitions:
        return self.transitions[key]
    # Deterministic fallback
    return (state * HASH_PRIME + token + 1) % self.num_states
```

Where `HASH_PRIME = 31` (a prime integer).

## Best Practices

When extending CircuitLM:

1. **Never use float literals**: Write `0` not `0.0`
2. **Use integer division**: `//` not `/`
3. **Avoid math module**: No `math.log`, `math.exp`, etc.
4. **Use integer weights**: Store counts as `int`, not `float`

## Testing

The test suite verifies integer-only constraints:

```bash
# Static scan for float literals
pytest tests/test_no_floats.py

# Runtime check for forbidden imports
pytest tests/test_forbidden_imports.py
```

## Performance

Integer operations are typically faster than floating-point:
- No special hardware needed
- No NaN/Inf handling
- Cache-friendly memory layout
