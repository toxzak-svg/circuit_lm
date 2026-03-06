# Benchmarks

CircuitLM includes several benchmark scripts for evaluating model performance and comparing different configurations.

## Running Benchmarks

### Small Benchmark

Basic smoke test and performance baseline:

```bash
python scripts/benchmark_small.py
```

Example output:
```
text_len=3105
effective vocab_size=30
sequences=13, total_tokens=3117
num_states=8
train_time=203ms
eval accuracy=20.84% (647/3104)
```

### Benchmark Matrix

Run a grid of experiments across different configurations:

```bash
python scripts/benchmark_matrix.py
```

Options:
```bash
# With CSV output
python scripts/benchmark_matrix.py --csv-out results.csv

# With TSV output
python scripts/benchmark_matrix.py --tsv-out results.tsv

# With snapshots
python scripts/benchmark_matrix.py --snapshot-dir ./snapshots --snapshot-prefix run1
```

The matrix explores:
- Tokenizer mode: `char`, `bpe`
- State bits: `2`, `3`, `4`
- CP-SAT steps: `1`, `2`, `5`

### Serialization Benchmark

Test JSON save/load performance:

```bash
python scripts/benchmark_serialization.py
```

Example output:
```
label     type  states  vocab     bytes  save_ms  load_ms  ok
fsm-sm    fsm        8     30      7020        1        0   1
pda-sm    pda        8     30    377610       23        6   1
pda-md    pda       16     30    739648       42       10   1
```

### Depth Generalization

Test model performance on sequences deeper than training:

```bash
python scripts/reproduce_depth_generalization.py
```

This tests models on sequences with nesting depth > training depth.

### Joint PDA Small Verification

Verify joint PDA can discover stack operations:

```bash
python scripts/verify_joint_pda_small.py
```

## Key Results

### Depth Generalization (2026-02-25)

Settings: train depth ≤ 3, 300 train seqs, 100 test seqs per depth, 20s CP-SAT budget

| depth | PDA-2ph | PDA-jt | FSM    | PPM    |
|-------|---------|--------|--------|--------|
| 3     | 44.66%  | 49.02% | 55.33% | 59.49% |
| 4*    | 50.66%  | 50.48% | 49.33% | 54.68% |
| 5*    | 50.60%  | 50.24% | 49.39% | 51.40% |
| 6*    | 53.61%  | 50.96% | 46.38% | 45.99% |
| 7*    | 55.09%  | 50.91% | 44.90% | 42.43% |
| 8*    | **57.21%** | 52.82% | 42.78% | 39.05% |

Key observations:
- **PDA-2ph**: Accuracy improves with depth (44.66% → 57.21%)
- **FSM/PPM**: Degrade with depth
- The stack encodes a depth-invariant feature

### Scalability Limits

| Solver | T_total max | num_states | vocab_size |
|--------|-------------|------------|------------|
| `train_joint_cpsat` | ≤ 4,000 | ≤ 16 | ≤ 64 |
| `train_joint_pda_cpsat` | ≤ 2,000 | ≤ 8 | ≤ 32 |

### Serialization Size

- FSM: ~7KB for small models
- PDA: ~378KB for small models (54× larger due to config_counts)

## Test Suite

Run the full test suite:

```bash
pytest
```

Key test modules:
- `test_forbidden_imports` - Runtime check for forbidden imports
- `test_no_floats` - Static scan for floating-point literals
- `test_circuit_eval` - Integration tests
- `test_pda`, `test_ppm`, `test_joint_pda` - Model-specific tests

## Performance Tips

1. **State bits**: More states = more memory, potentially better accuracy
2. **CP-SAT time budget**: More time = better solutions, but diminishing returns
3. **PDA stack depth**: Deeper stacks = more memory, better for deep nesting
4. **PPM order**: Higher order = more memory, better for long-range patterns
