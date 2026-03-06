# Installation Guide

## Requirements

- Python 3.10+
- OR-Tools 9.8+

## Install from Source

```bash
cd circuit_lm
pip install -e ".[dev]"
```

This installs the package in editable mode along with development dependencies (pytest).

## Verify Installation

Run the test suite to verify the installation:

```bash
pytest
```

You should see all tests pass. The test suite includes:
- `test_forbidden_imports` - Runtime check for forbidden imports (numpy, torch, etc.)
- `test_no_floats` - Static regex scan for floating-point literals
- `test_circuit_eval` - Integration and unit tests
- `test_pda`, `test_ppm`, `test_joint_pda` - Model-specific tests

## Optional Dependencies

### MessagePack Support

For compressed binary model format support:

```bash
pip install msgpack
```

### Development Tools

For running benchmarks and experiments:

```bash
pip install -e ".[dev]"
```

## Hard Constraints

CircuitLM enforces strict constraints throughout:

| Constraint | Enforcement |
|------------|--------------|
| No floating-point arithmetic | `tests/test_no_floats.py` — static regex scan |
| No numpy / torch / jax / scipy / tensorflow | `tests/test_forbidden_imports.py` — runtime check |
| No tensor / matmul | No such imports anywhere in the package |
| Solver: OR-Tools CP-SAT only | `pyproject.toml` dependency |

These constraints are verified by automated tests and must pass for the package to be considered valid.

## Directory Structure

After installation, you'll have:

```
circuit_lm/
├── circuit_lm/          # Main package
│   ├── circuits.py      # FSM implementation
│   ├── pda.py           # PDA implementation
│   ├── ppm.py           # PPM implementation
│   ├── tokenizer.py     # Tokenizer
│   ├── train_cpsat.py   # FSM training
│   ├── train_pda_cpsat.py  # PDA training
│   ├── eval.py          # Evaluation
│   ├── infer.py         # Inference
│   ├── io.py            # Serialization
│   ├── metrics.py       # Metrics
│   ├── data.py          # Data loading
│   └── cli.py           # Command-line interface
├── docs/                # Documentation
├── scripts/             # Benchmark and experiment scripts
└── tests/               # Test suite
```
