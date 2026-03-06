# CircuitLM Documentation

A **finite-state circuit language model** trained with OR-Tools CP-SAT, featuring zero floating-point arithmetic and integer-only computations throughout.

## Overview

CircuitLM is a language modeling system that uses finite-state machines (FSM), pushdown automata (PDA), or Prediction by Partial Matching (PPM) as its underlying architecture. All computations use integer arithmetic only—no floats, no numpy/torch/jax, no softmax, no temperature scaling.

## Key Features

- **Integer-only arithmetic**: All computations use Python integers, avoiding floating-point entirely
- **Multiple architectures**: FSM (finite-state machine), PDA (pushdown automaton), and PPM (prediction by partial matching)
- **CP-SAT optimization**: Uses OR-Tools CP-SAT solver for learning optimal state assignments and emissions
- **Hard constraints**: Zero floats enforced by automated tests

## Quick Links

- [Installation](installation.md)
- [Architecture Overview](architecture.md)
- [CLI Reference](cli.md)
- [API Reference](api/)
- [Examples](examples/)
- [Benchmarks](benchmarks.md)

## Model Types

| Model | Memory | Best For |
|-------|--------|----------|
| FSM | O(log num_states) | Simple sequential patterns |
| PDA | O(D × log V) | Nested structures, brackets, scope |
| PPM | O(K × V) | Variable-order n-grams |

Where D = stack depth, V = vocabulary size, K = number of distinct contexts

## Performance

On the depth-generalization benchmark:
- **PDA-2ph**: 57.21% at depth 8 (generalizes beyond training depth)
- **FSM**: 42.78% at depth 8 (degrades with depth)
- **PPM**: 39.05% at depth 8 (fastest degradation)

See [STATUS.md](../STATUS.md) for detailed benchmarks.

## Table of Contents

### Getting Started
- [Installation Guide](installation.md)
- [Quick Start Tutorial](quickstart.md)

### Core Concepts
- [Architecture Overview](architecture.md)
- [Integer Arithmetic](integer_arithmetic.md)
- [CP-SAT Solver](cpsat_solver.md)

### Models
- [FSM Circuit](models/fsm.md)
- [PDA Circuit](models/pda.md)
- [PPM Model](models/ppm.md)

### Training
- [Training FSM](training/fsm.md)
- [Training PDA](training/pda.md)
- [Training PPM](training/ppm.md)
- [Joint Training](training/joint.md)

### API Reference
- [CircuitLM API](api/circuits.md)
- [PDA API](api/pda.md)
- [PPM API](api/ppm.md)
- [Tokenizer API](api/tokenizer.md)
- [Training API](api/training.md)
- [Evaluation API](api/evaluation.md)
- [Inference API](api/inference.md)
- [I/O API](api/io.md)

### CLI Reference
- [CLI Overview](cli.md)
- [Train Command](cli/train.md)
- [Eval Command](cli/eval.md)
- [Sample Command](cli/sample.md)

### Advanced Topics
- [Model Serialization](advanced/serialization.md)
- [Constraint Satisfaction](advanced/constraints.md)
- [Performance Tuning](advanced/performance.md)
- [Extending CircuitLM](advanced/extending.md)

### Resources
- [Benchmark Scripts](benchmarks.md)
- [Test Suite](tests.md)
- [Troubleshooting](troubleshooting.md)
