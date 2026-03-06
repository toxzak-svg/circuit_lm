# Quick Start Guide

This guide will help you get started with CircuitLM in minutes.

## Installation

```bash
pip install -e ".[dev]"
```

Verify the installation:
```bash
pytest
```

## Step 1: Prepare Your Data

Create a simple text file for training:

```bash
echo "hello world hello again hello world" > data.txt
```

## Step 2: Train a Model

### Using the CLI

Train an FSM model:

```bash
circuit-lm train --data data.txt --out model.json --state_bits 4
```

Train a PDA model (for nested structures):

```bash
circuit-lm train --data data.txt --out pda_model.json --automaton pda --stack_depth 4
```

Train a PPM model:

```bash
circuit-lm train --data data.txt --out ppm_model.json --automaton ppm --order 4
```

### Using the Python API

```python
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.data import load_sequences
from circuit_lm.train_cpsat import train
from circuit_lm.io import save_model

# Prepare data
with open("data.txt") as f:
    text = f.read()

tokenizer = Tokenizer.from_text(text, vocab_size=128, mode="char")
sequences = load_sequences("data.txt", tokenizer)

# Train model
model = train(
    sequences=sequences,
    vocab_size=tokenizer.vocab_size,
    state_bits=4,
    steps=10,
)

# Save model
save_model(model, tokenizer, "model.json")
```

## Step 3: Evaluate

### CLI

```bash
circuit-lm eval --data data.txt --model model.json
```

### Python API

```python
from circuit_lm.eval import evaluate_any
from circuit_lm.io import load_model
from circuit_lm.data import load_sequences

# Load model
model, tokenizer = load_model("model.json")

# Evaluate
sequences = load_sequences("data.txt", tokenizer)
results = evaluate_any(model, sequences, per_token=True)

print(f"Correct: {results['correct']}, Total: {results['total']}")
```

## Step 4: Generate Text

### CLI

```bash
circuit-lm sample --prompt "hello" --model model.json --max_tokens 20
```

### Python API

```python
from circuit_lm.infer import decode_sample
from circuit_lm.io import load_model

# Load model
model, tokenizer = load_model("model.json")

# Generate
prompt = "hello"
prompt_ids = tokenizer.encode(prompt)

output_ids = decode_sample(
    model, 
    prompt_ids, 
    max_tokens=20,
    seed=42,
    top_k=10,
)

output = tokenizer.decode(output_ids)
print(output)
```

## Complete Example

Here's a complete end-to-end example:

```python
# Complete example script
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.data import load_sequences
from circuit_lm.train_cpsat import train
from circuit_lm.eval import evaluate_any
from circuit_lm.infer import decode_sample
from circuit_lm.io import save_model, load_model
from circuit_lm.metrics import format_accuracy

# Sample text
text = """
The quick brown fox jumps over the lazy dog.
The quick brown fox jumps over the lazy dog.
Pack my box with five dozen liquor jugs.
"""

# Save sample data
with open("sample.txt", "w") as f:
    f.write(text)

# Create tokenizer
tokenizer = Tokenizer.from_text(text, vocab_size=64, mode="char")

# Load sequences
sequences = load_sequences("sample.txt", tokenizer)
print(f"Loaded {len(sequences)} sequences")

# Train FSM model
model = train(
    sequences=sequences,
    vocab_size=tokenizer.vocab_size,
    state_bits=4,
    steps=10,
)

# Evaluate
results = evaluate_any(model, sequences)
print(f"Accuracy: {format_accuracy(results['correct'], results['total'])}")

# Save model
save_model(model, tokenizer, "sample_model.json")

# Generate text
prompt_ids = tokenizer.encode("The")
output_ids = decode_sample(model, prompt_ids, max_tokens=50, seed=42)
generated = tokenizer.decode(output_ids)
print(f"Generated: {generated}")
```

## Next Steps

- Explore different model types: FSM, PDA, PPM
- Adjust hyperparameters: `state_bits`, `stack_depth`, `order`
- Try BPE tokenization: `--tokenizer bpe`
- Run benchmarks: `python scripts/benchmark_small.py`

See the [Architecture Overview](architecture.md) for more details on the model types and their tradeoffs.
