# Inference API

CircuitLM provides inference functions for generating text from trained models.

## Greedy Decoding

### greedy_decode

Greedy decoding for FSM models.

```python
from circuit_lm.infer import greedy_decode
```

```python
def greedy_decode(
    model: CircuitLM,
    prompt_ids: list[int],
    max_tokens: int,
) -> list[int]
```

### pda_greedy_decode

Greedy decoding for PDA models.

```python
from circuit_lm.infer import pda_greedy_decode
```

```python
def pda_greedy_decode(
    model: PDACircuitLM,
    prompt_ids: list[int],
    max_tokens: int,
) -> list[int]
```

### ppm_greedy_decode

Greedy decoding for PPM models.

```python
from circuit_lm.infer import ppm_greedy_decode
```

```python
def ppm_greedy_decode(
    model: PPMModel,
    prompt_ids: list[int],
    max_tokens: int,
) -> list[int]
```

### decode_greedy

Unified greedy decoding for any model type.

```python
from circuit_lm.infer import decode_greedy
```

```python
def decode_greedy(
    model: CircuitLM | PDACircuitLM | PPMModel,
    prompt_ids: list[int],
    max_tokens: int,
) -> list[int]
```

## Sampling

### sample_tokens

Integer-weighted sampling for FSM models.

```python
from circuit_lm.infer import sample_tokens
```

```python
def sample_tokens(
    model: CircuitLM,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    top_k: int = 0,
    repeat_penalty_div: int = 1,
    repeat_window: int = 0,
) -> list[int]
```

**Parameters:**
- `model` (CircuitLM): Trained model
- `prompt_ids` (list[int]): Prompt token IDs
- `max_tokens` (int): Number of tokens to generate
- `seed` (int): Random seed
- `top_k` (int): Keep top-K weights (0 disables)
- `repeat_penalty_div` (int): Divide repeated weights (1 disables)
- `repeat_window` (int): Window for repetition penalty (0 = full history)

### pda_sample_tokens

Integer-weighted sampling for PDA models.

```python
from circuit_lm.infer import pda_sample_tokens
```

### ppm_sample_tokens

Integer-weighted sampling for PPM models.

```python
from circuit_lm.infer import ppm_sample_tokens
```

### decode_sample

Unified sampling for any model type.

```python
from circuit_lm.infer import decode_sample
```

```python
def decode_sample(
    model: CircuitLM | PDACircuitLM | PPMModel,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    top_k: int = 0,
    repeat_penalty_div: int = 1,
    repeat_window: int = 0,
) -> list[int]
```

## Integer-Weighted Sampling

All sampling uses integer arithmetic—no softmax or temperature:

1. Get integer count histogram `h[t]`
2. Apply optional repetition penalty
3. Apply optional top-k filtering
4. Draw uniform `r` in `[0, sum(h))`
5. Return index where cumulative sum exceeds `r`

## Usage Example

```python
from circuit_lm.infer import decode_greedy, decode_sample
from circuit_lm.io import load_model
from circuit_lm.tokenizer import Tokenizer

# Load model
model, tokenizer = load_model("model.json")

# Encode prompt
prompt = "Hello"
prompt_ids = tokenizer.encode(prompt)

# Greedy generation
output_ids = decode_greedy(model, prompt_ids, max_tokens=50)
output_text = tokenizer.decode(output_ids)

# Sampling with seed
output_ids = decode_sample(
    model, 
    prompt_ids, 
    max_tokens=50,
    seed=42,
    top_k=32,
    repeat_penalty_div=2,
    repeat_window=64
)
output_text = tokenizer.decode(output_ids)
```
