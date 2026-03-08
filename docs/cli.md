# CLI Reference

CircuitLM provides a command-line interface with four main commands: `train`, `eval`, `sample`, and `chat`. See [Chat](chat.md) for training and using a chat model.

## Basic Usage

```bash
circuit-lm <command> [options]
```

## Commands

### train

Train a new model from a text file.

```bash
circuit-lm train \
  --data data.txt \
  --out model.json \
  --vocab_size 128 \
  --tokenizer bpe \
  --bpe_merges 256 \
  --state_bits 4 \
  --transition_steps 12 \
  --emission_steps 18 \
  --refinement_rounds 1
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data` | PATH | required | Path to training text file (UTF-8) |
| `--out` | PATH | model.json | Output model JSON path |
| `--vocab_size` | int | 256 | Maximum vocabulary size |
| `--state_bits` | int | 4 | State width in bits; num_states = 2^S |
| `--steps` | int | 10 | Legacy CP-SAT time budget (seconds) |
| `--transition_steps` | int | null | Transition CP-SAT budget (seconds) |
| `--emission_steps` | int | null | Emission CP-SAT budget (seconds) |
| `--refinement_rounds` | int | 1 | Additional refinement rounds |
| `--context_len` | int | 4 | Context window length for state hashing |
| `--top_k_coverage` | int | 16 | Top-K token coverage constraint |
| `--tokenizer` | str | char | Tokenizer mode: char or bpe |
| `--bpe_merges` | int | 256 | Maximum BPE merge operations |
| `--automaton` | str | fsm | Automaton type: fsm, pda, or ppm |
| `--stack_depth` | int | 4 | Maximum stack depth for PDA |
| `--stack_steps` | int | null | PDA stack-policy CP-SAT budget |
| `--max_push` | int | 16 | Maximum number of PUSH tokens |
| `--max_pop` | int | 16 | Maximum number of POP tokens |
| `--top_k_pairs` | int | 256 | Top co-occurrence pairs for PDA Phase 1 |
| `--order` | int | 4 | Context order for PPM |

### eval

Evaluate next-token prediction accuracy.

```bash
circuit-lm eval --data data.txt --model model.json
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--data` | PATH | required | Path to evaluation text file |
| `--model` | PATH | model.json | Model JSON path |
| `--per_token` | flag | false | Print per-token accuracy breakdown |
| `--per_token_limit` | int | 20 | Max tokens to print in breakdown |

### sample

Generate text from a trained model.

```bash
circuit-lm sample \
  --prompt "Hello" \
  --model model.json \
  --max_tokens 64 \
  --seed 42 \
  --top_k 16 \
  --repeat_penalty_div 2 \
  --repeat_window 64
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--prompt` | TEXT | "" | Prompt string |
| `--model` | PATH | model.json | Model JSON path |
| `--max_tokens` | int | 64 | Number of tokens to generate |
| `--seed` | int | 42 | Integer random seed |
| `--top_k` | int | 0 | Keep top-K weights (0 disables) |
| `--repeat_penalty_div` | int | 1 | Divide repeated weights by D |
| `--repeat_window` | int | 0 | Penalize tokens in last N positions |

### chat

Interactive chat (User: / Assistant: format). The model must be trained on chat-style text; see [Chat](chat.md).

```bash
circuit-lm chat --model chat_model.json --max_tokens 128 --seed 42
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--model` | PATH | model.json | Model JSON path |
| `--max_tokens` | int | 128 | Max tokens per reply |
| `--seed` | int | 42 | Random seed |
| `--top_k` | int | 0 | Sampling top-k (0 disables) |
| `--repeat_penalty_div` | int | 1 | Repetition penalty divisor |
| `--repeat_window` | int | 0 | Repetition penalty window |

## Examples

### Train an FSM model

```bash
circuit-lm train --data data.txt --out fsm_model.json --state_bits 4
```

### Train a PDA model

```bash
circuit-lm train \
  --data data.txt \
  --out pda_model.json \
  --automaton pda \
  --state_bits 4 \
  --stack_depth 4
```

### Train a PPM model

```bash
circuit-lm train \
  --data data.txt \
  --out ppm_model.json \
  --automaton ppm \
  --order 6
```

### Evaluate with per-token breakdown

```bash
circuit-lm eval --data test.txt --model model.json --per_token --per_token_limit 10
```

### Sample with top-k filtering

```bash
circuit-lm sample --prompt "The" --model model.json --max_tokens 100 --top_k 32
```
