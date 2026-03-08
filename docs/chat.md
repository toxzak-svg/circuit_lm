# Chat language model

CircuitLM can be used as a **chat model**: train on conversation text in a fixed format, then run an interactive chat session where the model generates assistant replies.

## Format

Conversations use plain text so the existing tokenizer works without extra special tokens:

```
User: hello
Assistant: hi there
User: what's the weather?
Assistant: I don't have weather data.
```

- **Training data**: A single text file containing many such blocks (one after another). The model learns to predict the next token after `User: ...` and after `Assistant: ...`.
- **Inference**: You prompt with `User: {message}\nAssistant: ` and the model generates the reply (one line by default, stopping at newline).

## Training a chat model

### Option 1: Preformatted text file

Create a file `chat.txt` with the exact format above (many `User:` / `Assistant:` blocks). Then train as usual:

```bash
circuit-lm train \
  --data chat.txt \
  --out chat_model.json \
  --vocab_size 256 \
  --state_bits 4 \
  --context_len 8 \
  --transition_steps 15 \
  --emission_steps 15
```

For chat, use a **larger `--context_len`** (e.g. 8–16) so the model conditions on more of the recent conversation. PPM with a higher `--order` is another option for longer context.

### Option 2: Convert JSONL to text

If you have chat data as JSONL, convert it first:

**Turn-based JSONL** (one turn per line):

```json
{"user": "hello", "assistant": "hi there"}
{"user": "what's the weather?", "assistant": "I don't have weather data."}
```

**Conversation JSONL** (one conversation per line):

```json
{"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]}
```

From Python:

```python
from circuit_lm.data import chat_text_from_jsonl

text = chat_text_from_jsonl("chats.jsonl")
pathlib.Path("chat.txt").write_text(text, encoding="utf-8")
```

Then train with `--data chat.txt` as above.

### Option 3: OpenAI / ChatGPT export (zip or folder)

If you have a ChatGPT “Export data” zip (or an extracted folder with `conversations-*.json`):

```bash
python scripts/chat_export_to_txt.py --input chat_export --output chat.txt
```

Or from a zip:

```bash
python scripts/chat_export_to_txt.py --input path/to/export.zip --output chat.txt
```

Then train with `--data chat.txt` as above.

## Running chat

Interactive chat:

```bash
circuit-lm chat --model chat_model.json --max_tokens 128 --seed 42
```

You’ll see:

```
[chat] model='chat_model.json'  max_tokens=128  seed=42
User: hello
Assistant: hi there

User: _
```

Type your message and press Enter; the model generates a reply and stops at the first newline. Then you can type again for multi-turn chat.

### Chat options

| Option | Default | Description |
|--------|--------|-------------|
| `--model` | model.json | Model JSON path |
| `--max_tokens` | 128 | Max tokens per reply |
| `--seed` | 42 | Random seed |
| `--top_k` | 0 | Sampling top-k (0 = no limit) |
| `--repeat_penalty_div` | 1 | Repetition penalty divisor |
| `--repeat_window` | 0 | Repetition penalty window |

## Making chat work well

### Training choices

| Goal | What to change |
|------|----------------|
| **More context** | FSM/PDA: use `--context_len 12` or `16` (default is 4). Or use **PPM** for longer n-gram context: `--automaton ppm --order 6` or `--order 8`. |
| **Stronger model** | Increase `--state_bits` (e.g. 5 or 6 → 32–64 states). Give the solver more time: `--transition_steps 20 --emission_steps 20` (or higher). |
| **Less repetition** | Train on diverse data; at inference use `--repeat_penalty_div 2` or `3` and `--repeat_window 32`. |
| **Longer replies** | At inference use `--max_tokens 256` (or more). |

**Recommended chat training (FSM, more context and steps):**

```bash
circuit-lm train --data chat.txt --out chat_model.json --vocab_size 256 --state_bits 5 --context_len 12 --transition_steps 20 --emission_steps 20
```

(Multi-line form for bash: use `\` at end of each line. In PowerShell use `` ` `` or run as one line.)

**Alternative: PPM** (no CP-SAT; good for longer context, faster training):

```bash
circuit-lm train --data chat.txt --out chat_ppm.json --vocab_size 256 --automaton ppm --order 6
```

**Recommended chat run** (fewer boring repeats, longer replies):

```bash
circuit-lm chat --model chat_model.json --max_tokens 256 --repeat_penalty_div 2 --repeat_window 32
```

## Context limits

- **FSM / PDA**: During training, state is derived from a **short context window** (e.g. `context_len=4` = last 4 tokens). So the model only “sees” a few tokens when predicting. For chat, increase `--context_len` (e.g. 8–16) so it can use more of the current turn.
- **Multi-turn**: The full conversation is sent as the prompt at inference, so the FSM state after the prompt does depend on the whole prompt. Training quality still depends on how well the state captures long context; larger `context_len` or **PPM with higher order** improve that.
- **Single-turn vs multi-turn**: Training on many short “User: X\nAssistant: Y\n” pairs is enough for single-turn reply style. For better multi-turn coherence, train on long conversations and use a larger context (or future long-context / temporal extensions).

## Programmatic use

```python
from circuit_lm.chat import (
    USER_PREFIX,
    ASSISTANT_PREFIX,
    build_chat_prompt,
    prompt_for_assistant_reply,
    generate_reply,
)
from circuit_lm.io import load_model

model, tokenizer = load_model("chat_model.json")
turns = [("user", "hello")]
prompt_str = prompt_for_assistant_reply(turns)   # "User: hello\nAssistant: "
prompt_ids = tokenizer.encode(prompt_str)
reply_ids = generate_reply(
    model, tokenizer, prompt_ids,
    max_tokens=128, seed=42,
)
reply_text = tokenizer.decode(reply_ids)
print(reply_text)
```
