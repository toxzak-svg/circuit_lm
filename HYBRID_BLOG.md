# Hybrid Circuit-Neural Language Model: A Proof of Concept

**Date:** March 2026  
**Author:** Zachary Maronek  
**Status:** Proof of Concept — Training on laptop CPU

---

## The Idea

What if language models didn't have to be massive neural networks? What if we could combine the interpretability of finite-state machines with the flexibility of neural networks?

That's what this project explores: a **hybrid architecture** that combines:
- **CircuitLM** — a finite-state machine (FSM/PDA) trained via CP-SAT constraint solving
- **Neural corrector** — a small neural network that learns to fix the FSM's mistakes

The circuit handles structure. The neural handles nuance.

---

## Why This Matters

1. **Interpretability** — The circuit is a finite-state machine. You can literally trace what state it's in.
2. **CPU-friendly** — The circuit runs on integers. No GPU required.
3. **Differentiable** — The neural can learn end-to-end.
4. **Novel** — Nobody else is building this (as far as I know).

---

## Results

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| CircuitLM only (PDA) | 16.51% | baseline |
| + Small Neural (10k examples) | 34.78% | 2.1x |
| + Big Neural (100k examples) | **56.53%** | **3.4x** |

All training done on **laptop CPU** in minutes.

---

## Architecture

```
Input: "Hello"
    │
    ▼
┌─────────────────────┐
│  CircuitLM (PDA)    │ ← Finite-state machine
│  - State tracking   │   Tracks (state, stack)
│  - Structure        │   Integer-only
└─────────────────────┘
    │
    ▼ [state, stack, context, histogram]
┌─────────────────────┐
│  Neural Corrector   │ ← Small MLP
│  - 2 conv layers   │   ~200K params
│  - 3 FC layers      │   Learns corrections
└─────────────────────┘
    │
    ▼
Blended logits → Sample next token
```

The key insight: The circuit is great at spaces (78% accuracy) but terrible at letters (1-33%). The neural learns to correct where the circuit fails.

---

## The Problem

The output is still garbled. Why?

1. **Vocabulary is tiny** — 64 characters (char-level tokenizer)
2. **Model is small** — ~200K params for neural corrector
3. **Data is limited** — ~2MB of chat data

This is a proof of concept, not a production model.

---

## What Needs to Scale

To make this actually useful:

1. **Bigger vocabulary** — 64 chars is too small for English
2. **More training data** — Millions of tokens, not hundreds of thousands  
3. **Larger neural** — Current is tiny; could be 10M+ params
4. **Better tokenizer** — BPE instead of character-level

---

## Code

The hybrid module is in `src/hybrid.py`:

```python
from src.hybrid import train_hybrid, HybridModel

# Train
train_hybrid(
    circuit_path='model.json',
    data_path='data.txt',
    output_path='corrector.pt',
    epochs=15,
)

# Use
model = HybridModel.load(circuit_path, corrector_path)
```

---

## Reproduce

**One command (PowerShell, from repo root):**

```powershell
.\scripts\reproduce_hybrid.ps1
```

This will: create a small `data.txt` if missing, train a PDA circuit, train the corrector, run eval and sample, and print a state trace. Then run interactive chat:

```powershell
circuit-lm chat --model model.json --corrector corrector.pt
```

**Manual steps:**

```powershell
# 1. Train circuit
circuit-lm train --data data.txt --out model.json --vocab_size 128 --state_bits 4 --automaton pda --transition_steps 5 --emission_steps 5

# 2. Train corrector (requires PyTorch; run from repo root)
circuit-lm hybrid-train --circuit model.json --data data.txt --out corrector.pt --epochs 3 --max-examples 50000

# 3. Eval and sample
circuit-lm eval --data data.txt --model model.json
circuit-lm sample --prompt "Hello" --model model.json

# 4. Trace state (interpretability)
circuit-lm trace --prompt "Hello" --model model.json --top_k 5

# 5. Chat with hybrid
circuit-lm chat --model model.json --corrector corrector.pt
```

---

## What's Next

1. **Scale up** — Bigger vocab, more data, larger network
2. **Specialize** — Train on code (brackets/nesting is PDA's strength)
3. **Blog post** — Write up the full story
4. **Demo** — Make it interactive

---

## The Vision

A language model that:
- Runs on CPU (no GPU needed)
- Is interpretable (you can see the state)
- Generalizes on structure (PDA > transformers on bracket matching)
- Is different from everything else

That's the goal. This is step 1.

---

## Links

- CircuitLM: `C:\dev\research\circuit_lm`
- This project: `src/hybrid.py`
- Trained models: `big_corrector.pt`

---

*Built by Zach Maronek. Self-taught ML researcher. Building in public.*
