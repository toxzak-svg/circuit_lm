# State Machine Transformer (SMT) — circuit_lm Architecture Spec

## Overview

SMT = circuit_lm's FSM/PDA logic extended to be a full base model architecture.
The FSM layer enforces structural constraints; the neural corrector handles nuance.
QAT from day one — FSM weights are small and quantize cleanly.

**Goal:** Build a 1-2B base model that:
- Runs on CPU at 2B params (Q4/Q5 quantization)
- Has no alignment baked in (architecture enforces behavior, not training)
- Is QAT-native (designed for quantization from the start)
- Achieves quality competitive with 3-4B models through structural efficiency

---

## Architecture

### Layer Structure (stack of N blocks)

```
Input tokens
    ↓
Embedding Layer (vocab → d_model)
    ↓
[Block 0..N-1] ×
    ├── FSM Constraint Layer (learned state machine)
    │   ├── state: d_model bit (binary or small int)
    │   ├── transition: learned per-token state update
    │   └── constraint: hard rules on valid state sequences
    ├── Sparse Attention (only attends to relevant tokens)
    │   └── [For CPU: fixed low-rank attention, ~O(n) not O(n²)]
    ├── Feed-Forward (small: d_model → d_ff → d_model, d_ff = d_model×2)
    └── LayerNorm
    ↓
Output Projection (d_model → vocab)
    ↓
Generated tokens
```

### FSM Constraint Layer — Detail

**State representation:** 
- Each token position has a learned state vector (d_model bits compressed to ~16-64 states)
- FSM transition function: `state_new = f(state_old, token_embedding)`
- Valid transitions are learned during training (not programmed)
- Invalid transitions get a penalty signal (hard constraint via loss)

**Constraint enforcement:**
- At each step, FSM predicts a "valid token class" (e.g., "verb", "entity", "punctuation")
- If neural output violates FSM constraint, FSM signal dominates
- This is where "alignment via architecture" happens — FSM learned to enforce structure

**Quantization advantage:**
- FSM transition matrix is small (states × vocab_size, e.g., 32 × 512 = 16K params)
- These compress extremely well under QAT
- The FSM carries the "structural intelligence" — transformer can be smaller

### Neural Corrector

**Role:** Handle what FSM can't express — nuance, context, novel situations.

**Architecture:**
- Standard transformer attention + FFN
- Receives FSM state as an extra bias term at each layer
- Output = FSM_signal × α + transformer_output × (1-α), where α is FSM confidence

**Training:**
- Two-phase: (1) train FSM alone to learn valid state sequences, (2) train corrector
- Or joint training with FSM loss as auxiliary signal

### QAT Strategy

**Why SMT is QAT-friendly:**
1. FSM weights are structurally small — easy to quantize without accuracy loss
2. FSM operates on discrete states — naturally robust to low-precision
3. Corrector network can be aggressively quantized (Q4_K_M) while FSM stays Q6
4. The hard constraint architecture means model is robust to noise

**Implementation:**
- Use standard QAT framework (PyTorch FX + quantize_fx)
- Train with fake quantization in the loop
- FSM layers: Q6 quantization (preserve state precision)
- Transformer layers: Q4_K_M quantization
- Final precision: aim for Q4 with <1% quality loss vs FP16

---

## Training

### Dataset Requirements

For a 1B model to be genuinely useful (not just a demo):
- Minimum: 10B tokens (quality fine-tune) or 1T tokens (full pre-train)
- For circuit_lm SMT specifically: structured text (code, math, technical)
- FSM training: sequence pairs where structural validity is known

### Compute Requirements

| Hardware | Pre-train (1T tokens) | Fine-tune (10B tokens) |
|----------|----------------------|------------------------|
| A100 80GB | ~5-7 days | ~4-8 hours |
| T4 15GB (Kaggle) | ~40-60 days | ~1-2 days |
| CPU (32GB RAM) | not feasible | ~3-5 days (1B model) |

### Training Stack

- Framework: PyTorch + HuggingFace Transformers
- QAT: PyTorch Quantization (eager mode with FX graph)
- Checkpointing: llama.cpp compatible format at end
- Infrastructure: Kaggle free GPU for fine-tuning; vast.ai for pre-train

---

## GGUF Conversion

SMT → GGUF requires implementing a custom model type in llama.cpp:

**Option A — Implement custom model type:**
- Add `MODEL_SMT` to llama.cpp model definitions
- Implement forward pass that mirrors the FSM + corrector flow
- Significant C++ work but produces first-class GGUF model

**Option B — Compile SMT into transformer weights:**
- "Distill" the FSM behavior into additional transformer weights
- Result: standard transformer that behaves like SMT
- Easier GGUF path, but loses architectural advantage

**Option C — Custom ops in llama.cpp:**
- Implement FSM as a custom op that llama.cpp can call
- Keeps architecture clean, but requires more integration work

**Recommendation:** Option A — invest the C++ work once, get proper GGUF support.

---

## Roadmap

- [ ] Phase 1: Design SMT spec (this document) ✅
- [ ] Phase 2: Prototype SMT in PyTorch (toy scale, 50M params)
- [ ] Phase 3: Train FSM alone on structural task (PDA grammar)
- [ ] Phase 4: Train corrector + FSM jointly
- [ ] Phase 5: QAT implementation (PyTorch FX quantization)
- [ ] Phase 6: Evaluate vs baseline (TinyLlama, Qwen2.5-1.5B)
- [ ] Phase 7: GGUF conversion (llama.cpp custom model type)
- [ ] Phase 8: CPU inference optimization

---

## Key Questions to Resolve

1. **FSM state cardinality** — how many states? (16? 32? 64?) Affects FSM param count and expressiveness
2. **FSM training signal** — supervised (grammar rules) or self-supervised (reconstruct valid sequences)?
3. **Attention mechanism** — full attention (O(n²)) vs sparse vs linear? For CPU, linear attention is likely necessary
4. **Vocab size** — smaller vocab = more compression. 16K-32K reasonable for English.
5. **Context length** — target (512? 1024? 2048?) Affects memory and defines use case

---

## Related Work

- **DeltaNet / Linear Transformers** — linearized attention for O(n) complexity
- **State Space Models (Mamba)** — recurrent architecture with selection mechanism
- **GSS (Gated State Space)** — compression-focused recurrence
- **CRF (Conditional Random Fields)** — structured prediction with constraints
- **RLCD (Reinforcement Learning from Constraints)** — training with hard constraints

SMT combines ideas from all of these in a novel way specific to the circuit_lm goal.