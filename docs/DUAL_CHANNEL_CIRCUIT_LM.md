# Dual-Channel CircuitLM Architecture Spec

**Status:** Updated 2026-04-26
**Date:** 2026-04-15
**Goal:** 3M–10M parameter hybrid that beats plain tiny decoders at the same budget

---

## Core Idea

A dual-channel language model where:

1. **Symbolic channel** (PDA/FSM circuit) — discrete, integer-only state machine that tracks structural patterns. Zero floats. Inferred via CP-SAT.
2. **Neural corrector** — small MLP with SSD context encoder that handles distributional fluency and surface phrasing.
3. **Gated delta fusion** — learns per-state gate controlling how much the corrector delta modifies the circuit histogram.

The key claim: same parameter budget, better groundedness and controllability than a plain tiny LM.

---

## Architecture

```
Input: token stream
  │
  ├──► SYMBOLIC CHANNEL ──────────────────────────────────────
  │       │
  │       ├── PDA/FSM circuit (integer-only, CP-SAT trained)
  │       │   • Tracks structural patterns in token sequences
  │       │   • Emits: circuit_state, stack_top, histogram over vocab
  │       │
  │       └── State embeddings: Embed(state, embed_dim)
  │           Stack embeddings: Embed(stack_top, embed_dim)
  │           Histogram projection: Linear(vocab_size, embed_dim)
  │
  ├──► NEURAL CORRECTOR ──────────────────────────────────────
  │       │
  │       ├── StackedSSDContext (2-3 layers)
  │       │   • SSD: h_{t+1} = A @ h_t + B @ x_t (no gates)
  │       │   • Orthogonal init scaled by 1/sqrt(embed_dim)
  │       │   • Sequential: each layer's final h → next layer's initial
  │       │
  │       └── 2-layer MLP with gated delta output
  │           ├── LayerNorm on each input (state, stack, context, histogram)
  │           ├── combined = concat(state_emb, stack_emb, context_enc, hist_emb)
  │           ├── h1 = SiLU(W1 @ combined_norm); h1 = h1 + LayerNorm(h1)
  │           ├── h2 = SiLU(W2 @ h1); h2 = h2 + LayerNorm(h2)
  │           ├── delta = W_delta @ h2  (vocab_size logits)
  │           └── gate = sigmoid(gate_param[state])  (per-state scalar)
  │
  └──────► GATED DELTA FUSION ────────────────────────────────
              │
              ├── final_logits = circuit_histogram + gate * delta
              │   • gate ∈ (0,1) per state, init near 0 (sigmoid ≈ 0.5)
              │   • Circuit histogram is strong prior — gate controls override
              │
              └── CrossEntropyLoss on final_logits

Output: next token distribution (over vocab)
```

---

## Corrector Architecture Details

### Input Encoding

Each input is separately projected and LayerNormed before concatenation:

| Input | Shape | Projection |
|-------|-------|------------|
| circuit_state | (batch,) | Embed(num_states=4096, 256) |
| stack_top | (batch,) | Embed(stack_depth=5, 256) |
| context | (batch, max_ctx=32) | TokenEmbed → StackedSSD → 256 |
| histogram | (batch, vocab) | Linear(vocab, 256) |

Combined input: 4 × 256 = 1024 dimensions.

### Stacked SSD Context Encoder

SSD recurrence (no gates, two learnable matrices A and B):
```
h_{t+1} = A @ h_t + B @ token_embed(t)
h = h / (norm(h) + 1e-8)  # normalize for stability
```

Each layer independently processes the full sequence, with orthogonal initialization scaled by `1/sqrt(embed_dim)`. Multiple layers are applied sequentially — each layer's final hidden state becomes the initial state for the next layer.

### MLP with Residuals

```python
# Layer 1: 1024 → 512 with SiLU + residual + LayerNorm
h1 = SiLU(W1 @ combined) + LayerNorm(SiLU(W1 @ combined))

# Layer 2: 512 → 512 with SiLU + residual + LayerNorm
h2 = SiLU(W2 @ h1) + LayerNorm(SiLU(W2 @ h1))

# Delta output: 512 → vocab_size
delta = W_delta @ h2
```

### Gated Delta

```python
gate = sigmoid(gate_param[state])  # one scalar per state, init ~0
final_logits = circuit_histogram + gate * delta
```

Gate initialization near 0 (σ ≈ 0.5) ensures both circuit and corrector contribute equally at start. The circuit histogram is a strong prior — the gate learns how much the corrector should override it per-state.

---

## State Definitions

### Symbolic State (PDA Circuit)

The circuit tracks these as discrete state variables:

```
SLOT_TYPES:
  - USER_NAME: extracted from "I'm X" or conversation context
  - CURRENT_INTENT: one of [CODING, RESEARCH, EMOTIONAL, CASUAL, PLANNING, CREATIVE, IDENTITY, META]
  - TOPIC_ENTITY: what the conversation is about
  - EMOTIONAL_STATE: [POSITIVE, NEUTRAL, NEGATIVE, UNKNOWN]
  - GOAL_STACK: nested goals/subgoals (push on intent, pop on completion)
  - CONTEXT_SUMMARY: compressed summary of last N turns (hash fingerprint)

TRANSITIONS:
  - On USER_MESSAGE: extract entities, update slots, push if new goal
  - On ASSISTANT_MESSAGE: check goal completion, emit response
  - On SYSTEM_HINT: update emotional state or intent
```

### Text State (GRU)

```
INPUT: last 32-64 tokens (BPE or char-level)
HIDDEN: 128-256 dimensions
OUTPUT: text_hidden vector per position
```

---

## Parameter Budgets

### Tiny (3M params)

| Component | Params | Config |
|-----------|--------|--------|
| PDA Circuit | 0 (integer) | 16 states, stack_depth=2 |
| Symbol embeddings | ~2K | 16 states → 32 dim |
| GRU encoder | ~400K | 1 layer, hidden=128, input=vocab |
| Neural corrector | ~1.2M | embed_dim=64, hidden=128, 2 layers |
| Fusion gate | ~100K | Linear(192 → 1) + output proj |
| **Total** | **~1.7M** | |

### Small (5M params)

| Component | Params | Config |
|-----------|--------|--------|
| PDA Circuit | 0 (integer) | 32 states, stack_depth=3 |
| Symbol embeddings | ~4K | 32 states → 64 dim |
| GRU encoder | ~600K | 1 layer, hidden=192, input=vocab |
| Neural corrector | ~3.5M | embed_dim=128, hidden=256, 3 layers |
| Fusion gate | ~200K | Linear(320 → 1) + output proj |
| **Total** | **~4.3M** | |

### Medium (10M params)

| Component | Params | Config |
|-----------|--------|--------|
| PDA Circuit | 0 (integer) | 64 states, stack_depth=4 |
| Symbol embeddings | ~8K | 64 states → 64 dim |
| GRU encoder | ~1.2M | 2 layers, hidden=256, input=vocab |
| Neural corrector | ~8M | embed_dim=256, hidden=512, 4 layers |
| Fusion gate | ~400K | Linear(320 → 1) + output proj |
| **Total** | **~9.6M** | |

---

## Losses

### 1. Standard Cross-Entropy Loss on Gated Delta
```
final_logits = circuit_histogram + gate * delta
L_ce = CrossEntropyLoss(final_logits, target_token)
```

The gate is part of the computational graph — gradients flow through `gate_param[state]` and `delta` to all corrector weights.

### 2. Backprop Path
```
target ← CrossEntropyLoss ← final_logits = circuit_hist + gate * delta
                                              ↑
                              delta ← W_delta ← MLP layers
                                              ↑
                              combined ← [state_emb, stack_emb, context_enc, hist_emb]
                                              ↑
                              StackedSSDContext ← token_embed ← context
```

Backprop flows through: `ssd.forward()` → hidden_proj (W1, W2) → delta_proj (W_delta) → gate.

### 3. Gate Learning
Gate is a learned scalar per state (`gate_param[state]`), initialized near 0 so sigmoid ≈ 0.5:
- Gradient pushes gate higher when corrector delta is helpful
- Gradient pushes gate lower when delta hurts accuracy
- Circuit histogram is a strong prior — gate learns per-state how much to let corrector override

---

## Training Loop

### Phase 1: Train PDA/FSM Circuit (CP-SAT, integer-only)
```
Input: token sequences
Output: circuit.json (states, transitions, emission histograms)
Method: OR-Tools CP-SAT over:
  - state_bits (S states)
  - stack_depth (D)
  - push/pop/noop operations per token
  - emission distributions per (state, stack_top) config
Objective: maximize next-token accuracy on training data
```

### Phase 2: Train CorrectedCorrector (neural, full backprop)
```
Fix circuit.json
Train corrector on: circuit_state + stack_top + context + circuit_histogram → next_token
Loss: CrossEntropyLoss(circuit_histogram + gate * delta, target)
Backprop: ssd → hidden_proj → delta_proj → gate
Optimizer: Adam, lr=1e-3, batch=64
```

### Phase 3: Finetune on Personal Data
```
Use personal conversations
Narrow domain: personal AI assistant style/personality/memory patterns
Target: improve accuracy on personal data distribution
```

---

## Parameter Budgets

### Tiny (2-3M params)

| Component | Params | Config |
|-----------|--------|--------|
| PDA Circuit | 0 (integer) | 16 states, stack_depth=2 |
| CorrectedCorrector | ~2M | embed_dim=256, hidden_dim=512, 2 MLP layers |
| **Total** | **~2M** | |

### Small (5M params)

| Component | Params | Config |
|-----------|--------|--------|
| PDA/FSM Circuit | 0 (integer) | 32 states, stack_depth=3 |
| CorrectedCorrector | ~4M | embed_dim=256, hidden_dim=512, 2 layers, 2 SSD |
| **Total** | **~4M** | |

### Medium (10M params)

| Component | Params | Config |
|-----------|--------|--------|
| PDA/FSM Circuit | 0 (integer) | 64 states, stack_depth=4 |
| CorrectedCorrector | ~8M | embed_dim=256, hidden_dim=512, 3 MLP layers, 3 SSD |
| **Total** | **~8M** | |

---

## Data Requirements

### Phase 1 (CP-SAT circuit):
- Any text corpus — circuit learns structure, not semantics
- Personal corpus preferred (Zach's 1,758 convos = 6MB)
- ~10K-50K tokens sufficient for small circuits

### Phase 2-3 (neural training):
- Same corpus with circuit state precomputed
- ~100K tokens for corrector warmup
- ~500K tokens for full training

### Phase 4 (finetune):
- Zach's personal conversations (already have this)
- Target: voice/style/personality learning

---

## Baselines to Beat

| Model | Params | Task Accuracy |
|-------|--------|---------------|
| TinyLlama 1.1B | 1.1B | baseline (far above) |
| Bonsai-8B Q1_0 | 8B (Q1) | target |
| circuit_lm tiny (current) | ~1.5M | current circuit-only |
| **dual_channel tiny (this spec)** | **~1.7M** | **target: beat circuit-only** |
| **dual_channel small (this spec)** | **~4.3M** | **target: match Bonsai at personality tasks** |

---

## Novelty Over Standard circuit_lm

Current circuit_lm:
- Circuit histogram directly predicts next token
- Neural corrector blends with circuit via fixed weight (0.5)
- No learned gating based on confidence

This spec adds:
1. **Learned fusion gate** — not fixed weight, trained to decide per-token
2. **Text encoder (GRU)** — explicit context modeling separate from circuit state
3. **Circuit confidence loss** — trains gate to distrust circuit when it's wrong
4. **Symbolic consistency loss** — penalizes fluent-but-wrong when symbols contradict
5. **Explicit state tracking** — entities/intent/emotion as first-class citizens

---

## First Experiment Checklist

- [x] Build CorrectedCorrector with gated delta architecture (src/hybrid.py)
- [x] Build StackedSSDContext (2-3 layers with orthogonal init)
- [x] Add LayerNorm to each input encoding (state, stack, context, histogram)
- [x] Implement 2-layer MLP with SiLU + residual + LayerNorm
- [x] Implement train_corrected_hybrid() with CrossEntropyLoss on final_logits
- [x] Verify backprop through ssd → hidden_proj → delta_proj → gate
- [ ] Test on small corpus (verify loss converges)
- [ ] Evaluate accuracy improvement over circuit-only baseline
- [ ] Compare gated delta vs fixed blend (0.5 weight)
- [ ] Experiment with num_ssd_layers (2 vs 3)
- [ ] Experiment with embed_dim (256 vs 512)
- [ ] Finetune on personal data

---

## References

- Neuro-symbolic bridging: arxiv 2504.07640
- Hybrid recurrent/attention: nature 024-07930-y  
- Learned gate fusion: huggingface.co/blog/jjokah/small-language-model
- Consistency losses: arxiv 2511.17644
