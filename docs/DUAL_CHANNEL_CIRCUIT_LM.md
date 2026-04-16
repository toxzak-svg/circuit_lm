# Dual-Channel CircuitLM Architecture Spec

**Status:** Prototype-ready  
**Date:** 2026-04-15  
**Goal:** 3M–10M parameter hybrid that beats plain tiny decoders at the same budget

---

## Core Idea

A dual-channel language model where:

1. **Symbolic channel** (PDA circuit) — discrete, integer-only state machine that tracks entities, relations, intent slots, and conversation state. Zero floats. Inferred via CP-SAT.
2. **Text channel** (neural corrector) — small GRU/MLM that handles distributional fluency and surface phrasing.
3. **Learned stitcher** (fusion gate) — learns when to trust symbols vs text at every token position.

The key claim: same parameter budget, better groundedness and controllability than a plain tiny LM.

---

## Architecture

```
Input: token stream
  │
  ├──► SYMBOLIC CHANNEL ──────────────────────────────────────
  │       │
  │       ├── PDA circuit (integer-only, CP-SAT trained)
  │       │   • Stack operations (push/pop/noop)
  │       │   • Tracks: entities, relations, intent slots, memory keys
  │       │   • Emits: circuit_state vector (one-hot, S dimensions)
  │       │   • Emits: confidence histogram over vocab
  │       │
  │       └── Symbol embeddings: Linear(S → symbol_embed_dim)
  │
  ├──► TEXT CHANNEL ──────────────────────────────────────────
  │       │
  │       ├── GRU encoder (char or BPE tokens)
  │       │   • 1-2 layers, hidden_dim 128-256
  │       │   • Input: recent context window (32-64 tokens)
  │       │   • Output: text_hidden_dim
  │       │
  │       └── Neural corrector: MLP(circuit_state + text_hidden + context)
  │           • Predicts residual on top of circuit distribution
  │           • Output: logits over vocab
  │
  └──────► FUSION GATE ────────────────────────────────────────
              │
              ├── gate = sigmoid(Linear(circuit_state + text_hidden))
              │   • gate ∈ [0,1] per token
              │
              ├── circuit_logits = Linear(circuit_state_proj)
              ├── neural_logits = Linear(text_hidden_proj)
              │
              └── final_logits = gate * circuit_logits + (1 - gate) * neural_logits
                  • Alternately: weighted sum with learned per-class weights

Output: next token distribution (over vocab)
```

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

### 1. Standard NLL Loss
```
L_nll = -log p(token | circuit_state, text_hidden, gate)
```

### 2. Circuit Confidence Loss (novel)
When circuit is confident but wrong → penalize the gate for trusting it:
```
L_circuit_mismatch = |circuit_confidence - circuit_accuracy| * gate_weight
```
This teaches the gate to learn when the circuit is reliable without manual rules.

### 3. Symbol Consistency Loss (novel)
When symbolic state contradicts text → penalize fluent-but-wrong:
```
L_consistency = max(0, text_score(wrong_symbolic) - text_score(correct_symbolic))
```
Requires a symbolic ground-truth signal during training (extracted via rule-based parser).

### 4. Total Loss
```
L_total = L_nll + λ1 * L_circuit_mismatch + λ2 * L_consistency
λ1 = 0.1 (circuit mismatch weight)
λ2 = 0.2 (consistency weight)
```

---

## Training Loop

### Phase 1: Pretrain PDA Circuit (CP-SAT, integer-only)
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

### Phase 2: Warm Text Channel (MLM pretraining)
```
Fix circuit.json
Train GRU + corrector only (no fusion gate yet)
Objective: predict next token given circuit state + text context
L_warm = -log p(token | circuit_state, text_hidden)
```

### Phase 3: Train Fusion Gate (full end-to-end)
```
Unfreeze everything
Train with L_total (NLL + circuit mismatch + consistency)
Optimizer: AdamW, lr=2e-4, cosine schedule
Batch: 32-64, gradient_accumulation=4
```

### Phase 4: Finetune on Personal Data
```
Use Zach's Starfire conversations (1,758 examples)
Narrow domain: personal AI assistant
Target: learn personality + preferences + memory patterns
```

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

- [ ] Define vocabulary (BPE 512-1024 for personal data)
- [ ] Build text encoder (GRU, ~400K params)
- [ ] Build fusion gate MLP
- [ ] Write L_circuit_mismatch loss
- [ ] Write L_consistency loss (needs slot tracking annotation)
- [ ] Phase 1: train PDA circuit
- [ ] Phase 2: warm corrector (fix circuit)
- [ ] Phase 3: train full system
- [ ] Evaluate on held-out Starfire conversations
- [ ] Compare gate distribution vs fixed 0.5 blend

---

## References

- Neuro-symbolic bridging: arxiv 2504.07640
- Hybrid recurrent/attention: nature 024-07930-y  
- Learned gate fusion: huggingface.co/blog/jjokah/small-language-model
- Consistency losses: arxiv 2511.17644
