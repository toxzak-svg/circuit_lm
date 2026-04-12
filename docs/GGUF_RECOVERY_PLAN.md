# GGUF Weight Recovery — Constrained Recovery PTQ

## Overview

Given a quantized GGUF model, recover a better surrogate weight space that enables second-stage PTQ with less degradation than standard requantization.

**Core claim:** Given only a quantized GGUF, we can recover enough structure to do high-quality PTQ without needing FP16 source weights.

---

## Pipeline Architecture

```
Input: quantized GGUF (Q4_K_M, Q5_K_M, Q6_K, etc.)
    ↓
[1] GGUF Parser
    ↓
[2] Dequantize to Surrogate FP16
    ↓
[3] Attach Residual Parameters E
    ↓
[4] Calibrate on Dataset (measure error)
    ↓
[5] Optimize E (minimize perplexity / KL divergence)
    ↓
[6] Re-Quantize to GGUF
    ↓
Output: Improved GGUF
```

---

## Stage 1 — GGUF Parser

**Goal:** Read GGUF binary, extract metadata + tensor blocks.

**Key structures:**
- `GGUFHeader`: magic, version, tensor_count, metadata_len
- `TensorInfo`: name, shape, dtype, offset
- `TensorBlock`: quantized data for one weight tensor

**Supported dtypes:**
- Q4_0, Q4_1, Q5_K, Q6_K, Q8_0
- IQ4_NL, IQ3_S, IQ2_S, IQ2_XS (important for newer formats)
- F16, BF16, F32 (less relevant but needed for scales)

**Output:** Dictionary of tensor_name → {shape, dtype, data, scale, zero_point}

---

## Stage 2 — Dequantize to Surrogate FP16

**Goal:** Convert quantized blocks → approximate FP16 weights.

**Standard dequantization:**
```python
W_fp16[i] = scale * (quantized[i] - zero_point)
```

**Recovery layer approach:**
```python
W_approx = dequantize(quantized) + E
```
where E is learned correction, constrained so re-quantization lands in valid codebook region.

**Key constraint:** E must be small enough that re-quantization produces the same (or nearby) codebook index. This is what makes it "constrained" recovery rather than free-form fine-tuning.

---

## Stage 3 — Attach Residual Parameters E

**E parameterization options:**

| Type | Description | Param count |
|------|-------------|-------------|
| Per-tensor scalar | One α per tensor: W + α·D | 1 per tensor |
| Per-block affine | α·block + β per block | 2 per block |
| Low-rank | Low-rank matrix per tensor | very small |
| Codebook offset | Shift dequantization cluster assignment | tiny |

**Chosen approach:** Per-block affine correction (α·block + β) — balances expressiveness vs param count.

**Shape of E:**
- For Q4 blocks of 32 elements: 2 params per block
- For Q4_K_M (128-element blocks): 2 params per block
- Tiny overhead compared to full weight matrix

---

## Stage 4 — Calibration on Dataset

**Goal:** Measure quality degradation from quantization.

**Calibration dataset:**
- Use a small representative corpus (e.g., 512-1024 tokens)
- Run through model, collect hidden states or perplexity
- Compare FP16 surrogate vs quantized outputs

**Metrics:**
- Per-token perplexity difference
- KL divergence of output distributions
- Hidden state L2 distance (if accessible)

**For circuit_lm context:**
- Use Starfire's 1,758 training examples as calibration set
- Measure perplexity improvement on held-out examples

---

## Stage 5 — Optimize E

**Objective:** Minimize calibration loss subject to re-quantization constraint.

**Loss function:**
```python
L = KL(p_approx || p_true) + λ * constraint_penalty
```

Where:
- `p_approx` = model output with (W_approx + E)
- `p_true` = model output with FP16 surrogate
- `constraint_penalty` = how much E causes re-quantized weights to differ

**Optimization:**
- AdamW with small learning rate
- E parameters only (frozen base weights)
- Early stopping based on calibration perplexity

**Constraint enforcement:**
- After each update, project E so re-quantization produces valid codebook entries
- OR use straight-through estimator (STE) that approximates gradient through quantizer

---

## Stage 6 — Re-Quantize to GGUF

**Input:** Optimized E corrections + original quantization metadata

**Process:**
1. Compute W_final = W_surrogate + E
2. Quantize W_final using original dtype and quantization scheme
3. Update scale/zero_point if needed
4. Write new GGUF with corrected tensor blocks

**Verification:**
- Run perplexity check on calibration set
- Compare against naive requantization baseline
- Ensure no safety regressions

---

## Implementation Plan

### Phase 1 — GGUF Parser (foundation)
- `src/gguf/parser.py` — read GGUF binary, extract tensors
- `src/gguf/dequant.py` — convert to FP16 surrogate
- Test with a real GGUF file (DeepSeek-1.3B Q4 on HuggingFace)

### Phase 2 — Recovery Layer
- `src/recovery/residual.py` — E parameter definition
- `src/recovery/calibrate.py` — calibration data handling
- `src/recovery/optimize.py` — E optimization loop

### Phase 3 — Re-Quantization
- `src/emitter/quantize.py` — re-quantize with corrections
- `src/emitter/write_gguf.py` — write improved GGUF

### Phase 4 — Integration + Benchmarks
- `scripts/run_recovery.py` — full pipeline
- Compare against llama.cpp --allow-requantize baseline
- Report perplexity improvement

---

## Key Questions

1. **E parameter budget** — How small can E be while still improving quality?
2. **Calibration set size** — How many examples needed for meaningful signal?
3. **Re-quantization validity** — Can we always find valid codebook entries that E doesn't disturb?
4. **Which GGUF dtypes** — Does recovery work better for some formats (Q4 vs Q5 vs Q6)?

---

## Expected Outcome

Baseline: naive requantization degrades quality 3-8% perplexity
Target: constrained recovery reduces degradation to <1-2%

If we hit <1% degradation, we've shown that GGUF→GGUF recovery is viable without FP16 source.

---

## Related Work

- llama.cpp `--allow-requantize` — naive dequant + requant (baseline)
- QAT (Quantization-Aware Training) — training with quant in loop
- BRECQ / QDROP — calibration-based post-training quantization
- GPTQ / AWQ — importance-matrix guided quantization

Our contribution: bridging the gap between "have GGUF only" and "want better GGUF" without full model training.