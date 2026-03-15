# CircuitLM: Comprehensive Status Report

**Report date:** March 2026  
**Scope:** Research progress, product readiness, and next steps for both tracks.

---

## Executive summary

CircuitLM is a **finite-state circuit language model** (FSM and PDA) trained with OR-Tools CP-SAT under strict **integer-only, no-floats** constraints in the core package. A **hybrid** extension (circuit + small neural corrector) runs in a separate `src/` module with PyTorch. The project has a working CLI, reproducible benchmarks, and a clear research narrative (PDA depth generalization, hybrid accuracy gains). Remaining work splits into **research** (scaling, new benchmarks, publishability) and **productization** (reliability, packaging, demo, docs).

---

## 1. Current state

### 1.1 Core (circuit-only)

| Area | Status | Notes |
|------|--------|--------|
| **FSM training** | Done | Hash-bootstrap + CP-SAT emission; joint solver with states as free vars |
| **PDA training** | Done | Two-phase (stack then emission) and true joint PDA (stack + emissions in one model) |
| **Config-conditioned stack** | Done | `push_configs` / `pop_configs` as `(state, token, stack_top)` triples; JSON migration for old format |
| **Tokenizers** | Done | Char and BPE (`circuit_lm.tokenizer`); BPE merges and vocab_size configurable |
| **Formats** | Done | JSON + MessagePack; PDA ~14× smaller with msgpack |
| **CLI** | Done | `train`, `eval`, `sample`, `chat` with full flags |
| **Constraints** | Enforced | No floats in `circuit_lm/` and `scripts/`; no numpy/torch in core; CP-SAT only |
| **Tests** | Passing | 199 tests; forbidden imports, no-floats, circuit/eval, PDA, PPM, msgpack |

**Scalability (joint solvers):**

- FSM joint: T_total ≤ 4k, num_states ≤ 16, vocab ≤ 64 recommended.
- PDA joint: T_total ≤ 2k, num_states ≤ 8, vocab ≤ 32; full stack discovery at ~50 seqs + 60 s budget in sweep.

**Open TODOs (README):**

- Multi-pass CP-SAT for larger state spaces.
- Streaming data loading: **implemented** as `iter_sequences` / `iter_sequence_chunks` in `circuit_lm/data.py` (README TODO can be marked done).

---

### 1.2 Hybrid (circuit + neural corrector)

| Area | Status | Notes |
|------|--------|--------|
| **Architecture** | Done | CircuitLM (FSM/PDA) + NeuralCorrector (MLP/CNN); blend by circuit_weight |
| **Training** | Done | `train_hybrid()` in `src/hybrid.py`; builds dataset from circuit rollout |
| **BPE in pipeline** | Done | Tokenizer/vocab from circuit file; corrector uses circuit.vocab_size and stack_depth |
| **CLI** | Done | `circuit-lm hybrid-train`, `circuit-lm chat --corrector corrector.pt` |
| **Dependencies** | Optional | PyTorch only for `src/`; run from repo root or set PYTHONPATH |

**Reported results (PROGRESS_TRACKER / HYBRID_BLOG):**

- Circuit-only PDA: 16–25% next-token accuracy on chat data.
- Hybrid (100k examples, ~200K params): up to **56.53%** (2.1–3.4× over circuit).
- Larger circuit (198 vocab) + large corrector: ~35% on combined data.
- Generation still garbled at 64–198 vocab; structure (e.g. user:/assistant:) emerges with larger vocab.

**Limitations:**

- Vocab 64–198 in most runs; BPE path exists but needs a full 1k+ vocab pipeline run and report.
- Correctors ~50K–200K params; no 1M+ config yet.
- Hybrid chat is greedy (no sampling from combined logits in CLI).

---

### 1.3 Benchmarks and reproducibility

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_small.py` | Smoke/perf on small synthetic text |
| `scripts/benchmark_matrix.py` | Grid over tokenizer, state_bits, steps; CSV/TSV/snapshot export |
| `scripts/benchmark_serialization.py` | JSON vs MessagePack size and roundtrip |
| `scripts/reproduce_depth_generalization.py` | PDA vs FSM vs PPM across depths; PDA improves OOD (57% at depth 8 vs 43% FSM, 39% PPM) |
| `scripts/verify_joint_pda_small.py` | Joint-PDA stack discovery at T_total=1168 |
| `scripts/sweep_jpda_budget.py` | Budget sweep for full push+pop discovery |

**Gaps:**

- No single “run all” script that produces one benchmark table.
- No code/bracket-specific benchmark (only nested parens in depth-gen).
- Hybrid accuracy not wired into a standard benchmark script.

---

### 1.4 Data and I/O

- **In-memory:** `load_sequences(path, tokenizer, seq_len)` — full file load.
- **Streaming:** `iter_sequences(path, tokenizer, seq_len, stride)` and `iter_sequence_chunks(..., chunk_size)` — line-by-line, no full-file load.
- **Chat data:** `chat_text_from_jsonl`, `chat_text_from_openai_export`; scripts to convert to training text.
- **Trainers:** Still use `load_sequences`; streaming iterators available for future multi-pass or chunked training.

---

## 2. Research next steps

### 2.1 Scaling (circuit)

1. **Multi-pass CP-SAT**  
   - Implement two-phase or chunked training: e.g. train small (fewer states/vocab), fix or bias structure, then re-run with more states/budget.  
   - Document T_total and state/vocab limits per phase.  
   - **Output:** Script or CLI flag; update STATUS.md.

2. **Joint-PDA at larger scale**  
   - Test warm-start or constraint relaxation so joint PDA can find push+pop at higher T_total (e.g. 3k+ tokens) or with more vocab.  
   - **Output:** Notes in STATUS.md or a short experiment log.

3. **`load_model` format auto-detect**  
   - Auto-detect JSON vs MessagePack by file extension (e.g. `.json` vs `.msgpack`).  
   - **Output:** Small change in `circuit_lm/io.py`.

### 2.2 Scaling (hybrid)

4. **BPE end-to-end at 1k+ vocab**  
   - One reproducible run: BPE vocab ≥ 1024, train circuit then corrector on same data.  
   - Report eval accuracy and 1–2 sample outputs in PROGRESS_TRACKER or a one-page report.  
   - **Output:** Updated PROGRESS_TRACKER; optional `scripts/reproduce_hybrid_bpe.sh`/`.ps1`.

5. **Larger corrector and more data**  
   - Add “large” corrector config (e.g. embed_dim 64–128, hidden_dim 256+).  
   - Use `iter_sequence_chunks` or larger `max_examples` to train on 500k+ tokens.  
   - **Output:** Accuracy vs baseline in PROGRESS_TRACKER; optional config flag in `train_hybrid`.

### 2.3 Differentiation and publishability

6. **Code / bracket benchmark**  
   - New script (e.g. `scripts/benchmark_code.py` or `benchmark_brackets.py`): Python/JSON snippets or multi-type brackets; train on depth ≤ k, test on k+1, k+2.  
   - Compare FSM vs PDA (and optionally PPM, hybrid).  
   - **Output:** Script + short doc in STATUS.md or `docs/`.

7. **Interpretability: trace CLI**  
   - `circuit-lm trace --prompt "..." --model model.json`: for each token print state (and stack for PDA) and top-k predicted tokens.  
   - Optional JSON/Markdown export for a static “explainer” page.  
   - **Output:** New `circuit_lm/trace.py` or extend infer; CLI subcommand.

8. **Single benchmark table**  
   - Script (e.g. `scripts/run_all_benchmarks.py`) that runs: depth generalization, serialization, optional code benchmark, optional hybrid accuracy.  
   - Output one table (CSV/Markdown) with model type, dataset, metric, value; document seed and commands.  
   - **Output:** Script; link from README and STATUS.md.

---

## 3. Productization next steps

### 3.1 Reliability and packaging

9. **CI and test stability**  
   - Ensure full pytest run (and key scripts) in CI; fix any flaky or path-dependent tests.  
   - **Output:** Stable CI; README badge.

10. **Hybrid as optional extra**  
    - Optional dependency group, e.g. `pip install circuit-lm[hybrid]` (PyTorch + `src/` on path or installable).  
    - Document that `hybrid-train` and `chat --corrector` require this extra or running from repo root.  
    - **Output:** `pyproject.toml` optional dependency; README section.

11. **Version and changelog**  
    - Bump version for releases; maintain a short CHANGELOG (or release notes) for research and product milestones.  
    - **Output:** CHANGELOG.md or GitHub Releases.

### 3.2 Usability and docs

12. **Reproduce script**  
    - `scripts/reproduce_hybrid.ps1` (and/or `.sh`) that: optionally fetches or points to a small dataset, trains circuit, trains corrector, runs eval and one sample.  
    - **Output:** One-command reproduction for blog and users.

13. **Blog post and “Reproduce” section**  
    - Turn HYBRID_BLOG.md into a public post; add “Reproduce” with exact commands and link to reproduce script.  
    - **Output:** Published post; link from README.

14. **Demo**  
    - Minimal option A: Gradio/Streamlit app that loads circuit (+ optional corrector) and runs sample/chat.  
    - Option B: Static page with pre-generated samples and short “how it works” (state trace, hybrid blend).  
    - **Output:** `demo/` or `app/`; optional deps; link (e.g. Hugging Face Spaces or GitHub Pages).

### 3.3 Polish

15. **README TODOs**  
    - Mark “Streaming data loading” as done (point to `iter_sequences` / `iter_sequence_chunks`).  
    - **Output:** README update.

16. **Error messages and docs**  
    - Clear errors when hybrid module not found (e.g. “Run from repo root or install with pip install -e .[hybrid]”).  
    - One-page “Quick start” (train → eval → sample → hybrid-train → chat) in README or `docs/quickstart.md`.  
    - **Output:** README or docs update.

---

## 4. Prioritized roadmap

| Priority | Item | Track | Effort (rough) |
|----------|------|--------|------------------|
| P0 | BPE 1k+ vocab pipeline + report (2.2.4) | Research | 1–2 days |
| P0 | Run-all benchmark script (2.3.8) | Research | 0.5 day |
| P0 | Reproduce script + blog “Reproduce” section (3.2.12–13) | Product | 0.5–1 day |
| P1 | Code/bracket benchmark (2.3.6) | Research | 1–2 days |
| P1 | Trace CLI (2.3.7) | Research | 0.5–1 day |
| P1 | Hybrid optional extra + README (3.1.10, 3.3.16) | Product | 0.5 day |
| P2 | Multi-pass CP-SAT (2.1.1) | Research | 2–3 days |
| P2 | Larger corrector config + more data (2.2.5) | Research | 1 day |
| P2 | Demo (3.2.14) | Product | 1–2 days |
| P3 | Joint-PDA scaling experiments (2.1.2) | Research | 1–2 days |
| P3 | load_model auto-detect (2.1.3) | Research | 0.25 day |
| P3 | CHANGELOG / versioning (3.1.11) | Product | ongoing |

---

## 5. Summary

- **Research:** Core circuit (FSM/PDA, joint solvers, depth generalization, MessagePack) and hybrid (BPE-ready pipeline, CLI) are in place. Streaming data loaders exist. Next: scale (multi-pass, BPE 1k+ run, larger corrector), code benchmark, trace CLI, and a single benchmark table for clear, reproducible claims.
- **Productization:** CLI and formats are usable; hybrid requires repo root or optional extra. Next: optional `[hybrid]` extra, reproduce script, blog with Reproduce section, and a minimal demo. Then: README/CHANGELOG polish and stable CI.

This report can be updated as items are completed; suggested location: **`STATUS_REPORT.md`** in the repo root, with references from README and STATUS.md.
