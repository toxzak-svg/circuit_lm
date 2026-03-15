# Bringing CircuitLM to the Next Level

Concrete steps to level up the **circuit-only** and **hybrid** (circuit + neural corrector) work, based on the current codebase and progress tracker.

---

## 1. Quick Wins (1–2 days each)

### 1.1 Use BPE in the hybrid pipeline — **DONE**

- Train circuit with `--tokenizer bpe --bpe_merges 512`, then run `hybrid-train`; tokenizer and vocab come from the circuit file. `HybridModel.load` uses circuit `stack_depth` and `vocab_size`. (Previously: char-level vocab 64–198. PROGRESS_TRACKER calls this the main bottleneck.
- **Action:** Train circuit with `--tokenizer bpe --bpe_merges 512` (or 1024), then run `train_hybrid` using the same tokenizer and vocab. Ensure `HybridModel` / `NeuralCorrector` use `tokenizer.vocab_size` and that the corrector’s `vocab_size` matches the circuit.
- **Files:** `src/hybrid.py` (vocab_size from circuit/tokenizer), CLI or script that runs `train` then `train_hybrid` with shared tokenizer.

### 1.2 Hybrid in the CLI — **DONE**

- `circuit-lm hybrid-train --circuit ... --data ... --out ...` and `circuit-lm chat --model ... --corrector corrector.pt`. See README.

### 1.3 Strided / streaming-friendly data loading — **DONE**

- `iter_sequences(path, tokenizer, seq_len, stride)` and `iter_sequence_chunks(..., chunk_size)` in `circuit_lm/data.py`; `load_sequences` kept for small/medium files.

---

## 2. Scale (circuit + hybrid)

### 2.1 Multi-pass / coarse-to-fine CP-SAT (circuit)

- **Current:** Single-pass joint FSM/PDA; STATUS.md recommends T_total ≤ 4k (FSM) / ≤ 2k (PDA) and notes “multi-pass CP-SAT” in README TODOs.
- **Action:** Implement a two-phase or multi-phase training: (1) train a smaller model (fewer states / smaller vocab or context) to get a rough structure; (2) fix or bias states/transitions and re-run CP-SAT with more states or longer budget. Alternatively, chunk the corpus and train on chunks sequentially, then merge or refine (e.g. merge state counts, then re-solve emissions). Document recommended T_total and state/vocab limits per phase.
- **Files:** New script or module that composes `train_cpsat` / `train_joint_cpsat` (and PDA variants) in multiple passes; optionally extend `circuit_lm/cli.py` with a `--multi-pass` or similar.

### 2.2 Larger corrector and more data

- **Current:** ~200K-param corrector, 50k–100k examples; PROGRESS_TRACKER targets 1M+ params and “millions of tokens”.
- **Action:** Add an optional “large” corrector config (e.g. more layers, embed_dim 64–128, hidden_dim 256+) in `src/hybrid.py`. Use the new streaming/strided loader to train on 500k+ tokens (or more) and log accuracy vs. baseline. Keep a small config for fast iteration.
- **Files:** `src/hybrid.py` (NeuralCorrector configs, `train_hybrid` data source).

### 2.3 BPE end-to-end at 1k+ vocab

- **Current:** BPE exists in `circuit_lm.tokenizer` but hybrid and many experiments use small char vocab.
- **Action:** One reproducible pipeline: BPE with vocab_size ≥ 1000 (e.g. 1024 or 2048), train circuit on a fixed dataset, then train corrector. Report eval accuracy and a short sample (e.g. `circuit-lm sample` / hybrid chat) in PROGRESS_TRACKER or a small report. This directly addresses “vocabulary bottleneck” from HYBRID_BLOG.

---

## 3. Differentiation and research

### 3.1 Code / bracket benchmark

- **Current:** Depth-generalization script shows PDA > FSM/PPM on nested brackets; STATUS.md and PROGRESS_TRACKER suggest “try on code”.
- **Action:** Add a small “code” benchmark: e.g. Python or JSON snippets with balanced delimiters, or a synthetic bracket dataset with multiple bracket types. Compare FSM vs. PDA (and optionally PPM) on next-token accuracy and on depth-generalization (train on depth ≤ k, test on k+1, k+2). Optionally compare circuit-only vs. hybrid. Document in STATUS.md or a separate `docs/` or `scripts/` readme.
- **Files:** New script (e.g. `scripts/benchmark_code.py` or `scripts/benchmark_brackets.py`) and possibly a tiny dataset under `data/` or generated in the script.

### 3.2 Interpretability tooling

- **Current:** Circuit is a concrete FSM/PDA; “you can literally trace what state it’s in” (HYBRID_BLOG) but there’s no built-in tracing UI.
- **Action:** Add a small CLI or script: given a prompt and a model, print (or export) a step-by-step trace: token → state (and stack for PDA) → top-k predicted tokens. Optionally output JSON/Markdown for a simple static “explainer” page or blog. Keeps the interpretability story concrete.
- **Files:** `circuit_lm/infer.py` or a new `circuit_lm/trace.py`, plus a CLI subcommand (e.g. `circuit-lm trace --prompt "..." --model model.json`).

### 3.3 Publishable benchmark table

- **Current:** Depth-generalization and serialization benchmarks exist; results are in STATUS.md and PROGRESS_TRACKER.
- **Action:** Add a single script (e.g. `scripts/run_all_benchmarks.py`) that runs: (1) depth generalization, (2) serialization (JSON vs MessagePack), (3) optional code/bracket benchmark, (4) optional hybrid accuracy at 1–2 configs. Output a single table (CSV/Markdown) with model type, dataset, metric, value. Document how to reproduce (seed, paths, commands) so the table is one command away from regeneration.
- **Files:** New script; link from README and STATUS.md.

---

## 4. Story and visibility

### 4.1 Demo

- **Current:** CLI chat and sample exist; no hosted or shareable demo.
- **Action:** Minimal options: (1) a small Gradio/Streamlit app that loads a trained circuit (and optional corrector) and runs `sample` / chat in the browser; or (2) a static page with a few pre-generated samples and a short “how it works” (state trace, hybrid blend). Either way, one clear link (e.g. Hugging Face Spaces or GitHub Pages) for “try it”.
- **Files:** New `demo/` or `app/` directory; keep dependencies optional (e.g. `pip install streamlit` only for demo).

### 4.2 Blog post and replication

- **Current:** HYBRID_BLOG.md is a good draft; PROGRESS_TRACKER lists “Make demo/blog post”.
- **Action:** Turn HYBRID_BLOG into a short public post. Add a “Reproduce” section: exact commands to train circuit, train corrector, run eval and one sample (and optional trace). Optionally add `scripts/reproduce_hybrid.sh` (or `.ps1`) that downloads or points to a small dataset and runs the full pipeline. Ensures “next level” results are one copy-paste away.
- **Files:** `HYBRID_BLOG.md` (or `docs/`), `scripts/reproduce_hybrid.ps1` / `reproduce_hybrid.sh`.

---

## Suggested order

1. **BPE in hybrid** + **hybrid in CLI** — unblocks larger vocab and easier experimentation.
2. **Streaming data loader** — enables “more data” without OOM.
3. **Larger corrector + more data** — validate that scaling improves numbers.
4. **Code/bracket benchmark** — sharp, publishable angle (PDA vs FSM).
5. **Trace CLI** — makes “interpretable” tangible.
6. **Single benchmark script + reproduce script** — then blog/demo.

If you tell me which of these you want to do first (e.g. “BPE + CLI for hybrid” or “streaming loader”), I can outline exact code changes and call sites next.
