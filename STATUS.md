# Project Status

Last updated: 2026-04-08 (rev 4)

## Snapshot

- Repo state: working tree was clean when this status snapshot was created.
- Recent validation commit: `f3a47d5` (`kaggle notebook: 10K->6K vocab, 4 corrector sizes`)
- Python validation target: `3.12`
- Packaging: editable install works with `py -3.12 -m pip install --user -e .[dev]`
- ⚠️ Local test run not possible (no py -3.12 on this machine); last confirmed: 199 passed (2026-03-08)

## What Is Working

- Full test suite passes: `199 passed` via `py -3.12 -m pytest -q`
- **True joint FSM solver** (`train_joint_cpsat.train_joint`): states, transitions, and emissions as free CP-SAT decision variables; objective = prediction accuracy directly
- **True joint PDA solver** (`train_joint_pda_cpsat.train_joint_pda`): states, per-config `(state, token, stack_top)` push/pop policy, and per-config emissions jointly solved in a single CP-SAT model
- **Config-conditioned stack operations**: `PDACircuitLM` now stores `push_configs`/`pop_configs` as `frozenset[tuple[int,int,int]]`; `stack_op(state, token, stack_top)` dispatches on the full config triple. Two-phase PDA uses degenerate expansion (token-ID → all state/stack-top combos); joint PDA uses `S × V × ST_RANGE` CP-SAT boolean variables.
- JSON migration shim: old `push_tokens`/`pop_tokens` fields auto-expand to degenerate config triples on load.
- CLI train/eval/sample flows run with the new split CP-SAT budget flags
- Runtime CLI validation is covered for:
  - unpaired `--transition_steps` / `--emission_steps`
  - partial PDA explicit budgets missing `--stack_steps`
- Integer-only invariants remain enforced by tests (no floats / forbidden imports checks)

## Baseline Benchmark (Small Synthetic Dataset)

Command:

```powershell
py -3.12 scripts/benchmark_small.py
```

Results (2026-02-25):

- `text_len=3105`
- `effective vocab_size=30`
- `sequences=13`, `total_tokens=3117`
- `num_states=8`
- `train_time=203ms`
- `eval accuracy=20.84%` (`647/3104`)

Notes:

- This is a smoke/perf baseline only (small synthetic text, not a quality benchmark).
- Output confirms end-to-end train/eval/sample execution on current code.

## Benchmark Matrix Tooling (Added 2026-02-25)

- Added `scripts/benchmark_matrix.py` to run a small train/eval grid across:
  - tokenizer mode (`char`, `bpe`)
  - `state_bits` (`2`, `3`, `4`)
  - CP-SAT `steps` (`1`, `2`, `5`)
- Script emits a stable table with integer timings and integer metrics-derived accuracy strings.
- Script now supports optional file output for diffable baselines:
  - `--csv-out PATH`
  - `--tsv-out PATH`
- Script now supports timestamped snapshot exports:
  - `--snapshot-dir DIR`
  - optional `--snapshot-prefix NAME`

Quick observations from the first matrix run:

- On the synthetic benchmark text, increasing `state_bits` improved accuracy more than increasing `steps`.
- Accuracy is not monotonic in `steps` on this tiny dataset (for example some `steps=5` rows were worse than `steps=1`/`2`), so this matrix is useful for catching solver/objective behavior changes.

## Open Gaps (from README TODOs)

- Multi-pass CP-SAT scaling improvements for larger state spaces
- Streaming / stride-aware data loading for corpora larger than RAM

## Scalability Limits (joint solvers)

| Solver | Recommended T_total | num_states | vocab_size |
|--------|---------------------|------------|------------|
| `train_joint_cpsat` (FSM) | ≤ 4 000 | ≤ 16 | ≤ 64 |
| `train_joint_pda_cpsat` (PDA) | ≤ 2 000 | ≤ 8 | ≤ 32 |

The PDA joint solver is harder because the `S × V × ST_RANGE` config-conditioned
push/pop boolean variables couple all occurrences of each (state, token, stack_top)
triple simultaneously.

## Depth-Generalization Experiment (Four-Model Comparison, 2026-02-25)

Command:

```
py -3.12 scripts/reproduce_depth_generalization.py
```

Settings: train depth ≤ 3, 300 train seqs, 100 test seqs per depth, seed=42.
All models: 20 s CP-SAT budget. PDA/joint-PDA: 4 states. FSM: 16 states. PPM order 6.

| depth | PDA-2ph | PDA-jt | FSM    | PPM    |
|-------|---------|--------|--------|--------|
| 3     | 44.66%  | 49.02% | 55.33% | 59.49% |
| 4*    | 50.66%  | 50.48% | 49.33% | 54.68% |
| 5*    | 50.60%  | 50.24% | 49.39% | 51.40% |
| 6*    | 53.61%  | 50.96% | 46.38% | 45.99% |
| 7*    | 55.09%  | 50.91% | 44.90% | 42.43% |
| 8*    | **57.21%** | 52.82% | 42.78% | 39.05% |

Key observations:

- **PDA-2ph**: Correctly identified push=`(` pop=`)`. Accuracy *improves* with OOD depth (44.66% → 57.21%) while FSM and PPM degrade — the stack encodes a depth-invariant feature.
- **PDA-jt**: No stack learned (T_total ≈ 3000 > 2000 limit), settling at ~50–52%. See joint-PDA note below.
- **FSM** (16 states): 55% at depth 3, degrades to 43% at depth 8 — context window loses structural signal.
- **PPM** (order 6): Highest at depth 3 (59%), degrades fastest to 39% — negative transfer from depth-3 n-gram patterns.

### Bug Fixed: stack-top timing mismatch in two-phase PDA training

The `8415208` refactor changed `evaluate_pda` to **step first, then predict** (semantically correct)
but left both simulation functions (`_simulate_and_collect`, `_simulate_and_collect_runtime`)
recording configs with the **pre-update** stack top. At every PUSH/POP position, training and
evaluation looked up different config keys, causing the stack signal to be silently ignored.

Fix: move the stack update before the config recording in both simulation functions.
Result: PDA-2ph now demonstrates the expected depth-generalization behavior (57.21% at depth 8
vs 42.78% FSM, 39.05% PPM).

### Joint-PDA scalability note

With 300 training sequences the joint PDA still finds no stack (`push_configs` is empty). A 150-seq run
also produced no stack. This is consistent with the T_total ≤ 2000 limit — the search space is
too large for the 20 s budget to discover the push/pop structure.

## Config-Conditioned Stack Operations (2026-02-25)

Commits: `5d2c373` → `65d9525` (Tasks 1–5 of joint-PDA verification + config-ops plan).

- `push_tokens`/`pop_tokens` (`frozenset[int]`) replaced by `push_configs`/`pop_configs` (`frozenset[tuple[int,int,int]]`) throughout `pda.py`, `io.py`, `train_pda_cpsat.py`, `train_joint_pda_cpsat.py`, `cli.py`, and all scripts.
- `stack_op(state, token, stack_top)` now dispatches on the full `(src_state, token, stack_top_before_op)` triple.
- Two-phase PDA (`train_pda`): expands learned token-ID sets to all `(state, tok, stack_top)` triples at output (degenerate but correct).
- Joint PDA (`train_joint_pda`): uses `S × V × ST_RANGE` CP-SAT boolean variables — `is_push_flat` and `is_pop_flat` — with `add_element` lookup per occurrence. `max_push`/`max_pop` budget constraints count distinct token IDs via auxiliary `tok_ever_pushes`/`tok_ever_pops` booleans.
- JSON migration shim: old `push_tokens`/`pop_tokens` integer-list format auto-expands to degenerate config triples on load.

## Joint-PDA Small-Scale Verification (Task 6, 2026-02-25)

Command:

```
py -3.12 scripts/verify_joint_pda_small.py
```

Settings: train depth ≤ 3, 100 train seqs (T_total=1168), 30 test seqs per depth, seed=42.
Joint-PDA: 4 states, 30 s budget, max_push=1, max_pop=1. PDA-2ph: 4 states, 20 s budget.

```
  jpda push_tokens=[0]  pop_tokens=[]
  [PASS] joint-PDA discovered stack
  pda  push_tokens=[0]  pop_tokens=[1]
```

| depth | PDA-2ph | PDA-jt |
|-------|---------|--------|
| 3     | 44.20%  | 50.00% |
| 4*    | 49.85%  | 50.00% |
| 5*    | 51.08%  | 50.00% |
| 6*    | 52.98%  | 50.00% |

Key observations:

- **Stack discovery confirmed** at T_total=1168 — joint-PDA found push_tokens=[0] (`(`), satisfying the `[PASS]` condition.
- **Partial discovery**: POP token not learned in 30 s. The solver found the push rule but not the pop rule, leaving joint-PDA at a 50% plateau (predicts a fixed token regardless of depth).
- **PDA-2ph correctly finds push=[0] pop=[1]** and shows the expected OOD improvement (44% → 53%).
- **Interpretation**: push discovery is easier (every `(` unconditionally increases depth); pop discovery requires correlating `)` with the non-empty stack state. More budget or a smaller vocabulary may be needed for the joint solver to find both operations simultaneously.

## JSON Serialization Benchmark (2026-02-25)

Command:

```
py -3.12 scripts/benchmark_serialization.py
```

The script now runs both JSON and MessagePack per model (see MessagePack section below for full table).
Original JSON-only notes: all roundtrip checks pass; PDA JSON is ~54× larger than FSM at same state/vocab
due to `config_counts`. Timings are integer ms.

## Pop Discovery Sweep (2026-03-08)

Command:

```
py -3.12 scripts/sweep_jpda_budget.py --train-seqs 50 --steps 30 60
```

Smoke run (2 combos, seed=42):

| train_seqs | steps | T_total | push | pop | full |
|------------|-------|---------|------|-----|------|
| 50         | 30    | 580     | YES  | no  | no   |
| 50         | 60    | 580     | YES  | YES | YES  |

- **Full stack discovery achieved**: (50, 60) found both push and pop (1/2 combos in this run).
- **Minimum budget in this sweep**: (train_seqs=50, steps=60 s) is sufficient for the joint solver to learn both operations at T_total=580.
- At 30 s only push was discovered; 60 s allowed the solver to commit to the correlated pop policy.

## MessagePack Binary Format (2026-03-08)

`circuit_lm.io` provides `save_msgpack` / `load_msgpack`; `scripts/benchmark_serialization.py` reports both JSON and msgpack rows.

Results (seed=42, steps=3 s training per model):

```
label     type  fmt      states  vocab     bytes  save_ms  load_ms  ok
----------------------------------------------------------------------
fsm-sm    fsm   json          8     30      7020        2        0   1
fsm-sm    fsm   msgpack       8     30      1072        0        0   1
pda-sm    pda   json          8     30    377610       41       10   1
pda-sm    pda   msgpack       8     30     26680        5        6   1
pda-md    pda   json         16     30    739648       58       17   1
pda-md    pda   msgpack      16     30     51778       10       38   1
----------------------------------------------------------------------
```

- All six roundtrip checks pass (`ok=1`).
- **Compression ratio (PDA)**: PDA-sm msgpack 26,680 bytes ÷ JSON 377,610 ≈ 7.1% (~14× smaller). PDA-md 51,778 ÷ 739,648 ≈ 7.0%. Binary format is well within the earlier 20–40 KB target for PDA-sm.

## Kaggle Training Pipeline (2026-04-08)

**Kernel:** `zacharymaronek/circuit-lm-6k-training`
**Notebook:** `kaggle_training_notebook.ipynb`
**Dataset:** `zacharymaronek/circuit-lm-personal` → `all_personal_training.txt` (7.4 MB)

Trains a hybrid CircuitLM (integer PDA circuit + neural corrector) with 6,144-token BPE vocab.

| Step | What happens |
|------|-------------|
| 1 | Clone repo + install ortools, msgpack, sentencepiece, torch |
| 2 | Load personal training data from Kaggle dataset |
| 3 | Build 6K BPE tokenizer (sweet spot for 7MB personal data) |
| 4 | Train PDA circuit via OR-Tools CP-SAT (STATE_BITS=6, stack_depth=4) |
| 5 | Save circuit + tokenizer (shared across all corrector sizes) |
| 6 | Train 4 corrector sizes (tiny/small/medium/large) on GPU |
| 7 | Results summary + download links |
| 8 | Generate side-by-side samples from all 4 sizes |

**Corrector sizes:**

| Size | embed_dim | hidden_dim | num_layers | context | Params |
|------|-----------|------------|------------|---------|--------|
| Tiny | 64 | 128 | 2 | 32 | ~1.5 M |
| Small | 128 | 256 | 3 | 64 | ~5 M |
| Medium | 256 | 512 | 4 | 64 | ~20 M |
| Large | 512 | 1024 | 4 | 128 | ~80 M |

**Runtime:** GPU required. P100 (16GB) handles Medium comfortably. ~30-60 min total.

**Note:** Old kernel `zacharymaronek/circuitlm` errored because `notebook.ipynb` checked for a local file path. Fixed by pushing `kaggle_training_notebook.ipynb` with corrected Kaggle dataset path.

## Next Recommended Steps

1. Run the Kaggle training notebook (enable GPU, ~30-60 min)
2. Download trained correctors + circuit
3. Wire into Rust inference kernel (GPU-free local inference)
4. Consider extending `load_model` to auto-detect JSON vs msgpack by file extension

## How To Update This File

1. Re-run `py -3.12 -m pytest -q`
2. Re-run `py -3.12 scripts/benchmark_small.py`
3. Re-run `py -3.12 scripts/benchmark_matrix.py` (optional but recommended)
4. If needed, export `--csv-out` or `--tsv-out` for baseline comparison
5. If archiving a run, use `--snapshot-dir DIR`
6. Update the snapshot, results, and next steps with commit hashes and dates
