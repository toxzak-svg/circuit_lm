# Project Status

Last updated: 2026-02-25

## Snapshot

- Repo state: working tree was clean when this status snapshot was created.
- Recent validation commit: `16ad53e` (`Add true joint CP-SAT FSM learning and depth-generalization experiment scripts`)
- Python validation target: `3.12`
- Packaging: editable install works with `py -3.12 -m pip install --user -e .[dev]`

## What Is Working

- Full test suite passes: `175 passed` via `py -3.12 -m pytest -q`
- **True joint FSM solver** (`train_joint_cpsat.train_joint`): states, transitions, and emissions as free CP-SAT decision variables; objective = prediction accuracy directly
- **True joint PDA solver** (`train_joint_pda_cpsat.train_joint_pda`): states, per-token push/pop policy, and per-config emissions jointly solved in a single CP-SAT model
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

- Per-(state, token, stack_top) stack operations: current PDA assigns push/pop by token ID only; a richer formulation would condition on the full config
- Compressed binary model format (JSON remains the only tracked serialization path)
- Multi-pass CP-SAT scaling improvements for larger state spaces
- Streaming / stride-aware data loading for corpora larger than RAM

## Scalability Limits (joint solvers)

| Solver | Recommended T_total | num_states | vocab_size |
|--------|---------------------|------------|------------|
| `train_joint_cpsat` (FSM) | ≤ 4 000 | ≤ 16 | ≤ 64 |
| `train_joint_pda_cpsat` (PDA) | ≤ 2 000 | ≤ 8 | ≤ 32 |

The PDA joint solver is harder because shared `is_push`/`is_pop` decision
variables couple all occurrences of each token simultaneously.

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

With 300 training sequences the joint PDA still finds no stack (`push_tokens=[]`). A 150-seq run
also produced no stack. This is consistent with the T_total ≤ 2000 limit — the search space is
too large for the 20 s budget to discover the push/pop structure.

## Next Recommended Steps

1. Run joint-PDA experiment with a reduced vocabulary or smaller sequences to stay within T_total ≤ 2000 and verify stack discovery at that scale.
2. Add a serialization benchmark comparing JSON save/load sizes and times before introducing a binary format.

## How To Update This File

1. Re-run `py -3.12 -m pytest -q`
2. Re-run `py -3.12 scripts/benchmark_small.py`
3. Re-run `py -3.12 scripts/benchmark_matrix.py` (optional but recommended)
4. If needed, export `--csv-out` or `--tsv-out` for baseline comparison
5. If archiving a run, use `--snapshot-dir DIR`
6. Update the snapshot, results, and next steps with commit hashes and dates
