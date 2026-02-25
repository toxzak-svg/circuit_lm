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
| 3     | 31.84%  | 49.02% | 55.33% | 59.49% |
| 4*    | 30.16%  | 50.48% | 49.33% | 54.68% |
| 5*    | 30.45%  | 50.24% | 49.39% | 51.40% |
| 6*    | 29.56%  | 50.96% | 46.38% | 45.99% |
| 7*    | 29.10%  | 50.91% | 44.90% | 42.43% |
| 8*    | 28.51%  | 52.82% | 42.78% | 39.05% |

Key observations:

- **PDA-2ph**: Correctly identified push=`(` pop=`)` via co-occurrence, but accuracy is 28–31% — below random. The co-occurrence phase objective does not directly optimize prediction accuracy; the emission phase inherits a policy that may not help.
- **PDA-jt**: Learned no stack operations (`push_tokens=[]`), settling at ~50% (near-random for 3-token vocab). T_total ≈ 3000 exceeds the 2000-token recommended limit; the solver found a local optimum with no stack within the time budget.
- **FSM** (16 states): 55% at depth 3, degrades to 43% at depth 8 — loses structural context beyond its window.
- **PPM** (order 6): Highest at depth 3 (59%), degrades fastest to 39% at depth 8 — negative transfer from depth-3 n-gram patterns.

## Next Recommended Steps

1. Investigate the two-phase PDA emission quality: is the 28–31% accuracy a solver variance issue or a structural limitation of the co-occurrence stack-policy objective?
2. Re-run joint-PDA depth experiment with fewer training sequences (≤ 150) to stay within the T_total ≤ 2000 recommendation and see if the stack is discovered.
3. Add a serialization benchmark comparing JSON save/load sizes and times before introducing a binary format.

## How To Update This File

1. Re-run `py -3.12 -m pytest -q`
2. Re-run `py -3.12 scripts/benchmark_small.py`
3. Re-run `py -3.12 scripts/benchmark_matrix.py` (optional but recommended)
4. If needed, export `--csv-out` or `--tsv-out` for baseline comparison
5. If archiving a run, use `--snapshot-dir DIR`
6. Update the snapshot, results, and next steps with commit hashes and dates
