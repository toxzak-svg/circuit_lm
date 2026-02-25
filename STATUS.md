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

## Next Recommended Steps

1. Run `train_joint_pda_cpsat` on the balanced-parens depth-generalization experiment to verify that jointly-learned push/pop tokens match the ground-truth bracket tokens without supervision.
2. Compare depth-generalization curves: joint-PDA vs two-phase-PDA vs FSM vs PPM.
3. Add a serialization benchmark comparing JSON save/load sizes and times before introducing a binary format.

## How To Update This File

1. Re-run `py -3.12 -m pytest -q`
2. Re-run `py -3.12 scripts/benchmark_small.py`
3. Re-run `py -3.12 scripts/benchmark_matrix.py` (optional but recommended)
4. If needed, export `--csv-out` or `--tsv-out` for baseline comparison
5. If archiving a run, use `--snapshot-dir DIR`
6. Update the snapshot, results, and next steps with commit hashes and dates
