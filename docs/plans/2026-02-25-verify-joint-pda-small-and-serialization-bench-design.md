# Design: verify_joint_pda_small + benchmark_serialization

Date: 2026-02-25
Status: Approved

## Context

Two next-steps from STATUS.md:

1. **Task 6** — Confirm that `train_joint_pda` discovers the stack (`push_configs` non-empty)
   and reproduces the depth-generalization pattern when T_total is kept within the ≤ 2000
   solver limit.  With 300 training sequences T_total ≈ 3000, which exceeds the limit; the
   script reduces training to 100 sequences (T_total ≈ 1100).

2. **Serialization baseline** — Measure JSON round-trip byte size and integer ms for FSM and
   PDA models at three scale points before any binary format work is started.

---

## Script 1: `scripts/verify_joint_pda_small.py`

### Goal

Verify the structural-generalization hypothesis at tractable scale: when joint-PDA has enough
budget to search, it should discover push=`(` pop=`)` and show accuracy that improves (or
holds) with OOD depth while two-phase PDA remains the reference.

### Parameters

| Parameter            | Value        | Rationale |
|----------------------|--------------|-----------|
| `TRAIN_SEQS`         | 100          | T_total ≈ 1100 — comfortable margin below 2000 |
| `vocab_size`         | 3            | OPEN=0, CLOSE=1, EOS=2 — same as full experiment |
| `JPDA_NUM_STATES`    | 4 (2 bits)   | same as full experiment |
| `JPDA_STEPS`         | 30 s         | more budget per sequence than 300-seq run |
| `JPDA_MAX_PUSH/POP`  | 1 / 1        | one push token expected, one pop token |
| `PDA_STATE_BITS`     | 2 (4 states) | same as full experiment |
| `PDA_STEPS`          | 20 s         | same as full experiment |
| `TEST_DEPTHS`        | 3, 4, 5, 6   | narrowed — verifying pattern, not re-running full table |
| `TEST_SEQS_PER_DEPTH`| 30           | fast; adequate for verification |

### Output structure

```
=== Joint-PDA Small-Scale Verification ===
  T_total=<n>  train_seqs=100  seed=42
  JPDA push_tokens=<list>  pop_tokens=<list>
  [PASS] joint-PDA discovered stack  (or [WARN] joint-PDA found no stack — push_configs empty)

depth   PDA-2ph    PDA-jt   seqs
------  ---------  --------  ----
3       xx.xx%     xx.xx%    30
4*      xx.xx%     xx.xx%    30
5*      xx.xx%     xx.xx%    30
6*      xx.xx%     xx.xx%    30

Basis-points: PDA-2ph / PDA-jt
  depth 3 : ...
  ...
```

Only the two PDA variants are compared (FSM/PPM are already validated in the full experiment).

### Stack discovery check

If `push_configs` is empty after solving, print `[WARN]` and continue — never raise.  This
makes the script safe as a regression check.

### T_total reporting

Compute `T_total = sum(len(s) for s in train_data)` and print it in the header so future runs
can verify they remain within budget.

### CLI

`--seed` (default 42), `--train-seqs` (default 100), `--test-seqs-per-depth` (default 30),
`--steps` (default 30), `--quiet`

### Reuse

Data generation and evaluation reuse `reproduce_depth_generalization.py`'s
`gen_train_seqs`, `gen_test_seqs_at_depth`, `_eval_pda_on_seqs`, `_gen_one_balanced`,
`_max_depth_of`.  These will be imported directly (not duplicated).

---

## Script 2: `scripts/benchmark_serialization.py`

### Goal

Establish a stable, diffable baseline of JSON serialization byte size and round-trip time
for FSM and PDA models before any binary format (MessagePack, etc.) is introduced.

### Models benchmarked

| label   | type | num_states | vocab_size | notes |
|---------|------|-----------|------------|-------|
| `fsm-sm`| FSM  | 8         | 30         | small FSM |
| `pda-sm`| PDA  | 8         | 30         | small PDA, same state count as fsm-sm |
| `pda-md`| PDA  | 16        | 30         | medium PDA — tests state-count scaling |

Models are built from a small synthetic corpus using `gen_train_seqs` (reused from
`reproduce_depth_generalization.py`).  No external data needed.

### Metrics (integer-only)

| column         | type | source |
|----------------|------|--------|
| `bytes`        | int  | `len(path.read_bytes())` |
| `save_ms`      | int  | `(perf_counter_ns end − start) // 1_000_000` |
| `load_ms`      | int  | same for load path |
| `roundtrip_ok` | 0/1  | field-by-field equality check on transitions + configs |

No floats anywhere — consistent with project integer-only invariant.

### Temporary file strategy

Use `tempfile.TemporaryDirectory` as context manager so no test artifacts are left on disk.

### Output

Fixed-width table to stdout:

```
label    type  states  vocab  bytes   save_ms  load_ms  ok
-------  ----  ------  -----  ------  -------  -------  --
fsm-sm   fsm       8     30    1234        3        2    1
pda-sm   pda       8     30    2345        5        4    1
pda-md   pda      16     30    4567        8        7    1
```

Optional `--csv-out PATH` (same pattern as `benchmark_matrix.py`).

### CLI

`--seed` (default 42), `--csv-out PATH`

---

## File locations

```
scripts/verify_joint_pda_small.py      (new)
scripts/benchmark_serialization.py     (new)
```

No changes to existing source files.

---

## Success criteria

- `verify_joint_pda_small.py` prints `[PASS]` for stack discovery on default parameters.
- `benchmark_serialization.py` prints `roundtrip_ok=1` for all three rows.
- Both scripts complete without errors under `py -3.12`.
- Full test suite (`py -3.12 -m pytest -q`) continues to pass (no source changes).
