# verify_joint_pda_small + benchmark_serialization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two scripts — a small-scale joint-PDA depth-generalization verifier (Task 6) and a JSON serialization baseline benchmark — plus tests for both.

**Architecture:** Both scripts expose a `run_*()` function that returns a plain dict of integer results, making them directly importable by tests without subprocess. Tests use short CP-SAT budgets (≤ 3 s) to exercise structure; the full runs (30 s) are confirmed manually. Integer-only throughout — no float literals anywhere in `scripts/`.

**Tech Stack:** CP-SAT via `ortools.sat.python.cp_model`, existing `circuit_lm.train_*`, `circuit_lm.eval`, `circuit_lm.io`, `time.perf_counter_ns`, `tempfile`, `pathlib`, `argparse`.

---

## Background for the implementer

### Project constraints (read before touching anything)
- **No floats anywhere.** `tests/test_no_floats.py` scans both `circuit_lm/` and `scripts/` for float literals (`1.0`, `3.14`), `float()` calls, `math.log/exp/sqrt`, and numpy/torch/jax imports. If your script has any float, the test suite fails.
- `rng.randint(0, 1) == 0` not `rng.random() < 0.5`.
- Integer timing: `(time.perf_counter_ns() - t0) // 1_000_000` gives ms as `int`. Never divide without `//`.
- Stack tokens: `push_configs` is a `frozenset[tuple[int,int,int]]` where each triple is `(src_state, token_id, stack_top)` with `stack_top = -1` for STACK_EMPTY.
- Run tests with: `py -3.12 -m pytest -q`

### Key imports available
```python
from circuit_lm.train_joint_pda_cpsat import train_joint_pda
from circuit_lm.train_pda_cpsat       import train_pda
from circuit_lm.train_cpsat           import train as train_fsm
from circuit_lm.eval                  import evaluate_pda, evaluate
from circuit_lm.metrics               import format_accuracy, accuracy_pct_times100
from circuit_lm.io                    import save_model, load_model
from circuit_lm.tokenizer             import Tokenizer
```

### T_total calculation
```python
t_total: int = sum(len(s) for s in train_data)
```
With 100 balanced-paren sequences (depth ≤ 3, pairs 2–8, ~11 tokens each), T_total ≈ 1100, well below the 2000 limit.

### Vocab for balanced parens
```python
OPEN:       int = 0   # '('
CLOSE:      int = 1   # ')'
EOS:        int = 2   # end sentinel
VOCAB_SIZE: int = 3
```

---

## Task 1: Tests for `verify_joint_pda_small.py`

**Files:**
- Create: `tests/test_verify_joint_pda_small.py`

### Step 1: Create the test file

```python
"""Tests for scripts/verify_joint_pda_small.py.

Uses a minimal budget (train_seqs=20, steps=2) to test structural
properties only.  Actual stack discovery is verified by running the
script with its default parameters (100 seqs, 30 s).
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from verify_joint_pda_small import run_small, TEST_DEPTHS  # noqa: E402


TINY = dict(seed=42, train_seqs=20, test_seqs_per_depth=5, steps=2, quiet=True)


def test_run_small_returns_dict():
    result = run_small(**TINY)
    assert isinstance(result, dict)


def test_run_small_has_required_keys():
    result = run_small(**TINY)
    for key in ("t_total", "push_tokens", "pop_tokens", "stack_discovered", "results"):
        assert key in result, f"missing key: {key!r}"


def test_run_small_t_total_is_positive_int():
    result = run_small(**TINY)
    assert isinstance(result["t_total"], int)
    assert result["t_total"] > 0


def test_run_small_results_have_all_test_depths():
    result = run_small(**TINY)
    for depth in TEST_DEPTHS:
        assert depth in result["results"], f"depth {depth} missing from results"


def test_run_small_accuracy_pairs_are_int_tuples():
    result = run_small(**TINY)
    for depth, models in result["results"].items():
        for model_name in ("pda", "jpda"):
            assert model_name in models, f"model {model_name!r} missing at depth {depth}"
            correct, total = models[model_name]
            assert isinstance(correct, int)
            assert isinstance(total, int)
            assert total >= 0
            assert 0 <= correct <= total


def test_run_small_stack_discovered_is_bool():
    result = run_small(**TINY)
    assert isinstance(result["stack_discovered"], bool)


def test_run_small_push_pop_tokens_are_int_lists():
    result = run_small(**TINY)
    assert isinstance(result["push_tokens"], list)
    assert isinstance(result["pop_tokens"], list)
    for tok in result["push_tokens"]:
        assert isinstance(tok, int)
    for tok in result["pop_tokens"]:
        assert isinstance(tok, int)
```

### Step 2: Run to verify it fails

```
py -3.12 -m pytest tests/test_verify_joint_pda_small.py -v
```

Expected: `ModuleNotFoundError: No module named 'verify_joint_pda_small'`

### Step 3: Commit the failing test

```bash
git add tests/test_verify_joint_pda_small.py
git commit -m "test: add failing tests for verify_joint_pda_small"
```

---

## Task 2: Implement `scripts/verify_joint_pda_small.py`

**Files:**
- Create: `scripts/verify_joint_pda_small.py`

### Step 1: Write the script

```python
"""Small-scale joint-PDA depth-generalization verifier (Task 6).

Runs the same depth-generalization comparison as reproduce_depth_generalization.py
but with 100 training sequences (T_total ≈ 1100 < 2000 solver limit), so that
the joint-PDA solver has enough budget to discover the stack.

Usage
-----
    py -3.12 scripts/verify_joint_pda_small.py
    py -3.12 scripts/verify_joint_pda_small.py --train-seqs 80 --steps 60
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.eval    import evaluate_pda
from circuit_lm.metrics import format_accuracy, accuracy_pct_times100
from circuit_lm.train_joint_pda_cpsat import train_joint_pda
from circuit_lm.train_pda_cpsat       import train_pda

# ---------------------------------------------------------------------------
# Vocabulary (same as reproduce_depth_generalization.py)
# ---------------------------------------------------------------------------

OPEN:       int = 0
CLOSE:      int = 1
EOS:        int = 2
VOCAB_SIZE: int = 3

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

MAX_TRAIN_DEPTH:        int = 3
TEST_DEPTHS: tuple[int, ...] = (3, 4, 5, 6)
TRAIN_SEQS:             int = 100
TEST_SEQS_PER_DEPTH:    int = 30

PDA_STATE_BITS:  int = 2
PDA_STACK_DEPTH: int = 10
PDA_STEPS:       int = 20
PDA_MAX_PUSH:    int = 1
PDA_MAX_POP:     int = 1

JPDA_NUM_STATES:  int = 4
JPDA_STACK_DEPTH: int = 10
JPDA_STEPS:       int = 30
JPDA_MAX_PUSH:    int = 1
JPDA_MAX_POP:     int = 1

# ---------------------------------------------------------------------------
# Data generation (duplicated from reproduce_depth_generalization.py)
# ---------------------------------------------------------------------------


def _gen_one_balanced(
    max_depth: int,
    min_pairs: int,
    max_pairs: int,
    rng: random.Random,
) -> list[int]:
    n = rng.randint(min_pairs, max_pairs)
    tokens: list[int] = []
    depth = 0
    opens_left = n
    closes_left = n
    while opens_left + closes_left > 0:
        can_open  = opens_left > 0 and depth < max_depth and closes_left > depth
        can_close = closes_left > 0 and depth > 0
        if not can_open and not can_close:
            break
        if can_open and can_close:
            choice = OPEN if rng.randint(0, 1) == 0 else CLOSE
        elif can_open:
            choice = OPEN
        else:
            choice = CLOSE
        if choice == OPEN:
            tokens.append(OPEN);  depth += 1;  opens_left -= 1
        else:
            tokens.append(CLOSE); depth -= 1; closes_left -= 1
    tokens.append(EOS)
    return tokens


def _max_depth_of(tokens: list[int]) -> int:
    depth = 0
    max_d = 0
    for tok in tokens:
        if tok == OPEN:
            depth += 1
            if depth > max_d:
                max_d = depth
        elif tok == CLOSE:
            depth -= 1
    return max_d


def gen_train_seqs(
    max_depth: int, num_seqs: int, seed: int,
    min_pairs: int = 2, max_pairs: int = 8,
) -> list[list[int]]:
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    while len(seqs) < num_seqs:
        seqs.append(_gen_one_balanced(max_depth, min_pairs, max_pairs, rng))
    return seqs


def gen_test_seqs_at_depth(
    target_depth: int, num_seqs: int, seed: int,
    min_pairs: int | None = None, max_pairs: int = 16,
) -> list[list[int]]:
    if min_pairs is None:
        min_pairs = target_depth
    rng = random.Random(seed)
    seqs: list[list[int]] = []
    max_attempts = num_seqs * 10_000
    attempts = 0
    while len(seqs) < num_seqs and attempts < max_attempts:
        seq = _gen_one_balanced(target_depth, min_pairs, max_pairs, rng)
        if _max_depth_of(seq) == target_depth:
            seqs.append(seq)
        attempts += 1
    return seqs

# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


def _eval(model, seqs: list[list[int]]) -> tuple[int, int]:
    r = evaluate_pda(model, seqs)
    return r["correct"], r["total"]  # type: ignore[return-value]

# ---------------------------------------------------------------------------
# Core run function (importable for tests)
# ---------------------------------------------------------------------------


def run_small(
    seed:                int = 42,
    train_seqs:          int = TRAIN_SEQS,
    test_seqs_per_depth: int = TEST_SEQS_PER_DEPTH,
    steps:               int = JPDA_STEPS,
    quiet:               bool = False,
) -> dict:
    """Run the small-scale joint-PDA verification experiment.

    Returns a plain dict so callers (including tests) can inspect results
    without parsing stdout.

    Keys:
        t_total (int)          — total tokens in training data
        push_tokens (list[int])— distinct token IDs in jpda push_configs
        pop_tokens  (list[int])— distinct token IDs in jpda pop_configs
        stack_discovered (bool)— True iff push_configs is non-empty
        results (dict)         — {depth: {"pda": (correct, total),
                                          "jpda": (correct, total)}}
    """
    if not quiet:
        print()
        print("=== Joint-PDA Small-Scale Verification ===")
        print(f"  max_train_depth={MAX_TRAIN_DEPTH}  seed={seed}")
        print(f"  train_seqs={train_seqs}  test_seqs_per_depth={test_seqs_per_depth}")
        print(f"  jpda_steps={steps}s  pda_steps={PDA_STEPS}s")
        print()

    train_data = gen_train_seqs(MAX_TRAIN_DEPTH, train_seqs, seed)
    t_total: int = sum(len(s) for s in train_data)

    if not quiet:
        print(f"  T_total={t_total}  (limit: 2000)")
        print(f"  Training joint-PDA (num_states={JPDA_NUM_STATES}, steps={steps}s) ...")

    jpda_model = train_joint_pda(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        num_states=JPDA_NUM_STATES,
        stack_depth=JPDA_STACK_DEPTH,
        steps=steps,
        max_push=JPDA_MAX_PUSH,
        max_pop=JPDA_MAX_POP,
        top_k_coverage=VOCAB_SIZE,
    )

    push_tokens = sorted({tok for (_, tok, _) in jpda_model.push_configs})
    pop_tokens  = sorted({tok for (_, tok, _) in jpda_model.pop_configs})
    stack_discovered = len(jpda_model.push_configs) > 0

    if not quiet:
        print(f"  jpda push_tokens={push_tokens}  pop_tokens={pop_tokens}")
        if stack_discovered:
            print("  [PASS] joint-PDA discovered stack")
        else:
            print("  [WARN] joint-PDA found no stack — push_configs empty")
        print(f"  Training PDA-2ph (state_bits={PDA_STATE_BITS}, steps={PDA_STEPS}s) ...")

    pda_model = train_pda(
        sequences=train_data,
        vocab_size=VOCAB_SIZE,
        state_bits=PDA_STATE_BITS,
        stack_depth=PDA_STACK_DEPTH,
        steps=PDA_STEPS,
        max_push=PDA_MAX_PUSH,
        max_pop=PDA_MAX_POP,
        top_k_coverage=VOCAB_SIZE,
    )

    if not quiet:
        _pp = sorted({tok for (_, tok, _) in pda_model.push_configs})
        _pop = sorted({tok for (_, tok, _) in pda_model.pop_configs})
        print(f"  pda  push_tokens={_pp}  pop_tokens={_pop}")
        print()
        print("  Evaluating ...")
        _SEP = "-" * 52
        _HDR = f"{'depth':>7}  {'PDA-2ph':>10}  {'PDA-jt':>10}  {'seqs':>6}"
        print(_SEP)
        print(_HDR)
        print(_SEP)

    results: dict[int, dict[str, tuple[int, int]]] = {}

    for depth in TEST_DEPTHS:
        test_data = gen_test_seqs_at_depth(depth, test_seqs_per_depth, seed + depth)
        n = len(test_data)
        pda_res  = _eval(pda_model,  test_data)
        jpda_res = _eval(jpda_model, test_data)
        results[depth] = {"pda": pda_res, "jpda": jpda_res}

        if not quiet:
            ood = "*" if depth > MAX_TRAIN_DEPTH else " "
            print(
                f"{depth:>6}{ood}"
                f"  {format_accuracy(*pda_res):>10}"
                f"  {format_accuracy(*jpda_res):>10}"
                f"  {n:>6}"
            )

    if not quiet:
        print(_SEP)
        print()
        print("  Basis-points (10000 = 100%):  PDA-2ph / PDA-jt")
        for depth in TEST_DEPTHS:
            r = results[depth]
            pda_bp  = accuracy_pct_times100(*r["pda"])
            jpda_bp = accuracy_pct_times100(*r["jpda"])
            ood = "*" if depth > MAX_TRAIN_DEPTH else " "
            print(f"    depth {depth}{ood}: PDA-2ph={pda_bp:5d}  PDA-jt={jpda_bp:5d}")
        print()

    return {
        "t_total":          t_total,
        "push_tokens":      push_tokens,
        "pop_tokens":       pop_tokens,
        "stack_discovered": stack_discovered,
        "results":          results,
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seed",                 type=int, default=42)
    p.add_argument("--train-seqs",           type=int, default=TRAIN_SEQS)
    p.add_argument("--test-seqs-per-depth",  type=int, default=TEST_SEQS_PER_DEPTH)
    p.add_argument("--steps",                type=int, default=JPDA_STEPS)
    p.add_argument("--quiet",                action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_small(
        seed=args.seed,
        train_seqs=args.train_seqs,
        test_seqs_per_depth=args.test_seqs_per_depth,
        steps=args.steps,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
```

### Step 2: Run tests

```
py -3.12 -m pytest tests/test_verify_joint_pda_small.py -v
```

Expected: all 8 tests PASS (uses `steps=2`, so no long solver wait).

### Step 3: Run no-floats check

```
py -3.12 -m pytest tests/test_no_floats.py -v
```

Expected: PASS (no float literals in the new script).

### Step 4: Smoke-run the full script (manual, ~50 s)

```
py -3.12 scripts/verify_joint_pda_small.py
```

Check the output header for `[PASS] joint-PDA discovered stack`.

### Step 5: Commit

```bash
git add scripts/verify_joint_pda_small.py tests/test_verify_joint_pda_small.py
git commit -m "feat: add verify_joint_pda_small script and tests (Task 6)"
```

---

## Task 3: Tests for `scripts/benchmark_serialization.py`

**Files:**
- Create: `tests/test_benchmark_serialization.py`

### Step 1: Write the test file

```python
"""Tests for scripts/benchmark_serialization.py.

Tests structural correctness (roundtrip_ok, integer types) using a tiny
corpus so the benchmark itself runs in < 10 s.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from benchmark_serialization import run_benchmark  # noqa: E402


@pytest.fixture()
def bench_rows(tmp_path: pathlib.Path):
    return run_benchmark(seed=42, tmp_dir=tmp_path)


def test_benchmark_returns_three_rows(bench_rows):
    assert len(bench_rows) == 3


def test_benchmark_row_has_required_keys(bench_rows):
    required = {"label", "type", "states", "vocab", "bytes", "save_ms", "load_ms", "roundtrip_ok"}
    for row in bench_rows:
        assert required <= row.keys(), f"missing keys in row {row['label']!r}"


def test_benchmark_all_roundtrips_ok(bench_rows):
    for row in bench_rows:
        assert row["roundtrip_ok"] == 1, f"roundtrip failed for {row['label']!r}"


def test_benchmark_bytes_positive_int(bench_rows):
    for row in bench_rows:
        assert isinstance(row["bytes"], int), f"bytes not int for {row['label']!r}"
        assert row["bytes"] > 0


def test_benchmark_save_load_ms_nonneg_ints(bench_rows):
    for row in bench_rows:
        assert isinstance(row["save_ms"], int)
        assert isinstance(row["load_ms"], int)
        assert row["save_ms"] >= 0
        assert row["load_ms"] >= 0


def test_benchmark_roundtrip_ok_is_zero_or_one(bench_rows):
    for row in bench_rows:
        assert row["roundtrip_ok"] in (0, 1)


def test_benchmark_row_types_are_fsm_or_pda(bench_rows):
    for row in bench_rows:
        assert row["type"] in ("fsm", "pda")


def test_benchmark_labels_are_unique(bench_rows):
    labels = [r["label"] for r in bench_rows]
    assert len(labels) == len(set(labels))
```

### Step 2: Run to verify it fails

```
py -3.12 -m pytest tests/test_benchmark_serialization.py -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark_serialization'`

### Step 3: Commit the failing test

```bash
git add tests/test_benchmark_serialization.py
git commit -m "test: add failing tests for benchmark_serialization"
```

---

## Task 4: Implement `scripts/benchmark_serialization.py`

**Files:**
- Create: `scripts/benchmark_serialization.py`

### Step 1: Write the script

```python
"""JSON serialization benchmark for CircuitLM and PDACircuitLM models.

Measures round-trip byte size, save time, and load time for three model
configurations using the existing JSON save/load path in circuit_lm.io.
Establishes a stable integer-only baseline before any binary format work.

All timings are integer milliseconds.  No floats anywhere.

Usage
-----
    py -3.12 scripts/benchmark_serialization.py
    py -3.12 scripts/benchmark_serialization.py --csv-out results/ser_bench.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.circuits  import CircuitLM
from circuit_lm.io        import save_model, load_model
from circuit_lm.pda       import PDACircuitLM
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.train_cpsat      import train as train_fsm
from circuit_lm.train_pda_cpsat  import train_pda

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

import random as _random


def _make_random_seqs(
    num_seqs: int, seq_len: int, vocab_size: int, seed: int
) -> list[list[int]]:
    """Random integer token sequences — no linguistic structure needed for a serialisation bench."""
    rng = _random.Random(seed)
    return [
        [rng.randint(0, vocab_size - 1) for _ in range(seq_len)]
        for _ in range(num_seqs)
    ]


def _make_tokenizer(vocab_size: int) -> Tokenizer:
    """Build a minimal char tokenizer with *vocab_size* printable characters."""
    chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789!@#$%^&*()-+=[]{}|;:,.<>?"
    )
    return Tokenizer.from_chars(list(chars[:vocab_size]))

# ---------------------------------------------------------------------------
# Benchmark row
# ---------------------------------------------------------------------------


def _bench_one(
    label: str,
    model_type: str,
    num_states: int,
    vocab_size: int,
    seed: int,
    tmp_dir: pathlib.Path,
) -> dict:
    """Train, save, load, and compare one model. Returns an integer-only row dict."""
    seqs = _make_random_seqs(
        num_seqs=30, seq_len=20, vocab_size=vocab_size, seed=seed
    )
    tokenizer = _make_tokenizer(vocab_size)
    out_path = tmp_dir / f"{label}.json"

    if model_type == "fsm":
        model = train_fsm(
            sequences=seqs, vocab_size=vocab_size,
            state_bits=num_states.bit_length() - 1, steps=3,
        )
    else:
        model = train_pda(
            sequences=seqs, vocab_size=vocab_size,
            state_bits=num_states.bit_length() - 1,
            stack_depth=1, steps=3,
        )

    # --- save ---
    t0 = time.perf_counter_ns()
    save_model(model, tokenizer, out_path)
    save_ms: int = (time.perf_counter_ns() - t0) // 1_000_000

    file_bytes: int = len(out_path.read_bytes())

    # --- load ---
    t1 = time.perf_counter_ns()
    loaded_model, _ = load_model(out_path)
    load_ms: int = (time.perf_counter_ns() - t1) // 1_000_000

    # --- roundtrip check ---
    roundtrip_ok: int = 1 if _models_equal(model, loaded_model) else 0

    return {
        "label":        label,
        "type":         model_type,
        "states":       num_states,
        "vocab":        vocab_size,
        "bytes":        file_bytes,
        "save_ms":      save_ms,
        "load_ms":      load_ms,
        "roundtrip_ok": roundtrip_ok,
    }


def _models_equal(a, b) -> bool:
    """Field-by-field equality for CircuitLM and PDACircuitLM."""
    if type(a) is not type(b):
        return False
    if isinstance(a, PDACircuitLM):
        return (
            a.vocab_size           == b.vocab_size
            and a.num_states       == b.num_states
            and a.transitions      == b.transitions
            and a.push_configs     == b.push_configs
            and a.pop_configs      == b.pop_configs
            and a.config_pred_tokens == b.config_pred_tokens
        )
    # CircuitLM (FSM)
    return (
        a.vocab_size      == b.vocab_size
        and a.num_states  == b.num_states
        and a.transitions == b.transitions
        and a.pred_tokens == b.pred_tokens
    )

# ---------------------------------------------------------------------------
# BENCH_CONFIGS: (label, type, num_states, vocab_size)
# ---------------------------------------------------------------------------

BENCH_CONFIGS: list[tuple[str, str, int, int]] = [
    ("fsm-sm", "fsm",  8, 30),
    ("pda-sm", "pda",  8, 30),
    ("pda-md", "pda", 16, 30),
]

# ---------------------------------------------------------------------------
# Public run function (importable for tests)
# ---------------------------------------------------------------------------


def run_benchmark(
    seed: int = 42,
    tmp_dir: pathlib.Path | None = None,
) -> list[dict]:
    """Run the serialization benchmark and return a list of row dicts."""
    rows: list[dict] = []
    if tmp_dir is not None:
        for i, (label, mtype, ns, vs) in enumerate(BENCH_CONFIGS):
            rows.append(_bench_one(label, mtype, ns, vs, seed + i, tmp_dir))
        return rows

    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        for i, (label, mtype, ns, vs) in enumerate(BENCH_CONFIGS):
            rows.append(_bench_one(label, mtype, ns, vs, seed + i, td_path))
    return rows

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_HDR  = f"{'label':<8}  {'type':<4}  {'states':>6}  {'vocab':>5}  {'bytes':>8}  {'save_ms':>7}  {'load_ms':>7}  {'ok':>2}"
_SEP  = "-" * len(_HDR)


def _print_table(rows: list[dict]) -> None:
    print(_SEP)
    print(_HDR)
    print(_SEP)
    for r in rows:
        print(
            f"{r['label']:<8}  {r['type']:<4}  {r['states']:>6}  {r['vocab']:>5}"
            f"  {r['bytes']:>8}  {r['save_ms']:>7}  {r['load_ms']:>7}  {r['roundtrip_ok']:>2}"
        )
    print(_SEP)


def _write_csv(rows: list[dict], path: pathlib.Path) -> None:
    header = "label,type,states,vocab,bytes,save_ms,load_ms,roundtrip_ok"
    lines  = [header]
    for r in rows:
        lines.append(
            f"{r['label']},{r['type']},{r['states']},{r['vocab']},"
            f"{r['bytes']},{r['save_ms']},{r['load_ms']},{r['roundtrip_ok']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--csv-out",  type=pathlib.Path, default=None,
                   metavar="PATH", help="Write results to CSV file")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    print()
    print("=== JSON Serialization Benchmark ===")
    print(f"  seed={args.seed}  configs={len(BENCH_CONFIGS)}")
    print()
    rows = run_benchmark(seed=args.seed)
    _print_table(rows)
    if args.csv_out:
        _write_csv(rows, args.csv_out)
        print(f"  CSV written to {args.csv_out}")
    print()


if __name__ == "__main__":
    main()
```

**Note on `Tokenizer.from_chars`:** if that method does not exist, use `Tokenizer.from_text("".join(chars[:vocab_size]) * 5, vocab_size=vocab_size)` instead and adjust import. Check `circuit_lm/tokenizer.py` before writing.

### Step 2: Verify Tokenizer API

Run a quick check before finishing the implementation:

```
py -3.12 -c "from circuit_lm.tokenizer import Tokenizer; help(Tokenizer)"
```

Adjust `_make_tokenizer` accordingly if `from_chars` is not available.

### Step 3: Run tests

```
py -3.12 -m pytest tests/test_benchmark_serialization.py -v
```

Expected: all 8 tests PASS.

### Step 4: Run no-floats check

```
py -3.12 -m pytest tests/test_no_floats.py -v
```

Expected: PASS.

### Step 5: Smoke-run the full script (manual, ~30 s)

```
py -3.12 scripts/benchmark_serialization.py
```

Check that all three rows show `ok = 1`.

### Step 6: Commit

```bash
git add scripts/benchmark_serialization.py tests/test_benchmark_serialization.py
git commit -m "feat: add benchmark_serialization script and tests"
```

---

## Task 5: Full test suite + STATUS.md update

### Step 1: Run full test suite

```
py -3.12 -m pytest -q
```

Expected: all previously passing tests still pass, plus the new ones.

### Step 2: Update STATUS.md

Add two new sections under "What Is Working":
- `verify_joint_pda_small.py`: whether `[PASS]` was observed, T_total reported, push/pop tokens found.
- `benchmark_serialization.py`: the three-row table output (bytes / save_ms / load_ms).

Update "Next Recommended Steps" to remove completed items and add:
- Introduce compressed binary format (MessagePack) using the serialization benchmark as baseline.

Follow the "How To Update This File" instructions at the bottom of `STATUS.md`.

### Step 3: Commit STATUS.md

```bash
git add STATUS.md
git commit -m "docs: update STATUS.md with Task 6 verification and serialization bench results"
```

---

## Quick reference: integer-only checklist for new scripts

| Pattern | Instead of |
|---------|-----------|
| `(t1 - t0) // 1_000_000` | `(t1 - t0) / 1_000_000` |
| `rng.randint(0, 1) == 0` | `rng.random() < 0.5` |
| `accuracy_pct_times100(c, t)` from `circuit_lm.metrics` | manual `c/t*100` |
| `format_accuracy(c, t)` from `circuit_lm.metrics` | f-string with `%` |
| Integer constant `3` | float `3.0` |
