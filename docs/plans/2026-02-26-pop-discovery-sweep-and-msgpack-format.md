# Pop Discovery Sweep + MessagePack Binary Format Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** (1) Add `scripts/sweep_jpda_budget.py` to investigate which `(train_seqs, steps)` parameters enable full joint-PDA stack discovery (both push and pop); (2) add `save_msgpack`/`load_msgpack` to `circuit_lm/io.py` using the `msgpack` library; (3) extend `benchmark_serialization.py` with paired msgpack rows for direct JSON vs binary size comparison.

**Architecture:** TDD throughout. The sweep script imports `run_small` from `verify_joint_pda_small`. The msgpack functions live in `circuit_lm/io.py` alongside the existing JSON functions. The benchmark refactors `_bench_one` to accept a `format` parameter so the same trained model is reused for both JSON and msgpack rows (no extra CP-SAT solves).

**Tech Stack:** `msgpack>=1.0` (new dep), Python stdlib `pathlib`/`time`/`tempfile`; existing `circuit_lm.io`, `circuit_lm.train_*`, `ortools`.

---

## Background for the implementer

### Project constraints
- **No floats anywhere.** `tests/test_no_floats.py` scans `circuit_lm/` and `scripts/` for float literals, `float()`, `math.log/exp/sqrt`, numpy/torch/jax imports. `msgpack` is NOT on the forbidden list.
- `rng.randint(0, 1) == 0` not `rng.random() < 0.5`.
- Integer timing: `(time.perf_counter_ns() - t0) // 1_000_000` gives ms as `int`. Never divide without `//`.
- Run tests with: `py -3.12 -m pytest -q`

### Tokenizer API
`Tokenizer` has no `from_chars` method. Use `from_text`:
```python
text = "abcdefghijklmnopqrstuvwxyz0123456789!@#$"[:vocab_size] * 20
tok = Tokenizer.from_text(text, vocab_size=vocab_size)
```

### msgpack encoding keys
JSON uses string keys for sparse dicts (e.g., `"2,-1"` for a config).
msgpack uses integer keys — more compact:

```python
# Transition key:   state * vocab_size + token
# Config key:       state * ST_RANGE + st_enc
#   where ST_RANGE  = vocab_size + 1
#   and   st_enc    = vocab_size   if stack_top == STACK_EMPTY (-1)
#                   = stack_top    otherwise
```

Both encodings are injective (reversible given `vocab_size`).

### Key imports
```python
from circuit_lm.circuits  import CircuitLM
from circuit_lm.pda       import PDACircuitLM, STACK_EMPTY
from circuit_lm.io        import save_model, load_model, save_msgpack, load_msgpack
from circuit_lm.tokenizer import Tokenizer
import msgpack
```

---

## Task 1: Failing tests for `sweep_jpda_budget.py`

**Files:**
- Create: `tests/test_sweep_jpda_budget.py`

### Step 1: Write the test file

```python
"""Tests for scripts/sweep_jpda_budget.py.

Uses a 2-entry TINY_GRID (steps=2) to verify structure only.
Full stack discovery is NOT expected at steps=2 — only dict shape matters.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
from sweep_jpda_budget import run_sweep  # noqa: E402


TINY_GRID = [(10, 2), (15, 2)]
TINY_KWARGS = dict(param_grid=TINY_GRID, seed=42, test_seqs_per_depth=3, quiet=True)


def test_run_sweep_returns_list():
    rows = run_sweep(**TINY_KWARGS)
    assert isinstance(rows, list)


def test_run_sweep_row_count_matches_grid():
    rows = run_sweep(**TINY_KWARGS)
    assert len(rows) == len(TINY_GRID)


def test_run_sweep_row_keys():
    rows = run_sweep(**TINY_KWARGS)
    required = {
        "train_seqs", "steps", "t_total",
        "push_discovered", "pop_discovered", "full_stack_discovered",
        "push_tokens", "pop_tokens",
    }
    for row in rows:
        assert required <= row.keys(), f"missing keys in row {row}"


def test_run_sweep_types():
    rows = run_sweep(**TINY_KWARGS)
    for row in rows:
        assert isinstance(row["train_seqs"], int)
        assert isinstance(row["steps"], int)
        assert isinstance(row["t_total"], int)
        assert row["t_total"] > 0
        assert isinstance(row["push_discovered"], bool)
        assert isinstance(row["pop_discovered"], bool)
        assert isinstance(row["full_stack_discovered"], bool)
        assert isinstance(row["push_tokens"], list)
        assert isinstance(row["pop_tokens"], list)


def test_run_sweep_full_stack_is_conjunction():
    rows = run_sweep(**TINY_KWARGS)
    for row in rows:
        expected = row["push_discovered"] and row["pop_discovered"]
        assert row["full_stack_discovered"] == expected


def test_run_sweep_params_match_grid():
    rows = run_sweep(**TINY_KWARGS)
    for row, (ns, ss) in zip(rows, TINY_GRID):
        assert row["train_seqs"] == ns
        assert row["steps"] == ss
```

### Step 2: Run to verify it fails

```
py -3.12 -m pytest tests/test_sweep_jpda_budget.py -v
```

Expected: `ModuleNotFoundError: No module named 'sweep_jpda_budget'`

### Step 3: Commit the failing test

```bash
git add tests/test_sweep_jpda_budget.py
git commit -m "test: add failing tests for sweep_jpda_budget"
```

---

## Task 2: Implement `scripts/sweep_jpda_budget.py`

**Files:**
- Create: `scripts/sweep_jpda_budget.py`

### Step 1: Write the script

```python
"""Joint-PDA pop-discovery parameter sweep (investigation script).

Calls run_small() from verify_joint_pda_small with each (train_seqs, steps)
combination in a parameter grid and reports which configurations enable the
solver to discover both push and pop operations.

Usage
-----
    py -3.12 scripts/sweep_jpda_budget.py
    py -3.12 scripts/sweep_jpda_budget.py --train-seqs 50 100 --steps 60 120
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from verify_joint_pda_small import run_small  # noqa: E402

# ---------------------------------------------------------------------------
# Default sweep grid: (train_seqs, steps)
# ---------------------------------------------------------------------------

SWEEP_GRID: list[tuple[int, int]] = [
    (50,  30),
    (50,  60),
    (50, 120),
    (100,  60),
    (100, 120),
]

# ---------------------------------------------------------------------------
# Core sweep function (importable for tests)
# ---------------------------------------------------------------------------


def run_sweep(
    param_grid: list[tuple[int, int]],
    seed: int = 42,
    test_seqs_per_depth: int = 10,
    quiet: bool = False,
) -> list[dict]:
    """Run run_small for each (train_seqs, steps) combo in param_grid.

    Returns a list of row dicts with these keys:
        train_seqs (int)             — number of training sequences
        steps (int)                  — CP-SAT time budget in seconds
        t_total (int)                — total tokens in training data
        push_discovered (bool)       — True iff push_configs is non-empty
        pop_discovered (bool)        — True iff pop_configs is non-empty
        full_stack_discovered (bool) — True iff both push AND pop discovered
        push_tokens (list[int])      — distinct token IDs in push_configs
        pop_tokens (list[int])       — distinct token IDs in pop_configs
    """
    rows: list[dict] = []
    for train_seqs, steps in param_grid:
        if not quiet:
            print(
                f"  [{len(rows) + 1}/{len(param_grid)}]"
                f"  train_seqs={train_seqs}  steps={steps}s ..."
            )
        result = run_small(
            seed=seed,
            train_seqs=train_seqs,
            test_seqs_per_depth=test_seqs_per_depth,
            steps=steps,
            quiet=True,
        )
        push_discovered       = len(result["push_tokens"]) > 0
        pop_discovered        = len(result["pop_tokens"])  > 0
        full_stack_discovered = push_discovered and pop_discovered
        rows.append({
            "train_seqs":            train_seqs,
            "steps":                 steps,
            "t_total":               result["t_total"],
            "push_discovered":       push_discovered,
            "pop_discovered":        pop_discovered,
            "full_stack_discovered": full_stack_discovered,
            "push_tokens":           result["push_tokens"],
            "pop_tokens":            result["pop_tokens"],
        })
    return rows

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_HDR = (
    f"{'train_seqs':>10}  {'steps':>5}  {'T_total':>7}"
    f"  {'push':>5}  {'pop':>5}  {'full':>5}"
    f"  push_toks  pop_toks"
)
_SEP = "-" * 72


def _print_table(rows: list[dict]) -> None:
    print(_SEP)
    print(_HDR)
    print(_SEP)
    for r in rows:
        push_s = "YES" if r["push_discovered"] else "no"
        pop_s  = "YES" if r["pop_discovered"]  else "no"
        full_s = "YES" if r["full_stack_discovered"] else "no"
        print(
            f"{r['train_seqs']:>10}  {r['steps']:>5}  {r['t_total']:>7}"
            f"  {push_s:>5}  {pop_s:>5}  {full_s:>5}"
            f"  {r['push_tokens']}  {r['pop_tokens']}"
        )
    print(_SEP)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--seed",                type=int, default=42)
    p.add_argument("--test-seqs-per-depth", type=int, default=10)
    p.add_argument("--train-seqs", type=int, nargs="+", default=None,
                   metavar="N", help="Override train_seqs values (replaces default grid)")
    p.add_argument("--steps",      type=int, nargs="+", default=None,
                   metavar="S", help="Override steps values (replaces default grid)")
    p.add_argument("--quiet",      action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.train_seqs is not None and args.steps is not None:
        grid: list[tuple[int, int]] = [
            (n, s) for n in args.train_seqs for s in args.steps
        ]
    elif args.train_seqs is not None:
        default_steps = sorted({s for (_, s) in SWEEP_GRID})
        grid = [(n, s) for n in args.train_seqs for s in default_steps]
    elif args.steps is not None:
        default_seqs = sorted({n for (n, _) in SWEEP_GRID})
        grid = [(n, s) for n in default_seqs for s in args.steps]
    else:
        grid = list(SWEEP_GRID)

    print()
    print("=== Joint-PDA Pop Discovery Sweep ===")
    print(f"  seed={args.seed}  grid_size={len(grid)}")
    print()

    rows = run_sweep(
        param_grid=grid,
        seed=args.seed,
        test_seqs_per_depth=args.test_seqs_per_depth,
        quiet=args.quiet,
    )
    _print_table(rows)

    full_found = [r for r in rows if r["full_stack_discovered"]]
    if full_found:
        print(
            f"\n  [PASS] {len(full_found)}/{len(rows)} configs"
            " achieved full stack discovery (push+pop)."
        )
    else:
        print(
            f"\n  [WARN] No config achieved full stack discovery"
            " (push+pop) in this sweep."
        )
    print()


if __name__ == "__main__":
    main()
```

### Step 2: Run tests

```
py -3.12 -m pytest tests/test_sweep_jpda_budget.py -v
```

Expected: all 6 tests PASS.

### Step 3: Run no-floats check

```
py -3.12 -m pytest tests/test_no_floats.py -v
```

Expected: PASS.

### Step 4: Commit

```bash
git add scripts/sweep_jpda_budget.py tests/test_sweep_jpda_budget.py
git commit -m "feat: add sweep_jpda_budget script and tests for pop discovery investigation"
```

---

## Task 3: Add msgpack dependency

**Files:**
- Modify: `pyproject.toml`

### Step 1: Add msgpack to dependencies

In `pyproject.toml`, change:

```toml
dependencies = [
    "ortools>=9.8",
]
```

to:

```toml
dependencies = [
    "ortools>=9.8",
    "msgpack>=1.0",
]
```

### Step 2: Install

```
py -3.12 -m pip install --user -e .[dev]
```

### Step 3: Verify import

```
py -3.12 -c "import msgpack; print(msgpack.__version__)"
```

Expected: version string printed (e.g., `1.1.1`).

### Step 4: Commit

```bash
git add pyproject.toml
git commit -m "feat: add msgpack dependency for binary model serialization"
```

---

## Task 4: Failing tests for msgpack save/load

**Files:**
- Create: `tests/test_io_msgpack.py`

### Step 1: Write the test file

```python
"""Tests for save_msgpack / load_msgpack in circuit_lm.io.

Verifies roundtrip correctness for CircuitLM (FSM) and PDACircuitLM.
Also checks that msgpack files are strictly smaller than equivalent JSON.
"""
from __future__ import annotations

import pathlib
import random

import pytest

from circuit_lm.circuits  import CircuitLM
from circuit_lm.io        import save_model, load_model, save_msgpack, load_msgpack
from circuit_lm.pda       import PDACircuitLM, STACK_EMPTY
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.train_cpsat     import train as train_fsm
from circuit_lm.train_pda_cpsat import train_pda


def _make_tokenizer(vocab_size: int) -> Tokenizer:
    chars = "abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()"
    text = chars[:vocab_size] * 20
    return Tokenizer.from_text(text, vocab_size=vocab_size)


def _make_fsm(vocab_size: int = 4, num_states: int = 4) -> CircuitLM:
    return CircuitLM(
        vocab_size=vocab_size,
        num_states=num_states,
        state_bits=2,
        transitions={(0, 0): 1, (1, 1): 2, (2, 2): 3},
        state_counts={0: [5, 2, 1, 0], 1: [0, 4, 1, 0], 2: [0, 0, 3, 1]},
        pred_tokens={0: 1, 1: 1, 2: 3},
    )


def _make_pda(vocab_size: int = 3, num_states: int = 4) -> PDACircuitLM:
    return PDACircuitLM(
        vocab_size=vocab_size,
        num_states=num_states,
        state_bits=2,
        stack_depth=10,
        push_configs=frozenset({(0, 0, STACK_EMPTY), (1, 0, 0)}),
        pop_configs=frozenset({(0, 1, 0)}),
        transitions={(0, 0): 1, (1, 1): 0},
        config_counts={
            (0, STACK_EMPTY): [3, 1, 0],
            (0, 0):           [0, 2, 1],
            (1, STACK_EMPTY): [1, 0, 2],
        },
        config_pred_tokens={(0, STACK_EMPTY): 0, (0, 0): 1},
    )


# ---------------------------------------------------------------------------
# FSM roundtrip
# ---------------------------------------------------------------------------

def test_fsm_save_creates_file(tmp_path: pathlib.Path):
    model = _make_fsm()
    tok = _make_tokenizer(model.vocab_size)
    p = tmp_path / "model.msgpack"
    save_msgpack(model, tok, p)
    assert p.exists()
    assert p.stat().st_size > 0


def test_fsm_load_returns_circuit_lm(tmp_path: pathlib.Path):
    model = _make_fsm()
    tok = _make_tokenizer(model.vocab_size)
    p = tmp_path / "model.msgpack"
    save_msgpack(model, tok, p)
    loaded, _ = load_msgpack(p)
    assert isinstance(loaded, CircuitLM)


def test_fsm_roundtrip_fields(tmp_path: pathlib.Path):
    model = _make_fsm()
    tok = _make_tokenizer(model.vocab_size)
    p = tmp_path / "model.msgpack"
    save_msgpack(model, tok, p)
    loaded, loaded_tok = load_msgpack(p)
    assert loaded.vocab_size  == model.vocab_size
    assert loaded.num_states  == model.num_states
    assert loaded.state_bits  == model.state_bits
    assert loaded.transitions == model.transitions
    assert loaded.state_counts == model.state_counts
    assert loaded.pred_tokens == model.pred_tokens


def test_fsm_msgpack_smaller_than_json(tmp_path: pathlib.Path):
    rng = random.Random(42)
    seqs = [[rng.randint(0, 7) for _ in range(20)] for _ in range(30)]
    tok = _make_tokenizer(8)
    model = train_fsm(sequences=seqs, vocab_size=8, state_bits=3, steps=2)
    json_path    = tmp_path / "model.json"
    msgpack_path = tmp_path / "model.msgpack"
    save_model(model, tok, json_path)
    save_msgpack(model, tok, msgpack_path)
    assert msgpack_path.stat().st_size < json_path.stat().st_size


# ---------------------------------------------------------------------------
# PDA roundtrip
# ---------------------------------------------------------------------------

def test_pda_save_creates_file(tmp_path: pathlib.Path):
    model = _make_pda()
    tok = _make_tokenizer(model.vocab_size)
    p = tmp_path / "model.msgpack"
    save_msgpack(model, tok, p)
    assert p.exists()
    assert p.stat().st_size > 0


def test_pda_load_returns_pda_circuit_lm(tmp_path: pathlib.Path):
    model = _make_pda()
    tok = _make_tokenizer(model.vocab_size)
    p = tmp_path / "model.msgpack"
    save_msgpack(model, tok, p)
    loaded, _ = load_msgpack(p)
    assert isinstance(loaded, PDACircuitLM)


def test_pda_roundtrip_fields(tmp_path: pathlib.Path):
    model = _make_pda()
    tok = _make_tokenizer(model.vocab_size)
    p = tmp_path / "model.msgpack"
    save_msgpack(model, tok, p)
    loaded, _ = load_msgpack(p)
    assert loaded.vocab_size          == model.vocab_size
    assert loaded.num_states          == model.num_states
    assert loaded.state_bits          == model.state_bits
    assert loaded.stack_depth         == model.stack_depth
    assert loaded.push_configs        == model.push_configs
    assert loaded.pop_configs         == model.pop_configs
    assert loaded.transitions         == model.transitions
    assert loaded.config_counts       == model.config_counts
    assert loaded.config_pred_tokens  == model.config_pred_tokens


def test_pda_msgpack_smaller_than_json(tmp_path: pathlib.Path):
    rng = random.Random(42)
    seqs = [[rng.randint(0, 2) for _ in range(15)] for _ in range(30)]
    tok = _make_tokenizer(3)
    model = train_pda(
        sequences=seqs, vocab_size=3, state_bits=2, stack_depth=5, steps=2
    )
    json_path    = tmp_path / "model.json"
    msgpack_path = tmp_path / "model.msgpack"
    save_model(model, tok, json_path)
    save_msgpack(model, tok, msgpack_path)
    assert msgpack_path.stat().st_size < json_path.stat().st_size
```

### Step 2: Run to verify it fails

```
py -3.12 -m pytest tests/test_io_msgpack.py -v
```

Expected: `ImportError: cannot import name 'save_msgpack' from 'circuit_lm.io'`

### Step 3: Commit the failing test

```bash
git add tests/test_io_msgpack.py
git commit -m "test: add failing tests for msgpack save/load"
```

---

## Task 5: Implement msgpack save/load in `circuit_lm/io.py`

**Files:**
- Modify: `circuit_lm/io.py`

### Step 1: Add `import msgpack` at the top of `io.py`

After `import json` and `import pathlib`, add:

```python
import msgpack
```

### Step 2: Add encoding comment block after `AnyModel` typedef

After the line `AnyModel = Union[CircuitLM, PDACircuitLM, PPMModel]`, add:

```python

# ---------------------------------------------------------------------------
# MessagePack key-encoding helpers
# ---------------------------------------------------------------------------
#
# Transitions: (state, token) → int key = state * vocab_size + token
# Configs:     (state, stack_top) → int key = state * ST_RANGE + st_enc
#   where ST_RANGE = vocab_size + 1
#   and   st_enc   = vocab_size  if stack_top == STACK_EMPTY (-1)
#                  = stack_top   otherwise
#
# Both encodings are injective and exactly reversible given vocab_size.
```

### Step 3: Add `save_msgpack` and `load_msgpack` at the end of `io.py`

Append these functions after `_load_pda`:

```python
# ---------------------------------------------------------------------------
# MessagePack save / load
# ---------------------------------------------------------------------------


def save_msgpack(
    model: AnyModel,
    tokenizer: Tokenizer,
    path: str | pathlib.Path,
) -> None:
    """Serialise *model* and *tokenizer* to a binary MessagePack file at *path*.

    Produces a more compact representation than JSON by using integer map
    keys and no whitespace.  Only FSM and PDA models are supported (PPM is
    JSON-only).
    """
    if isinstance(model, PDACircuitLM):
        payload = _encode_pda_msgpack(model, tokenizer)
    else:
        payload = _encode_fsm_msgpack(model, tokenizer)
    pathlib.Path(path).write_bytes(msgpack.packb(payload, use_bin_type=True))


def _encode_fsm_msgpack(model: CircuitLM, tokenizer: Tokenizer) -> dict:
    vs = model.vocab_size
    return {
        "model_type":   "fsm",
        "vocab_size":   vs,
        "num_states":   model.num_states,
        "state_bits":   model.state_bits,
        "transitions":  {s * vs + t: ns for (s, t), ns in model.transitions.items()},
        "state_counts": {s: counts for s, counts in model.state_counts.items()},
        "pred_tokens":  {s: tok for s, tok in model.pred_tokens.items()},
        "tokenizer":    tokenizer.to_dict(),
    }


def _encode_pda_msgpack(model: PDACircuitLM, tokenizer: Tokenizer) -> dict:
    vs = model.vocab_size
    st_range  = vs + 1
    empty_enc = vs

    def _st_enc(stack_top: int) -> int:
        return empty_enc if stack_top == STACK_EMPTY else stack_top

    return {
        "model_type":         "pda",
        "vocab_size":         vs,
        "num_states":         model.num_states,
        "state_bits":         model.state_bits,
        "stack_depth":        model.stack_depth,
        "push_configs":       sorted([s, tok, st] for (s, tok, st) in model.push_configs),
        "pop_configs":        sorted([s, tok, st] for (s, tok, st) in model.pop_configs),
        "transitions":        {s * vs + t: ns for (s, t), ns in model.transitions.items()},
        "config_counts":      {
            s * st_range + _st_enc(st): counts
            for (s, st), counts in model.config_counts.items()
        },
        "config_pred_tokens": {
            s * st_range + _st_enc(st): tok
            for (s, st), tok in model.config_pred_tokens.items()
        },
        "tokenizer":          tokenizer.to_dict(),
    }


def load_msgpack(path: str | pathlib.Path) -> tuple[AnyModel, Tokenizer]:
    """Load a model and tokenizer from a binary MessagePack file.

    Auto-detects model type from the ``"model_type"`` field.

    Returns:
        ``(model, tokenizer)``

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError / ValueError: If the file is malformed.
    """
    data = msgpack.unpackb(
        pathlib.Path(path).read_bytes(),
        raw=False,
        strict_map_key=False,
    )
    model_type = data.get("model_type", "fsm")
    tokenizer  = Tokenizer.from_dict(data["tokenizer"])
    if model_type == "pda":
        return _decode_pda_msgpack(data), tokenizer
    return _decode_fsm_msgpack(data), tokenizer


def _decode_fsm_msgpack(data: dict) -> CircuitLM:
    vs = int(data["vocab_size"])
    return CircuitLM(
        vocab_size  = vs,
        num_states  = int(data["num_states"]),
        state_bits  = int(data["state_bits"]),
        transitions = {(k // vs, k % vs): int(ns)
                       for k, ns in data["transitions"].items()},
        state_counts = {int(s): [int(c) for c in counts]
                        for s, counts in data["state_counts"].items()},
        pred_tokens  = {int(s): int(tok)
                        for s, tok in data.get("pred_tokens", {}).items()},
    )


def _decode_pda_msgpack(data: dict) -> PDACircuitLM:
    vs        = int(data["vocab_size"])
    st_range  = vs + 1
    empty_enc = vs

    def _decode_st(st_enc: int) -> int:
        return STACK_EMPTY if int(st_enc) == empty_enc else int(st_enc)

    return PDACircuitLM(
        vocab_size   = vs,
        num_states   = int(data["num_states"]),
        state_bits   = int(data["state_bits"]),
        stack_depth  = int(data["stack_depth"]),
        push_configs = frozenset(
            (int(t[0]), int(t[1]), int(t[2])) for t in data["push_configs"]
        ),
        pop_configs  = frozenset(
            (int(t[0]), int(t[1]), int(t[2])) for t in data.get("pop_configs", [])
        ),
        transitions  = {(k // vs, k % vs): int(ns)
                        for k, ns in data["transitions"].items()},
        config_counts = {
            (k // st_range, _decode_st(k % st_range)): [int(c) for c in counts]
            for k, counts in data["config_counts"].items()
        },
        config_pred_tokens = {
            (k // st_range, _decode_st(k % st_range)): int(tok)
            for k, tok in data.get("config_pred_tokens", {}).items()
        },
    )
```

### Step 4: Run tests

```
py -3.12 -m pytest tests/test_io_msgpack.py -v
```

Expected: all 8 tests PASS.

### Step 5: Run no-floats and forbidden-imports checks

```
py -3.12 -m pytest tests/test_no_floats.py tests/test_forbidden_imports.py -v
```

Expected: both PASS. (`msgpack` is not in FORBIDDEN_IMPORTS.)

### Step 6: Commit

```bash
git add circuit_lm/io.py
git commit -m "feat: add save_msgpack/load_msgpack binary serialization to io.py"
```

---

## Task 6: Extend benchmark_serialization.py with msgpack rows

**Files:**
- Modify: `scripts/benchmark_serialization.py`
- Modify: `tests/test_benchmark_serialization.py`

### Step 1: Update test to expect 6 rows

In `tests/test_benchmark_serialization.py`:

1. Rename `test_benchmark_returns_three_rows` → `test_benchmark_returns_six_rows` and change `== 3` to `== 6`.
2. Change `test_benchmark_row_types_are_fsm_or_pda` to also allow `"fsm-mp"` and `"pda-mp"`:

```python
def test_benchmark_returns_six_rows(bench_rows):
    assert len(bench_rows) == 6


def test_benchmark_row_types_are_valid(bench_rows):
    valid_types = {"fsm", "pda", "fsm-mp", "pda-mp"}
    for row in bench_rows:
        assert row["type"] in valid_types
```

### Step 2: Run to verify it fails

```
py -3.12 -m pytest tests/test_benchmark_serialization.py -v
```

Expected: `test_benchmark_returns_six_rows` FAILS (still 3 rows).

### Step 3: Update `benchmark_serialization.py`

**a)** Add `save_msgpack, load_msgpack` to the import line:

```python
from circuit_lm.io        import save_model, load_model, save_msgpack, load_msgpack
```

**b)** Refactor `_bench_one` to accept a `use_msgpack: bool = False` parameter.
Change the file extension and save/load calls based on the flag.
Add `"format"` field to the returned row dict.

Replace `_bench_one` with:

```python
def _bench_one(
    label: str,
    model_type: str,
    num_states: int,
    vocab_size: int,
    seed: int,
    tmp_dir: pathlib.Path,
    use_msgpack: bool = False,
) -> dict:
    """Train, save, load, and compare one model. Returns an integer-only row dict."""
    seqs = _make_random_seqs(
        num_seqs=30, seq_len=20, vocab_size=vocab_size, seed=seed
    )
    tokenizer = _make_tokenizer(vocab_size)
    ext = ".msgpack" if use_msgpack else ".json"
    out_path = tmp_dir / f"{label}{ext}"

    if model_type in ("fsm", "fsm-mp"):
        base_type = "fsm"
        model = train_fsm(
            sequences=seqs, vocab_size=vocab_size,
            state_bits=num_states.bit_length() - 1, steps=3,
        )
    else:
        base_type = "pda"
        model = train_pda(
            sequences=seqs, vocab_size=vocab_size,
            state_bits=num_states.bit_length() - 1,
            stack_depth=1, steps=3,
        )

    # --- save ---
    t0 = time.perf_counter_ns()
    if use_msgpack:
        save_msgpack(model, tokenizer, out_path)
    else:
        save_model(model, tokenizer, out_path)
    save_ms: int = (time.perf_counter_ns() - t0) // 1_000_000

    file_bytes: int = len(out_path.read_bytes())

    # --- load ---
    t1 = time.perf_counter_ns()
    if use_msgpack:
        loaded_model, _ = load_msgpack(out_path)
    else:
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
```

**c)** Replace `BENCH_CONFIGS` with a 6-entry version that includes a `use_msgpack` bool:

```python
# (label, type, num_states, vocab_size, use_msgpack)
BENCH_CONFIGS: list[tuple[str, str, int, int, bool]] = [
    ("fsm-sm",    "fsm",    8, 30, False),
    ("pda-sm",    "pda",    8, 30, False),
    ("pda-md",    "pda",   16, 30, False),
    ("fsm-sm-mp", "fsm-mp", 8, 30, True),
    ("pda-sm-mp", "pda-mp", 8, 30, True),
    ("pda-md-mp", "pda-mp",16, 30, True),
]
```

**d)** Update `run_benchmark` to unpack 5 elements per config:

```python
def run_benchmark(
    seed: int = 42,
    tmp_dir: pathlib.Path | None = None,
) -> list[dict]:
    """Run the serialization benchmark and return a list of row dicts."""
    rows: list[dict] = []
    if tmp_dir is not None:
        for i, (label, mtype, ns, vs, use_mp) in enumerate(BENCH_CONFIGS):
            rows.append(_bench_one(label, mtype, ns, vs, seed + i, tmp_dir, use_mp))
        return rows

    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        for i, (label, mtype, ns, vs, use_mp) in enumerate(BENCH_CONFIGS):
            rows.append(_bench_one(label, mtype, ns, vs, seed + i, td_path, use_mp))
    return rows
```

**Note on model reuse:** The same `seed + i` offset means msgpack rows train fresh models. This is intentional for benchmark isolation — each row is independent.

### Step 4: Run tests

```
py -3.12 -m pytest tests/test_benchmark_serialization.py -v
```

Expected: all tests PASS.

### Step 5: Run no-floats check

```
py -3.12 -m pytest tests/test_no_floats.py -v
```

Expected: PASS.

### Step 6: Smoke-run the script

```
py -3.12 scripts/benchmark_serialization.py
```

Expected: 6-row table. Confirm msgpack rows show `ok=1` and smaller byte counts than their JSON counterparts for the same model size.

### Step 7: Commit

```bash
git add scripts/benchmark_serialization.py tests/test_benchmark_serialization.py
git commit -m "feat: extend benchmark_serialization with msgpack rows for JSON vs binary comparison"
```

---

## Task 7: Full test suite + STATUS.md update

### Step 1: Run full test suite

```
py -3.12 -m pytest -q
```

Expected: all previously-passing tests still pass, plus all new ones.

### Step 2: Smoke-run sweep (optional, ~10–15 min)

```
py -3.12 scripts/sweep_jpda_budget.py --train-seqs 50 --steps 30 60
```

Record which `(train_seqs, steps)` combos, if any, achieve full stack discovery.

### Step 3: Update `STATUS.md`

Under "What Is Working", add two sections:

**Pop Discovery Sweep:**
- Report whether any combo in the sweep found both push and pop.
- Note the minimum `(train_seqs, steps)` that achieves full discovery (if any).
- If none found, note that the joint solver still requires a longer budget than tested.

**MessagePack Binary Format:**
- The 6-row benchmark table (bytes / save_ms / load_ms / ok).
- Note the compression ratio: PDA msgpack bytes ÷ PDA JSON bytes.

Update "Next Recommended Steps" to remove the two items just completed and add:
- Consider extending `load_model` to auto-detect JSON vs msgpack by file extension.
- If joint-PDA pop still not found at 120 s, investigate solver warm-start or constraint relaxation.

### Step 4: Commit STATUS.md

```bash
git add STATUS.md
git commit -m "docs: update STATUS.md with sweep results and msgpack benchmark"
```

---

## Quick reference: integer-only checklist

| Pattern | Instead of |
|---------|-----------|
| `(t1 - t0) // 1_000_000` | `(t1 - t0) / 1_000_000` |
| `rng.randint(0, 1) == 0` | `rng.random() < 0.5` |
| Integer constant `3` | float `3.0` |
| `accuracy_pct_times100(c, t)` | manual `c/t*100` |
