"""Tests for scripts/sweep_jpda_budget.py.

Uses a tiny parameter grid to verify result structure only.
"""

from __future__ import annotations

import pathlib
import sys

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
        "train_seqs",
        "steps",
        "t_total",
        "push_discovered",
        "pop_discovered",
        "full_stack_discovered",
        "push_tokens",
        "pop_tokens",
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
    for row, (train_seqs, steps) in zip(rows, TINY_GRID):
        assert row["train_seqs"] == train_seqs
        assert row["steps"] == steps
