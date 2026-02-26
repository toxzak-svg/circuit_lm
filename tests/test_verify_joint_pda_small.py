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
