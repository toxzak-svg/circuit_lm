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
from circuit_lm.io import has_msgpack


@pytest.fixture()
def bench_rows(tmp_path: pathlib.Path):
    return run_benchmark(seed=42, tmp_dir=tmp_path)


def test_benchmark_returns_expected_row_count(bench_rows):
    expected_formats = 2 if has_msgpack() else 1
    assert len(bench_rows) == 3 * expected_formats


def test_benchmark_row_has_required_keys(bench_rows):
    required = {
        "label",
        "type",
        "format",
        "states",
        "vocab",
        "bytes",
        "save_ms",
        "load_ms",
        "roundtrip_ok",
    }
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


def test_benchmark_format_is_json_or_msgpack(bench_rows):
    for row in bench_rows:
        assert row["format"] in ("json", "msgpack")


def test_benchmark_labels_are_unique(bench_rows):
    labels_with_format = [f"{r['label']}:{r['format']}" for r in bench_rows]
    assert len(labels_with_format) == len(set(labels_with_format))
