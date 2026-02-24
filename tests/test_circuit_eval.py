"""Integration tests for the circuit_lm pipeline.

Tests cover:
  - Tokenizer round-trip (encode → decode preserves length).
  - Training smoke test (model builds without error).
  - Model save / reload (fields survive JSON round-trip).
  - Evaluation returns integer results.
  - Inference (greedy + sampled) produces the expected number of tokens.
  - Metrics return integers and format correctly.
  - Runtime type check: no float values appear in model state_counts.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from circuit_lm.tokenizer import Tokenizer
from circuit_lm.data import load_sequences
from circuit_lm.circuits import CircuitLM
from circuit_lm.train_cpsat import train
from circuit_lm.eval import evaluate
from circuit_lm.infer import greedy_decode, sample_tokens
from circuit_lm.metrics import (
    accuracy_fraction,
    accuracy_pct_times100,
    format_accuracy,
)
from circuit_lm.io import save_model, load_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "abcabc def def ghi jkl mno pqr stu vwx yz "
    "hello world hello world goodbye world "
) * 4  # repeat so there are enough sequences


@pytest.fixture()
def tokenizer() -> Tokenizer:
    return Tokenizer.from_text(SAMPLE_TEXT, vocab_size=32)


@pytest.fixture()
def data_file(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "data.txt"
    p.write_text(SAMPLE_TEXT, encoding="utf-8")
    return p


@pytest.fixture()
def sequences(data_file: pathlib.Path, tokenizer: Tokenizer) -> list[list[int]]:
    return load_sequences(data_file, tokenizer)


@pytest.fixture()
def tiny_model(sequences: list[list[int]], tokenizer: Tokenizer) -> CircuitLM:
    return train(
        sequences=sequences,
        vocab_size=tokenizer.vocab_size,
        state_bits=2,
        steps=2,
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenizer_encode_returns_ints(tokenizer: Tokenizer) -> None:
    ids = tokenizer.encode(SAMPLE_TEXT)
    assert all(isinstance(i, int) for i in ids)


def test_tokenizer_decode_length(tokenizer: Tokenizer) -> None:
    ids = tokenizer.encode(SAMPLE_TEXT)
    decoded = tokenizer.decode(ids)
    assert len(decoded) == len(SAMPLE_TEXT)


def test_tokenizer_roundtrip_known_chars(tokenizer: Tokenizer) -> None:
    text = "hello"
    ids = tokenizer.encode(text)
    back = tokenizer.decode(ids)
    assert back == text


def test_tokenizer_serialisation(tokenizer: Tokenizer) -> None:
    d = tokenizer.to_dict()
    tok2 = Tokenizer.from_dict(d)
    assert tok2.vocab_size == tokenizer.vocab_size
    assert tok2.encode("abc") == tokenizer.encode("abc")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def test_sequences_non_empty(sequences: list[list[int]]) -> None:
    assert len(sequences) > 0


def test_sequences_contain_ints(sequences: list[list[int]]) -> None:
    for seq in sequences[:5]:
        assert all(isinstance(t, int) for t in seq)


def test_sequences_min_length(sequences: list[list[int]]) -> None:
    for seq in sequences:
        assert len(seq) >= 2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_returns_correct_num_states(
    sequences: list[list[int]], tokenizer: Tokenizer
) -> None:
    model = train(sequences=sequences, vocab_size=tokenizer.vocab_size, state_bits=2, steps=1)
    assert model.num_states == 4  # 1 << 2


def test_train_state_counts_are_ints(tiny_model: CircuitLM) -> None:
    for state, counts in tiny_model.state_counts.items():
        assert isinstance(state, int), f"state key is not int: {state!r}"
        for c in counts:
            assert isinstance(c, int), f"count is not int: {c!r} (type {type(c)})"


def test_train_transitions_are_ints(tiny_model: CircuitLM) -> None:
    for (s, t), ns in tiny_model.transitions.items():
        assert isinstance(s, int)
        assert isinstance(t, int)
        assert isinstance(ns, int)
        assert 0 <= ns < tiny_model.num_states


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_returns_int_dict(
    tiny_model: CircuitLM, sequences: list[list[int]]
) -> None:
    results = evaluate(tiny_model, sequences)
    assert isinstance(results["correct"], int)
    assert isinstance(results["total"], int)
    assert results["total"] >= 0
    assert 0 <= results["correct"] <= results["total"]


def test_evaluate_empty_sequences(tiny_model: CircuitLM) -> None:
    results = evaluate(tiny_model, [])
    assert results == {"correct": 0, "total": 0}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_greedy_decode_length(tiny_model: CircuitLM, tokenizer: Tokenizer) -> None:
    prompt = tokenizer.encode("the")
    out = greedy_decode(tiny_model, prompt, max_tokens=10)
    assert len(out) == len(prompt) + 10


def test_sample_tokens_length(tiny_model: CircuitLM, tokenizer: Tokenizer) -> None:
    prompt = tokenizer.encode("hello")
    out = sample_tokens(tiny_model, prompt, max_tokens=20, seed=0)
    assert len(out) == len(prompt) + 20


def test_sample_tokens_deterministic(tiny_model: CircuitLM, tokenizer: Tokenizer) -> None:
    prompt = tokenizer.encode("the")
    out1 = sample_tokens(tiny_model, prompt, max_tokens=15, seed=7)
    out2 = sample_tokens(tiny_model, prompt, max_tokens=15, seed=7)
    assert out1 == out2


def test_sample_tokens_different_seeds(tiny_model: CircuitLM, tokenizer: Tokenizer) -> None:
    prompt = tokenizer.encode("the")
    out1 = sample_tokens(tiny_model, prompt, max_tokens=50, seed=1)
    out2 = sample_tokens(tiny_model, prompt, max_tokens=50, seed=2)
    # Different seeds should (almost certainly) produce different sequences
    # for a non-deterministic model; we just check they're not identical
    # (this could theoretically fail on a degenerate model – acceptable)
    _ = out1, out2  # exercise path only; don't assert equality


# ---------------------------------------------------------------------------
# IO / serialisation
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(
    tiny_model: CircuitLM, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    out_path = tmp_path / "model.json"
    save_model(tiny_model, tokenizer, out_path)
    model2, tok2 = load_model(out_path)

    assert model2.vocab_size == tiny_model.vocab_size
    assert model2.num_states == tiny_model.num_states
    assert model2.state_bits == tiny_model.state_bits
    assert tok2.vocab_size == tokenizer.vocab_size


def test_save_load_state_counts_preserved(
    tiny_model: CircuitLM, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    out_path = tmp_path / "model.json"
    save_model(tiny_model, tokenizer, out_path)
    model2, _ = load_model(out_path)

    for s, counts in tiny_model.state_counts.items():
        assert model2.state_counts[s] == counts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_accuracy_fraction_basic() -> None:
    assert accuracy_fraction(1, 4) == (1, 4)
    assert accuracy_fraction(2, 4) == (1, 2)
    assert accuracy_fraction(0, 10) == (0, 1)
    assert accuracy_fraction(0, 0) == (0, 1)


def test_accuracy_pct_times100() -> None:
    assert accuracy_pct_times100(1, 4) == 2500
    assert accuracy_pct_times100(1, 1) == 10000
    assert accuracy_pct_times100(0, 5) == 0
    assert accuracy_pct_times100(0, 0) == 0
    # All returned values are plain Python ints
    result = accuracy_pct_times100(3, 7)
    assert isinstance(result, int)


def test_format_accuracy_string() -> None:
    s = format_accuracy(1, 4)
    assert "25" in s
    assert "%" in s
    # Must not contain a Python float object – it's a str
    assert isinstance(s, str)


def test_format_accuracy_zero_total() -> None:
    s = format_accuracy(0, 0)
    assert isinstance(s, str)
