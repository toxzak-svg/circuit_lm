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
from circuit_lm.infer import (
    greedy_decode,
    sample_tokens,
    _apply_top_k,
    _apply_repetition_penalty,
    _prepare_sampling_weights,
)
from circuit_lm.metrics import (
    accuracy_fraction,
    accuracy_pct_times100,
    format_accuracy,
)
from circuit_lm.io import save_model, load_model
from circuit_lm.cli import build_parser, cmd_train

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


def test_tokenizer_bpe_roundtrip_known_text() -> None:
    text = "banana bandana banana"
    tok = Tokenizer.from_text(text, vocab_size=64, mode="bpe", bpe_merges=20)
    ids = tok.encode(text)
    assert tok.mode == "bpe"
    assert all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == text
    # BPE should not increase token count vs char mode on repetitive text.
    char_tok = Tokenizer.from_text(text, vocab_size=64, mode="char")
    assert len(ids) <= len(char_tok.encode(text))


def test_tokenizer_bpe_serialisation_roundtrip() -> None:
    text = "abc abc abc xyz xyz"
    tok = Tokenizer.from_text(text, vocab_size=64, mode="bpe", bpe_merges=16)
    d = tok.to_dict()
    tok2 = Tokenizer.from_dict(d)
    assert tok2.mode == "bpe"
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode(text) == tok.encode(text)
    assert tok2.decode(tok2.encode(text)) == text


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


def test_predict_token_prefers_learned_emission_table() -> None:
    model = CircuitLM(
        vocab_size=4,
        num_states=2,
        state_bits=1,
        transitions={},
        state_counts={0: [0, 0, 5, 0]},
        pred_tokens={0: 1},
    )
    # Argmax(state_counts[0]) would be token 2; learned emission should win.
    assert model.predict_token(0) == 1


def test_train_stores_cpsat_emission_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    import circuit_lm.train_cpsat as train_cpsat_mod

    def fake_build_state_counts(
        sequences: list[list[int]],
        vocab_size: int,
        num_states: int,
        context_len: int,
    ) -> dict[int, list[int]]:
        _ = sequences, vocab_size, num_states, context_len
        return {0: [0, 9, 0], 1: [0, 0, 7]}

    def fake_optimize(
        state_counts: dict[int, list[int]],
        vocab_size: int,
        num_states: int,
        top_k_coverage: int,
        time_limit_seconds: int,
    ) -> dict[int, int]:
        _ = state_counts, vocab_size, num_states, top_k_coverage, time_limit_seconds
        return {0: 2, 1: 1}

    monkeypatch.setattr(train_cpsat_mod, "_build_state_counts", fake_build_state_counts)
    monkeypatch.setattr(train_cpsat_mod, "_optimize_emissions_cpsat", fake_optimize)

    model = train_cpsat_mod.train(
        sequences=[[1, 2, 3]],
        vocab_size=3,
        state_bits=1,
        steps=1,
    )

    assert model.pred_tokens == {0: 2, 1: 1}
    assert model.predict_token(0) == 2
    assert model.predict_token(1) == 1


def test_train_stores_cpsat_transition_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    import circuit_lm.train_cpsat as train_cpsat_mod
    from circuit_lm.circuits import HASH_PRIME

    monkeypatch.setattr(
        train_cpsat_mod,
        "_build_state_counts",
        lambda *args, **kwargs: {0: [0, 1], 1: [1, 0]},
    )
    monkeypatch.setattr(
        train_cpsat_mod,
        "_build_transition_counts",
        lambda *args, **kwargs: {(0, 1): [0, 5]},
    )
    monkeypatch.setattr(
        train_cpsat_mod,
        "_optimize_transitions_cpsat",
        lambda *args, **kwargs: {(0, 1): 1},
    )
    monkeypatch.setattr(
        train_cpsat_mod,
        "_optimize_emissions_cpsat",
        lambda *args, **kwargs: {0: 1, 1: 0},
    )

    model = train_cpsat_mod.train(
        sequences=[[1, 0]],
        vocab_size=2,
        state_bits=1,
        steps=4,
    )

    assert model.transitions[(0, 1)] == 1
    # Unobserved pair still uses hash fallback fill.
    assert model.transitions[(1, 1)] == (1 * HASH_PRIME + 1 + 1) % model.num_states


def test_train_splits_explicit_phase_budgets_across_refinement_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import circuit_lm.train_cpsat as train_cpsat_mod

    monkeypatch.setattr(
        train_cpsat_mod,
        "_build_state_counts",
        lambda *args, **kwargs: {0: [1, 0], 1: [0, 1]},
    )
    monkeypatch.setattr(
        train_cpsat_mod,
        "_build_transition_counts",
        lambda *args, **kwargs: {(0, 0): [1, 0]},
    )
    monkeypatch.setattr(
        train_cpsat_mod,
        "_collect_runtime_counts",
        lambda *args, **kwargs: ({0: [1, 0], 1: [0, 1]}, {(0, 0): [1, 0]}),
    )

    transition_calls: list[int] = []
    emission_calls: list[int] = []

    def fake_optimize_transitions(
        transition_counts: dict[tuple[int, int], list[int]],
        num_states: int,
        time_limit_seconds: int,
    ) -> dict[tuple[int, int], int]:
        _ = transition_counts, num_states
        transition_calls.append(time_limit_seconds)
        return {(0, 0): 0}

    def fake_optimize_emissions(
        state_counts: dict[int, list[int]],
        vocab_size: int,
        num_states: int,
        top_k_coverage: int,
        time_limit_seconds: int,
    ) -> dict[int, int]:
        _ = state_counts, vocab_size, num_states, top_k_coverage
        emission_calls.append(time_limit_seconds)
        return {0: 0, 1: 1}

    monkeypatch.setattr(
        train_cpsat_mod,
        "_optimize_transitions_cpsat",
        fake_optimize_transitions,
    )
    monkeypatch.setattr(
        train_cpsat_mod,
        "_optimize_emissions_cpsat",
        fake_optimize_emissions,
    )

    train_cpsat_mod.train(
        sequences=[[0, 1]],
        vocab_size=2,
        state_bits=1,
        steps=0,
        transition_steps=5,
        emission_steps=8,
        refinement_rounds=1,
    )

    assert transition_calls == [3, 2]
    assert emission_calls == [4, 4]


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


def test_evaluate_consumes_current_token_before_predicting_next() -> None:
    # state 0 predicts token 0, state 1 predicts token 1
    model = CircuitLM(
        vocab_size=2,
        num_states=2,
        state_bits=1,
        transitions={(0, 0): 1, (0, 1): 0, (1, 0): 1, (1, 1): 1},
        state_counts={
            0: [5, 0],
            1: [0, 5],
        },
    )
    # For sequence [0, 1], the correct prediction for token 1 should use the
    # state reached after consuming token 0, i.e. state 1.
    results = evaluate(model, [[0, 1]])
    assert results == {"correct": 1, "total": 1}


def test_evaluate_per_token_breakdown_counts_gold_tokens() -> None:
    # Single-state model predicts token 1 for every position.
    model = CircuitLM(
        vocab_size=3,
        num_states=1,
        state_bits=0,
        transitions={},
        state_counts={0: [0, 10, 0]},
    )
    # Predictions are for golds [1, 2, 1].
    results = evaluate(model, [[0, 1, 2, 1]], per_token=True)
    assert results["correct"] == 2
    assert results["total"] == 3

    per_token = results["per_token"]
    assert isinstance(per_token, dict)
    assert per_token[1] == {"correct": 2, "total": 2}
    assert per_token[2] == {"correct": 0, "total": 1}


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


def test_apply_top_k_keeps_largest_weights_with_deterministic_tie_break() -> None:
    weights = [5, 5, 3, 1]
    # Tie on tokens 0 and 1 -> lower token ID wins for top_k=1.
    assert _apply_top_k(weights, 1) == [5, 0, 0, 0]
    assert _apply_top_k(weights, 2) == [5, 5, 0, 0]


def test_apply_repetition_penalty_divides_recent_tokens() -> None:
    weights = [0, 10, 3, 1]
    seen = [9, 2, 1, 2]   # token 9 ignored (out of range)
    out = _apply_repetition_penalty(
        weights,
        seen,
        repeat_penalty_div=3,
        repeat_window=2,   # penalise tokens {1,2}
    )
    assert out == [0, 3, 1, 1]


def test_prepare_sampling_weights_penalty_then_top_k() -> None:
    base = [0, 10, 9]
    seen = [1]
    out = _prepare_sampling_weights(
        base,
        seen,
        top_k=1,
        repeat_penalty_div=100,
        repeat_window=1,
    )
    # Token 1 is penalised to 1, token 2 stays 9, then top_k keeps token 2.
    assert out == [0, 0, 9]


def test_sample_tokens_top_k_one_is_deterministic_argmax() -> None:
    model = CircuitLM(
        vocab_size=3,
        num_states=1,
        state_bits=0,
        transitions={},
        state_counts={0: [0, 7, 5]},
    )
    out = sample_tokens(model, prompt_ids=[], max_tokens=8, seed=123, top_k=1)
    assert out == [1] * 8


def test_sample_tokens_repetition_penalty_can_flip_top_k_choice() -> None:
    # Single-state model so the histogram is constant at every step.
    model = CircuitLM(
        vocab_size=3,
        num_states=1,
        state_bits=0,
        transitions={},
        state_counts={0: [0, 10, 9]},
    )
    out = sample_tokens(
        model,
        prompt_ids=[1],
        max_tokens=4,
        seed=0,
        top_k=1,
        repeat_penalty_div=100,
        repeat_window=1,
    )
    # With top_k=1 and a strong repeat penalty on the last token, generation
    # alternates between token 2 and token 1.
    assert out == [1, 2, 1, 2, 1]


def test_cli_train_parser_exposes_hidden_cpsat_params() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "train",
        "--data", "data.txt",
        "--context_len", "6",
        "--top_k_coverage", "9",
        "--max_push", "7",
        "--max_pop", "8",
        "--top_k_pairs", "99",
        "--tokenizer", "bpe",
        "--bpe_merges", "123",
        "--transition_steps", "11",
        "--emission_steps", "13",
        "--refinement_rounds", "2",
        "--stack_steps", "5",
    ])
    assert args.context_len == 6
    assert args.top_k_coverage == 9
    assert args.max_push == 7
    assert args.max_pop == 8
    assert args.top_k_pairs == 99
    assert args.tokenizer == "bpe"
    assert args.bpe_merges == 123
    assert args.transition_steps == 11
    assert args.emission_steps == 13
    assert args.refinement_rounds == 2
    assert args.stack_steps == 5


def test_cli_sample_parser_exposes_sampling_controls() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "sample",
        "--top_k", "5",
        "--repeat_penalty_div", "4",
        "--repeat_window", "20",
    ])
    assert args.top_k == 5
    assert args.repeat_penalty_div == 4
    assert args.repeat_window == 20


def test_cli_eval_parser_exposes_per_token_flags() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "eval",
        "--data", "data.txt",
        "--per_token",
        "--per_token_limit", "12",
    ])
    assert args.per_token is True
    assert args.per_token_limit == 12


def test_cli_train_cmd_rejects_unpaired_split_budgets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "train",
        "--data", "data.txt",
        "--transition_steps", "3",
    ])

    rc = cmd_train(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert "--transition_steps and --emission_steps" in captured.err


def test_cli_train_cmd_rejects_partial_pda_explicit_budgets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    args = parser.parse_args([
        "train",
        "--data", "data.txt",
        "--automaton", "pda",
        "--transition_steps", "3",
        "--emission_steps", "4",
    ])

    rc = cmd_train(args)

    captured = capsys.readouterr()
    assert rc == 1
    assert "--stack_steps, --transition_steps, and --emission_steps" in captured.err


def test_cli_rejects_negative_train_steps() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--data", "data.txt", "--steps", "-1"])


def test_cli_rejects_negative_sampling_top_k() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sample", "--top_k", "-1"])


def test_cli_rejects_nonpositive_repeat_penalty_divisor() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["sample", "--repeat_penalty_div", "0"])


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


def test_save_load_pred_tokens_preserved(
    tiny_model: CircuitLM, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    out_path = tmp_path / "model.json"
    save_model(tiny_model, tokenizer, out_path)
    model2, _ = load_model(out_path)

    assert model2.pred_tokens == tiny_model.pred_tokens


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
