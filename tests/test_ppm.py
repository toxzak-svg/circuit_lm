"""Tests for the PPM (Prediction by Partial Matching) context-tree language model.

Coverage:
  - PPMModel dataclass: predict_token, context_histogram, step, run.
  - Context backoff: longest-match fallback to shorter contexts.
  - train_ppm: pure integer counting (tiny corpus).
  - PPM evaluation and inference (greedy + stochastic).
  - PPM save / load round-trip (all integer counts preserved).
  - No float types appear at runtime.
  - test_no_floats scan implicitly covers ppm.py and train_ppm.py.
"""

from __future__ import annotations

import pathlib

import pytest

from circuit_lm.ppm import PPMModel
from circuit_lm.train_ppm import train_ppm
from circuit_lm.eval import evaluate_ppm, evaluate_any
from circuit_lm.infer import ppm_greedy_decode, ppm_sample_tokens, decode_sample
from circuit_lm.tokenizer import Tokenizer
from circuit_lm.data import load_sequences
from circuit_lm.io import save_model, load_model

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "( hello world ) ( foo bar ) [ abc def ] ( the quick brown fox ) "
    "hello ( world hello ) [ bar baz ] "
) * 8


@pytest.fixture()
def tokenizer() -> Tokenizer:
    return Tokenizer.from_text(SAMPLE_TEXT, vocab_size=24)


@pytest.fixture()
def data_file(tmp_path: pathlib.Path, tokenizer: Tokenizer) -> pathlib.Path:
    p = tmp_path / "data.txt"
    p.write_text(SAMPLE_TEXT, encoding="utf-8")
    return p


@pytest.fixture()
def sequences(data_file: pathlib.Path, tokenizer: Tokenizer) -> list[list[int]]:
    return load_sequences(data_file, tokenizer)


@pytest.fixture()
def tiny_ppm(sequences: list[list[int]], tokenizer: Tokenizer) -> PPMModel:
    return train_ppm(
        sequences=sequences,
        vocab_size=tokenizer.vocab_size,
        order=3,
    )


# ---------------------------------------------------------------------------
# Hand-crafted minimal model for deterministic unit tests
# ---------------------------------------------------------------------------


def _make_tiny_model(vocab_size: int = 8, order: int = 2) -> PPMModel:
    """Build a PPMModel with known counts for deterministic testing.

    Counts:
      ()      → token 3 is most common (count 10)
      (1,)    → token 4 is most common (count 5)
      (2,)    → token 5 is most common (count 8)
      (1, 2)  → token 6 is most common (count 7)
    """
    counts: dict[tuple[int, ...], list[int]] = {
        ():      [0, 0, 0, 10, 0, 0, 0, 0],
        (1,):    [0, 0, 0,  0, 5, 0, 0, 0],
        (2,):    [0, 0, 0,  0, 0, 8, 0, 0],
        (1, 2):  [0, 0, 0,  0, 0, 0, 7, 0],
    }
    return PPMModel(vocab_size=vocab_size, order=order, counts=counts)


# ---------------------------------------------------------------------------
# PPMModel field types
# ---------------------------------------------------------------------------


def test_vocab_size_is_int() -> None:
    m = _make_tiny_model()
    assert isinstance(m.vocab_size, int)


def test_order_is_int() -> None:
    m = _make_tiny_model()
    assert isinstance(m.order, int)
    assert m.order == 2


def test_counts_keys_are_tuples_of_ints() -> None:
    m = _make_tiny_model()
    for ctx, hist in m.counts.items():
        assert isinstance(ctx, tuple)
        for t in ctx:
            assert isinstance(t, int)
        assert isinstance(hist, list)
        for c in hist:
            assert isinstance(c, int)


# ---------------------------------------------------------------------------
# step() – context sliding
# ---------------------------------------------------------------------------


def test_step_empty_context_grows() -> None:
    m = _make_tiny_model(order=3)
    ctx = m.step((), 5)
    assert ctx == (5,)


def test_step_grows_to_order() -> None:
    m = _make_tiny_model(order=2)
    ctx = m.step((), 1)
    assert ctx == (1,)
    ctx = m.step(ctx, 2)
    assert ctx == (1, 2)


def test_step_slides_window_at_order() -> None:
    m = _make_tiny_model(order=2)
    ctx = (1, 2)
    ctx = m.step(ctx, 3)
    assert ctx == (2, 3)    # oldest token dropped


def test_step_order_zero_always_empty() -> None:
    m = PPMModel(vocab_size=4, order=0, counts={})
    ctx = m.step((), 7)
    assert ctx == ()
    ctx = m.step(ctx, 3)
    assert ctx == ()


# ---------------------------------------------------------------------------
# predict_token() – longest-match backoff
# ---------------------------------------------------------------------------


def test_predict_token_argmax_empty_context() -> None:
    m = _make_tiny_model()
    assert m.predict_token(()) == 3   # global argmax from ()


def test_predict_token_uses_length1_context() -> None:
    m = _make_tiny_model()
    # context (1,) has count for token 4
    assert m.predict_token((1,)) == 4


def test_predict_token_uses_length2_context() -> None:
    m = _make_tiny_model()
    # context (1, 2) has count for token 6
    assert m.predict_token((1, 2)) == 6


def test_predict_token_backoff_to_length1() -> None:
    m = _make_tiny_model()
    # context (99, 1) – (99,1) not in counts, back off to (1,) → token 4
    assert m.predict_token((99, 1)) == 4


def test_predict_token_backoff_to_empty() -> None:
    m = _make_tiny_model()
    # context (99,) not in counts → backoff to () → token 3
    assert m.predict_token((99,)) == 3


def test_predict_token_empty_model_returns_zero() -> None:
    m = PPMModel(vocab_size=4, order=2, counts={})
    assert m.predict_token((1, 2)) == 0


def test_predict_token_returns_int() -> None:
    m = _make_tiny_model()
    tok = m.predict_token((1,))
    assert isinstance(tok, int)


def test_predict_token_all_zero_counts_backsoff() -> None:
    # A context node with all-zero counts should trigger backoff.
    counts = {
        ():   [0, 0, 0, 5, 0, 0, 0, 0],  # fallback: token 3
        (1,): [0, 0, 0, 0, 0, 0, 0, 0],  # all zeros – must back off
    }
    m = PPMModel(vocab_size=8, order=1, counts=counts)
    assert m.predict_token((1,)) == 3


# ---------------------------------------------------------------------------
# context_histogram()
# ---------------------------------------------------------------------------


def test_context_histogram_empty_context() -> None:
    m = _make_tiny_model()
    h = m.context_histogram(())
    # Only level 0 (weight 1) contributes from ()
    assert h[3] == 10
    assert all(c == 0 for i, c in enumerate(h) if i != 3)


def test_context_histogram_returns_int_list() -> None:
    m = _make_tiny_model()
    h = m.context_histogram((1,))
    assert isinstance(h, list)
    assert all(isinstance(c, int) for c in h)


def test_context_histogram_length_is_vocab_size() -> None:
    m = _make_tiny_model(vocab_size=8)
    assert len(m.context_histogram((1, 2))) == 8


def test_context_histogram_empty_model_all_zeros() -> None:
    m = PPMModel(vocab_size=4, order=2, counts={})
    assert m.context_histogram((1, 2)) == [0, 0, 0, 0]


def test_context_histogram_longer_context_weighted_more() -> None:
    m = _make_tiny_model()
    h = m.context_histogram((1,))
    # Level 1 (context=(1,), weight=2): token 4 gets 5*2=10
    # Level 0 (context=(), weight=1): token 3 gets 10*1=10
    assert h[4] == 10   # 5 * 2
    assert h[3] == 10   # 10 * 1


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


def test_run_returns_correct_length() -> None:
    m = _make_tiny_model()
    tokens = [0, 1, 2, 3]
    ctxs = m.run(tokens)
    assert len(ctxs) == len(tokens)


def test_run_first_context_is_empty() -> None:
    m = _make_tiny_model(order=2)
    ctxs = m.run([1, 2, 3])
    assert ctxs[0] == ()


def test_run_context_grows_then_slides() -> None:
    m = PPMModel(vocab_size=4, order=2, counts={})
    ctxs = m.run([0, 1, 2, 3])
    assert ctxs[0] == ()       # before token 0
    assert ctxs[1] == (0,)     # after consuming 0
    assert ctxs[2] == (0, 1)   # after consuming 0, 1
    assert ctxs[3] == (1, 2)   # window slides: drops 0


def test_run_respects_initial_context() -> None:
    m = PPMModel(vocab_size=4, order=2, counts={})
    ctxs = m.run([5], initial_context=(3, 4))
    assert ctxs[0] == (3, 4)


def test_run_contexts_are_tuples_of_ints() -> None:
    m = _make_tiny_model()
    for ctx in m.run([0, 1, 2]):
        assert isinstance(ctx, tuple)
        for x in ctx:
            assert isinstance(x, int)


# ---------------------------------------------------------------------------
# train_ppm – smoke tests
# ---------------------------------------------------------------------------


def test_train_ppm_returns_ppm_model(
    sequences: list[list[int]], tokenizer: Tokenizer
) -> None:
    model = train_ppm(sequences=sequences, vocab_size=tokenizer.vocab_size, order=2)
    assert isinstance(model, PPMModel)


def test_train_ppm_order_preserved(
    sequences: list[list[int]], tokenizer: Tokenizer
) -> None:
    model = train_ppm(sequences=sequences, vocab_size=tokenizer.vocab_size, order=3)
    assert model.order == 3


def test_train_ppm_vocab_size_preserved(
    sequences: list[list[int]], tokenizer: Tokenizer
) -> None:
    model = train_ppm(sequences=sequences, vocab_size=tokenizer.vocab_size, order=2)
    assert model.vocab_size == tokenizer.vocab_size


def test_train_ppm_empty_context_always_present(
    sequences: list[list[int]], tokenizer: Tokenizer
) -> None:
    model = train_ppm(sequences=sequences, vocab_size=tokenizer.vocab_size, order=2)
    assert () in model.counts


def test_train_ppm_counts_are_ints(tiny_ppm: PPMModel) -> None:
    for ctx, hist in tiny_ppm.counts.items():
        for c in hist:
            assert isinstance(c, int), f"Non-int count: {c!r}"


def test_train_ppm_hist_length_is_vocab_size(tiny_ppm: PPMModel) -> None:
    for hist in tiny_ppm.counts.values():
        assert len(hist) == tiny_ppm.vocab_size


def test_train_ppm_order_zero_only_empty_context(
    sequences: list[list[int]], tokenizer: Tokenizer
) -> None:
    model = train_ppm(sequences=sequences, vocab_size=tokenizer.vocab_size, order=0)
    assert model.order == 0
    # Only the empty context should be present
    assert set(model.counts.keys()) == {()}


def test_train_ppm_context_keys_bounded_by_order(tiny_ppm: PPMModel) -> None:
    for ctx in tiny_ppm.counts:
        assert len(ctx) <= tiny_ppm.order


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_ppm_returns_int_dict(
    tiny_ppm: PPMModel, sequences: list[list[int]]
) -> None:
    results = evaluate_ppm(tiny_ppm, sequences)
    assert isinstance(results["correct"], int)
    assert isinstance(results["total"], int)
    assert 0 <= results["correct"] <= results["total"]


def test_evaluate_ppm_empty_sequences(tiny_ppm: PPMModel) -> None:
    results = evaluate_ppm(tiny_ppm, [])
    assert results == {"correct": 0, "total": 0}


def test_evaluate_any_dispatches_ppm(
    tiny_ppm: PPMModel, sequences: list[list[int]]
) -> None:
    results = evaluate_any(tiny_ppm, sequences)
    assert isinstance(results["total"], int)


def test_evaluate_ppm_perfect_on_known_data() -> None:
    """A model trained on a tiny corpus should score well on the same corpus."""
    seqs = [[1, 2, 3, 2, 3, 2, 3]]
    model = train_ppm(sequences=seqs, vocab_size=4, order=2)
    results = evaluate_ppm(model, seqs)
    # Should predict 3 after 2 perfectly (bigram context always seen)
    assert results["correct"] > 0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_ppm_greedy_decode_length(tiny_ppm: PPMModel, tokenizer: Tokenizer) -> None:
    prompt = tokenizer.encode("( hello")
    out = ppm_greedy_decode(tiny_ppm, prompt, max_tokens=10)
    assert len(out) == len(prompt) + 10


def test_ppm_sample_tokens_length(tiny_ppm: PPMModel, tokenizer: Tokenizer) -> None:
    prompt = tokenizer.encode("hello")
    out = ppm_sample_tokens(tiny_ppm, prompt, max_tokens=20, seed=0)
    assert len(out) == len(prompt) + 20


def test_ppm_sample_tokens_deterministic(
    tiny_ppm: PPMModel, tokenizer: Tokenizer
) -> None:
    prompt = tokenizer.encode("foo")
    out1 = ppm_sample_tokens(tiny_ppm, prompt, max_tokens=15, seed=7)
    out2 = ppm_sample_tokens(tiny_ppm, prompt, max_tokens=15, seed=7)
    assert out1 == out2


def test_ppm_sample_tokens_different_seeds_differ(
    tiny_ppm: PPMModel, tokenizer: Tokenizer
) -> None:
    prompt = tokenizer.encode("hello")
    out1 = ppm_sample_tokens(tiny_ppm, prompt, max_tokens=20, seed=1)
    out2 = ppm_sample_tokens(tiny_ppm, prompt, max_tokens=20, seed=2)
    # Different seeds should (almost certainly) produce different output
    assert out1 != out2


def test_decode_sample_dispatches_ppm(
    tiny_ppm: PPMModel, tokenizer: Tokenizer
) -> None:
    prompt = tokenizer.encode("( bar")
    out = decode_sample(tiny_ppm, prompt, max_tokens=8, seed=1)
    assert len(out) == len(prompt) + 8


def test_ppm_greedy_decode_all_ints(
    tiny_ppm: PPMModel, tokenizer: Tokenizer
) -> None:
    prompt = tokenizer.encode("hello")
    out = ppm_greedy_decode(tiny_ppm, prompt, max_tokens=5)
    assert all(isinstance(t, int) for t in out)


# ---------------------------------------------------------------------------
# IO round-trip
# ---------------------------------------------------------------------------


def test_ppm_save_load_roundtrip(
    tiny_ppm: PPMModel, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    out_path = tmp_path / "ppm_model.json"
    save_model(tiny_ppm, tokenizer, out_path)
    model2, tok2 = load_model(out_path)

    assert isinstance(model2, PPMModel)
    assert model2.vocab_size == tiny_ppm.vocab_size
    assert model2.order      == tiny_ppm.order
    assert tok2.vocab_size   == tokenizer.vocab_size


def test_ppm_save_load_counts_preserved(
    tiny_ppm: PPMModel, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    out_path = tmp_path / "ppm_model.json"
    save_model(tiny_ppm, tokenizer, out_path)
    model2, _ = load_model(out_path)
    assert isinstance(model2, PPMModel)

    for ctx, hist in tiny_ppm.counts.items():
        assert model2.counts[ctx] == hist


def test_ppm_save_load_empty_context_preserved(
    tiny_ppm: PPMModel, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    """Empty context key () must survive JSON round-trip."""
    out_path = tmp_path / "ppm_model.json"
    save_model(tiny_ppm, tokenizer, out_path)
    model2, _ = load_model(out_path)
    assert isinstance(model2, PPMModel)
    assert () in model2.counts


def test_ppm_save_load_counts_are_ints(
    tiny_ppm: PPMModel, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    out_path = tmp_path / "ppm_model.json"
    save_model(tiny_ppm, tokenizer, out_path)
    model2, _ = load_model(out_path)
    assert isinstance(model2, PPMModel)

    for ctx, hist in model2.counts.items():
        assert isinstance(ctx, tuple)
        for t in ctx:
            assert isinstance(t, int)
        for c in hist:
            assert isinstance(c, int), f"Non-int count after load: {c!r}"


# ---------------------------------------------------------------------------
# No float types at runtime
# ---------------------------------------------------------------------------


def test_no_float_values_in_ppm_model(tiny_ppm: PPMModel) -> None:
    """All PPMModel fields must be integer-typed at runtime."""
    assert isinstance(tiny_ppm.vocab_size, int)
    assert isinstance(tiny_ppm.order, int)

    for ctx, hist in tiny_ppm.counts.items():
        assert isinstance(ctx, tuple)
        for t in ctx:
            assert isinstance(t, int)
        for c in hist:
            assert isinstance(c, int), f"Float count found: {c!r}"


def test_context_histogram_no_floats(tiny_ppm: PPMModel) -> None:
    h = tiny_ppm.context_histogram(())
    for c in h:
        assert isinstance(c, int)


def test_predict_token_no_float(tiny_ppm: PPMModel) -> None:
    tok = tiny_ppm.predict_token(())
    assert isinstance(tok, int)
