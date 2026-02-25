"""Tests for train_joint_pda_cpsat – true joint CP-SAT PDA training.

Coverage:
  - train_joint_pda returns a PDACircuitLM with correct structural fields.
  - push_tokens and pop_tokens are disjoint frozensets of integers.
  - All model fields are integers (no floats at runtime).
  - Transitions are within [0, num_states).
  - Config counts and config_pred_tokens are keyed by (int, int) pairs.
  - max_push / max_pop budget constraints are respected.
  - top_k_coverage constraint is respected.
  - Argument validation raises ValueError for invalid inputs.
  - Zero-length / single-token sequences are handled gracefully.
  - Fallback produces a valid PDACircuitLM when CP-SAT finds no solution.
  - test_no_floats scan covers train_joint_pda_cpsat.py implicitly.
"""

from __future__ import annotations

import pytest

from circuit_lm.pda import PDACircuitLM, STACK_EMPTY
from circuit_lm.train_joint_pda_cpsat import train_joint_pda, _hash_fallback


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BRACKET_SEQS: list[list[int]] = [
    # Token 0 = "(", token 1 = ")", token 2 = "x"
    # Balanced bracket sequences – ideal for push/pop discovery
    [0, 2, 1, 0, 2, 1],
    [0, 0, 2, 1, 2, 1],
    [0, 2, 2, 1, 0, 1],
    [0, 2, 1, 2, 0, 2, 1],
    [2, 0, 2, 1, 2],
] * 4  # repeat for more signal


@pytest.fixture()
def tiny_joint_pda() -> PDACircuitLM:
    return train_joint_pda(
        sequences=BRACKET_SEQS,
        vocab_size=3,
        num_states=4,
        stack_depth=1,
        steps=3,
    )


# ---------------------------------------------------------------------------
# Return type and structural fields
# ---------------------------------------------------------------------------


def test_train_joint_pda_returns_pda_circuit_lm(tiny_joint_pda: PDACircuitLM) -> None:
    assert isinstance(tiny_joint_pda, PDACircuitLM)


def test_train_joint_pda_correct_num_states(tiny_joint_pda: PDACircuitLM) -> None:
    assert tiny_joint_pda.num_states == 4


def test_train_joint_pda_correct_state_bits(tiny_joint_pda: PDACircuitLM) -> None:
    assert tiny_joint_pda.state_bits == 2


def test_train_joint_pda_stack_depth_preserved(tiny_joint_pda: PDACircuitLM) -> None:
    assert tiny_joint_pda.stack_depth == 1


def test_train_joint_pda_vocab_size_preserved(tiny_joint_pda: PDACircuitLM) -> None:
    assert tiny_joint_pda.vocab_size == 3


# ---------------------------------------------------------------------------
# Push/pop token constraints
# ---------------------------------------------------------------------------


def test_train_joint_pda_push_pop_disjoint(tiny_joint_pda: PDACircuitLM) -> None:
    assert tiny_joint_pda.push_tokens.isdisjoint(tiny_joint_pda.pop_tokens)


def test_train_joint_pda_push_pop_are_frozensets(tiny_joint_pda: PDACircuitLM) -> None:
    assert isinstance(tiny_joint_pda.push_tokens, frozenset)
    assert isinstance(tiny_joint_pda.pop_tokens, frozenset)


def test_train_joint_pda_push_tokens_in_vocab(tiny_joint_pda: PDACircuitLM) -> None:
    for t in tiny_joint_pda.push_tokens:
        assert isinstance(t, int)
        assert 0 <= t < tiny_joint_pda.vocab_size


def test_train_joint_pda_pop_tokens_in_vocab(tiny_joint_pda: PDACircuitLM) -> None:
    for t in tiny_joint_pda.pop_tokens:
        assert isinstance(t, int)
        assert 0 <= t < tiny_joint_pda.vocab_size


def test_train_joint_pda_max_push_constraint() -> None:
    model = train_joint_pda(
        sequences=BRACKET_SEQS,
        vocab_size=3,
        num_states=4,
        stack_depth=1,
        steps=4,
        max_push=1,
    )
    assert len(model.push_tokens) <= 1


def test_train_joint_pda_max_pop_constraint() -> None:
    model = train_joint_pda(
        sequences=BRACKET_SEQS,
        vocab_size=3,
        num_states=4,
        stack_depth=1,
        steps=4,
        max_pop=1,
    )
    assert len(model.pop_tokens) <= 1


def test_train_joint_pda_max_push_zero_means_no_push() -> None:
    model = train_joint_pda(
        sequences=BRACKET_SEQS,
        vocab_size=3,
        num_states=4,
        stack_depth=1,
        steps=4,
        max_push=0,
    )
    assert len(model.push_tokens) == 0


# ---------------------------------------------------------------------------
# Integer discipline (no floats)
# ---------------------------------------------------------------------------


def test_train_joint_pda_transitions_are_integers(tiny_joint_pda: PDACircuitLM) -> None:
    for (s, t), ns in tiny_joint_pda.transitions.items():
        assert isinstance(s, int)
        assert isinstance(t, int)
        assert isinstance(ns, int)


def test_train_joint_pda_transitions_in_range(tiny_joint_pda: PDACircuitLM) -> None:
    for (s, t), ns in tiny_joint_pda.transitions.items():
        assert 0 <= ns < tiny_joint_pda.num_states, (
            f"transition ({s},{t}) → {ns} out of range"
        )


def test_train_joint_pda_config_counts_keys_are_int_pairs(
    tiny_joint_pda: PDACircuitLM,
) -> None:
    for cfg, counts in tiny_joint_pda.config_counts.items():
        s, st = cfg
        assert isinstance(s, int)
        assert isinstance(st, int)
        assert len(counts) == tiny_joint_pda.vocab_size
        for c in counts:
            assert isinstance(c, int), f"Non-int count: {c!r}"


def test_train_joint_pda_config_pred_tokens_are_integers(
    tiny_joint_pda: PDACircuitLM,
) -> None:
    for (s, st), pt in tiny_joint_pda.config_pred_tokens.items():
        assert isinstance(s, int)
        assert isinstance(st, int)
        assert isinstance(pt, int)
        assert 0 <= pt < tiny_joint_pda.vocab_size


def test_train_joint_pda_config_pred_tokens_stack_tops_are_valid(
    tiny_joint_pda: PDACircuitLM,
) -> None:
    """Stack top in config_pred_tokens must be STACK_EMPTY or a valid token ID."""
    for (s, st) in tiny_joint_pda.config_pred_tokens:
        assert st == STACK_EMPTY or (0 <= st < tiny_joint_pda.vocab_size), (
            f"Invalid stack top {st} in config_pred_tokens"
        )


# ---------------------------------------------------------------------------
# top_k_coverage constraint
# ---------------------------------------------------------------------------


def test_train_joint_pda_top_k_coverage_token_predicted() -> None:
    """With top_k_coverage=1, the most-frequent token must appear in some config emission."""
    seqs = [[0, 1, 0, 1, 0, 1]] * 10
    model = train_joint_pda(
        sequences=seqs,
        vocab_size=2,
        num_states=4,
        stack_depth=1,
        steps=5,
        top_k_coverage=1,
    )
    # Token 0 is the most frequent next-token; must appear in at least one config_pred
    assert any(pt == 0 for pt in model.config_pred_tokens.values())


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_train_joint_pda_rejects_nonpositive_steps() -> None:
    with pytest.raises(ValueError, match="steps"):
        train_joint_pda(
            sequences=[[0, 1]],
            vocab_size=2,
            num_states=2,
            stack_depth=1,
            steps=0,
        )


def test_train_joint_pda_rejects_non_power_of_two_states() -> None:
    with pytest.raises(ValueError, match="power of 2"):
        train_joint_pda(
            sequences=[[0, 1]],
            vocab_size=2,
            num_states=3,
            stack_depth=1,
            steps=1,
        )


def test_train_joint_pda_rejects_negative_stack_depth() -> None:
    with pytest.raises(ValueError, match="stack_depth"):
        train_joint_pda(
            sequences=[[0, 1]],
            vocab_size=2,
            num_states=2,
            stack_depth=-1,
            steps=1,
        )


def test_train_joint_pda_rejects_zero_vocab_size() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        train_joint_pda(
            sequences=[[0, 1]],
            vocab_size=0,
            num_states=2,
            stack_depth=1,
            steps=1,
        )


def test_train_joint_pda_rejects_zero_num_states() -> None:
    with pytest.raises(ValueError, match="num_states"):
        train_joint_pda(
            sequences=[[0, 1]],
            vocab_size=2,
            num_states=0,
            stack_depth=1,
            steps=1,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_train_joint_pda_empty_sequences_returns_model() -> None:
    model = train_joint_pda(
        sequences=[],
        vocab_size=4,
        num_states=2,
        stack_depth=1,
        steps=1,
    )
    assert isinstance(model, PDACircuitLM)
    assert model.num_states == 2


def test_train_joint_pda_single_token_sequences_handled() -> None:
    """Sequences shorter than 2 tokens provide no prediction targets and are skipped."""
    model = train_joint_pda(
        sequences=[[0], [1]],
        vocab_size=2,
        num_states=2,
        stack_depth=1,
        steps=1,
    )
    assert isinstance(model, PDACircuitLM)


def test_train_joint_pda_sym_break_false_still_returns_model() -> None:
    model = train_joint_pda(
        sequences=BRACKET_SEQS,
        vocab_size=3,
        num_states=2,
        stack_depth=1,
        steps=2,
        sym_break=False,
    )
    assert isinstance(model, PDACircuitLM)


# ---------------------------------------------------------------------------
# Hash fallback
# ---------------------------------------------------------------------------


def test_hash_fallback_returns_pda_circuit_lm() -> None:
    model = _hash_fallback(
        sequences=[[0, 1, 0, 1]],
        vocab_size=2,
        num_states=4,
        state_bits=2,
        stack_depth=1,
    )
    assert isinstance(model, PDACircuitLM)


def test_hash_fallback_no_push_pop_tokens() -> None:
    model = _hash_fallback(
        sequences=[[0, 1, 0]],
        vocab_size=2,
        num_states=4,
        state_bits=2,
        stack_depth=1,
    )
    assert len(model.push_tokens) == 0
    assert len(model.pop_tokens) == 0


def test_hash_fallback_all_configs_have_stack_empty() -> None:
    """Without push tokens, all configs should use STACK_EMPTY."""
    model = _hash_fallback(
        sequences=[[0, 1, 0, 1, 0]],
        vocab_size=2,
        num_states=4,
        state_bits=2,
        stack_depth=1,
    )
    for (s, st) in model.config_counts:
        assert st == STACK_EMPTY, f"Unexpected stack_top {st} in fallback"


def test_hash_fallback_transitions_in_range() -> None:
    model = _hash_fallback(
        sequences=[[0, 1, 0]],
        vocab_size=2,
        num_states=4,
        state_bits=2,
        stack_depth=0,
    )
    for (s, t), ns in model.transitions.items():
        assert 0 <= ns < 4


def test_hash_fallback_config_counts_are_integers() -> None:
    model = _hash_fallback(
        sequences=[[0, 1, 0, 1]],
        vocab_size=2,
        num_states=4,
        state_bits=2,
        stack_depth=1,
    )
    for cfg, counts in model.config_counts.items():
        s, st = cfg
        assert isinstance(s, int)
        assert isinstance(st, int)
        for c in counts:
            assert isinstance(c, int)


# ---------------------------------------------------------------------------
# Stack top encoding round-trip
# ---------------------------------------------------------------------------


def test_train_joint_pda_stack_empty_key_never_equals_vocab_size(
    tiny_joint_pda: PDACircuitLM,
) -> None:
    """EMPTY_ENC (vocab_size) must be decoded to STACK_EMPTY (-1), never left as vocab_size."""
    V = tiny_joint_pda.vocab_size
    for (s, st) in tiny_joint_pda.config_pred_tokens:
        assert st != V, (
            f"EMPTY_ENC={V} leaked into config_pred_tokens as stack_top; "
            f"expected STACK_EMPTY={STACK_EMPTY}"
        )
    for (s, st) in tiny_joint_pda.config_counts:
        assert st != V
