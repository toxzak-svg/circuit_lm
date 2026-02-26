"""Tests for the PDA (Pushdown Automaton) circuit language model.

Coverage:
  - PDACircuitLM dataclass: step, predict_token, run, config_histogram.
  - Stack operations: PUSH, POP, NOP; depth bound; empty-stack sentinel.
  - train_pda: two-phase CP-SAT training (tiny corpus, short time limit).
  - PDA evaluation and inference (greedy + stochastic).
  - PDA save / load round-trip (all integer fields preserved).
  - No float types appear at runtime.
  - test_no_floats scan implicitly covers pda.py and train_pda_cpsat.py.
"""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from circuit_lm.pda import (
    PDACircuitLM,
    STACK_EMPTY,
    OP_NOP,
    OP_PUSH,
    OP_POP,
)
from circuit_lm.train_pda_cpsat import train_pda
from circuit_lm.eval import evaluate_pda, evaluate_any
from circuit_lm.infer import pda_greedy_decode, pda_sample_tokens, decode_sample
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
def tiny_pda(sequences: list[list[int]], tokenizer: Tokenizer) -> PDACircuitLM:
    return train_pda(
        sequences=sequences,
        vocab_size=tokenizer.vocab_size,
        state_bits=2,
        stack_depth=3,
        steps=3,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_stack_empty_is_integer() -> None:
    assert isinstance(STACK_EMPTY, int)
    assert STACK_EMPTY == -1


def test_op_codes_are_distinct_integers() -> None:
    ops = {OP_NOP, OP_PUSH, OP_POP}
    assert len(ops) == 3
    assert all(isinstance(o, int) for o in ops)


# ---------------------------------------------------------------------------
# PDACircuitLM – unit tests with hand-crafted model
# ---------------------------------------------------------------------------


def _make_tiny_model(vocab_size: int = 8, num_states: int = 4, stack_depth: int = 3) -> PDACircuitLM:
    """Build a minimal PDA with deterministic push/pop configs for testing.

    Token 1 = PUSH (from any state / any stack_top), token 2 = POP (from any
    state / any stack_top), everything else = NOP.
    """
    # Build config triples covering all (state, stack_top) combinations so
    # that token-1 always pushes and token-2 always pops regardless of context.
    stack_tops = [STACK_EMPTY] + list(range(vocab_size))
    push_configs: frozenset[tuple[int, int, int]] = frozenset(
        (s, 1, st)
        for s in range(num_states)
        for st in stack_tops
    )
    pop_configs: frozenset[tuple[int, int, int]] = frozenset(
        (s, 2, st)
        for s in range(num_states)
        for st in stack_tops
    )

    # Simple transitions: (s, t) -> (s + t) % num_states
    transitions: dict[tuple[int, int], int] = {
        (s, t): (s + t) % num_states
        for s in range(num_states)
        for t in range(vocab_size)
    }

    # config_counts: each config predicts token 3
    config_counts: dict[tuple[int, int], list[int]] = {}
    for s in range(num_states):
        for st in stack_tops:
            counts = [0] * vocab_size
            counts[3] = 10   # argmax is token 3
            config_counts[(s, st)] = counts

    return PDACircuitLM(
        vocab_size=vocab_size,
        num_states=num_states,
        state_bits=2,
        stack_depth=stack_depth,
        push_configs=push_configs,
        pop_configs=pop_configs,
        transitions=transitions,
        config_counts=config_counts,
    )


def test_stack_op_nop() -> None:
    m = _make_tiny_model()
    assert m.stack_op(0, 0, STACK_EMPTY) == OP_NOP
    assert m.stack_op(0, 3, STACK_EMPTY) == OP_NOP
    assert m.stack_op(0, 7, STACK_EMPTY) == OP_NOP


def test_stack_op_push() -> None:
    m = _make_tiny_model()
    assert m.stack_op(0, 1, STACK_EMPTY) == OP_PUSH


def test_stack_op_pop() -> None:
    m = _make_tiny_model()
    assert m.stack_op(0, 2, STACK_EMPTY) == OP_POP


def test_step_push_adds_to_stack() -> None:
    m = _make_tiny_model(stack_depth=4)
    state, stack = m.step(0, [], 1)   # token 1 = PUSH
    assert stack == [1]
    assert isinstance(state, int)


def test_step_pop_removes_from_stack() -> None:
    m = _make_tiny_model()
    _, stack_after_push = m.step(0, [], 1)
    assert stack_after_push == [1]
    _, stack_after_pop = m.step(0, stack_after_push, 2)
    assert stack_after_pop == []


def test_step_pop_on_empty_stack_is_noop() -> None:
    m = _make_tiny_model()
    _, stack = m.step(0, [], 2)   # token 2 = POP on empty stack
    assert stack == []


def test_step_push_respects_depth_limit() -> None:
    m = _make_tiny_model(stack_depth=2)
    state, s1 = m.step(0, [], 1)
    state, s2 = m.step(state, s1, 1)
    assert len(s2) == 2
    # Third push should not grow beyond limit
    state, s3 = m.step(state, s2, 1)
    assert len(s3) == 2


def test_step_nop_does_not_change_stack() -> None:
    m = _make_tiny_model()
    stack_before = [1, 3]
    _, stack_after = m.step(0, stack_before, 0)   # token 0 = NOP
    assert stack_after == [1, 3]


def test_predict_token_returns_int() -> None:
    m = _make_tiny_model()
    tok = m.predict_token(0, [])
    assert isinstance(tok, int)
    assert tok == 3    # argmax from config_counts


def test_predict_token_with_nonempty_stack() -> None:
    m = _make_tiny_model()
    tok = m.predict_token(1, [1])
    assert isinstance(tok, int)


def test_predict_token_prefers_learned_config_emission() -> None:
    m = _make_tiny_model()
    # Argmax for (0, STACK_EMPTY) is token 3 in the fixture model.
    m.config_pred_tokens[(0, STACK_EMPTY)] = 5
    assert m.predict_token(0, []) == 5


def test_predict_token_empty_config_returns_zero() -> None:
    m = PDACircuitLM(
        vocab_size=4, num_states=2, state_bits=1, stack_depth=2,
        push_configs=frozenset(), pop_configs=frozenset(),
        transitions={}, config_counts={},
    )
    assert m.predict_token(0, []) == 0


def test_config_histogram_returns_int_list() -> None:
    m = _make_tiny_model()
    h = m.config_histogram(0, [])
    assert isinstance(h, list)
    assert all(isinstance(c, int) for c in h)


def test_config_histogram_unseen_config_zeros() -> None:
    m = PDACircuitLM(
        vocab_size=4, num_states=2, state_bits=1, stack_depth=2,
        push_configs=frozenset(), pop_configs=frozenset(),
        transitions={}, config_counts={},
    )
    h = m.config_histogram(0, [])
    assert h == [0, 0, 0, 0]


def test_run_returns_correct_length() -> None:
    m = _make_tiny_model()
    tokens = [0, 1, 2, 3, 0]
    configs = m.run(tokens)
    assert len(configs) == len(tokens)


def test_run_configs_are_tuples_of_ints() -> None:
    m = _make_tiny_model()
    for state, stack in m.run([0, 1, 2]):
        assert isinstance(state, int)
        assert isinstance(stack, list)
        assert all(isinstance(x, int) for x in stack)


def test_run_tracks_stack_correctly() -> None:
    m = _make_tiny_model(stack_depth=4)
    # token 1 = PUSH, token 2 = POP, token 0 = NOP
    configs = m.run([1, 1, 2, 0])
    # config[0]: before any token, stack empty
    _, s0 = configs[0]
    assert s0 == []
    # config[1]: after PUSH(1), stack=[1]
    _, s1 = configs[1]
    assert s1 == [1]
    # config[2]: after second PUSH(1), stack=[1,1]
    _, s2 = configs[2]
    assert s2 == [1, 1]
    # config[3]: after POP, stack=[1]
    _, s3 = configs[3]
    assert s3 == [1]


# ---------------------------------------------------------------------------
# train_pda – smoke tests
# ---------------------------------------------------------------------------


def test_train_pda_returns_correct_num_states(
    sequences: list[list[int]], tokenizer: Tokenizer
) -> None:
    model = train_pda(
        sequences=sequences,
        vocab_size=tokenizer.vocab_size,
        state_bits=2,
        stack_depth=2,
        steps=1,
    )
    assert model.num_states == 4


def test_train_pda_push_pop_disjoint(tiny_pda: PDACircuitLM) -> None:
    assert tiny_pda.push_configs.isdisjoint(tiny_pda.pop_configs)


def test_train_pda_push_pop_are_frozensets(tiny_pda: PDACircuitLM) -> None:
    assert isinstance(tiny_pda.push_configs, frozenset)
    assert isinstance(tiny_pda.pop_configs, frozenset)


def test_train_pda_config_counts_are_ints(tiny_pda: PDACircuitLM) -> None:
    for cfg, counts in tiny_pda.config_counts.items():
        s, st = cfg
        assert isinstance(s, int)
        assert isinstance(st, int)
        for c in counts:
            assert isinstance(c, int), f"Non-int count: {c!r}"


def test_train_pda_transitions_are_ints(tiny_pda: PDACircuitLM) -> None:
    for (s, t), ns in tiny_pda.transitions.items():
        assert isinstance(s, int)
        assert isinstance(t, int)
        assert isinstance(ns, int)
        assert 0 <= ns < tiny_pda.num_states


def test_train_pda_stores_cpsat_config_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    import circuit_lm.train_pda_cpsat as train_pda_mod

    def fake_simulate_and_collect(
        sequences: list[list[int]],
        vocab_size: int,
        num_states: int,
        stack_depth: int,
        push_tokens: frozenset[int],
        pop_tokens: frozenset[int],
        context_len: int,
    ) -> dict[tuple[int, int], list[int]]:
        _ = (
            sequences, vocab_size, num_states, stack_depth,
            push_tokens, pop_tokens, context_len,
        )
        return {(0, STACK_EMPTY): [0, 10, 0], (1, STACK_EMPTY): [0, 0, 8]}

    def fake_optimize(
        config_counts: dict[tuple[int, int], list[int]],
        vocab_size: int,
        top_k_coverage: int,
        time_limit_seconds: int,
    ) -> dict[tuple[int, int], int]:
        _ = config_counts, vocab_size, top_k_coverage, time_limit_seconds
        return {(0, STACK_EMPTY): 2, (1, STACK_EMPTY): 1}

    monkeypatch.setattr(train_pda_mod, "_simulate_and_collect", fake_simulate_and_collect)
    monkeypatch.setattr(train_pda_mod, "_optimize_config_emissions_cpsat", fake_optimize)

    model = train_pda_mod.train_pda(
        sequences=[[1, 2, 3]],
        vocab_size=3,
        state_bits=1,
        stack_depth=0,   # skips phase 1 solver
        steps=1,
    )

    assert model.config_pred_tokens == {(0, STACK_EMPTY): 2, (1, STACK_EMPTY): 1}
    assert model.predict_token(0, []) == 2
    assert model.predict_token(1, []) == 1


def test_train_pda_stores_learned_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    import circuit_lm.train_pda_cpsat as train_pda_mod
    from circuit_lm.circuits import HASH_PRIME

    monkeypatch.setattr(
        train_pda_mod,
        "_build_transition_counts",
        lambda *args, **kwargs: {(0, 1): [0, 5]},
    )
    monkeypatch.setattr(
        train_pda_mod,
        "_optimize_transitions_cpsat",
        lambda *args, **kwargs: {(0, 1): 1},
    )
    monkeypatch.setattr(
        train_pda_mod,
        "_simulate_and_collect",
        lambda *args, **kwargs: {(0, STACK_EMPTY): [0, 1], (1, STACK_EMPTY): [1, 0]},
    )
    monkeypatch.setattr(
        train_pda_mod,
        "_optimize_config_emissions_cpsat",
        lambda *args, **kwargs: {(0, STACK_EMPTY): 1, (1, STACK_EMPTY): 0},
    )

    model = train_pda_mod.train_pda(
        sequences=[[1, 0]],
        vocab_size=2,
        state_bits=1,
        stack_depth=0,
        steps=4,
    )

    assert model.transitions[(0, 1)] == 1
    assert model.transitions[(1, 1)] == (1 * HASH_PRIME + 1 + 1) % model.num_states


def test_train_pda_splits_explicit_budgets_across_refinement_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import circuit_lm.train_pda_cpsat as train_pda_mod

    stack_calls: list[int] = []
    transition_calls: list[int] = []
    emission_calls: list[int] = []

    monkeypatch.setattr(
        train_pda_mod,
        "_collect_pair_scores",
        lambda *args, **kwargs: {(0, 1): 1},
    )

    def fake_learn_push_pop(
        pair_scores: dict[tuple[int, int], int],
        vocab_size: int,
        max_push: int,
        max_pop: int,
        top_k_pairs: int,
        time_limit_seconds: int,
    ) -> tuple[frozenset[int], frozenset[int]]:
        _ = pair_scores, vocab_size, max_push, max_pop, top_k_pairs
        stack_calls.append(time_limit_seconds)
        return frozenset({0}), frozenset({1})

    monkeypatch.setattr(train_pda_mod, "_learn_push_pop_cpsat", fake_learn_push_pop)
    monkeypatch.setattr(
        train_pda_mod,
        "_build_transition_counts",
        lambda *args, **kwargs: {(0, 0): [1, 0]},
    )
    monkeypatch.setattr(
        train_pda_mod,
        "_simulate_and_collect",
        lambda *args, **kwargs: {(0, STACK_EMPTY): [1, 0], (1, STACK_EMPTY): [0, 1]},
    )
    monkeypatch.setattr(
        train_pda_mod,
        "_simulate_and_collect_runtime",
        lambda *args, **kwargs: (
            {(0, STACK_EMPTY): [1, 0], (1, STACK_EMPTY): [0, 1]},
            {(0, 0): [1, 0]},
        ),
    )

    def fake_optimize_transitions(
        transition_counts: dict[tuple[int, int], list[int]],
        num_states: int,
        time_limit_seconds: int,
    ) -> dict[tuple[int, int], int]:
        _ = transition_counts, num_states
        transition_calls.append(time_limit_seconds)
        return {(0, 0): 0}

    def fake_optimize_emissions(
        config_counts: dict[tuple[int, int], list[int]],
        vocab_size: int,
        top_k_coverage: int,
        time_limit_seconds: int,
    ) -> dict[tuple[int, int], int]:
        _ = config_counts, vocab_size, top_k_coverage
        emission_calls.append(time_limit_seconds)
        return {(0, STACK_EMPTY): 0, (1, STACK_EMPTY): 1}

    monkeypatch.setattr(
        train_pda_mod,
        "_optimize_transitions_cpsat",
        fake_optimize_transitions,
    )
    monkeypatch.setattr(
        train_pda_mod,
        "_optimize_config_emissions_cpsat",
        fake_optimize_emissions,
    )

    train_pda_mod.train_pda(
        sequences=[[0, 1]],
        vocab_size=2,
        state_bits=1,
        stack_depth=1,
        steps=0,
        stack_steps=7,
        transition_steps=5,
        emission_steps=3,
        refinement_rounds=1,
    )

    assert stack_calls == [7]
    assert transition_calls == [3, 2]
    assert emission_calls == [2, 1]


def test_train_pda_stack_depth_zero(
    sequences: list[list[int]], tokenizer: Tokenizer
) -> None:
    """With stack_depth=0 the PDA degrades to plain FSM behaviour."""
    model = train_pda(
        sequences=sequences,
        vocab_size=tokenizer.vocab_size,
        state_bits=2,
        stack_depth=0,
        steps=1,
    )
    assert model.stack_depth == 0
    # All configs should have stack_top == STACK_EMPTY
    for cfg in model.config_counts:
        assert cfg[1] == STACK_EMPTY


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def test_evaluate_pda_returns_int_dict(
    tiny_pda: PDACircuitLM, sequences: list[list[int]]
) -> None:
    results = evaluate_pda(tiny_pda, sequences)
    assert isinstance(results["correct"], int)
    assert isinstance(results["total"], int)
    assert 0 <= results["correct"] <= results["total"]


def test_evaluate_any_dispatches_pda(
    tiny_pda: PDACircuitLM, sequences: list[list[int]]
) -> None:
    results = evaluate_any(tiny_pda, sequences)
    assert isinstance(results["total"], int)


def test_evaluate_pda_empty_sequences(tiny_pda: PDACircuitLM) -> None:
    results = evaluate_pda(tiny_pda, [])
    assert results == {"correct": 0, "total": 0}


def test_evaluate_pda_consumes_current_token_before_predicting_next() -> None:
    model = PDACircuitLM(
        vocab_size=3,
        num_states=2,
        state_bits=1,
        stack_depth=0,
        push_configs=frozenset(),
        pop_configs=frozenset(),
        transitions={(0, 1): 1, (0, 0): 0, (1, 0): 1, (1, 1): 1, (0, 2): 0, (1, 2): 1},
        config_counts={
            (0, STACK_EMPTY): [5, 0, 0],
            (1, STACK_EMPTY): [0, 0, 5],
        },
    )
    results = evaluate_pda(model, [[1, 2]])
    assert results == {"correct": 1, "total": 1}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def test_pda_greedy_decode_length(tiny_pda: PDACircuitLM, tokenizer: Tokenizer) -> None:
    prompt = tokenizer.encode("( hello")
    out = pda_greedy_decode(tiny_pda, prompt, max_tokens=10)
    assert len(out) == len(prompt) + 10


def test_pda_sample_tokens_length(tiny_pda: PDACircuitLM, tokenizer: Tokenizer) -> None:
    prompt = tokenizer.encode("hello")
    out = pda_sample_tokens(tiny_pda, prompt, max_tokens=20, seed=0)
    assert len(out) == len(prompt) + 20


def test_pda_sample_tokens_deterministic(
    tiny_pda: PDACircuitLM, tokenizer: Tokenizer
) -> None:
    prompt = tokenizer.encode("foo")
    out1 = pda_sample_tokens(tiny_pda, prompt, max_tokens=15, seed=9)
    out2 = pda_sample_tokens(tiny_pda, prompt, max_tokens=15, seed=9)
    assert out1 == out2


def test_decode_sample_dispatches_pda(
    tiny_pda: PDACircuitLM, tokenizer: Tokenizer
) -> None:
    prompt = tokenizer.encode("( bar")
    out = decode_sample(tiny_pda, prompt, max_tokens=8, seed=1)
    assert len(out) == len(prompt) + 8


# ---------------------------------------------------------------------------
# IO round-trip
# ---------------------------------------------------------------------------


def test_pda_save_load_roundtrip(
    tiny_pda: PDACircuitLM, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    out_path = tmp_path / "pda_model.json"
    save_model(tiny_pda, tokenizer, out_path)
    model2, tok2 = load_model(out_path)

    assert isinstance(model2, PDACircuitLM)
    assert model2.vocab_size    == tiny_pda.vocab_size
    assert model2.num_states    == tiny_pda.num_states
    assert model2.state_bits    == tiny_pda.state_bits
    assert model2.stack_depth   == tiny_pda.stack_depth
    assert model2.push_configs  == tiny_pda.push_configs
    assert model2.pop_configs   == tiny_pda.pop_configs
    assert tok2.vocab_size      == tokenizer.vocab_size


def test_pda_save_load_config_counts_preserved(
    tiny_pda: PDACircuitLM, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    out_path = tmp_path / "pda_model.json"
    save_model(tiny_pda, tokenizer, out_path)
    model2, _ = load_model(out_path)
    assert isinstance(model2, PDACircuitLM)

    for cfg, counts in tiny_pda.config_counts.items():
        assert model2.config_counts[cfg] == counts


def test_pda_save_load_config_pred_tokens_preserved(
    tiny_pda: PDACircuitLM, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    out_path = tmp_path / "pda_model.json"
    save_model(tiny_pda, tokenizer, out_path)
    model2, _ = load_model(out_path)
    assert isinstance(model2, PDACircuitLM)

    assert model2.config_pred_tokens == tiny_pda.config_pred_tokens


def test_pda_save_load_stack_empty_key_preserved(
    tiny_pda: PDACircuitLM, tokenizer: Tokenizer, tmp_path: pathlib.Path
) -> None:
    """Keys with stack_top = STACK_EMPTY (-1) must survive JSON round-trip."""
    out_path = tmp_path / "pda_model.json"
    save_model(tiny_pda, tokenizer, out_path)
    model2, _ = load_model(out_path)
    assert isinstance(model2, PDACircuitLM)

    empty_configs = [cfg for cfg in model2.config_counts if cfg[1] == STACK_EMPTY]
    # At least the (state=0, STACK_EMPTY) config should exist
    assert len(empty_configs) > 0
    for cfg in empty_configs:
        assert cfg[1] == STACK_EMPTY
        assert isinstance(cfg[0], int)


# ---------------------------------------------------------------------------
# Config-conditioned push/pop fields
# ---------------------------------------------------------------------------

def test_pda_has_push_configs_field() -> None:
    """PDACircuitLM should have push_configs, not push_tokens."""
    m = PDACircuitLM(
        vocab_size=3, num_states=4, state_bits=2, stack_depth=1,
        push_configs=frozenset([(0, 0, STACK_EMPTY)]),
        pop_configs=frozenset([(0, 1, 0)]),
    )
    assert isinstance(m.push_configs, frozenset)
    assert (0, 0, STACK_EMPTY) in m.push_configs


def test_pda_stack_op_three_arg() -> None:
    """stack_op(state, token, stack_top) dispatches via push_configs/pop_configs."""
    m = PDACircuitLM(
        vocab_size=3, num_states=4, state_bits=2, stack_depth=1,
        push_configs=frozenset([(0, 0, -1)]),   # push tok=0 when state=0, stack=EMPTY
        pop_configs=frozenset([(0, 1, 0)]),      # pop  tok=1 when state=0, stack=0
    )
    assert m.stack_op(0, 0, STACK_EMPTY) == OP_PUSH
    assert m.stack_op(0, 1, 0)           == OP_POP
    assert m.stack_op(0, 2, STACK_EMPTY) == OP_NOP
    # Config not in push_configs, even if same token
    assert m.stack_op(1, 0, STACK_EMPTY) == OP_NOP  # state=1, not in push_configs


def test_pda_step_uses_source_state() -> None:
    """step() resolves op with src state, not dst state."""
    # State 0 → state 1 on tok=0.  Push only fires on (src=0, tok=0, stack=EMPTY).
    transitions = {(0, 0): 1, (1, 0): 0}
    m = PDACircuitLM(
        vocab_size=3, num_states=2, state_bits=1, stack_depth=2,
        push_configs=frozenset([(0, 0, STACK_EMPTY)]),
        pop_configs=frozenset(),
        transitions=transitions,
    )
    # First step: state=0 (src), tok=0, stack=[] → push should fire
    next_s, new_stack = m.step(0, [], 0)
    assert next_s == 1
    assert new_stack == [0]   # tok=0 was pushed

    # Second step: state=1 (src), tok=0, stack=[0] → push should NOT fire
    next_s2, new_stack2 = m.step(1, [0], 0)
    assert next_s2 == 0
    assert new_stack2 == [0]  # unchanged


# ---------------------------------------------------------------------------
# Serialization round-trip for push_configs / pop_configs
# ---------------------------------------------------------------------------

def test_pda_save_load_push_configs_round_trip(tmp_path: pathlib.Path) -> None:
    """push_configs / pop_configs survive a JSON save/load cycle."""
    from circuit_lm.io import save_model, load_model
    from circuit_lm.tokenizer import Tokenizer
    push = frozenset([(0, 0, STACK_EMPTY), (1, 0, 0)])
    pop  = frozenset([(0, 1, 0)])
    m = PDACircuitLM(
        vocab_size=3, num_states=2, state_bits=1, stack_depth=2,
        push_configs=push, pop_configs=pop,
        transitions={(0, 0): 1, (0, 1): 0, (0, 2): 0,
                     (1, 0): 0, (1, 1): 1, (1, 2): 0},
        config_counts={(0, STACK_EMPTY): [0, 5, 3]},
        config_pred_tokens={(0, STACK_EMPTY): 1},
    )
    tok = Tokenizer.from_text("( ) x ( ) x", vocab_size=4)
    path = tmp_path / "model.json"
    save_model(m, tok, path)
    loaded, _ = load_model(path)
    assert isinstance(loaded, PDACircuitLM)
    assert loaded.push_configs == push
    assert loaded.pop_configs == pop


def test_pda_load_old_push_tokens_migrates(tmp_path: pathlib.Path) -> None:
    """Old JSON with push_tokens=[0] / pop_tokens=[1] loads as degenerate push_configs."""
    import json
    from circuit_lm.io import load_model
    old_payload = {
        "model_type": "pda",
        "vocab_size": 3, "num_states": 2, "state_bits": 1, "stack_depth": 1,
        "push_tokens": [0],
        "pop_tokens":  [1],
        "transitions": {"0,0": 1, "0,1": 0, "0,2": 0,
                        "1,0": 0, "1,1": 1, "1,2": 0},
        "config_counts": {"0,-1": [0, 3, 1]},
        "config_pred_tokens": {"0,-1": 1},
        "tokenizer": {"mode": "char", "chars": ["<PAD>", "<UNK>", "(", ")", "x"]},
    }
    p = tmp_path / "old_model.json"
    p.write_text(json.dumps(old_payload), encoding="utf-8")
    loaded, _ = load_model(p)
    assert isinstance(loaded, PDACircuitLM)
    # Every (state, tok=0, any_stack_top) must be in push_configs
    for s in range(2):
        for st in [STACK_EMPTY, 0, 1, 2]:
            assert (s, 0, st) in loaded.push_configs
    # pop token = 1 should also cover all (state, tok=1, stack_top) combos
    for s in range(2):
        for st in [STACK_EMPTY, 0, 1, 2]:
            assert (s, 1, st) in loaded.pop_configs
    # No push_tokens attribute
    assert not hasattr(loaded, "push_tokens")


# ---------------------------------------------------------------------------
# No float types at runtime
# ---------------------------------------------------------------------------


def test_no_float_values_in_pda_model(tiny_pda: PDACircuitLM) -> None:
    """All PDACircuitLM fields must be integer-typed at runtime."""
    assert isinstance(tiny_pda.vocab_size,  int)
    assert isinstance(tiny_pda.num_states,  int)
    assert isinstance(tiny_pda.state_bits,  int)
    assert isinstance(tiny_pda.stack_depth, int)

    for triple in tiny_pda.push_configs:
        assert all(isinstance(x, int) for x in triple)
    for triple in tiny_pda.pop_configs:
        assert all(isinstance(x, int) for x in triple)

    for (s, t), ns in tiny_pda.transitions.items():
        assert isinstance(s, int)
        assert isinstance(t, int)
        assert isinstance(ns, int)

    for (s, st), counts in tiny_pda.config_counts.items():
        assert isinstance(s, int)
        assert isinstance(st, int)
        for c in counts:
            assert isinstance(c, int), f"float count found: {c!r}"
