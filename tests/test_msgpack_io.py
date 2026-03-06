"""Tests for circuit_lm.io MessagePack save/load helpers."""

from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("msgpack")

from circuit_lm.circuits import CircuitLM
from circuit_lm.io import has_msgpack, load_msgpack, save_msgpack
from circuit_lm.pda import PDACircuitLM, STACK_EMPTY
from circuit_lm.ppm import PPMModel
from circuit_lm.tokenizer import Tokenizer


def _tok() -> Tokenizer:
    return Tokenizer.from_text("abbaabba", vocab_size=8, mode="char")


def test_has_msgpack_true_when_dependency_installed() -> None:
    assert has_msgpack() is True


def test_fsm_msgpack_roundtrip(tmp_path: pathlib.Path) -> None:
    model = CircuitLM(
        vocab_size=4,
        num_states=2,
        state_bits=1,
        transitions={(0, 0): 1, (1, 1): 0},
        state_counts={0: [1, 2, 3, 4], 1: [4, 3, 2, 1]},
        pred_tokens={0: 2, 1: 0},
    )
    path = tmp_path / "fsm.msgpack"
    tokenizer = _tok()

    save_msgpack(model, tokenizer, path)
    model2, tok2 = load_msgpack(path)

    assert isinstance(model2, CircuitLM)
    assert model2.transitions == model.transitions
    assert model2.state_counts == model.state_counts
    assert model2.pred_tokens == model.pred_tokens
    assert tok2.to_dict() == tokenizer.to_dict()


def test_pda_msgpack_roundtrip(tmp_path: pathlib.Path) -> None:
    model = PDACircuitLM(
        vocab_size=4,
        num_states=2,
        state_bits=1,
        stack_depth=3,
        push_configs=frozenset({(0, 0, STACK_EMPTY), (1, 2, 1)}),
        pop_configs=frozenset({(0, 1, 0)}),
        transitions={(0, 0): 1, (1, 1): 0},
        config_counts={
            (0, STACK_EMPTY): [0, 4, 0, 0],
            (1, 1): [1, 0, 2, 3],
        },
        config_pred_tokens={(0, STACK_EMPTY): 1, (1, 1): 3},
    )
    path = tmp_path / "pda.msgpack"
    tokenizer = _tok()

    save_msgpack(model, tokenizer, path)
    model2, tok2 = load_msgpack(path)

    assert isinstance(model2, PDACircuitLM)
    assert model2.transitions == model.transitions
    assert model2.push_configs == model.push_configs
    assert model2.pop_configs == model.pop_configs
    assert model2.config_counts == model.config_counts
    assert model2.config_pred_tokens == model.config_pred_tokens
    assert tok2.to_dict() == tokenizer.to_dict()


def test_ppm_msgpack_roundtrip(tmp_path: pathlib.Path) -> None:
    model = PPMModel(
        vocab_size=4,
        order=3,
        counts={
            (): [1, 2, 3, 4],
            (0,): [0, 1, 0, 0],
            (0, 1): [0, 0, 2, 0],
        },
    )
    path = tmp_path / "ppm.msgpack"
    tokenizer = _tok()

    save_msgpack(model, tokenizer, path)
    model2, tok2 = load_msgpack(path)

    assert isinstance(model2, PPMModel)
    assert model2.counts == model.counts
    assert model2.order == model.order
    assert tok2.to_dict() == tokenizer.to_dict()
