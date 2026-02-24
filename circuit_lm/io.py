"""JSON serialisation for CircuitLM (FSM) and PDACircuitLM models.

All numeric values written to / read from disk are stored as JSON integers.
No floating-point values appear in the file format.

A ``"model_type"`` field distinguishes FSM from PDA files:
  ``"model_type": "fsm"``  → CircuitLM
  ``"model_type": "pda"``  → PDACircuitLM

FSM file format
---------------
{
  "model_type":  "fsm",
  "vocab_size":  <int>,
  "num_states":  <int>,
  "state_bits":  <int>,
  "transitions": { "<state>,<token>": <next_state>, ... },
  "state_counts": { "<state>": [<count_tok0>, ...], ... },
  "tokenizer":   { "chars": [...] }
}

PDA file format (extends FSM format)
--------------------------------------
{
  "model_type":    "pda",
  "vocab_size":    <int>,
  "num_states":    <int>,
  "state_bits":    <int>,
  "stack_depth":   <int>,
  "push_tokens":   [<int>, ...],
  "pop_tokens":    [<int>, ...],
  "transitions":   { "<state>,<token>": <next_state>, ... },
  "config_counts": { "<state>,<stack_top>": [<count_tok0>, ...], ... },
  "tokenizer":     { "chars": [...] }
}

TODO: Compressed binary format (MessagePack) for large models.
TODO: Incremental / streaming save for models that don't fit in RAM.
"""

from __future__ import annotations

import json
import pathlib
from typing import Union

from circuit_lm.circuits import CircuitLM
from circuit_lm.pda import STACK_EMPTY, PDACircuitLM
from circuit_lm.ppm import PPMModel
from circuit_lm.tokenizer import Tokenizer

AnyModel = Union[CircuitLM, PDACircuitLM, PPMModel]


# ---------------------------------------------------------------------------
# FSM save / load (unchanged interface, adds model_type field)
# ---------------------------------------------------------------------------


def save_model(
    model: AnyModel,
    tokenizer: Tokenizer,
    path: str | pathlib.Path,
) -> None:
    """Serialise *model* and *tokenizer* to a JSON file at *path*.

    Accepts :class:`~circuit_lm.circuits.CircuitLM` (FSM),
    :class:`~circuit_lm.pda.PDACircuitLM`, or
    :class:`~circuit_lm.ppm.PPMModel`; the ``"model_type"`` field in the
    output distinguishes them.
    """
    if isinstance(model, PDACircuitLM):
        _save_pda(model, tokenizer, path)
    elif isinstance(model, PPMModel):
        _save_ppm(model, tokenizer, path)
    else:
        _save_fsm(model, tokenizer, path)


def _save_fsm(
    model: CircuitLM,
    tokenizer: Tokenizer,
    path: str | pathlib.Path,
) -> None:
    transitions_out: dict[str, int] = {
        f"{s},{t}": ns for (s, t), ns in model.transitions.items()
    }
    state_counts_out: dict[str, list[int]] = {
        str(s): counts for s, counts in model.state_counts.items()
    }
    payload = {
        "model_type":   "fsm",
        "vocab_size":   model.vocab_size,
        "num_states":   model.num_states,
        "state_bits":   model.state_bits,
        "transitions":  transitions_out,
        "state_counts": state_counts_out,
        "tokenizer":    tokenizer.to_dict(),
    }
    pathlib.Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _save_pda(
    model: PDACircuitLM,
    tokenizer: Tokenizer,
    path: str | pathlib.Path,
) -> None:
    transitions_out: dict[str, int] = {
        f"{s},{t}": ns for (s, t), ns in model.transitions.items()
    }
    # config_counts keys: (state, stack_top) where stack_top may be STACK_EMPTY (-1)
    config_counts_out: dict[str, list[int]] = {
        f"{s},{st}": counts for (s, st), counts in model.config_counts.items()
    }
    payload = {
        "model_type":    "pda",
        "vocab_size":    model.vocab_size,
        "num_states":    model.num_states,
        "state_bits":    model.state_bits,
        "stack_depth":   model.stack_depth,
        "push_tokens":   sorted(model.push_tokens),
        "pop_tokens":    sorted(model.pop_tokens),
        "transitions":   transitions_out,
        "config_counts": config_counts_out,
        "tokenizer":     tokenizer.to_dict(),
    }
    pathlib.Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# PPM save / load
# ---------------------------------------------------------------------------


def _save_ppm(
    model: PPMModel,
    tokenizer: Tokenizer,
    path: str | pathlib.Path,
) -> None:
    # Serialise context tuple keys as comma-separated strings.
    # Empty tuple () → ""
    # (3,) → "3"
    # (3, 5) → "3,5"
    counts_out: dict[str, list[int]] = {
        ",".join(str(t) for t in ctx): counts
        for ctx, counts in model.counts.items()
    }
    payload = {
        "model_type": "ppm",
        "vocab_size":  model.vocab_size,
        "order":       model.order,
        "counts":      counts_out,
        "tokenizer":   tokenizer.to_dict(),
    }
    pathlib.Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_ppm(data: dict) -> PPMModel:
    counts: dict[tuple[int, ...], list[int]] = {}
    for key, hist in data["counts"].items():
        if key == "":
            ctx: tuple[int, ...] = ()
        else:
            ctx = tuple(int(x) for x in key.split(","))
        counts[ctx] = [int(c) for c in hist]

    return PPMModel(
        vocab_size=int(data["vocab_size"]),
        order=int(data["order"]),
        counts=counts,
    )


# ---------------------------------------------------------------------------
# Load – returns (AnyModel, Tokenizer); model type auto-detected
# ---------------------------------------------------------------------------


def load_model(path: str | pathlib.Path) -> tuple[AnyModel, Tokenizer]:
    """Load a model and tokenizer from a JSON file.

    Auto-detects model type from the ``"model_type"`` field.  Files written
    before that field was added are treated as FSM models.

    Returns:
        ``(model, tokenizer)`` where model is either
        :class:`~circuit_lm.circuits.CircuitLM` or
        :class:`~circuit_lm.pda.PDACircuitLM`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError / ValueError: If the file is malformed.
    """
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    model_type = data.get("model_type", "fsm")

    tokenizer = Tokenizer.from_dict(data["tokenizer"])

    if model_type == "pda":
        return _load_pda(data), tokenizer
    if model_type == "ppm":
        return _load_ppm(data), tokenizer
    return _load_fsm(data), tokenizer


def _load_fsm(data: dict) -> CircuitLM:
    transitions: dict[tuple[int, int], int] = {}
    for key, ns in data["transitions"].items():
        s_str, t_str = key.split(",", 1)
        transitions[(int(s_str), int(t_str))] = int(ns)

    state_counts: dict[int, list[int]] = {
        int(s): [int(c) for c in counts]
        for s, counts in data["state_counts"].items()
    }

    return CircuitLM(
        vocab_size=int(data["vocab_size"]),
        num_states=int(data["num_states"]),
        state_bits=int(data["state_bits"]),
        transitions=transitions,
        state_counts=state_counts,
    )


def _load_pda(data: dict) -> PDACircuitLM:
    transitions: dict[tuple[int, int], int] = {}
    for key, ns in data["transitions"].items():
        s_str, t_str = key.split(",", 1)
        transitions[(int(s_str), int(t_str))] = int(ns)

    # config_counts keys: "<state>,<stack_top>" where stack_top may be -1
    config_counts: dict[tuple[int, int], list[int]] = {}
    for key, counts in data["config_counts"].items():
        # Split on last comma to handle negative stack_top (-1)
        last_comma = key.rfind(",")
        s   = int(key[:last_comma])
        st  = int(key[last_comma + 1:])
        config_counts[(s, st)] = [int(c) for c in counts]

    return PDACircuitLM(
        vocab_size=int(data["vocab_size"]),
        num_states=int(data["num_states"]),
        state_bits=int(data["state_bits"]),
        stack_depth=int(data["stack_depth"]),
        push_tokens=frozenset(int(t) for t in data["push_tokens"]),
        pop_tokens=frozenset(int(t) for t in data["pop_tokens"]),
        transitions=transitions,
        config_counts=config_counts,
    )
