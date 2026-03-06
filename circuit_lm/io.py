"""JSON + MessagePack serialisation for CircuitLM and PDACircuitLM models.

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
  "pred_tokens": { "<state>": <predicted_token>, ... },
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
  "push_configs":  [[state, token, stack_top], ...],
  "pop_configs":   [[state, token, stack_top], ...],
  "transitions":   { "<state>,<token>": <next_state>, ... },
  "config_counts": { "<state>,<stack_top>": [<count_tok0>, ...], ... },
  "config_pred_tokens": { "<state>,<stack_top>": <predicted_token>, ... },
  "tokenizer":     { "chars": [...] }
}

TODO: Incremental / streaming save for models that don't fit in RAM.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Union

from circuit_lm.circuits import CircuitLM
from circuit_lm.pda import STACK_EMPTY, PDACircuitLM
from circuit_lm.ppm import PPMModel
from circuit_lm.tokenizer import Tokenizer

AnyModel = Union[CircuitLM, PDACircuitLM, PPMModel]


# ---------------------------------------------------------------------------
# Optional MessagePack dependency
# ---------------------------------------------------------------------------


def has_msgpack() -> bool:
    """Return True when the optional ``msgpack`` dependency is importable."""
    try:
        import msgpack  # noqa: F401
    except ImportError:
        return False
    return True


def _require_msgpack() -> Any:
    """Import and return ``msgpack``, or raise a clear runtime error."""
    try:
        import msgpack
    except ImportError as exc:
        raise RuntimeError(
            "MessagePack support requires the 'msgpack' package. "
            "Install dependencies with: pip install -e ."
        ) from exc
    return msgpack


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
    pred_tokens_out: dict[str, int] = {
        str(s): tok for s, tok in model.pred_tokens.items()
    }
    payload = {
        "model_type":   "fsm",
        "vocab_size":   model.vocab_size,
        "num_states":   model.num_states,
        "state_bits":   model.state_bits,
        "transitions":  transitions_out,
        "state_counts": state_counts_out,
        "pred_tokens":  pred_tokens_out,
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
    config_pred_tokens_out: dict[str, int] = {
        f"{s},{st}": tok for (s, st), tok in model.config_pred_tokens.items()
    }
    payload = {
        "model_type":    "pda",
        "vocab_size":    model.vocab_size,
        "num_states":    model.num_states,
        "state_bits":    model.state_bits,
        "stack_depth":   model.stack_depth,
        "push_configs":  sorted([s, tok, st] for (s, tok, st) in model.push_configs),
        "pop_configs":   sorted([s, tok, st] for (s, tok, st) in model.pop_configs),
        "transitions":   transitions_out,
        "config_counts": config_counts_out,
        "config_pred_tokens": config_pred_tokens_out,
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
    pred_tokens: dict[int, int] = {
        int(s): int(tok)
        for s, tok in data.get("pred_tokens", {}).items()
    }

    return CircuitLM(
        vocab_size=int(data["vocab_size"]),
        num_states=int(data["num_states"]),
        state_bits=int(data["state_bits"]),
        transitions=transitions,
        state_counts=state_counts,
        pred_tokens=pred_tokens,
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

    config_pred_tokens: dict[tuple[int, int], int] = {}
    for key, tok in data.get("config_pred_tokens", {}).items():
        last_comma = key.rfind(",")
        s = int(key[:last_comma])
        st = int(key[last_comma + 1:])
        config_pred_tokens[(s, st)] = int(tok)

    # New format: push_configs / pop_configs as [[s, tok, st], ...]
    if "push_configs" in data:
        push_configs: frozenset[tuple[int, int, int]] = frozenset(
            (int(triple[0]), int(triple[1]), int(triple[2]))
            for triple in data["push_configs"]
        )
        pop_configs: frozenset[tuple[int, int, int]] = frozenset(
            (int(triple[0]), int(triple[1]), int(triple[2]))
            for triple in data.get("pop_configs", [])
        )
    else:
        # Migration shim: old format with push_tokens / pop_tokens as int lists.
        vocab_size_v = int(data["vocab_size"])
        num_states_v = int(data["num_states"])
        all_stack_tops = [STACK_EMPTY] + list(range(vocab_size_v))
        push_tokens_old = [int(t) for t in data.get("push_tokens", [])]
        pop_tokens_old  = [int(t) for t in data.get("pop_tokens", [])]
        push_configs = frozenset(
            (s, tok, st)
            for tok in push_tokens_old
            for s in range(num_states_v)
            for st in all_stack_tops
        )
        pop_configs = frozenset(
            (s, tok, st)
            for tok in pop_tokens_old
            for s in range(num_states_v)
            for st in all_stack_tops
        )

    return PDACircuitLM(
        vocab_size=int(data["vocab_size"]),
        num_states=int(data["num_states"]),
        state_bits=int(data["state_bits"]),
        stack_depth=int(data["stack_depth"]),
        push_configs=push_configs,
        pop_configs=pop_configs,
        transitions=transitions,
        config_counts=config_counts,
        config_pred_tokens=config_pred_tokens,
    )


# ---------------------------------------------------------------------------
# MessagePack save / load (compact integer-key format)
# ---------------------------------------------------------------------------


def _encode_stack_top(stack_top: int, vocab_size: int) -> int:
    if stack_top == STACK_EMPTY:
        return vocab_size
    return stack_top


def _decode_stack_top(encoded: int, vocab_size: int) -> int:
    if encoded == vocab_size:
        return STACK_EMPTY
    return encoded


def _encode_transition_key(state: int, token: int, vocab_size: int) -> int:
    return state * vocab_size + token


def _decode_transition_key(key: int, vocab_size: int) -> tuple[int, int]:
    return key // vocab_size, key % vocab_size


def _encode_config_key(state: int, stack_top: int, vocab_size: int) -> int:
    st_range = vocab_size + 1
    return state * st_range + _encode_stack_top(stack_top, vocab_size)


def _decode_config_key(key: int, vocab_size: int) -> tuple[int, int]:
    st_range = vocab_size + 1
    state = key // st_range
    st_encoded = key % st_range
    return state, _decode_stack_top(st_encoded, vocab_size)


def _encode_config_triple_key(
    state: int,
    token: int,
    stack_top: int,
    vocab_size: int,
) -> int:
    st_range = vocab_size + 1
    return (state * vocab_size + token) * st_range + _encode_stack_top(stack_top, vocab_size)


def _decode_config_triple_key(key: int, vocab_size: int) -> tuple[int, int, int]:
    st_range = vocab_size + 1
    pair = key // st_range
    st_encoded = key % st_range
    state = pair // vocab_size
    token = pair % vocab_size
    return state, token, _decode_stack_top(st_encoded, vocab_size)


def save_msgpack(
    model: AnyModel,
    tokenizer: Tokenizer,
    path: str | pathlib.Path,
) -> None:
    """Serialise *model* and *tokenizer* to MessagePack at *path*."""
    msgpack = _require_msgpack()

    if isinstance(model, PDACircuitLM):
        payload = _msgpack_payload_pda(model, tokenizer)
    elif isinstance(model, PPMModel):
        payload = _msgpack_payload_ppm(model, tokenizer)
    else:
        payload = _msgpack_payload_fsm(model, tokenizer)

    packed: bytes = msgpack.packb(payload, use_bin_type=True)
    pathlib.Path(path).write_bytes(packed)


def load_msgpack(path: str | pathlib.Path) -> tuple[AnyModel, Tokenizer]:
    """Load a model and tokenizer from a MessagePack file."""
    msgpack = _require_msgpack()
    data = msgpack.unpackb(
        pathlib.Path(path).read_bytes(),
        raw=False,
        strict_map_key=False,
    )
    model_type = data.get("model_type", "fsm")
    tokenizer = Tokenizer.from_dict(data["tokenizer"])

    if model_type == "pda":
        return _load_pda_msgpack(data), tokenizer
    if model_type == "ppm":
        return _load_ppm_msgpack(data), tokenizer
    return _load_fsm_msgpack(data), tokenizer


def _msgpack_payload_fsm(model: CircuitLM, tokenizer: Tokenizer) -> dict:
    transitions_out: dict[int, int] = {
        _encode_transition_key(s, t, model.vocab_size): ns
        for (s, t), ns in model.transitions.items()
    }
    state_counts_out: dict[int, list[int]] = {
        int(s): [int(c) for c in counts]
        for s, counts in model.state_counts.items()
    }
    pred_tokens_out: dict[int, int] = {
        int(s): int(tok)
        for s, tok in model.pred_tokens.items()
    }
    return {
        "model_type": "fsm",
        "vocab_size": int(model.vocab_size),
        "num_states": int(model.num_states),
        "state_bits": int(model.state_bits),
        "transitions": transitions_out,
        "state_counts": state_counts_out,
        "pred_tokens": pred_tokens_out,
        "tokenizer": tokenizer.to_dict(),
    }


def _msgpack_payload_pda(model: PDACircuitLM, tokenizer: Tokenizer) -> dict:
    transitions_out: dict[int, int] = {
        _encode_transition_key(s, t, model.vocab_size): ns
        for (s, t), ns in model.transitions.items()
    }
    config_counts_out: dict[int, list[int]] = {
        _encode_config_key(s, st, model.vocab_size): [int(c) for c in counts]
        for (s, st), counts in model.config_counts.items()
    }
    config_pred_tokens_out: dict[int, int] = {
        _encode_config_key(s, st, model.vocab_size): int(tok)
        for (s, st), tok in model.config_pred_tokens.items()
    }
    push_configs_out: list[int] = sorted(
        _encode_config_triple_key(s, tok, st, model.vocab_size)
        for (s, tok, st) in model.push_configs
    )
    pop_configs_out: list[int] = sorted(
        _encode_config_triple_key(s, tok, st, model.vocab_size)
        for (s, tok, st) in model.pop_configs
    )
    return {
        "model_type": "pda",
        "vocab_size": int(model.vocab_size),
        "num_states": int(model.num_states),
        "state_bits": int(model.state_bits),
        "stack_depth": int(model.stack_depth),
        "push_configs": push_configs_out,
        "pop_configs": pop_configs_out,
        "transitions": transitions_out,
        "config_counts": config_counts_out,
        "config_pred_tokens": config_pred_tokens_out,
        "tokenizer": tokenizer.to_dict(),
    }


def _msgpack_payload_ppm(model: PPMModel, tokenizer: Tokenizer) -> dict:
    counts_out: dict[str, list[int]] = {
        ",".join(str(t) for t in ctx): [int(c) for c in counts]
        for ctx, counts in model.counts.items()
    }
    return {
        "model_type": "ppm",
        "vocab_size": int(model.vocab_size),
        "order": int(model.order),
        "counts": counts_out,
        "tokenizer": tokenizer.to_dict(),
    }


def _load_fsm_msgpack(data: dict) -> CircuitLM:
    vocab_size = int(data["vocab_size"])

    transitions: dict[tuple[int, int], int] = {}
    for key, ns in data["transitions"].items():
        s, t = _decode_transition_key(int(key), vocab_size)
        transitions[(s, t)] = int(ns)

    state_counts: dict[int, list[int]] = {
        int(s): [int(c) for c in counts]
        for s, counts in data["state_counts"].items()
    }
    pred_tokens: dict[int, int] = {
        int(s): int(tok)
        for s, tok in data.get("pred_tokens", {}).items()
    }
    return CircuitLM(
        vocab_size=vocab_size,
        num_states=int(data["num_states"]),
        state_bits=int(data["state_bits"]),
        transitions=transitions,
        state_counts=state_counts,
        pred_tokens=pred_tokens,
    )


def _load_pda_msgpack(data: dict) -> PDACircuitLM:
    vocab_size = int(data["vocab_size"])

    transitions: dict[tuple[int, int], int] = {}
    for key, ns in data["transitions"].items():
        s, t = _decode_transition_key(int(key), vocab_size)
        transitions[(s, t)] = int(ns)

    config_counts: dict[tuple[int, int], list[int]] = {}
    for key, counts in data["config_counts"].items():
        s, st = _decode_config_key(int(key), vocab_size)
        config_counts[(s, st)] = [int(c) for c in counts]

    config_pred_tokens: dict[tuple[int, int], int] = {}
    for key, tok in data.get("config_pred_tokens", {}).items():
        s, st = _decode_config_key(int(key), vocab_size)
        config_pred_tokens[(s, st)] = int(tok)

    push_configs = frozenset(
        _decode_config_triple_key(int(key), vocab_size)
        for key in data.get("push_configs", [])
    )
    pop_configs = frozenset(
        _decode_config_triple_key(int(key), vocab_size)
        for key in data.get("pop_configs", [])
    )

    return PDACircuitLM(
        vocab_size=vocab_size,
        num_states=int(data["num_states"]),
        state_bits=int(data["state_bits"]),
        stack_depth=int(data["stack_depth"]),
        push_configs=push_configs,
        pop_configs=pop_configs,
        transitions=transitions,
        config_counts=config_counts,
        config_pred_tokens=config_pred_tokens,
    )


def _load_ppm_msgpack(data: dict) -> PPMModel:
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
