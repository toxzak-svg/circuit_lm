"""Evaluation loop for next-token prediction.

All return values are plain Python integers; no floats are produced or
consumed at any point.

TODO: Per-token breakdown by token ID.
TODO: Sequence-level metrics (e.g. exact-match count).
TODO: Windowed evaluation for very long sequences.
"""

from __future__ import annotations

from circuit_lm.circuits import CircuitLM
from circuit_lm.pda import PDACircuitLM
from circuit_lm.ppm import PPMModel


# ---------------------------------------------------------------------------
# FSM evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: CircuitLM,
    sequences: list[list[int]],
) -> dict[str, int]:
    """Evaluate next-token prediction accuracy for a plain FSM model.

    For each position *i* in each sequence the model predicts the token at
    position *i+1* using the FSM state reached after consuming
    ``sequence[0:i+1]``.

    Args:
        model:     Trained :class:`~circuit_lm.circuits.CircuitLM`.
        sequences: List of integer token-ID sequences (length >= 2 each).

    Returns:
        Dictionary with integer values:
        ``{"correct": int, "total": int}``
    """
    correct = 0
    total = 0

    for seq in sequences:
        state = 0
        for pos in range(len(seq) - 1):
            tok  = seq[pos]
            gold = seq[pos + 1]
            if model.predict_token(state) == gold:
                correct += 1
            total += 1
            state = model.next_state(state, tok)

    return {"correct": correct, "total": total}


# ---------------------------------------------------------------------------
# PDA evaluation
# ---------------------------------------------------------------------------


def evaluate_pda(
    model: PDACircuitLM,
    sequences: list[list[int]],
) -> dict[str, int]:
    """Evaluate next-token prediction accuracy for a PDA model.

    Maintains a live integer stack alongside the FSM state so that
    predictions are conditioned on ``(state, stack_top)`` configs.

    Args:
        model:     Trained :class:`~circuit_lm.pda.PDACircuitLM`.
        sequences: List of integer token-ID sequences (length >= 2 each).

    Returns:
        Dictionary with integer values:
        ``{"correct": int, "total": int}``

    TODO: Return per-config accuracy breakdown.
    TODO: Track stack-depth statistics (all-integer counters).
    """
    correct = 0
    total = 0

    for seq in sequences:
        state = 0
        stack: list[int] = []
        for pos in range(len(seq) - 1):
            tok  = seq[pos]
            gold = seq[pos + 1]
            if model.predict_token(state, stack) == gold:
                correct += 1
            total += 1
            state, stack = model.step(state, stack, tok)

    return {"correct": correct, "total": total}


# ---------------------------------------------------------------------------
# PPM evaluation
# ---------------------------------------------------------------------------


def evaluate_ppm(
    model: PPMModel,
    sequences: list[list[int]],
) -> dict[str, int]:
    """Evaluate next-token prediction accuracy for a PPM model.

    Maintains a sliding context window of length at most *model.order*
    so that predictions are conditioned on recent history.

    Args:
        model:     Trained :class:`~circuit_lm.ppm.PPMModel`.
        sequences: List of integer token-ID sequences (length >= 2 each).

    Returns:
        Dictionary with integer values:
        ``{"correct": int, "total": int}``
    """
    correct = 0
    total = 0

    for seq in sequences:
        context: tuple[int, ...] = ()
        for pos in range(len(seq) - 1):
            tok  = seq[pos]
            gold = seq[pos + 1]
            if model.predict_token(context) == gold:
                correct += 1
            total += 1
            context = model.step(context, tok)

    return {"correct": correct, "total": total}


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def evaluate_any(
    model: CircuitLM | PDACircuitLM | PPMModel,
    sequences: list[list[int]],
) -> dict[str, int]:
    """Evaluate *model* regardless of type (FSM, PDA, or PPM).

    Dispatches to :func:`evaluate`, :func:`evaluate_pda`, or
    :func:`evaluate_ppm` based on the model type.  All return the same
    ``{"correct": int, "total": int}`` schema.
    """
    if isinstance(model, PDACircuitLM):
        return evaluate_pda(model, sequences)
    if isinstance(model, PPMModel):
        return evaluate_ppm(model, sequences)
    return evaluate(model, sequences)
