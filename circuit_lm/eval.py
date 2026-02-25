"""Evaluation loop for next-token prediction.

All return values are plain Python integers; no floats are produced or
consumed at any point.

TODO: Sequence-level metrics (e.g. exact-match count).
TODO: Windowed evaluation for very long sequences.
"""

from __future__ import annotations

from circuit_lm.circuits import CircuitLM
from circuit_lm.pda import PDACircuitLM
from circuit_lm.ppm import PPMModel

PerTokenBreakdown = dict[int, dict[str, int]]
EvalResult = dict[str, int | PerTokenBreakdown]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _update_per_token_breakdown(
    per_token_stats: PerTokenBreakdown | None,
    gold_token: int,
    is_correct: bool,
) -> None:
    """Update optional per-token accuracy counters keyed by gold token ID."""
    if per_token_stats is None:
        return
    if gold_token not in per_token_stats:
        per_token_stats[gold_token] = {"correct": 0, "total": 0}
    stats = per_token_stats[gold_token]
    stats["total"] += 1
    if is_correct:
        stats["correct"] += 1


# ---------------------------------------------------------------------------
# FSM evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: CircuitLM,
    sequences: list[list[int]],
    per_token: bool = False,
) -> EvalResult:
    """Evaluate next-token prediction accuracy for a plain FSM model.

    For each position *i* in each sequence the model predicts the token at
    position *i+1* using the FSM state reached after consuming
    ``sequence[0:i+1]``.

    Args:
        model:     Trained :class:`~circuit_lm.circuits.CircuitLM`.
        sequences: List of integer token-ID sequences (length >= 2 each).
        per_token: If True, include per-gold-token ``{"correct", "total"}``
                   breakdowns keyed by token ID under ``"per_token"``.

    Returns:
        By default: ``{"correct": int, "total": int}``
        If ``per_token``: also includes ``"per_token"`` mapping token IDs
        to integer counters.
    """
    correct = 0
    total = 0
    per_token_stats: PerTokenBreakdown | None = {} if per_token else None

    for seq in sequences:
        state = 0
        for pos in range(len(seq) - 1):
            tok  = seq[pos]
            gold = seq[pos + 1]
            state = model.next_state(state, tok)
            is_correct = model.predict_token(state) == gold
            if is_correct:
                correct += 1
            _update_per_token_breakdown(per_token_stats, gold, is_correct)
            total += 1

    out: EvalResult = {"correct": correct, "total": total}
    if per_token_stats is not None:
        out["per_token"] = per_token_stats
    return out


# ---------------------------------------------------------------------------
# PDA evaluation
# ---------------------------------------------------------------------------


def evaluate_pda(
    model: PDACircuitLM,
    sequences: list[list[int]],
    per_token: bool = False,
) -> EvalResult:
    """Evaluate next-token prediction accuracy for a PDA model.

    Maintains a live integer stack alongside the FSM state so that
    predictions are conditioned on ``(state, stack_top)`` configs.

    Args:
        model:     Trained :class:`~circuit_lm.pda.PDACircuitLM`.
        sequences: List of integer token-ID sequences (length >= 2 each).
        per_token: If True, include per-gold-token ``{"correct", "total"}``
                   breakdowns keyed by token ID under ``"per_token"``.

    Returns:
        By default: ``{"correct": int, "total": int}``
        If ``per_token``: also includes ``"per_token"`` mapping token IDs
        to integer counters.

    TODO: Return per-config accuracy breakdown.
    TODO: Track stack-depth statistics (all-integer counters).
    """
    correct = 0
    total = 0
    per_token_stats: PerTokenBreakdown | None = {} if per_token else None

    for seq in sequences:
        state = 0
        stack: list[int] = []
        for pos in range(len(seq) - 1):
            tok  = seq[pos]
            gold = seq[pos + 1]
            state, stack = model.step(state, stack, tok)
            is_correct = model.predict_token(state, stack) == gold
            if is_correct:
                correct += 1
            _update_per_token_breakdown(per_token_stats, gold, is_correct)
            total += 1

    out: EvalResult = {"correct": correct, "total": total}
    if per_token_stats is not None:
        out["per_token"] = per_token_stats
    return out


# ---------------------------------------------------------------------------
# PPM evaluation
# ---------------------------------------------------------------------------


def evaluate_ppm(
    model: PPMModel,
    sequences: list[list[int]],
    per_token: bool = False,
) -> EvalResult:
    """Evaluate next-token prediction accuracy for a PPM model.

    Maintains a sliding context window of length at most *model.order*
    so that predictions are conditioned on recent history.

    Args:
        model:     Trained :class:`~circuit_lm.ppm.PPMModel`.
        sequences: List of integer token-ID sequences (length >= 2 each).
        per_token: If True, include per-gold-token ``{"correct", "total"}``
                   breakdowns keyed by token ID under ``"per_token"``.

    Returns:
        By default: ``{"correct": int, "total": int}``
        If ``per_token``: also includes ``"per_token"`` mapping token IDs
        to integer counters.
    """
    correct = 0
    total = 0
    per_token_stats: PerTokenBreakdown | None = {} if per_token else None

    for seq in sequences:
        context: tuple[int, ...] = ()
        for pos in range(len(seq) - 1):
            tok  = seq[pos]
            gold = seq[pos + 1]
            context = model.step(context, tok)
            is_correct = model.predict_token(context) == gold
            if is_correct:
                correct += 1
            _update_per_token_breakdown(per_token_stats, gold, is_correct)
            total += 1

    out: EvalResult = {"correct": correct, "total": total}
    if per_token_stats is not None:
        out["per_token"] = per_token_stats
    return out


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def evaluate_any(
    model: CircuitLM | PDACircuitLM | PPMModel,
    sequences: list[list[int]],
    per_token: bool = False,
) -> EvalResult:
    """Evaluate *model* regardless of type (FSM, PDA, or PPM).

    Dispatches to :func:`evaluate`, :func:`evaluate_pda`, or
    :func:`evaluate_ppm` based on the model type.  All return the same
    top-level ``{"correct": int, "total": int}`` schema; when
    ``per_token=True`` they also include ``"per_token"``.
    """
    if isinstance(model, PDACircuitLM):
        return evaluate_pda(model, sequences, per_token=per_token)
    if isinstance(model, PPMModel):
        return evaluate_ppm(model, sequences, per_token=per_token)
    return evaluate(model, sequences, per_token=per_token)
