"""Inference and token sampling using integer arithmetic only.

Sampling strategy
-----------------
Without floats we cannot use softmax or temperature scaling in the
conventional sense.  Instead we use *integer-weighted random sampling*:

  1. Obtain the integer count histogram h[t] for the current state/config.
  2. Draw a uniform integer r in [0, sum(h)).
  3. Return the index t such that the cumulative sum first exceeds r.

This is equivalent to sampling proportionally to empirical frequencies –
no division, no log, no float required.

Both FSM and PDA variants are provided; a unified helper dispatches to the
correct one based on model type.

TODO: Beam search over FSM states / PDA configurations.
"""

from __future__ import annotations

import random

from circuit_lm.circuits import CircuitLM
from circuit_lm.pda import PDACircuitLM
from circuit_lm.ppm import PPMModel


# ---------------------------------------------------------------------------
# Shared integer-weighted sampler
# ---------------------------------------------------------------------------


def _weighted_choice(weights: list[int], rng: random.Random) -> int:
    """Sample an index proportional to *weights* using only integer arithmetic.

    Args:
        weights: Non-negative integer weights.  If all are zero a uniform
                 choice is made over all indices.
        rng:     Seeded :class:`random.Random` instance.

    Returns:
        Sampled index in [0, len(weights)).
    """
    total = sum(weights)
    if total == 0:
        return rng.randrange(len(weights))

    r = rng.randrange(total)   # uniform integer in [0, total)
    cumsum = 0
    for i, w in enumerate(weights):
        cumsum += w
        if r < cumsum:
            return i
    return len(weights) - 1    # unreachable guard


def _apply_top_k(weights: list[int], top_k: int) -> list[int]:
    """Zero out all but the top-*k* weights (integer-only).

    Ties are broken by lower token ID first for deterministic behaviour.
    Returns a new list.
    """
    if top_k <= 0 or top_k >= len(weights):
        return list(weights)

    ranked_ids = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
    keep = set(ranked_ids[:top_k])
    return [w if i in keep else 0 for i, w in enumerate(weights)]


def _apply_repetition_penalty(
    weights: list[int],
    seen_ids: list[int],
    repeat_penalty_div: int,
    repeat_window: int,
) -> list[int]:
    """Apply an integer repetition penalty to recently seen tokens.

    Penalised tokens have their weight divided by ``repeat_penalty_div`` with
    integer floor division, but a positive weight is kept at least 1 so the
    token remains sampleable.

    ``repeat_window <= 0`` means "use the full history in *seen_ids*".
    Returns a new list.
    """
    if repeat_penalty_div <= 1 or not seen_ids:
        return list(weights)

    history = seen_ids[-repeat_window:] if repeat_window > 0 else seen_ids
    repeated = {tok for tok in history if 0 <= tok < len(weights)}
    if not repeated:
        return list(weights)

    out = list(weights)
    for tok in repeated:
        w = out[tok]
        if w > 0:
            out[tok] = max(1, w // repeat_penalty_div)
    return out


def _prepare_sampling_weights(
    base_weights: list[int],
    seen_ids: list[int],
    top_k: int = 0,
    repeat_penalty_div: int = 1,
    repeat_window: int = 0,
) -> list[int]:
    """Apply integer sampling controls to a histogram.

    Order of operations:
      1. repetition penalty
      2. top-k filter
    """
    weights = _apply_repetition_penalty(
        base_weights, seen_ids, repeat_penalty_div, repeat_window
    )
    weights = _apply_top_k(weights, top_k)
    return weights


# ---------------------------------------------------------------------------
# FSM greedy / stochastic
# ---------------------------------------------------------------------------


def greedy_decode(
    model: CircuitLM,
    prompt_ids: list[int],
    max_tokens: int,
) -> list[int]:
    """Autoregressively generate tokens by argmax (greedy) for an FSM model.

    Args:
        model:      Trained :class:`~circuit_lm.circuits.CircuitLM`.
        prompt_ids: Integer token IDs forming the prompt / context.
        max_tokens: Number of new tokens to generate.

    Returns:
        Full token-ID list: prompt + generated tokens.

    TODO: Beam search over FSM states.
    """
    ids = list(prompt_ids)
    state = 0
    for tok in ids:
        state = model.next_state(state, tok)

    for _ in range(max_tokens):
        next_tok = model.predict_token(state)
        ids.append(next_tok)
        state = model.next_state(state, next_tok)

    return ids


def sample_tokens(
    model: CircuitLM,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    top_k: int = 0,
    repeat_penalty_div: int = 1,
    repeat_window: int = 0,
) -> list[int]:
    """Autoregressively sample tokens using integer-weighted random sampling for FSM.

    Fully deterministic for a given *seed*.  No floats.

    Args:
        model:      Trained :class:`~circuit_lm.circuits.CircuitLM`.
        prompt_ids: Integer token IDs forming the prompt / context.
        max_tokens: Number of new tokens to generate.
        seed:       Integer random seed for reproducibility.
        top_k:      Keep only the top-k integer weights before sampling
                    (0 disables).
        repeat_penalty_div: Divide repeated-token weights by this integer
                    (1 disables).
        repeat_window: Penalise repeats seen within the last N tokens;
                    0 means use the full generated history.

    Returns:
        Full token-ID list: prompt + generated tokens.
    """
    rng = random.Random(seed)
    ids = list(prompt_ids)
    state = 0
    for tok in ids:
        state = model.next_state(state, tok)

    for _ in range(max_tokens):
        weights = _prepare_sampling_weights(
            model.state_histogram(state),
            ids,
            top_k=top_k,
            repeat_penalty_div=repeat_penalty_div,
            repeat_window=repeat_window,
        )
        next_tok = _weighted_choice(weights, rng)
        ids.append(next_tok)
        state = model.next_state(state, next_tok)

    return ids


# ---------------------------------------------------------------------------
# PDA greedy / stochastic
# ---------------------------------------------------------------------------


def pda_greedy_decode(
    model: PDACircuitLM,
    prompt_ids: list[int],
    max_tokens: int,
) -> list[int]:
    """Autoregressively generate tokens by argmax (greedy) for a PDA model.

    Maintains a bounded integer stack alongside the FSM state so that
    predictions are conditioned on the full ``(state, stack_top)``
    configuration.

    Args:
        model:      Trained :class:`~circuit_lm.pda.PDACircuitLM`.
        prompt_ids: Integer token IDs forming the prompt / context.
        max_tokens: Number of new tokens to generate.

    Returns:
        Full token-ID list: prompt + generated tokens.

    TODO: Beam search over PDA configurations (state × stack prefix).
    """
    ids = list(prompt_ids)
    state = 0
    stack: list[int] = []
    for tok in ids:
        state, stack = model.step(state, stack, tok)

    for _ in range(max_tokens):
        next_tok = model.predict_token(state, stack)
        ids.append(next_tok)
        state, stack = model.step(state, stack, next_tok)

    return ids


def pda_sample_tokens(
    model: PDACircuitLM,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    top_k: int = 0,
    repeat_penalty_div: int = 1,
    repeat_window: int = 0,
) -> list[int]:
    """Autoregressively sample tokens using integer-weighted random sampling for PDA.

    Maintains a bounded integer stack.  Sampling is proportional to the
    integer count histogram of the current ``(state, stack_top)`` config.
    Fully deterministic for a given *seed*.  No floats.

    Args:
        model:      Trained :class:`~circuit_lm.pda.PDACircuitLM`.
        prompt_ids: Integer token IDs forming the prompt / context.
        max_tokens: Number of new tokens to generate.
        seed:       Integer random seed for reproducibility.
        top_k:      Keep only the top-k integer weights before sampling
                    (0 disables).
        repeat_penalty_div: Divide repeated-token weights by this integer
                    (1 disables).
        repeat_window: Penalise repeats seen within the last N tokens;
                    0 means use the full generated history.

    Returns:
        Full token-ID list: prompt + generated tokens.
    """
    rng = random.Random(seed)
    ids = list(prompt_ids)
    state = 0
    stack: list[int] = []
    for tok in ids:
        state, stack = model.step(state, stack, tok)

    for _ in range(max_tokens):
        weights = _prepare_sampling_weights(
            model.config_histogram(state, stack),
            ids,
            top_k=top_k,
            repeat_penalty_div=repeat_penalty_div,
            repeat_window=repeat_window,
        )
        next_tok = _weighted_choice(weights, rng)
        ids.append(next_tok)
        state, stack = model.step(state, stack, next_tok)

    return ids


# ---------------------------------------------------------------------------
# PPM greedy / stochastic
# ---------------------------------------------------------------------------


def ppm_greedy_decode(
    model: PPMModel,
    prompt_ids: list[int],
    max_tokens: int,
) -> list[int]:
    """Autoregressively generate tokens by argmax (greedy) for a PPM model.

    Maintains a sliding context window of length at most *model.order* so
    that predictions are conditioned on the recent token history.

    Args:
        model:      Trained :class:`~circuit_lm.ppm.PPMModel`.
        prompt_ids: Integer token IDs forming the prompt / context.
        max_tokens: Number of new tokens to generate.

    Returns:
        Full token-ID list: prompt + generated tokens.
    """
    ids = list(prompt_ids)
    context: tuple[int, ...] = ()
    for tok in ids:
        context = model.step(context, tok)

    for _ in range(max_tokens):
        next_tok = model.predict_token(context)
        ids.append(next_tok)
        context = model.step(context, next_tok)

    return ids


def ppm_sample_tokens(
    model: PPMModel,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    top_k: int = 0,
    repeat_penalty_div: int = 1,
    repeat_window: int = 0,
) -> list[int]:
    """Autoregressively sample tokens using integer-weighted random sampling for PPM.

    Uses the blended integer histogram from all context levels for sampling.
    Fully deterministic for a given *seed*.  No floats.

    Args:
        model:      Trained :class:`~circuit_lm.ppm.PPMModel`.
        prompt_ids: Integer token IDs forming the prompt / context.
        max_tokens: Number of new tokens to generate.
        seed:       Integer random seed for reproducibility.
        top_k:      Keep only the top-k integer weights before sampling
                    (0 disables).
        repeat_penalty_div: Divide repeated-token weights by this integer
                    (1 disables).
        repeat_window: Penalise repeats seen within the last N tokens;
                    0 means use the full generated history.

    Returns:
        Full token-ID list: prompt + generated tokens.
    """
    rng = random.Random(seed)
    ids = list(prompt_ids)
    context: tuple[int, ...] = ()
    for tok in ids:
        context = model.step(context, tok)

    for _ in range(max_tokens):
        weights = _prepare_sampling_weights(
            model.context_histogram(context),
            ids,
            top_k=top_k,
            repeat_penalty_div=repeat_penalty_div,
            repeat_window=repeat_window,
        )
        next_tok = _weighted_choice(weights, rng)
        ids.append(next_tok)
        context = model.step(context, next_tok)

    return ids


# ---------------------------------------------------------------------------
# Unified dispatchers
# ---------------------------------------------------------------------------


def decode_greedy(
    model: CircuitLM | PDACircuitLM | PPMModel,
    prompt_ids: list[int],
    max_tokens: int,
) -> list[int]:
    """Greedy decode for FSM, PDA, or PPM model."""
    if isinstance(model, PDACircuitLM):
        return pda_greedy_decode(model, prompt_ids, max_tokens)
    if isinstance(model, PPMModel):
        return ppm_greedy_decode(model, prompt_ids, max_tokens)
    return greedy_decode(model, prompt_ids, max_tokens)


def decode_sample(
    model: CircuitLM | PDACircuitLM | PPMModel,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int,
    top_k: int = 0,
    repeat_penalty_div: int = 1,
    repeat_window: int = 0,
) -> list[int]:
    """Integer-weighted sampling for FSM, PDA, or PPM model."""
    if isinstance(model, PDACircuitLM):
        return pda_sample_tokens(
            model,
            prompt_ids,
            max_tokens,
            seed,
            top_k=top_k,
            repeat_penalty_div=repeat_penalty_div,
            repeat_window=repeat_window,
        )
    if isinstance(model, PPMModel):
        return ppm_sample_tokens(
            model,
            prompt_ids,
            max_tokens,
            seed,
            top_k=top_k,
            repeat_penalty_div=repeat_penalty_div,
            repeat_window=repeat_window,
        )
    return sample_tokens(
        model,
        prompt_ids,
        max_tokens,
        seed,
        top_k=top_k,
        repeat_penalty_div=repeat_penalty_div,
        repeat_window=repeat_window,
    )
