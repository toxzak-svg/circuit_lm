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

TODO: Top-k filtering (integer: zero out all but the k largest counts).
TODO: Repetition penalty (integer: divide repeated token counts by a
      small integer factor, e.g. counts[t] //= penalty).
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
) -> list[int]:
    """Autoregressively sample tokens using integer-weighted random sampling for FSM.

    Fully deterministic for a given *seed*.  No floats.

    Args:
        model:      Trained :class:`~circuit_lm.circuits.CircuitLM`.
        prompt_ids: Integer token IDs forming the prompt / context.
        max_tokens: Number of new tokens to generate.
        seed:       Integer random seed for reproducibility.

    Returns:
        Full token-ID list: prompt + generated tokens.

    TODO: Integer top-k filtering; integer repetition penalty.
    """
    rng = random.Random(seed)
    ids = list(prompt_ids)
    state = 0
    for tok in ids:
        state = model.next_state(state, tok)

    for _ in range(max_tokens):
        weights = model.state_histogram(state)
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

    Returns:
        Full token-ID list: prompt + generated tokens.

    TODO: Integer top-k filtering; integer repetition penalty.
    """
    rng = random.Random(seed)
    ids = list(prompt_ids)
    state = 0
    stack: list[int] = []
    for tok in ids:
        state, stack = model.step(state, stack, tok)

    for _ in range(max_tokens):
        weights = model.config_histogram(state, stack)
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
) -> list[int]:
    """Autoregressively sample tokens using integer-weighted random sampling for PPM.

    Uses the blended integer histogram from all context levels for sampling.
    Fully deterministic for a given *seed*.  No floats.

    Args:
        model:      Trained :class:`~circuit_lm.ppm.PPMModel`.
        prompt_ids: Integer token IDs forming the prompt / context.
        max_tokens: Number of new tokens to generate.
        seed:       Integer random seed for reproducibility.

    Returns:
        Full token-ID list: prompt + generated tokens.
    """
    rng = random.Random(seed)
    ids = list(prompt_ids)
    context: tuple[int, ...] = ()
    for tok in ids:
        context = model.step(context, tok)

    for _ in range(max_tokens):
        weights = model.context_histogram(context)
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
) -> list[int]:
    """Integer-weighted sampling for FSM, PDA, or PPM model."""
    if isinstance(model, PDACircuitLM):
        return pda_sample_tokens(model, prompt_ids, max_tokens, seed)
    if isinstance(model, PPMModel):
        return ppm_sample_tokens(model, prompt_ids, max_tokens, seed)
    return sample_tokens(model, prompt_ids, max_tokens, seed)
