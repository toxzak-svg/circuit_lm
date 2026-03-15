"""Step-by-step trace of circuit state and predictions (interpretability).

All values are integers. No floats. Yields one record per prompt position:
  (step, token_id, state, stack_top, top_k_token_ids)

For FSM, stack_top is None. For PDA, stack_top is the top of stack or -1 if empty.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from circuit_lm.circuits import CircuitLM
from circuit_lm.pda import PDACircuitLM, STACK_EMPTY
from circuit_lm.ppm import PPMModel


@dataclass
class TraceStep:
    """One step in the trace: state/config before consuming token at this index."""
    step: int
    token_id: int
    state: int
    stack_top: int | None  # None for FSM
    top_k_token_ids: list[int]


def _top_k_from_histogram(hist: list[int], k: int) -> list[int]:
    """Return token IDs with highest counts (descending), up to k. Integer-only."""
    if k <= 0:
        return []
    ranked = sorted(range(len(hist)), key=lambda i: (-hist[i], i))
    return ranked[:k]


def trace_steps(
    model: CircuitLM | PDACircuitLM | PPMModel,
    prompt_ids: list[int],
    top_k: int = 5,
) -> Iterator[TraceStep]:
    """Yield a trace step for each position in prompt_ids.

    At step i we have consumed prompt_ids[0:i], we are in (state, stack), and
    we are about to predict the next token (the gold next token is prompt_ids[i]).
    Yields (i, prompt_ids[i], state, stack_top, top_k predicted token IDs).
    """
    if isinstance(model, PDACircuitLM):
        state = 0
        stack: list[int] = []
        for i, token_id in enumerate(prompt_ids):
            stack_top = stack[-1] if stack else STACK_EMPTY
            hist = model.config_histogram(state, stack)
            top_k_ids = _top_k_from_histogram(hist, top_k)
            yield TraceStep(
                step=i,
                token_id=token_id,
                state=state,
                stack_top=stack_top,
                top_k_token_ids=top_k_ids,
            )
            state, stack = model.step(state, stack, token_id)
        return

    if isinstance(model, PPMModel):
        # PPM: no single "state"; use context. For trace we show context and top-k.
        for i in range(len(prompt_ids)):
            token_id = prompt_ids[i]
            start = max(0, i - model.order)
            context = tuple(prompt_ids[start:i])
            hist = model.context_histogram(context)
            top_k_ids = _top_k_from_histogram(hist, top_k)
            yield TraceStep(
                step=i,
                token_id=token_id,
                state=0,  # PPM has no state index
                stack_top=None,
                top_k_token_ids=top_k_ids,
            )
        return

    # FSM
    state = 0
    for i, token_id in enumerate(prompt_ids):
        hist = model.state_histogram(state)
        top_k_ids = _top_k_from_histogram(hist, top_k)
        yield TraceStep(
            step=i,
            token_id=token_id,
            state=state,
            stack_top=None,
            top_k_token_ids=top_k_ids,
        )
        state = model.next_state(state, token_id)
