"""Pushdown Automaton Circuit Language Model (PDA-CLM).

Extends the plain FSM in circuits.py by adding a bounded integer stack.

Architecture
============
A PDA configuration is the pair::

    config = (state: int, stack_top: int)

where ``state`` is the FSM state (integer in [0, num_states)) and
``stack_top`` is the topmost element of the stack (an integer token ID)
or ``STACK_EMPTY`` (-1) when the stack is empty.

Stack operations (integer-encoded)
-----------------------------------
    OP_NOP  = 0   keep stack unchanged
    OP_PUSH = 1   push the current token onto the stack
    OP_POP  = 2   pop the top element off the stack

The operation for each (src_state, token, stack_top_before_op) triple is
determined by membership in ``push_configs`` or ``pop_configs`` (two disjoint
sets of integer triples learned by CP-SAT in ``train_pda_cpsat.py``).  If a
triple belongs to neither set the operation is OP_NOP.

Expressiveness lift over plain FSM
------------------------------------
* A plain FSM has memory O(log num_states) bits – essentially a bounded
  sliding window of recent context.
* A PDA with a stack of depth D has memory O(D × log vocab_size) bits and
  can express many context-free patterns: nested brackets, scope tracking,
  paired delimiters, repeated-token long-range dependencies.

All fields, operations, and return values are integers.  No floats.

TODO: Expose multiple stack elements (top-k) as configuration features.
TODO: Support multi-stack / counter automaton extension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from circuit_lm.circuits import HASH_PRIME

# ---------------------------------------------------------------------------
# Constants (all integers)
# ---------------------------------------------------------------------------

STACK_EMPTY: int = -1   # sentinel used when stack is empty
OP_NOP:      int = 0    # no stack change
OP_PUSH:     int = 1    # push current token
OP_POP:      int = 2    # pop top element


# ---------------------------------------------------------------------------
# PDACircuitLM
# ---------------------------------------------------------------------------


@dataclass
class PDACircuitLM:
    """Pushdown automaton language model.

    Configuration
    -------------
    ``(state: int, stack_top: int)`` where ``stack_top`` is either an
    integer token ID or ``STACK_EMPTY`` (-1).

    Transition
    ----------
    ``next_state = transitions.get((state, token), hash_fallback)``
    Stack operation is determined by triple membership in push_configs/pop_configs.

    Emission
    --------
    ``config_counts[(state, stack_top)]`` is an integer count histogram
    of length ``vocab_size`` over next tokens observed in training.
    Optional ``config_pred_tokens`` stores learned emissions per config.

    Attributes
    ----------
    vocab_size:    Number of distinct token IDs.
    num_states:    Total FSM states (= 2 ** state_bits).
    state_bits:    Bit-width of state representation.
    stack_depth:   Maximum stack depth (integer bound).
    push_configs:  Frozen set of (src_state, token, stack_top_before_op) triples that trigger OP_PUSH.
    pop_configs:   Frozen set of (src_state, token, stack_top_before_op) triples that trigger OP_POP.
    transitions:   ``{(state, token): next_state}`` integer mapping.
    config_counts: ``{(state, stack_top): list[int]}`` histograms.
    config_pred_tokens: ``{(state, stack_top): predicted_token}`` mapping.
    """

    vocab_size:    int
    num_states:    int
    state_bits:    int
    stack_depth:   int
    push_configs:  frozenset[tuple[int, int, int]] = field(default_factory=frozenset)
    pop_configs:   frozenset[tuple[int, int, int]] = field(default_factory=frozenset)
    transitions:   dict[tuple[int, int], int] = field(default_factory=dict)
    config_counts: dict[tuple[int, int], list[int]] = field(default_factory=dict)
    config_pred_tokens: dict[tuple[int, int], int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Stack operation
    # ------------------------------------------------------------------

    def stack_op(self, state: int, token: int, stack_top: int) -> int:
        """Return the stack operation for the given (state, token, stack_top) triple.

        Uses push_configs / pop_configs; returns OP_NOP if the triple appears in
        neither set.  All arguments are integers; STACK_EMPTY (-1) is the sentinel
        for an empty stack.
        """
        key = (state, token, stack_top)
        if key in self.push_configs:
            return OP_PUSH
        if key in self.pop_configs:
            return OP_POP
        return OP_NOP

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def next_state(self, state: int, token: int) -> int:
        """Return the next FSM state (falls back to integer rolling hash)."""
        key = (state, token)
        if key in self.transitions:
            return self.transitions[key]
        return (state * HASH_PRIME + token + 1) % self.num_states

    # ------------------------------------------------------------------
    # Single PDA step
    # ------------------------------------------------------------------

    def step(
        self, state: int, stack: list[int], token: int
    ) -> tuple[int, list[int]]:
        """Execute one PDA step and return (next_state, new_stack).

        Stack operation is resolved using (state, token, stack_top) before the
        FSM transition so that the source state governs the op decision.

        The stack is bounded by ``stack_depth``; OP_PUSH is silently ignored
        when the stack is already full.  OP_POP on an empty stack is a no-op.

        Args:
            state:  Current FSM state (integer).
            stack:  Current stack contents (list of integer token IDs).
            token:  Current input token (integer).

        Returns:
            ``(next_state, new_stack)`` both using only integer values.
        """
        stack_top = stack[-1] if stack else STACK_EMPTY
        op = self.stack_op(state, token, stack_top)
        next_s = self.next_state(state, token)

        new_stack = list(stack)
        if op == OP_PUSH and len(new_stack) < self.stack_depth:
            new_stack.append(token)
        elif op == OP_POP and new_stack:
            new_stack.pop()

        return next_s, new_stack

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_token(self, state: int, stack: list[int]) -> int:
        """Return the argmax next token for the current configuration.

        Uses ``config_pred_tokens[(state, stack_top)]`` when available;
        otherwise falls back to ``config_counts[(state, stack_top)]`` where
        ``stack_top = stack[-1]`` if the stack is non-empty, else
        ``STACK_EMPTY``.  Returns 0 (PAD) if no observations exist.
        """
        stack_top = stack[-1] if stack else STACK_EMPTY
        learned = self.config_pred_tokens.get((state, stack_top))
        if learned is not None:
            return learned
        counts = self.config_counts.get((state, stack_top))
        if not counts:
            return 0
        best_tok = 0
        best_cnt = -1
        for tok, cnt in enumerate(counts):
            if cnt > best_cnt:
                best_cnt = cnt
                best_tok = tok
        return best_tok

    def config_histogram(self, state: int, stack: list[int]) -> list[int]:
        """Return the integer count histogram for the current configuration.

        Returns a zero-filled list of length ``vocab_size`` for unseen configs.
        """
        stack_top = stack[-1] if stack else STACK_EMPTY
        counts = self.config_counts.get((state, stack_top))
        if counts is not None:
            return counts
        return [0] * self.vocab_size

    # ------------------------------------------------------------------
    # Full sequence run
    # ------------------------------------------------------------------

    def run(
        self,
        tokens: list[int],
        initial_state: int = 0,
        initial_stack: list[int] | None = None,
    ) -> list[tuple[int, list[int]]]:
        """Run the PDA over *tokens*, returning a list of configurations.

        Element *i* of the returned list is the ``(state, stack_copy)``
        **before** consuming ``tokens[i]``.

        Args:
            tokens:        Integer token-ID sequence.
            initial_state: Starting FSM state (default 0).
            initial_stack: Starting stack contents (default empty).

        Returns:
            List of ``(state, stack_snapshot)`` tuples, same length as
            *tokens*.  Each ``stack_snapshot`` is a shallow copy of the
            stack at that step.

        TODO: Return full config sequence including post-sequence state.
        """
        configs: list[tuple[int, list[int]]] = []
        state = initial_state
        stack = list(initial_stack) if initial_stack is not None else []

        for tok in tokens:
            configs.append((state, list(stack)))
            state, stack = self.step(state, stack, tok)

        return configs
