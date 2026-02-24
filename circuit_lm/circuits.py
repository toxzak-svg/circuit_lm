"""FSM-based circuit language model data structure.

All fields and computations are strictly integer; no floats anywhere.

Architecture
------------
The model is a Mealy-style finite-state machine:

  next_state = delta(state, token)   # transition function
  prediction = argmax(emit[state])   # emission function (integer histogram)

States are non-negative integers in [0, num_states).
Tokens are non-negative integers in [0, vocab_size).

The *transitions* dict stores learned (or hash-derived) delta values.
The *state_counts* dict stores integer histograms over next tokens, one
histogram per state.

TODO: Expose circuit as a boolean DAG for interpretability.
TODO: Support multi-step lookahead states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Multiplier used in the default (hash-based) fallback transition.
# Must be a prime > 1; must be an integer literal (no floats).
HASH_PRIME: int = 31


@dataclass
class CircuitLM:
    """Finite-state circuit language model.

    Attributes:
        vocab_size:   Number of distinct token IDs.
        num_states:   Total number of FSM states (= 2 ** state_bits).
        state_bits:   Bit-width of the state representation.
        transitions:  Mapping (state, token) → next_state (integers).
        state_counts: Mapping state → list[int] of length vocab_size,
                      representing an integer count histogram over next tokens.
    """

    vocab_size: int
    num_states: int
    state_bits: int
    transitions: dict[tuple[int, int], int] = field(default_factory=dict)
    state_counts: dict[int, list[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def next_state(self, state: int, token: int) -> int:
        """Return the next state given the current state and observed token.

        Falls back to a deterministic rolling-hash formula if the
        (state, token) pair is not in *transitions*.
        """
        key = (state, token)
        if key in self.transitions:
            return self.transitions[key]
        # Integer-only fallback: deterministic hash
        return (state * HASH_PRIME + token + 1) % self.num_states

    def run(self, tokens: list[int], initial_state: int = 0) -> list[int]:
        """Run the FSM over *tokens* and return the resulting state sequence.

        The returned list has the same length as *tokens*: element *i* is
        the state **before** consuming ``tokens[i]``.
        """
        states: list[int] = []
        state = initial_state
        for tok in tokens:
            states.append(state)
            state = self.next_state(state, tok)
        return states

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_token(self, state: int) -> int:
        """Return the argmax (most frequent) next token from *state*.

        Returns token 0 (PAD) if the state has no observations.
        """
        counts = self.state_counts.get(state)
        if not counts:
            return 0
        best_tok = 0
        best_cnt = -1
        for tok, cnt in enumerate(counts):
            if cnt > best_cnt:
                best_cnt = cnt
                best_tok = tok
        return best_tok

    def state_histogram(self, state: int) -> list[int]:
        """Return the integer count histogram for *state*.

        Returns a zero-filled list of length *vocab_size* for unseen states.
        """
        counts = self.state_counts.get(state)
        if counts is not None:
            return counts
        return [0] * self.vocab_size
