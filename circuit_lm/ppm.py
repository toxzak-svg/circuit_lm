"""Prediction by Partial Matching (PPM) context-tree language model.

Integer-only: all counts are Python ints; no floats anywhere.

The model is a variable-order n-gram trie with longest-match backoff.
Each node in the trie stores an integer count histogram over the vocabulary.

Prediction strategy (longest-match backoff)
-------------------------------------------
1. Start with the current context of length min(len(seen), order).
2. Walk back through shorter context suffixes until a node with nonzero
   counts is found.
3. If no node is found at any depth, fall back to token 0.

Blended histogram (for stochastic sampling)
--------------------------------------------
Each context level l contributes its counts multiplied by integer weight
(l + 1).  Longer contexts are favoured.  All arithmetic stays integer; no
floats introduced.  The result is an unnormalised integer histogram suitable
for integer-weighted sampling.

Complexity
----------
- Prediction: O(order * V) where V = vocab_size.
- Memory: O(K * V) where K = number of distinct context tuples seen.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPMModel:
    """Variable-order n-gram model with longest-match backoff.

    Attributes:
        vocab_size: Number of distinct token IDs (V).
        order:      Maximum context length in tokens.  0 = unigram only.
        counts:     Trie stored as a flat dict mapping context tuples to
                    integer count histograms of length *vocab_size*.
    """

    vocab_size: int
    order: int
    counts: dict[tuple[int, ...], list[int]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    def predict_token(self, context: tuple[int, ...]) -> int:
        """Return argmax next token using longest-match backoff.

        Args:
            context: Tuple of up to *order* preceding token IDs (oldest first).

        Returns:
            Token ID with the highest count at the deepest available context.
            Falls back to 0 if no context node has any counts.
        """
        for ctx_len in range(len(context), -1, -1):
            ctx = context[-ctx_len:] if ctx_len > 0 else ()
            hist = self.counts.get(ctx)
            if hist is not None:
                best_tok = 0
                best_cnt = 0
                for t, c in enumerate(hist):
                    if c > best_cnt:
                        best_cnt = c
                        best_tok = t
                if best_cnt > 0:
                    return best_tok
        return 0

    def context_histogram(self, context: tuple[int, ...]) -> list[int]:
        """Integer-blended count histogram across all context levels.

        Each level *l* (0 = empty context, 1 = last-token context, …) is
        weighted by ``l + 1`` so longer contexts are favoured.  All
        arithmetic is integer; no floats introduced.

        Args:
            context: Tuple of up to *order* preceding token IDs.

        Returns:
            Blended integer histogram of length *vocab_size*.
        """
        blended = [0] * self.vocab_size
        for ctx_len in range(len(context), -1, -1):
            ctx = context[-ctx_len:] if ctx_len > 0 else ()
            hist = self.counts.get(ctx)
            if hist is not None:
                weight = ctx_len + 1
                for t in range(self.vocab_size):
                    blended[t] += hist[t] * weight
        return blended

    # ------------------------------------------------------------------
    # State transition
    # ------------------------------------------------------------------

    def step(self, context: tuple[int, ...], token: int) -> tuple[int, ...]:
        """Advance the context window by consuming *token*.

        Args:
            context: Current context tuple (length <= order).
            token:   Newly observed token ID.

        Returns:
            Updated context tuple of length at most *order*.
        """
        if self.order == 0:
            return ()
        new_ctx = context + (token,)
        return new_ctx[-self.order:]

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run(
        self,
        tokens: list[int],
        initial_context: tuple[int, ...] | None = None,
    ) -> list[tuple[int, ...]]:
        """Run the context window over *tokens*.

        Returns the context *before* each token is consumed (same convention
        as :meth:`~circuit_lm.pda.PDACircuitLM.run`).

        Args:
            tokens:          Token sequence to process.
            initial_context: Starting context (default: empty tuple).

        Returns:
            List of context tuples, one per position in *tokens*.
        """
        context: tuple[int, ...] = initial_context if initial_context is not None else ()
        result: list[tuple[int, ...]] = []
        for tok in tokens:
            result.append(context)
            context = self.step(context, tok)
        return result
