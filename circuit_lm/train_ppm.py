"""Training for the PPM (Prediction by Partial Matching) context-tree model.

No CP-SAT required – training is pure integer n-gram counting.

Complexity
----------
- Time:  O(N * order) where N = total tokens across all sequences.
- Space: O(K * V) where K = number of distinct context tuples observed and
         V = vocab_size.

Algorithm
---------
For each position *pos* in each training sequence, the token to predict is
``seq[pos + 1]``.  The observed context at depth *ctx_len* is the tuple of
the last *ctx_len* tokens ending at *pos*::

    ctx_len = 0  →  context = ()              (empty / unigram fallback)
    ctx_len = 1  →  context = (seq[pos],)
    ctx_len = 2  →  context = (seq[pos-1], seq[pos])
    ...
    ctx_len = k  →  context = tuple(seq[pos-k+1 : pos+1])

The empty context ``()`` always accumulates counts from every position,
guaranteeing a unigram fallback is always available.
"""

from __future__ import annotations

from circuit_lm.ppm import PPMModel


def train_ppm(
    sequences: list[list[int]],
    vocab_size: int,
    order: int,
) -> PPMModel:
    """Build a PPM context-tree model by counting n-grams up to *order*.

    Args:
        sequences:  List of integer token-ID sequences from training data.
        vocab_size: Size of the token vocabulary (V).
        order:      Maximum context length (0 = unigram only, 1 = bigram, …).

    Returns:
        Trained :class:`~circuit_lm.ppm.PPMModel` with integer-only counts.
    """
    counts: dict[tuple[int, ...], list[int]] = {}

    for seq in sequences:
        n = len(seq)
        for pos in range(n - 1):
            next_tok = seq[pos + 1]
            # can't look back further than pos tokens
            max_ctx = min(pos + 1, order)
            for ctx_len in range(max_ctx + 1):
                if ctx_len == 0:
                    ctx: tuple[int, ...] = ()
                else:
                    ctx = tuple(seq[pos - ctx_len + 1 : pos + 1])
                if ctx not in counts:
                    counts[ctx] = [0] * vocab_size
                counts[ctx][next_tok] += 1

    return PPMModel(vocab_size=vocab_size, order=order, counts=counts)
