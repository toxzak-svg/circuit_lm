"""Data loading and integer sequence utilities.

All values are non-negative Python ints.  No floats, no numpy.
"""

from __future__ import annotations

import pathlib

from circuit_lm.tokenizer import Tokenizer


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_text(path: str | pathlib.Path) -> str:
    """Read a UTF-8 text file and return its contents as a string."""
    return pathlib.Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


def load_sequences(
    path: str | pathlib.Path,
    tokenizer: Tokenizer,
    seq_len: int = 256,
) -> list[list[int]]:
    """Tokenise a text file into fixed-length integer sequences.

    Each returned sequence has length ≤ seq_len + 1 and contains at least
    two tokens (so that at least one (input, target) pair can be formed).

    TODO: Streaming support for files larger than RAM.
    TODO: Overlap / stride parameter.

    Args:
        path:      Path to a plain-text UTF-8 file.
        tokenizer: Tokenizer used to convert characters to integer IDs.
        seq_len:   Maximum number of input tokens per chunk.

    Returns:
        List of integer token-ID sequences.
    """
    text = load_text(path)
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return []

    sequences: list[list[int]] = []
    for start in range(0, len(ids) - 1, seq_len):
        chunk = ids[start : start + seq_len + 1]
        if len(chunk) >= 2:
            sequences.append(chunk)
    return sequences


# ---------------------------------------------------------------------------
# Vocabulary statistics
# ---------------------------------------------------------------------------


def count_vocab(sequences: list[list[int]], vocab_size: int) -> list[int]:
    """Count token frequencies across all sequences.

    Returns a list of length *vocab_size* where index *t* holds the total
    count of token *t* across every sequence.  All values are integers.
    """
    counts = [0] * vocab_size
    for seq in sequences:
        for tok in seq:
            if 0 <= tok < vocab_size:
                counts[tok] += 1
    return counts
