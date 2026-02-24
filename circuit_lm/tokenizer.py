"""Character-level tokenizer using integer IDs only.

All operations map characters to/from non-negative integers.
No floats, no external dependencies.
"""

from __future__ import annotations


class Tokenizer:
    """Maps characters to integer token IDs and back.

    Reserved IDs:
      0 → <PAD>   (padding / unknown)
      1 → <UNK>   (out-of-vocabulary character)
    User characters start at ID 2.

    TODO: Support BPE / subword tokenisation.
    TODO: Support byte-level fallback for arbitrary Unicode.
    """

    PAD_ID: int = 0
    UNK_ID: int = 1

    def __init__(self, vocab: list[str] | None = None) -> None:
        base = ["<PAD>", "<UNK>"]
        self._chars: list[str] = base + list(vocab or [])
        self._char_to_id: dict[str, int] = {c: i for i, c in enumerate(self._chars)}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_text(cls, text: str, vocab_size: int | None = None) -> "Tokenizer":
        """Build a tokenizer from raw text.

        Characters are sorted by descending frequency so the most common
        characters get the lowest IDs (after PAD/UNK).

        Args:
            text:       Input text to derive the vocabulary from.
            vocab_size: Maximum vocabulary size (including PAD + UNK).
                        Pass None to include every character in *text*.
        """
        freq: dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1

        sorted_chars = sorted(freq, key=freq.__getitem__, reverse=True)
        if vocab_size is not None:
            # Reserve 2 slots for PAD and UNK
            sorted_chars = sorted_chars[: max(0, vocab_size - 2)]
        return cls(sorted_chars)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of integer token IDs."""
        return [self._char_to_id.get(ch, self.UNK_ID) for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Decode a list of integer token IDs back to a string.

        PAD and UNK tokens are rendered as the replacement character U+FFFD.
        """
        out: list[str] = []
        for i in ids:
            if 0 <= i < len(self._chars):
                ch = self._chars[i]
                if ch in ("<PAD>", "<UNK>"):
                    out.append("\ufffd")
                else:
                    out.append(ch)
            else:
                out.append("\ufffd")
        return "".join(out)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe, integer-keyed internally)."""
        return {"chars": self._chars}

    @classmethod
    def from_dict(cls, d: dict) -> "Tokenizer":
        """Restore a Tokenizer from a dict produced by :meth:`to_dict`."""
        obj = cls.__new__(cls)
        obj._chars = list(d["chars"])
        obj._char_to_id = {c: i for i, c in enumerate(obj._chars)}
        return obj
