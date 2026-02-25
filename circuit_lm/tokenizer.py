"""Integer tokenizer with character and simple BPE modes.

All operations map strings to/from non-negative integers.
No floats, no external dependencies.
"""

from __future__ import annotations


class Tokenizer:
    """Maps text units (chars or BPE pieces) to integer token IDs and back.

    Reserved IDs:
      0 → <PAD>   (padding / unknown)
      1 → <UNK>   (out-of-vocabulary character)
    User pieces start at ID 2.

    Modes:
      ``"char"`` – character-level tokenizer (default)
      ``"bpe"``  – simple greedy BPE over the raw character stream

    TODO: Support byte-level fallback for arbitrary Unicode.
    """

    PAD_ID: int = 0
    UNK_ID: int = 1

    def __init__(
        self,
        vocab: list[str] | None = None,
        mode: str = "char",
    ) -> None:
        if mode not in ("char", "bpe"):
            raise ValueError(f"Unsupported tokenizer mode: {mode!r}")
        base = ["<PAD>", "<UNK>"]
        self._mode = mode
        # Deduplicate while preserving order.
        seen: set[str] = set()
        user_pieces: list[str] = []
        for piece in (vocab or []):
            if piece in ("<PAD>", "<UNK>"):
                continue
            if piece not in seen:
                seen.add(piece)
                user_pieces.append(piece)
        self._pieces: list[str] = base + user_pieces
        # Backward-compatible alias used by older code/docs and legacy payloads.
        self._chars = self._pieces
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self._piece_to_id: dict[str, int] = {p: i for i, p in enumerate(self._pieces)}
        # Char-mode encode uses direct map lookup per character.
        self._char_to_id: dict[str, int] = (
            self._piece_to_id if self._mode == "char" else {}
        )
        user_pieces = self._pieces[2:]
        self._bpe_piece_set: set[str] = set(user_pieces)
        self._max_piece_len: int = 1
        for piece in user_pieces:
            if len(piece) > self._max_piece_len:
                self._max_piece_len = len(piece)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab_size: int | None = None,
        mode: str = "char",
        bpe_merges: int | None = None,
    ) -> "Tokenizer":
        """Build a tokenizer from raw text.

        In ``"char"`` mode, characters are sorted by descending frequency so
        the most common characters get the lowest IDs (after PAD/UNK).
        In ``"bpe"`` mode, a simple deterministic BPE is learned over the raw
        character stream and the resulting piece vocabulary is stored.

        Args:
            text:       Input text to derive the vocabulary from.
            vocab_size: Maximum vocabulary size (including PAD + UNK).
                        Pass None to include every character in *text*.
            mode:       ``"char"`` (default) or ``"bpe"``.
            bpe_merges: Maximum number of BPE merges to apply in ``"bpe"``
                        mode.  Ignored in ``"char"`` mode.  ``None`` means
                        continue merging until no valid pair remains or
                        ``vocab_size`` is reached.
        """
        if mode not in ("char", "bpe"):
            raise ValueError(f"Unsupported tokenizer mode: {mode!r}")
        if bpe_merges is not None and bpe_merges < 0:
            raise ValueError("bpe_merges must be >= 0 or None")

        if mode == "bpe":
            pieces = cls._build_bpe_pieces(text, vocab_size, bpe_merges)
            return cls(pieces, mode="bpe")

        freq: dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1

        sorted_chars = sorted(freq, key=freq.__getitem__, reverse=True)
        if vocab_size is not None:
            # Reserve 2 slots for PAD and UNK
            sorted_chars = sorted_chars[: max(0, vocab_size - 2)]
        return cls(sorted_chars, mode="char")

    @classmethod
    def _build_bpe_pieces(
        cls,
        text: str,
        vocab_size: int | None,
        bpe_merges: int | None,
    ) -> list[str]:
        """Learn a simple deterministic BPE vocabulary over raw text.

        This implementation operates directly on the full character stream
        (including whitespace) so that decoding is exact string concatenation.
        """
        # Base alphabet: same frequency-based truncation policy as char mode.
        char_freq: dict[str, int] = {}
        for ch in text:
            char_freq[ch] = char_freq.get(ch, 0) + 1

        base_chars = sorted(char_freq, key=char_freq.__getitem__, reverse=True)
        if vocab_size is not None:
            base_chars = base_chars[: max(0, vocab_size - 2)]

        if not text or not base_chars:
            return list(base_chars)

        base_set = set(base_chars)
        seq: list[str] = [ch if ch in base_set else "<UNK>" for ch in text]

        pieces: list[str] = list(base_chars)
        piece_set: set[str] = set(pieces)
        merges_left = bpe_merges if bpe_merges is not None else -1

        while True:
            if merges_left == 0:
                break
            if vocab_size is not None and (len(pieces) + 2) >= vocab_size:
                break

            pair_counts: dict[tuple[str, str], int] = {}
            for i in range(len(seq) - 1):
                a = seq[i]
                b = seq[i + 1]
                if a == "<UNK>" or b == "<UNK>":
                    continue
                pair = (a, b)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            ranked_pairs = sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            best_pair: tuple[str, str] | None = None
            merged_piece = ""
            for pair, count in ranked_pairs:
                if count < 2:
                    break
                candidate = pair[0] + pair[1]
                if candidate in ("<PAD>", "<UNK>"):
                    continue
                if candidate in piece_set:
                    continue
                best_pair = pair
                merged_piece = candidate
                break

            if best_pair is None:
                break

            new_seq: list[str] = []
            i = 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == best_pair[0] and seq[i + 1] == best_pair[1]:
                    new_seq.append(merged_piece)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq

            pieces.append(merged_piece)
            piece_set.add(merged_piece)
            if merges_left > 0:
                merges_left -= 1

        return pieces

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._pieces)

    @property
    def mode(self) -> str:
        return self._mode

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of integer token IDs."""
        if self._mode == "char":
            return [self._char_to_id.get(ch, self.UNK_ID) for ch in text]
        return self._encode_bpe(text)

    def _encode_bpe(self, text: str) -> list[int]:
        """Greedy longest-piece encoding for BPE mode."""
        ids: list[int] = []
        i = 0
        n = len(text)
        while i < n:
            max_len = self._max_piece_len
            if max_len > (n - i):
                max_len = n - i

            matched_id: int | None = None
            matched_len = 0
            for piece_len in range(max_len, 0, -1):
                piece = text[i : i + piece_len]
                tok_id = self._piece_to_id.get(piece)
                if tok_id is not None and tok_id >= 2:
                    matched_id = tok_id
                    matched_len = piece_len
                    break

            if matched_id is None:
                ids.append(self.UNK_ID)
                i += 1
            else:
                ids.append(matched_id)
                i += matched_len
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of integer token IDs back to a string.

        PAD and UNK tokens are rendered as the replacement character U+FFFD.
        """
        out: list[str] = []
        for i in ids:
            if 0 <= i < len(self._pieces):
                ch = self._pieces[i]
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
        if self._mode == "char":
            # Preserve legacy "chars" payload for backward compatibility.
            return {"mode": "char", "chars": self._pieces}
        return {"mode": "bpe", "pieces": self._pieces}

    @classmethod
    def from_dict(cls, d: dict) -> "Tokenizer":
        """Restore a Tokenizer from a dict produced by :meth:`to_dict`."""
        obj = cls.__new__(cls)
        mode = d.get("mode", "char")
        if "pieces" in d:
            pieces = list(d["pieces"])
        else:
            # Legacy serialisation format (char-only)
            pieces = list(d["chars"])
            mode = "char"
        obj._mode = mode
        obj._pieces = pieces
        obj._chars = obj._pieces
        obj._rebuild_indices()
        return obj
