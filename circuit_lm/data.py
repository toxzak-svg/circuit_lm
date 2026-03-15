"""Data loading and integer sequence utilities.

All values are non-negative Python ints.  No floats, no numpy.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterator

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

    Loads the entire file into memory. For large corpora use
    :func:`iter_sequences` or :func:`iter_sequence_chunks` instead.

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


def iter_sequences(
    path: str | pathlib.Path,
    tokenizer: Tokenizer,
    seq_len: int = 256,
    stride: int | None = None,
) -> Iterator[list[int]]:
    """Yield fixed-length sequences from a text file without loading it all into memory.

    Reads the file line by line, tokenizes incrementally, and yields each
    sequence of length seq_len + 1 (so that at least one (input, target) pair
    can be formed). Use this for corpora larger than RAM.

    Args:
        path:      Path to a plain-text UTF-8 file.
        tokenizer: Tokenizer used to convert characters to integer IDs.
        seq_len:   Maximum number of input tokens per sequence (yielded length is seq_len + 1).
        stride:    Advance by this many tokens after each sequence (default: seq_len).

    Yields:
        Integer token-ID lists of length seq_len + 1.
    """
    if stride is None:
        stride = seq_len
    ids: list[int] = []
    path = pathlib.Path(path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            ids.extend(tokenizer.encode(line))
            while len(ids) >= seq_len + 1:
                yield ids[: seq_len + 1]
                ids = ids[stride:]


def iter_sequence_chunks(
    path: str | pathlib.Path,
    tokenizer: Tokenizer,
    seq_len: int = 256,
    stride: int | None = None,
    chunk_size: int = 5000,
) -> Iterator[list[list[int]]]:
    """Yield batches of sequences from a text file without loading it all into memory.

    Same as :func:`iter_sequences` but yields lists of up to *chunk_size*
    sequences at a time, for use with trainers that process batches.

    Args:
        path:       Path to a plain-text UTF-8 file.
        tokenizer:  Tokenizer used to convert characters to integer IDs.
        seq_len:    Maximum number of input tokens per sequence.
        stride:     Advance by this many tokens after each sequence (default: seq_len).
        chunk_size: Maximum number of sequences per yielded batch.

    Yields:
        Lists of integer token-ID sequences (each of length seq_len + 1).
    """
    if stride is None:
        stride = seq_len
    batch: list[list[int]] = []
    for seq in iter_sequences(path, tokenizer, seq_len=seq_len, stride=stride):
        batch.append(seq)
        if len(batch) >= chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Chat data (User: / Assistant: format)
# ---------------------------------------------------------------------------

def chat_text_from_jsonl(path: str | pathlib.Path) -> str:
    """Read a JSONL chat file and return one string in User:/Assistant: format.

    Each line of the file is one JSON object. Supported shapes:

    - Turn pair: ``{"user": "hello", "assistant": "hi there"}``
    - Full convo: ``{"messages": [{"role": "user", "content": "..."}, ...]}``

    Conversations are concatenated with newlines so the result can be
    saved to a text file and used with :func:`load_sequences` for training.

    Args:
        path: Path to UTF-8 JSONL file.

    Returns:
        Single string: "User: ...\\nAssistant: ...\\nUser: ...\\n"
    """
    from circuit_lm.chat import ASSISTANT_PREFIX, USER_PREFIX

    lines: list[str] = []
    with pathlib.Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                for msg in obj["messages"]:
                    role = (msg.get("role") or "").strip().lower()
                    content = (msg.get("content") or "").strip()
                    if role == "user":
                        lines.append(USER_PREFIX + content + "\n")
                    elif role == "assistant":
                        lines.append(ASSISTANT_PREFIX + content + "\n")
            elif "user" in obj or "assistant" in obj:
                user = (obj.get("user") or "").strip()
                assistant = (obj.get("assistant") or "").strip()
                if user:
                    lines.append(USER_PREFIX + user + "\n")
                if assistant:
                    lines.append(ASSISTANT_PREFIX + assistant + "\n")
    return "".join(lines)


def chat_text_from_openai_export(
    paths: str | pathlib.Path | list[str] | list[pathlib.Path],
) -> str:
    """Read OpenAI/ChatGPT export conversation JSON and return User:/Assistant: text.

    Expects the format from ChatGPT "Export data" (conversations-*.json): each
    file is a JSON array of conversation objects with a "mapping" of nodes.
    Each node has "message": {"author": {"role": "user"|"assistant"},
    "content": {"content_type": "text", "parts": ["..."]}}, "parent", "children".
    Walks the tree from root(s) by children to preserve order.

    Args:
        paths: One or more paths to conversations-*.json files.

    Returns:
        Single string in "User: ...\\nAssistant: ...\\n" format for training.
    """
    from circuit_lm.chat import ASSISTANT_PREFIX, USER_PREFIX

    if isinstance(paths, (str, pathlib.Path)):
        paths = [paths]
    lines: list[str] = []
    for path in paths:
        path = pathlib.Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = [data]
        for conv in data:
            mapping = conv.get("mapping") or {}
            if not mapping:
                continue
            # Root nodes: parent is null or not in this mapping
            root_ids = [
                nid
                for nid, n in mapping.items()
                if not n.get("parent") or n["parent"] not in mapping
            ]

            def walk(node_id: str) -> None:
                node = mapping.get(node_id)
                if not node:
                    return
                msg = node.get("message")
                if msg:
                    role = (msg.get("author") or {}).get("role") or ""
                    role = str(role).strip().lower()
                    content = msg.get("content") or {}
                    if content.get("content_type") == "text":
                        parts = content.get("parts") or []
                        text = " ".join(str(p).strip() for p in parts if p).strip()
                        if text and role in ("user", "assistant"):
                            if role == "user":
                                lines.append(USER_PREFIX + text + "\n")
                            else:
                                lines.append(ASSISTANT_PREFIX + text + "\n")
                for child_id in node.get("children") or []:
                    walk(child_id)

            for rid in root_ids:
                walk(rid)
    return "".join(lines)


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
