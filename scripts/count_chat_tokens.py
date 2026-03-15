"""Count tokens in chat data files using the project tokenizer (BPE or char).

Use this to check whether you have enough data for the 500k+ token hybrid
training target (see NEXT_LEVEL.md / STATUS_REPORT.md).

Usage
-----
  python scripts/count_chat_tokens.py
  python scripts/count_chat_tokens.py --files chat_data.txt chat_data_full.txt
  python scripts/count_chat_tokens.py --tokenizer char
  python scripts/count_chat_tokens.py --bpe-merges 1024
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from circuit_lm.data import load_text
from circuit_lm.tokenizer import Tokenizer

TARGET_TOKENS = 500_000


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count tokens in chat data files (BPE or char tokenizer)."
    )
    parser.add_argument(
        "--files",
        "-f",
        nargs="*",
        type=pathlib.Path,
        default=None,
        help="Chat text files to count (default: chat_data.txt, chat_data_full.txt, chat_data_500k.txt in repo root).",
    )
    parser.add_argument(
        "--tokenizer",
        "-t",
        choices=("bpe", "char"),
        default="bpe",
        help="Tokenizer mode (default: bpe).",
    )
    parser.add_argument(
        "--bpe-merges",
        type=int,
        default=512,
        help="BPE merge count when --tokenizer bpe (default: 512).",
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    if args.files:
        paths = [p.resolve() for p in args.files]
    else:
        defaults = [
            root / "chat_data.txt",
            root / "chat_data_full.txt",
            root / "chat_data_500k.txt",
        ]
        paths = [p for p in defaults if p.exists()]
        if not paths:
            print("No default chat data files found. Use --files to specify paths.", file=sys.stderr)
            return 1

    for p in paths:
        if not p.exists():
            print(f"File not found: {p}", file=sys.stderr)
            return 1

    # Build tokenizer from all file contents so BPE vocab matches the data
    combined = ""
    for p in paths:
        combined += load_text(p)
    if not combined.strip():
        print("All files are empty.", file=sys.stderr)
        return 1

    if args.tokenizer == "bpe":
        tokenizer = Tokenizer.from_text(
            combined,
            mode="bpe",
            bpe_merges=args.bpe_merges,
        )
    else:
        tokenizer = Tokenizer.from_text(combined, mode="char")

    print(f"Tokenizer: {args.tokenizer}" + (f" (merges={args.bpe_merges})" if args.tokenizer == "bpe" else ""))
    print(f"Vocab size: {tokenizer.vocab_size}")
    print()

    total = 0
    for p in paths:
        text = load_text(p)
        ids = tokenizer.encode(text)
        n = len(ids)
        total += n
        rel = root / p if p.is_absolute() else p
        try:
            rel = rel.relative_to(root)
        except ValueError:
            pass
        print(f"  {rel}: {n:,} tokens")

    print()
    print(f"  Total: {total:,} tokens")
    if total >= TARGET_TOKENS:
        print(f"  Target {TARGET_TOKENS:,}+ tokens: yes")
    else:
        print(f"  Target {TARGET_TOKENS:,}+ tokens: no (need {TARGET_TOKENS - total:,} more)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
