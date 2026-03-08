"""Convert JSONL chat data to User:/Assistant: text for training.

Reads JSONL from stdin or --input and writes formatted chat text to
stdout or --output. Use the output as training data:

  python scripts/chat_jsonl_to_txt.py --input chats.jsonl > chat.txt
  circuit-lm train --data chat.txt --out chat_model.json ...

Supported JSONL shapes:
  - {"user": "...", "assistant": "..."}  (one turn per line)
  - {"messages": [{"role": "user", "content": "..."}, ...]}  (one convo per line)

Usage
-----
  python scripts/chat_jsonl_to_txt.py < chats.jsonl > chat.txt
  python scripts/chat_jsonl_to_txt.py --input chats.jsonl --output chat.txt
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from circuit_lm.data import chat_text_from_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert JSONL chat to User:/Assistant: text.")
    parser.add_argument("--input", "-i", type=pathlib.Path, default=None,
                        help="Input JSONL file (default: stdin).")
    parser.add_argument("--output", "-o", type=pathlib.Path, default=None,
                        help="Output text file (default: stdout).")
    args = parser.parse_args()

    if args.input is not None:
        text = chat_text_from_jsonl(args.input)
    else:
        # Read from stdin: write to temp then use same API, or implement line-by-line.
        lines: list[str] = []
        for line in sys.stdin:
            lines.append(line)
        # chat_text_from_jsonl expects a path; we have lines. Use a small helper.
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.writelines(lines)
            path = pathlib.Path(f.name)
        try:
            text = chat_text_from_jsonl(path)
        finally:
            path.unlink(missing_ok=True)

    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
