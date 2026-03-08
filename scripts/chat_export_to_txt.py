"""Convert OpenAI/ChatGPT export conversation JSON to User:/Assistant: chat.txt.

Reads conversations-*.json from an extracted export directory (or zip) and
writes a single chat.txt suitable for CircuitLM training:

  python scripts/chat_export_to_txt.py --input chat_export --output chat.txt
  circuit-lm train --data chat.txt --out chat_model.json ...

If --input is a directory, looks for conversations-000.json, conversations-001.json, ...
If --input is a zip, extracts conversation JSONs and converts them.
"""

from __future__ import annotations

import argparse
import pathlib
import zipfile

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from circuit_lm.data import chat_text_from_openai_export


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert OpenAI/ChatGPT export to User:/Assistant: chat.txt"
    )
    parser.add_argument(
        "--input", "-i",
        type=pathlib.Path,
        required=True,
        help="Export directory (with conversations-*.json) or zip file.",
    )
    parser.add_argument(
        "--output", "-o",
        type=pathlib.Path,
        default=pathlib.Path("chat.txt"),
        help="Output chat text file (default: chat.txt).",
    )
    args = parser.parse_args()

    inp = args.input.resolve()
    paths: list[pathlib.Path] = []

    if inp.is_file() and inp.suffix.lower() == ".zip":
        with zipfile.ZipFile(inp, "r") as z:
            names = sorted(z.namelist())
            for name in names:
                if name.startswith("conversations-") and name.endswith(".json"):
                    # Extract to a temp dir or read from zip
                    base = inp.parent / (inp.stem + "_extracted")
                    base.mkdir(exist_ok=True)
                    z.extract(name, base)
                    paths.append(base / name)
        if not paths:
            print("No conversations-*.json found in zip.", file=sys.stderr)
            return 1
    elif inp.is_dir():
        for p in sorted(inp.glob("conversations-*.json")):
            paths.append(p)
        if not paths:
            print("No conversations-*.json found in directory.", file=sys.stderr)
            return 1
    else:
        print("--input must be a directory or a .zip file.", file=sys.stderr)
        return 1

    text = chat_text_from_openai_export(paths)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {len(text)} chars from {len(paths)} file(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
