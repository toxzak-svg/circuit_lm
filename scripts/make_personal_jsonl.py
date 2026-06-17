"""Convert personal data files into a unified JSONL chat dataset.

Input files (from build_personal_dataset/):
  - personal_training.jsonl  (3125 lines, alternating role/text)
  - training_data.txt        (generic text, one conversation per block)
  - chatgpt_data.txt         (ChatGPT exports)
  - chat_data_full.txt       (user:/assistant: format)
  - research_evolver_data.txt
  - marble_data.txt
  - nova_data.txt

Output: personal_chat_dataset.jsonl
  One JSON object per line:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "build_personal_dataset"
OUTPUT_FILE = ROOT / "personal_chat_dataset.jsonl"


def process_personal_jsonl(path: Path) -> list[dict]:
    """Load personal_training.jsonl and group into multi-turn conversations.

    The file has 3125 alternating lines:
      {"role": "user", "text": "..."}
      {"role": "assistant", "text": "..."}
      ...

    Group consecutive exchanges into conversations. Break on "System:" role.
    """
    convos = []
    current = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = obj.get("role", "").lower()
            text = obj.get("text", "").strip()
            if not text:
                continue

            if role == "system":
                # New conversation starts
                if current:
                    convos.append({"messages": current})
                    current = []
                continue

            if role not in ("user", "assistant"):
                continue

            # Multiple consecutive assistants = single response to user
            if current and role == "assistant":
                # Append to last assistant message
                current[-1]["content"] += "\n" + text
            else:
                current.append({"role": role, "content": text})

    if current:
        convos.append({"messages": current})
    return convos


def process_txt_conversations(path: Path) -> list[dict]:
    """Load a plain-text conversation file.

    Format expected:
      user: message
      assistant: message
      user: message
      ...

    Blank lines separate conversations.
    """
    convos = []
    current = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                if current:
                    convos.append({"messages": current})
                    current = []
                continue
            lower = line.lower()
            if lower.startswith("user:") or lower.startswith("user "):
                role = "user"
                content = line[5 if lower.startswith("user ") else 5:].strip()
            elif lower.startswith("assistant:") or lower.startswith("assistant "):
                role = "assistant"
                content = line[10 if lower.startswith("assistant ") else 10:].strip()
            elif lower.startswith("human:") or lower.startswith("human "):
                role = "user"
                content = line[6:].strip()
            elif lower.startswith("bot:") or lower.startswith("bot "):
                role = "assistant"
                content = line[4:].strip()
            elif lower.startswith("ai:") or lower.startswith("ai "):
                role = "assistant"
                content = line[3:].strip()
            elif lower.startswith("system:"):
                continue
            else:
                # Continuation of previous message
                if current:
                    current[-1]["content"] += " " + line
                continue
            current.append({"role": role, "content": content})
    if current:
        convos.append({"messages": current})
    return convos


def process_research_evolver(path: Path) -> list[dict]:
    """Load research_evolver_data.txt."""
    return process_txt_conversations(path)


def process_chatgpt_export(path: Path) -> list[dict]:
    """Load ChatGPT export format — blocks separated by ---."""
    convos = []
    current = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if line == "---":
                if current:
                    convos.append({"messages": current})
                    current = []
                continue
            lower = line.lower()
            if lower.startswith("user:") or lower.startswith("user "):
                role = "user"
                content = line[5:].strip()
            elif lower.startswith("assistant:") or lower.startswith("assistant "):
                role = "assistant"
                content = line[10:].strip()
            elif lower.startswith("human:") or lower.startswith("human "):
                role = "user"
                content = line[6:].strip()
            else:
                if current:
                    current[-1]["content"] += " " + line
                continue
            current.append({"role": role, "content": content})
    if current:
        convos.append({"messages": current})
    return convos


def dedupe_and_filter(convos: list[dict], min_messages: int = 2) -> list[dict]:
    """Remove duplicates and filter short/incorrect conversations."""
    seen = set()
    result = []
    for convo in convos:
        msgs = convo.get("messages", [])
        if len(msgs) < min_messages:
            continue
        roles = [m.get("role") for m in msgs]
        if not all(r in ("user", "assistant") for r in roles):
            continue
        if msgs[0].get("role") != "user":
            continue
        # Dedupe by content hash of first few messages
        content_hash = "|".join(m.get("content", "")[:100] for m in msgs[:4])
        if content_hash in seen:
            continue
        seen.add(content_hash)
        result.append(convo)
    return result


def main():
    all_convos = []

    files = [
        ("personal_training.jsonl", process_personal_jsonl),
        ("research_evolver_data.txt", process_research_evolver),
        ("chatgpt_data.txt", process_chatgpt_export),
        ("chat_data_full.txt", process_txt_conversations),
        ("training_data.txt", process_txt_conversations),
        ("marble_data.txt", process_txt_conversations),
        ("nova_data.txt", process_txt_conversations),
    ]

    for fname, processor in files:
        path = INPUT_DIR / fname
        if not path.exists():
            print(f"Skipping {fname} (not found)")
            continue
        print(f"Processing {fname}...")
        try:
            c = processor(path)
            print(f"  Found {len(c)} conversations")
            all_convos.extend(c)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nTotal before dedup: {len(all_convos)}")
    all_convos = dedupe_and_filter(all_convos)
    print(f"Total after dedup + filter: {len(all_convos)}")

    total_messages = sum(len(c["messages"]) for c in all_convos)
    avg_turns = total_messages / len(all_convos) if all_convos else 0
    print(f"Avg turns/conversation: {avg_turns:.1f}")
    print(f"Total messages: {total_messages:,}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for convo in all_convos:
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")

    print(f"\nWritten to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
