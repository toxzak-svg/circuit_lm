#!/usr/bin/env python3
"""
Build personal training dataset for circuit_lm from Zach's workspace.
Sources: OpenClaw sessions, ChatGPT exports, git commits -> JSONL.
"""

import argparse
import json
import os
import re
from pathlib import Path


WORKSPACE = Path("C:/Users/Zwmar/.openclaw")
SESSIONS_DIR = WORKSPACE / "agents" / "main" / "sessions"
CHATGPT_DIR = WORKSPACE / "workspace" / "projects" / "mydata" / "chatgpt"
GIT_COMMITS_PATH = WORKSPACE / "workspace" / "projects" / "mydata" / "git_commits.txt"
OUT_DIR = Path("C:/Users/Zwmar/.openclaw/workspace/projects/circuit_lm")


def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_openclaw_sessions(min_len: int = 20) -> list[dict]:
    print(f"Scanning OpenClaw sessions: {SESSIONS_DIR}")
    messages = []

    if not SESSIONS_DIR.exists():
        print(f"  NOT FOUND")
    else:
        for sf in SESSIONS_DIR.glob("*.jsonl"):
            try:
                with open(sf, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if obj.get("type") == "message":
                            msg = obj.get("message", {})
                            role = msg.get("role", "")
                            if role not in ("user", "assistant"):
                                continue

                            content = msg.get("content", [])
                            text = ""
                            if isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text += block.get("text", "")
                            elif isinstance(content, str):
                                text = content

                            text = text.strip()
                            if len(text) >= min_len:
                                messages.append({"role": role, "text": text})
            except (json.JSONDecodeError, IOError):
                pass

    print(f"  -> {len(messages)} messages from OpenClaw")
    return messages


def extract_chatgpt(min_len: int = 20) -> list[dict]:
    print(f"Scanning ChatGPT: {CHATGPT_DIR}")
    messages = []

    if not CHATGPT_DIR.exists():
        print(f"  NOT FOUND")
        return messages

    json_files = sorted(CHATGPT_DIR.glob("conversations-*.json"))
    print(f"  Found {len(json_files)} conversation files")

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8", errors="replace") as f:
                conversations = json.load(f)

            for conv in conversations:
                mapping = conv.get("mapping", {})
                for node_id, node in mapping.items():
                    msg = node.get("message", {})
                    if not msg:
                        continue

                    author_role = msg.get("author", {}).get("role", "")
                    if author_role not in ("user", "assistant"):
                        continue

                    content = msg.get("content", {})
                    text = ""
                    if isinstance(content, dict):
                        parts = content.get("parts", [])
                        text = " ".join(p for p in parts if isinstance(p, str))
                    elif isinstance(content, str):
                        text = content

                    text = clean_text(text)
                    if len(text) >= min_len:
                        messages.append({"role": author_role, "text": text})
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Error {jf.name}: {e}")

    print(f"  -> {len(messages)} messages from ChatGPT")
    return messages


def extract_git_commits(min_len: int = 15) -> list[dict]:
    print(f"Scanning git: {GIT_COMMITS_PATH}")
    messages = []

    if GIT_COMMITS_PATH.exists():
        try:
            with open(GIT_COMMITS_PATH, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if len(line) >= min_len:
                        messages.append({
                            "role": "user",
                            "text": clean_text(f"Write a commit message for: {line}")
                        })
        except Exception as e:
            print(f"  Error: {e}")

    print(f"  -> {len(messages)} commits")
    return messages


def deduplicate(messages: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for m in messages:
        key = (m["role"], m["text"])
        if key not in seen:
            seen.add(key)
            unique.append(m)
    print(f"  Dedup: {len(messages) - len(unique)} removed, {len(unique)} unique")
    return unique


def save(messages: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path} ({len(messages)} msgs, {mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "personal_training.jsonl"))
    args = ap.parse_args()

    out_path = Path(args.out)
    messages = []

    msgs = extract_openclaw_sessions()
    messages.extend(msgs)

    msgs = extract_chatgpt()
    messages.extend(msgs)

    msgs = extract_git_commits()
    messages.extend(msgs)

    print(f"\nBefore dedup: {len(messages)}")
    messages = deduplicate(messages)

    for m in messages:
        m["text"] = clean_text(m["text"])
    messages = [m for m in messages if m["text"]]

    roles = {}
    for m in messages:
        roles[m["role"]] = roles.get(m["role"], 0) + 1
    avg_len = sum(len(m["text"]) for m in messages) / max(len(messages), 1)
    print(f"\nFinal: {len(messages)} messages")
    print(f"Roles: {roles}")
    print(f"Avg length: {avg_len:.0f} chars")

    save(messages, out_path)


if __name__ == "__main__":
    main()


# TODO (rotator): **BPE in hybrid** + **hybrid in CLI** — unblocks larger vocab and easier experimentation.

# TODO (rotator): ~~**BPE in hybrid** + **hybrid in CLI** — unblocks larger vocab and easier experimentation.~~ ✅ (done)