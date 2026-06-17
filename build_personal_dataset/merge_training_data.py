"""
Merge all personal conversational data into one deduplicated training file.
Output: circuit_lm/all_personal_training.txt
"""
import os

DATA_DIR = "C:/Users/Zwmar/.openclaw/workspace/projects/circuit_lm"
OUT_FILE = os.path.join(DATA_DIR, "all_personal_training.txt")

FILES = [
    ("chatgpt_data.txt", "ChatGPT conversations"),
    ("marble_data.txt", "Marble/Zach conversations"),
    ("research_evolver_data.txt", "Research Evolver combined (has overlap with above)"),
]

print("Loading all files...")
all_lines = []
for fname, desc in FILES:
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {fname}")
        continue
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    print(f"  {fname}: {len(lines)} lines ({os.path.getsize(path)/1e6:.1f} MB) — {desc}")
    all_lines.extend(lines)

print(f"\nTotal lines (with duplicates): {len(all_lines)}")

# Deduplicate by first 100 chars (good enough for conversation turns)
print("Deduplicating...")
seen = set()
unique = []
for line in all_lines:
    key = line[:100].strip()
    if key and key not in seen:
        seen.add(key)
        unique.append(line)

print(f"Unique lines: {len(unique)}")
print(f"Removed: {len(all_lines) - len(unique)} duplicates")

# Write output
with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.writelines(unique)

size_mb = os.path.getsize(OUT_FILE) / 1e6
print(f"\nWritten: {OUT_FILE}")
print(f"Size: {len(unique)} lines, {size_mb:.1f} MB")

if size_mb > 100:
    print("WARNING: file > 100 MB, may be too large for some uses")
