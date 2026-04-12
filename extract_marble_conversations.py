import os
import re

MEMORY_DIR = "C:/Users/Zwmar/.openclaw/workspace/memory"
OUTPUT_FILE = "C:/Users/Zwmar/.openclaw/workspace/projects/circuit_lm/marble_conversations.txt"
STAR_CHAT_DIR = "C:/Users/Zwmar/.openclaw/workspace/projects/star/data/personal_chats"
PERSONALITY_FILES = [
    "C:/Users/Zwmar/.openclaw/workspace/SOUL.md",
    "C:/Users/Zwmar/.openclaw/workspace/IDENTITY.md",
    "C:/Users/Zwmar/.openclaw/workspace/USER.md",
    "C:/Users/Zwmar/.openclaw/workspace/AGENTS.md",
]

def extract_turns_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    turns = []
    
    # Pattern: raw user/assistant format
    user_pattern = re.compile(r'^user:\s*(.+?)$', re.MULTILINE | re.IGNORECASE)
    assistant_pattern = re.compile(r'^assistant:\s*(.+?)$', re.MULTILINE | re.IGNORECASE)
    
    users = user_pattern.findall(content)
    assistants = assistant_pattern.findall(content)
    
    if users or assistants:
        for u in users:
            u = u.strip()
            if len(u) > 10:
                turns.append(('user', u))
        for a in assistants:
            a = a.strip()
            if len(a) > 10:
                turns.append(('assistant', a))
        return turns
    
    # Pattern: Star-style raw conversations "Name · timestamp" format
    star_pattern = re.compile(r'^(Zach|Marble|Star|★)\s*[·\u2022]\s*(\d{1,2}:\d{2}\s*(AM|PM)?)\s*(.+?)$', re.MULTILINE)
    star_matches = star_pattern.findall(content)
    
    if star_matches:
        for role, time, ampm, msg in star_matches:
            msg = msg.strip()
            if len(msg) > 5:
                role_map = {'Zach': 'user', 'Star': 'assistant', 'Marble': 'assistant', '★': 'assistant'}
                role_clean = role_map.get(role, 'user')
                turns.append((role_clean, msg))
        return turns
    
    return turns

def extract_from_md(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    turns = []
    
    # [Day 2026-04-03 HH:MM EDT] message lines
    line_pattern = re.compile(r'^\[(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+(EDT|UTC)\]\s*(.+?)$', re.MULTILINE)
    
    for match in line_pattern.finditer(content):
        msg = match.group(3).strip()
        if len(msg) > 10 and not msg.startswith('Sender') and not msg.startswith('Conversation'):
            turns.append(('user', msg))
    
    assistant_pattern = re.compile(r'^assistant:\s*(.+?)$', re.MULTILINE | re.IGNORECASE)
    for m in assistant_pattern.finditer(content):
        msg = m.group(1).strip()
        if len(msg) > 10:
            turns.append(('assistant', msg))
    
    return turns

def clean_text(text):
    text = re.sub(r'\[TOOL_CALL\].*?\[/TOOL_CALL\]', '', text, flags=re.DOTALL)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = text.strip()
    return text

def main():
    all_turns = []
    
    # Daily memory files
    for fname in os.listdir(MEMORY_DIR):
        if fname.startswith('2026-04') and fname.endswith('.md'):
            fpath = os.path.join(MEMORY_DIR, fname)
            print(f"Processing {fname}...")
            turns = extract_from_md(fpath)
            print(f"  -> {len(turns)} turns")
            all_turns.extend(turns)
    
    # Star personal chat files
    for fname in os.listdir(STAR_CHAT_DIR):
        fpath = os.path.join(STAR_CHAT_DIR, fname)
        print(f"Processing star chat: {fname}...")
        turns = extract_turns_from_file(fpath)
        print(f"  -> {len(turns)} turns")
        all_turns.extend(turns)
    
    # Workspace personality files
    for fpath in PERSONALITY_FILES:
        if os.path.exists(fpath):
            fname = os.path.basename(fpath)
            print(f"Processing {fname}...")
            turns = extract_turns_from_file(fpath)
            print(f"  -> {len(turns)} turns")
            all_turns.extend(turns)
    
    print(f"\nTotal raw turns: {len(all_turns)}")
    
    # Deduplicate and filter
    seen = set()
    cleaned = []
    for role, text in all_turns:
        text = clean_text(text)
        if len(text) < 15:
            continue
        key = text[:100].lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append((role, text))
    
    print(f"After dedup + filter: {len(cleaned)} turns")
    
    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for role, text in cleaned:
            f.write(f"{role}: {text}\n")
    
    print(f"\nWritten to: {OUTPUT_FILE}")
    
    user_count = sum(1 for r, _ in cleaned if r == 'user')
    asst_count = sum(1 for r, _ in cleaned if r == 'assistant')
    print(f"User turns: {user_count}")
    print(f"Assistant turns: {asst_count}")
    total_chars = sum(len(t) for _, t in cleaned)
    print(f"Total chars: {total_chars:,}")
    print(f"Total KB: {total_chars / 1024:.1f}")

if __name__ == '__main__':
    main()
