import os
import re

MEMORY_DIR = 'C:/Users/Zwmar/.openclaw/workspace/memory'
OUTPUT_FILE = 'C:/Users/Zwmar/.openclaw/workspace/projects/circuit_lm/marble_conversations.txt'
STAR_CHAT_DIR = 'C:/Users/Zwmar/.openclaw/workspace/projects/star/data/personal_chats'

def read_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='cp1252') as f:
            return f.read()

def extract_from_md(filepath):
    content = read_file(filepath)
    turns = []
    
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

def extract_star_chat(filepath):
    content = read_file(filepath)
    turns = []
    
    star_pattern = re.compile(r'^(Zach|Marble|Star|★)\s*\xb7\s*(.+?)$', re.MULTILINE)
    for match in star_pattern.finditer(content):
        role, rest = match.groups()
        parts = rest.strip().split('\n', 1)
        if len(parts) == 2:
            msg = parts[1].strip()
        else:
            continue
        if len(msg) > 5:
            role_map = {'Zach': 'user', 'Star': 'assistant', 'Marble': 'assistant', '\u2606': 'assistant'}
            role_clean = role_map.get(role, 'user')
            turns.append((role_clean, msg))
    
    return turns

def clean_text(text):
    text = re.sub(r'\[TOOL_CALL\].*?\[/TOOL_CALL\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\[\[reply_to_current\]\]', '', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

all_turns = []

for fname in sorted(os.listdir(MEMORY_DIR)):
    if fname.startswith('2026-04') and fname.endswith('.md'):
        fpath = os.path.join(MEMORY_DIR, fname)
        turns = extract_from_md(fpath)
        if turns:
            print(f'{fname}: {len(turns)} turns')
            all_turns.extend(turns)

for fname in os.listdir(STAR_CHAT_DIR):
    fpath = os.path.join(STAR_CHAT_DIR, fname)
    turns = extract_star_chat(fpath)
    if turns:
        print(f'star chat {fname}: {len(turns)} turns')
        all_turns.extend(turns)

print(f'\nTotal raw turns: {len(all_turns)}')

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

print(f'After dedup + filter: {len(cleaned)} turns')

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for role, text in cleaned:
        f.write(f'{role}: {text}\n')

print(f'\nWritten to: {OUTPUT_FILE}')
user_count = sum(1 for r, _ in cleaned if r == 'user')
asst_count = sum(1 for r, _ in cleaned if r == 'assistant')
print(f'User turns: {user_count}')
print(f'Assistant turns: {asst_count}')
total_chars = sum(len(t) for _, t in cleaned)
print(f'Total chars: {total_chars:,}')
print(f'Total KB: {total_chars / 1024:.1f}')
