import json

# Check what the processor is producing for personal_training.jsonl
path = 'build_personal_dataset/personal_training.jsonl'
convos = []
current = []
system_count = 0
assistant_runs = 0
with open(path, encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except:
            continue
        role = obj.get('role', '').lower()
        text = obj.get('text', '').strip()
        if not text:
            continue
        if role == 'system':
            if current:
                convos.append({'messages': current})
                current = []
            continue
        if role not in ('user', 'assistant'):
            continue
        if current and role == 'assistant':
            current[-1]['content'] += '\n' + text
            assistant_runs += 1
        else:
            current.append({'role': role, 'content': text})

if current:
    convos.append({'messages': current})

print(f'Found {len(convos)} conversations')
print(f'Assistant merge events: {assistant_runs}')

# Show first 3 convos
for i, c in enumerate(convos[:3]):
    msgs = c['messages']
    print(f'\nConvo {i}: {len(msgs)} messages')
    for j, m in enumerate(msgs[:6]):
        print(f'  {j} {m["role"]}: {m["content"][:80]}')
