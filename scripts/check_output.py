import json

# Check what the converter produced
path = 'personal_chat_dataset.jsonl'
with open(path, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')

# Show first 10 convos
for i, l in enumerate(lines[:10]):
    convo = json.loads(l)
    msgs = convo['messages']
    print(f'\n{i}: {len(msgs)} msgs')
    for j, m in enumerate(msgs[:6]):
        role = m['role']
        content = m['content'][:80]
        print(f'  {j} {role}: {content}')
