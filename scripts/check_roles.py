import json
import sys

with open('build_personal_dataset/personal_training.jsonl', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

with open('tmp_roles_output.txt', 'w', encoding='utf-8', errors='replace') as out:
    out.write(f'Total lines: {len(lines)}\n')
    for i, l in enumerate(lines[:40]):
        obj = json.loads(l)
        role = obj.get('role', '')
        text = obj.get('text', '')[:100]
        out.write(f'{i} {role}: {text}\n')

print('Done, written to tmp_roles_output.txt')
