import json

# Check chatgpt_data.txt format
path = 'build_personal_dataset/chatgpt_data.txt'
with open(path, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')
for i, l in enumerate(lines[:40]):
    print(repr(l[:100]))
