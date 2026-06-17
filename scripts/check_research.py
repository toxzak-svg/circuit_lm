import json

# Check research_evolver_data.txt format
path = 'build_personal_dataset/research_evolver_data.txt'
with open(path, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print(f'Total lines: {len(lines)}')
for i, l in enumerate(lines[:30]):
    print(repr(l[:100]))
