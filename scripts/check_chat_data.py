import json

# Check chat_data_full.txt
with open('build_personal_dataset/chat_data_full.txt', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

print('Total lines:', len(lines))
for i, l in enumerate(lines[:20]):
    print(repr(l[:100]))
