import json
with open('kaggle_run.ipynb', 'r') as f:
    nb = json.load(f)
for cell in nb['cells']:
    new_source = []
    for line in cell['source']:
        if 'entity=' in line:
            line = line.replace('entity="zacharymaronek",\n', '')
        new_source.append(line)
    cell['source'] = new_source
with open('kaggle_run.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print('Done - entity removed')
