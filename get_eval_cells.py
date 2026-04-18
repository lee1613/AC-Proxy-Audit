import json

with open('notebook/Data Cleaning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'agent_results_full =' in source or 'results_full =' in source:
            print(f"--- CELL {i} ---")
            print(source)
