import json

with open('notebook/Data Cleaning.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 19 (agent result evaluation)
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'agent_results_full = evaluate_rag_dataset(' in source:
            suffix_agent = """
cost_full = sum(m.evaluation_cost for t in agent_results_full.test_results for m in t.metrics_data if getattr(m, 'evaluation_cost', None) is not None)
cost_na = sum(m.evaluation_cost for t in results_na_agent.test_results for m in t.metrics_data if getattr(m, 'evaluation_cost', None) is not None)
print(f"Agent Full Evaluation Cost: ${cost_full:.15f}")
print(f"Agent Restricted Evaluation Cost: ${cost_na:.15f}")
print(f"Total Agent Evaluation Cost: ${cost_full + cost_na:.15f}")
"""
            if "cost_full" not in source:
                cell['source'][-1] = cell['source'][-1] + "\n"
                cell['source'].append(suffix_agent)

        if 'results_full = evaluate_rag_dataset(' in source:
            suffix_native = """
cost_full = sum(m.evaluation_cost for t in results_full.test_results for m in t.metrics_data if getattr(m, 'evaluation_cost', None) is not None)
cost_na = sum(m.evaluation_cost for t in results_na.test_results for m in t.metrics_data if getattr(m, 'evaluation_cost', None) is not None)
print(f"Native Full Evaluation Cost: ${cost_full:.15f}")
print(f"Native Restricted Evaluation Cost: ${cost_na:.15f}")
print(f"Total Native Evaluation Cost: ${cost_full + cost_na:.15f}")
"""
            if "cost_full" not in source:
                cell['source'][-1] = cell['source'][-1] + "\n"
                cell['source'].append(suffix_native)

with open('notebook/Data Cleaning.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Updated notebook successfully!")
