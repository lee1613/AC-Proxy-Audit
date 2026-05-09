import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_notebook():
    with open('notebook/Data Cleaning.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    output_text = ""
    # find original eval output (cell 26 usually)
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                if 'text' in out:
                    text = "".join(out['text'])
                    # Check for original eval (it doesn't have "Reranker Pipeline" prefix)
                    if "Metrics Summary" in text and "Reranker Pipeline" not in text:
                        output_text += text

    test_cases = output_text.split("======================================================================")
    data = []
    for case in test_cases:
        if "Metrics Summary" not in case: continue
        metrics = {}
        matches = re.finditer(r'- [✅❌] ([\w\s]+) \(score: ([\d.]+),', case)
        for m in matches:
            metrics[m.group(1).strip()] = float(m.group(2))
        if metrics: data.append(metrics)
    return pd.DataFrame(data)

def parse_reranker_log():
    # Since I deleted the log file in the previous turn, I'll have to re-extract it from the notebook
    # because I appended the reranker eval to the notebook too!
    with open('notebook/Data Cleaning.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    output_text = ""
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                if 'text' in out:
                    text = "".join(out['text'])
                    if "Metrics Summary" in text and "Reranker Pipeline" in text:
                        output_text += text

    test_cases = output_text.split("======================================================================")
    data = []
    for case in test_cases:
        if "Metrics Summary" not in case: continue
        metrics = {}
        matches = re.finditer(r'- [✅❌] ([\w\s]+) \(score: ([\d.]+),', case)
        for m in matches:
            metrics[m.group(1).strip()] = float(m.group(2))
        if metrics: data.append(metrics)
    return pd.DataFrame(data)

def generate_box_plot(df, title, save_path):
    plt.figure(figsize=(12, 6))
    df.boxplot()
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Box plot saved to {save_path}")

if __name__ == '__main__':
    dest_dir = "/Users/june/Desktop/Project/AC-Proxy-Audit/artifacts"
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Baseline
    df_base = parse_notebook()
    if not df_base.empty:
        generate_box_plot(df_base, "Baseline RAG Metrics Distribution", os.path.join(dest_dir, "baseline_metrics_boxplot.png"))
    
    # 2. Reranker
    # Note: If the notebook hasn't been run yet by the user, this might be empty.
    # But I ran it as a script earlier and deleted the log. 
    # Let me check if the notebook has the output now.
    df_rerank = parse_reranker_log()
    if not df_rerank.empty:
        generate_box_plot(df_rerank, "Reranker RAG Metrics Distribution", os.path.join(dest_dir, "reranker_metrics_boxplot.png"))
    else:
        print("Reranker data not found in notebook. Did you run the cell yet?")
