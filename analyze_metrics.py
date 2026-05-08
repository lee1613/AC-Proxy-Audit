import json
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_notebook():
    with open('notebook/Data Cleaning.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    output_text = ""
    # find cell 23 (or the one with "Prepared 50 samples for full DeepEval execution.")
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            outputs = cell.get('outputs', [])
            for out in outputs:
                if 'text' in out:
                    text = "".join(out['text'])
                    if "Metrics Summary" in text:
                        output_text += text

    # Parse metrics
    test_cases = output_text.split("======================================================================")
    
    data = []
    
    for case in test_cases:
        if "Metrics Summary" not in case:
            continue
            
        metrics = {}
        # Regex to find metrics
        matches = re.finditer(r'- [✅❌] ([\w\s]+) \(score: ([\d.]+),', case)
        for m in matches:
            metric_name = m.group(1).strip()
            score = float(m.group(2))
            metrics[metric_name] = score
            
        if metrics:
            data.append(metrics)
            
    df = pd.DataFrame(data)
    print(f"Extracted data for {len(df)} test cases.")
    print("Summary Statistics:")
    print(df.describe())
    
    # Check for odd phenomenon
    print("\nValue counts for each metric:")
    for col in df.columns:
        print(f"\n{col}:")
        print(df[col].value_counts().sort_index())

    # Generate 5 bar charts, one for each metric
    artifact_dir = "/Users/june/.gemini/antigravity/brain/e3d60365-caf6-4c24-a5ca-65065cf04e4e/artifacts"
    import os
    os.makedirs(artifact_dir, exist_ok=True)
    
    colors = ['skyblue', 'orange', 'green', 'red', 'purple']
    
    for idx, col in enumerate(df.columns):
        plt.figure(figsize=(14, 6))
        # Plot each index (1-50) for the current metric
        x_indices = range(1, len(df) + 1)
        plt.bar(x_indices, df[col], color=colors[idx % len(colors)])
        
        plt.title(f'{col} Scores across 50 Datasets')
        plt.xlabel('Dataset Index')
        plt.ylabel('Score')
        plt.xticks(x_indices, rotation=90)
        plt.ylim(0, 1.05)  # scores are 0 to 1
        
        plt.tight_layout()
        
        # Save the plot
        clean_col_name = col.replace(" ", "_").lower()
        plot_path = os.path.join(artifact_dir, f'evaluation_metrics_{clean_col_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Bar chart for {col} saved to {plot_path}")

if __name__ == '__main__':
    parse_notebook()
