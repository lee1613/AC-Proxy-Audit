import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig

load_dotenv()

def build_deepeval_metrics(model_name="gpt-5.4-mini"):
    return [
        FaithfulnessMetric(threshold=0.7, model=model_name, async_mode=False),
        AnswerRelevancyMetric(threshold=0.7, model=model_name, async_mode=False),
        ContextualPrecisionMetric(threshold=0.7, model=model_name, async_mode=False),
        ContextualRecallMetric(threshold=0.7, model=model_name, async_mode=False),
        ContextualRelevancyMetric(threshold=0.7, model=model_name, async_mode=False)
    ]

def evaluate_rag_dataset(dataset_samples, model_name="gpt-5.4-mini"):
    test_cases = []
    for sample in dataset_samples:
        test_case = LLMTestCase(
            input=sample["input"],
            actual_output=sample["actual_output"],
            expected_output=sample.get("expected_output", ""),
            retrieval_context=sample.get("retrieval_context", [])
        )
        test_cases.append(test_case)
    metrics = build_deepeval_metrics(model_name)
    results = evaluate(test_cases, metrics=metrics, async_config=AsyncConfig(run_async=True, max_concurrent=3))
    return results

def generate_visualizations(df, prefix, dest_dir):
    colors = ['skyblue', 'orange', 'green', 'red', 'purple']
    # Bar charts
    for idx, col in enumerate(df.columns):
        plt.figure(figsize=(14, 6))
        x_indices = range(1, len(df) + 1)
        plt.bar(x_indices, df[col], color=colors[idx % len(colors)])
        plt.title(f'{col} Scores ({prefix})')
        plt.xlabel('Dataset Index')
        plt.ylabel('Score')
        plt.xticks(x_indices, rotation=90)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        clean_col_name = col.replace(" ", "_").lower()
        plt.savefig(os.path.join(dest_dir, f'{prefix.lower()}_bar_{clean_col_name}.png'))
        plt.close()

    # Box plot
    plt.figure(figsize=(12, 6))
    df.boxplot()
    plt.title(f"{prefix} Metrics Distribution")
    plt.ylabel('Score')
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(dest_dir, f'{prefix.lower()}_boxplot.png'))
    plt.close()

def parse_notebook_baseline():
    with open('notebook/Data Cleaning.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    output_text = ""
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            for out in cell.get('outputs', []):
                if 'text' in out:
                    text = "".join(out['text'])
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

def run_all():
    dest_dir = "/Users/june/Desktop/Project/AC-Proxy-Audit/artifacts"
    os.makedirs(dest_dir, exist_ok=True)

    # 1. Baseline from Notebook
    print("Extracting baseline metrics...")
    df_base = parse_notebook_baseline()
    if not df_base.empty:
        generate_visualizations(df_base, "Baseline", dest_dir)
        print("Baseline visualizations generated.")

    # 2. Reranker Evaluation (Run Fresh)
    print("Running Reranker evaluation...")
    dataset_path = 'data/Evaluation Dataset (Answer) V0.5.csv'
    df_eval = pd.read_csv(dataset_path)
    contexts_path = 'data/retrieved context/reranker_context.json'
    with open(contexts_path, 'r') as f:
        contexts_data = json.load(f)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant answering corporate compliance or company-related questions based ONLY on the provided context.\n"
                   "Context:\n{context}"),
        ("human", "{question}")
    ])
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)
    chain = prompt | llm

    dataset_samples = []
    for idx, row in df_eval.iterrows():
        qid = str(row["Index"])
        retrieval_context = [c["content"] for c in contexts_data[qid]["contexts"]] if qid in contexts_data else []
        actual_output = chain.invoke({"context": "\n\n".join(retrieval_context), "question": row["Question"]}).content
        dataset_samples.append({
            "input": row["Question"],
            "actual_output": actual_output,
            "expected_output": str(row["Expected Answer"]),
            "retrieval_context": retrieval_context
        })

    results = evaluate_rag_dataset(dataset_samples)
    
    # Extract scores from deepeval results
    rerank_data = []
    for test_result in results.test_results:
        metrics = {m.name: m.score for m in test_result.metrics_data}
        rerank_data.append(metrics)
    
    df_rerank = pd.DataFrame(rerank_data)
    generate_visualizations(df_rerank, "Reranker", dest_dir)
    print("Reranker visualizations generated.")

if __name__ == '__main__':
    run_all()
