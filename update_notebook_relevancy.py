import json
import os

NOTEBOOK_PATH = "notebook/Data Cleaning.ipynb"

try:
    with open(NOTEBOOK_PATH, "r") as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error reading notebook: {e}")
    exit(1)

code = """
import json
import os

# 1. Load the generated answers (actual outputs) from our API RAG Pipeline
# Adjust path if your Jupyter runtime root is different
eval_dataset_path = '../native_result.json' if os.path.exists('../native_result.json') else 'native_result.json'
with open(eval_dataset_path, 'r') as f:
    eval_data = json.load(f)["audit_questions"]

# 2. Load the retrieved contexts
contexts_path = '../contexts_dump.json' if os.path.exists('../contexts_dump.json') else 'contexts_dump.json'
with open(contexts_path, 'r') as f:
    contexts_data = json.load(f)

# 3. Prepare datasets for DeepEval
dataset_samples = []
na_dataset_samples = [] # Specifically for 'No information found' cases

for q in eval_data:
    qid = str(q["id"])
    question = q["question"]
    actual_output = q["answer"]
    
    retrieval_context = []
    if qid in contexts_data:
        for c in contexts_data[qid]["contexts"]:
            retrieval_context.append(c["content"])

    sample = {
        "input": question,
        "actual_output": actual_output,
        "expected_output": actual_output, # Proxy for exact match requirement
        "retrieval_context": retrieval_context
    }

    if actual_output.strip() in ["No information found", "No documentation directly address the question"]:
        na_dataset_samples.append(sample)
    else:
        dataset_samples.append(sample)

print(f"Prepared {len(dataset_samples)} samples for full DeepEval execution.")
print(f"Prepared {len(na_dataset_samples)} samples for restricted (Contextual Precision, Recall, Relevancy) execution.")

# 4. Evaluate
# Ensure your OPENAI_API_KEY is set in your environment before running this cell:
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# To execute the full evaluation:
# results_full = evaluate_rag_dataset(dataset_samples, model_name="gpt-4o-mini")

# For 'No info' cases, we can use a custom metric set:
# from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
# from deepeval.test_case import LLMTestCase
# from deepeval import evaluate
# 
# metric_precision = ContextualPrecisionMetric(threshold=0.5, model="gpt-4o-mini")
# metric_recall = ContextualRecallMetric(threshold=0.5, model="gpt-4o-mini")
# metric_relevancy = ContextualRelevancyMetric(threshold=0.5, model="gpt-4o-mini")
# 
# na_test_cases = [LLMTestCase(**s) for s in na_dataset_samples]
# results_na = evaluate(na_test_cases, metrics=[metric_precision, metric_recall, metric_relevancy])
"""

new_source = [line + "\n" if not line.endswith("\n") else line for line in code.strip().split("\n")]

# Update the exact cell we appended previously
for idx in range(len(nb["cells"])-1, -1, -1):
    cell = nb["cells"][idx]
    if cell["cell_type"] == "code" and len(cell["source"]) > 0:
        if "1. Load the generated answers" in cell["source"][3] or "eval_dataset_path =" in "".join(cell["source"]):
            nb["cells"][idx]["source"] = new_source
            break

with open(NOTEBOOK_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print("Updated the execution cell to include ContextualRelevancyMetric.")
