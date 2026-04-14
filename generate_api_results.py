import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# We will use the standalone openai library since it is installed, or we can use langchain.
# Let's use the explicit OpenAI client for maximum control over the output.
client = OpenAI()

def generate_answer(question, context_list):
    context_str = "\n---\n".join([c["content"] for c in context_list])
    
    sys_prompt = (
        "You are an Auditor verifying to which extent GitLab complies with the NIST SP 800-53 Access Control framework.\n"
        "You are provided with a question and several context documents from the GitLab handbook.\n"
        "Your goal is to answer the question strictly based on the provided context.\n"
        "\nRULES:\n"
        "1. Only answers that directly address the question are acceptable.\n"
        "2. For ambiguous answers found, provide the exact ambiguous statement and then effectively answer: 'No documentation directly address the question'.\n"
        "3. If no answer is found within the text at all, state explicitly: 'No information found'.\n"
        "4. DO NOT provide any reasoning in your final output, ONLY provide the answer itself."
    )
    
    user_prompt = f"Contexts:\n{context_str}\n\nQuestion: {question}"

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "No information found"


def main():
    with open("eval_dataset.json", "r") as f:
        eval_data = json.load(f)["audit_questions"]
        
    with open("contexts_dump.json", "r") as f:
        contexts_data = json.load(f)

    results = []
    
    print("Generating answers using gpt-4o-mini...")
    
    for q in eval_data:
        qid = str(q["id"])
        question = q["question"]
        
        contexts = contexts_data.get(qid, {}).get("contexts", [])
        
        answer = generate_answer(question, contexts)
        
        # We try to keep the original structure
        new_q = {
            "id": q["id"],
            "family": q["family"],
            "control_id": q["control_id"],
            "question": question,
            "answer": answer,
            "source": q.get("source", "N/A") # Retain original source or we could extract it
        }
        
        # If no internal contexts or no information found, force N/A source
        if answer in ["No information found", "No documentation directly address the question"]:
            new_q["source"] = "N/A"
            
        print(f"Processed Q{qid} - Answer snippet: {answer[:40]}...")
        results.append(new_q)
        
    with open("native_result.json", "w") as f:
        json.dump({"audit_questions": results}, f, indent=4)
        
    print("Done! Saved to native_result.json")

if __name__ == "__main__":
    main()
