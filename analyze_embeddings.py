import json
import numpy as np
from langchain_openai import OpenAIEmbeddings

# Initialize embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def analyze_sets():
    with open("context.json", "r") as f:
        data = json.load(f)
        
    sets = {
        "Set 1 (Typos - Simple)": ["1", "2", "3"],
        "Set 2 (Typos - Procedural)": ["21", "22", "23"],
        "Set 3 (Rephrasing - Complex)": ["36", "37", "38", "39"]
    }
    
    results = {}
    
    for set_name, indices in sets.items():
        print(f"\\n{'='*50}\\nAnalyzing {set_name}\\n{'='*50}")
        
        # 1. Embed questions
        questions = []
        for idx in indices:
            if idx in data:
                questions.append((idx, data[idx]["question"]))
                
        if not questions:
            continue
            
        base_idx, base_q = questions[0]
        base_emb = embeddings_model.embed_query(base_q)
        
        print(f"Base Question ({base_idx}): {base_q}")
        
        # 2. Compare variations
        for idx, q in questions[1:]:
            q_emb = embeddings_model.embed_query(q)
            sim = cosine_similarity(base_emb, q_emb)
            print(f"Variation ({idx}): {q}")
            print(f"  -> Cosine Similarity to Base: {sim:.4f}")
            
        print("\\n--- Context Relevancy ---")
        # 3. Compare Question to its Contexts
        for idx, q in questions:
            contexts = data[idx]["contexts"][:3] # Analyze top 3
            if not contexts:
                continue
                
            q_emb = embeddings_model.embed_query(q)
            
            print(f"Question ({idx}): {q}")
            for i, ctx in enumerate(contexts):
                content = ctx["content"]
                ctx_emb = embeddings_model.embed_query(content)
                sim = cosine_similarity(q_emb, ctx_emb)
                
                # Snippet
                snippet = content.replace('\\n', ' ')[:80] + "..."
                print(f"  Context {i+1} Similarity: {sim:.4f} | Preview: {snippet}")

if __name__ == "__main__":
    analyze_sets()
