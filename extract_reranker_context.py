import json
import os
import pandas as pd
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

def extract_contexts():
    # 1. Load the evaluation dataset
    dataset_path = 'data/Evaluation Dataset (Answer) V0.6_Atomic.csv'
    df = pd.read_csv(dataset_path)

    # 2. Setup Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_dir = "chroma_rag_db"
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # 3. Setup BAAI Cross-Encoder for reranking
    print("Loading BAAI/bge-reranker-base cross-encoder...")
    cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    TOP_K_BASE = 20    # Docs fetched from vector DB
    TOP_N_RERANKED = 5 # Final docs after reranking

    # 4. Extract contexts for each question
    print(f"\nExtracting reranked contexts for {len(df)} questions...")
    results = {}

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        qid = str(row["Index"])
        question = row["Question"]

        # Fetch top-20 candidate docs from ChromaDB
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_BASE})
        candidate_docs = base_retriever.invoke(question)

        # Score each (question, passage) pair with the cross-encoder
        pairs = [(question, doc.page_content) for doc in candidate_docs]
        scores = cross_encoder.predict(pairs)

        # Sort by score descending and take top-N
        scored_docs = sorted(zip(scores, candidate_docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[:TOP_N_RERANKED]]

        contexts = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
            }
            for doc in top_docs
        ]

        results[qid] = {
            "question": question,
            "contexts": contexts,
        }

    # 5. Ensure output directory exists and save
    output_dir = 'data/retrieved context'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'reranker_context.json')

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    print(f"\nSuccessfully saved reranked contexts to {output_path}")

if __name__ == "__main__":
    extract_contexts()
