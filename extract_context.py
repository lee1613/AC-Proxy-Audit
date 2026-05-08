import pandas as pd
import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def main():
    load_dotenv()

    # Setup Vector DB
    print("Loading vector database...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_dir = "./chroma_rag_db"
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # Read dataset
    dataset_path = "data/Evalaution Dataset V0.5.csv"
    print(f"Reading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)

    results = {}

    print("Extracting contexts...")
    for idx, row in df.iterrows():
        question = row["Question"]
        index = str(row["Index"])
        
        retrieved_docs = retriever.invoke(question)
        
        contexts = []
        for d in retrieved_docs:
            contexts.append({
                "source": d.metadata.get("source", "Unknown"),
                "content": d.page_content
            })
            
        results[index] = {
            "question": question,
            "contexts": contexts
        }

    output_path = "context.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Successfully extracted {len(results)} contexts and saved to {output_path}")

if __name__ == "__main__":
    main()
