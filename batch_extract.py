import json
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# This uses the API minimally only to embed 50 query strings, bypassing LLM generation
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
persist_dir = "./chroma_rag_db"
vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

with open("eval_dataset.json", "r") as f:
    eval_data = json.load(f)

extracted_contexts = {}

for q in eval_data["audit_questions"]:
    qid = q["id"]
    question = q["question"]
    print(f"Retrieving context for Q{qid}...")
    
    docs = retriever.invoke(question)
    contexts = []
    for d in docs:
        contexts.append({
            "source": d.metadata.get("source", "N/A"),
            "content": d.page_content
        })
    extracted_contexts[qid] = {
        "question": question,
        "contexts": contexts
    }

with open("contexts_dump.json", "w") as f:
    json.dump(extracted_contexts, f, indent=4)

print("Finished extracting contexts.")
