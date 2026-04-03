import os
import re
import random
import nbformat
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

def clean_for_rag(text):
    text = re.sub(r'^---[\s\S]*?---\n', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def get_rag_corpus(handbook_dir='./handbook/content/handbook'):
    rag_corpus = {}
    for root, _, files in os.walk(handbook_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                section_name = os.path.relpath(file_path, handbook_dir)
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                clean_text = clean_for_rag(raw_content)
                if len(clean_text) > 200:
                    rag_corpus[section_name] = clean_text
    return rag_corpus

def generate_eval_qa(corpus, n=15):
    keys = random.sample(list(corpus.keys()), min(n, len(corpus)))
    qa_pairs = []
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are generating a test dataset. Given the following document text, generate exactly one factual query whose answer is explicitly in the text. Output ONLY the query, nothing else."),
        ("human", "{text}")
    ])
    
    print("Generating QA Evaluation Pairs using LLM...")
    for key in tqdm(keys):
        text = corpus[key][:1500]
        chain = prompt | llm
        try:
            q = chain.invoke({"text": text}).content
            qa_pairs.append({"question": q, "ground_truth_doc": key})
        except Exception as e:
            continue
    return qa_pairs, keys

def run_iteration(corpus, qa_pairs, doc_keys, chunk_size, chunk_overlap, k):
    print(f"\n--- Running Iteration: Chunk Size={chunk_size}, Overlap={chunk_overlap}, K={k} ---")
    
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for key in doc_keys:
        chunks = splitter.split_text(corpus[key])
        for chunk in chunks:
            docs.append(Document(page_content=chunk, metadata={"source": key}))
            
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    
    hits = 0
    mrr_sum = 0
    
    for qa in qa_pairs:
        query = qa["question"]
        truth = qa["ground_truth_doc"]
        
        results = retriever.invoke(query)
        retrieved_sources = [doc.metadata.get("source", "") for doc in results]
        
        if truth in retrieved_sources:
            hits += 1
            rank = retrieved_sources.index(truth) + 1
            mrr_sum += 1.0 / rank
            
    hit_rate = hits / len(qa_pairs) if qa_pairs else 0
    mrr = mrr_sum / len(qa_pairs) if qa_pairs else 0
    
    vectorstore.delete_collection() # clean up memory
    
    metric = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "k": k,
        "hit_rate": hit_rate,
        "mrr": mrr
    }
    print(f"Results -> Hit Rate: {hit_rate:.2f}, MRR: {mrr:.2f}")
    return metric

def append_to_notebook(best_config, results):
    nb_path = "Data Cleaning.ipynb"
    if not os.path.exists(nb_path):
        print(f"Error: {nb_path} not found.")
        return
        
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    md_content = f"""## Automated RAG Pipeline (Optimized Iterations)

Based on rigorous iterative testing using LLM synthetic query generation across 10 different combinations of `chunk_size` and retrieval `k` limits, the optimal retrieval parameters have been dynamically identified for generating the highest Mean Reciprocal Rank (MRR) and Hit Rate.

**Best Parameters Found:**
- **Chunk Size:** {best_config['chunk_size']} characters
- **Chunk Overlap:** {best_config['chunk_overlap']} characters
- **Top K Retrieval:** {best_config['k']}

### Evaluation Results (10 Iterations)
| Chunk Size | Overlap | K | Hit Rate | MRR |
|---|---|---|---|---|
"""
    for res in results:
        md_content += f"| {res['chunk_size']} | {res['chunk_overlap']} | {res['k']} | {res['hit_rate']:.2f} | {res['mrr']:.2f} |\n"

    markdown_cell = nbformat.v4.new_markdown_cell(source=md_content)
    
    code_source = f"""# ==========================================
# 1. SPLIT DOCUMENTS & INITIALIZE VECTOR STORE
# ==========================================
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import os

print("Split and Chunking Using Optimized Parameters...")
docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size={best_config['chunk_size']}, chunk_overlap={best_config['chunk_overlap']})

# Iterating over the full rag_corpus generated in the earlier cells
for key, text in rag_corpus.items():
    chunks = splitter.split_text(text)
    for chunk in chunks:
        docs.append(Document(page_content=chunk, metadata={{"source": key}}))

print(f"Created {{len(docs)}} chunks from {{len(rag_corpus)}} documents.")

# Initialize Chroma (this builds the vector database on disk)
persist_dir = "./chroma_rag_db"
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

print("Building Vector Store (Chroma)... This will take a few moments for the full dataset.")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
print("Vector Store successfully built and saved!")"""
    code_cell_1 = nbformat.v4.new_code_cell(source=code_source)

    code_source_2 = f"""# ==========================================
# 2. RETRIEVAL & QUESTION ANSWERING
# ==========================================

def retrieve_and_answer(question):
    # Use the best K value from optimization
    retriever = vectorstore.as_retriever(search_kwargs={{"k": {best_config['k']}}})
    retrieved_docs = retriever.invoke(question)
    
    context = "\\n\\n".join([d.page_content for d in retrieved_docs])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant answering corporate compliance or company-related questions based ONLY on the provided context.\\n"
                   "If you do not know the answer based on the context, say 'The context does not provide this information.'\\n\\n"
                   "Context:\\n{{context}}"),
        ("human", "{{question}}")
    ])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm
    
    response = chain.invoke({{"context": context, "question": question}}).content
    
    # Show sources
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))
    
    return response, sources

# --- Example Evaluation Query ---
sample_question = "What are the rules regarding the creation and review of access control policies?"
answer, sources = retrieve_and_answer(sample_question)

print("QUESTION:", sample_question)
print("\\nANSWER:\\n", answer)
print("\\nSOURCES:\\n", "\\n ".join(sources))
"""
    code_cell_2 = nbformat.v4.new_code_cell(source=code_source_2)
    
    nb.cells.extend([markdown_cell, code_cell_1, code_cell_2])
    
    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
        
    print(f"\\n✅ Optimization complete! Successfully appended markdown and code cells to {nb_path}.")

if __name__ == "__main__":
    print("Loading Corpus...")
    corpus = get_rag_corpus()
    
    random.seed(42)
    subset_keys = random.sample(list(corpus.keys()), min(300, len(corpus)))
    subset_corpus = {k: corpus[k] for k in subset_keys}
    print(f"Sampled {len(subset_corpus)} documents for fast parameter optimization.")
    
    qa_pairs, qa_keys = generate_eval_qa(subset_corpus, n=15)
    
    iterations = [
        {"chunk_size": 500, "overlap": 50, "k": 3},
        {"chunk_size": 500, "overlap": 50, "k": 5},
        {"chunk_size": 500, "overlap": 50, "k": 10},
        {"chunk_size": 1000, "overlap": 100, "k": 3},
        {"chunk_size": 1000, "overlap": 100, "k": 5},
        {"chunk_size": 1000, "overlap": 100, "k": 10},
        {"chunk_size": 2000, "overlap": 200, "k": 3},
        {"chunk_size": 2000, "overlap": 200, "k": 5},
        {"chunk_size": 2000, "overlap": 200, "k": 10},
        {"chunk_size": 3000, "overlap": 300, "k": 5},
    ]
    
    results = []
    for i, it in enumerate(iterations):
        print(f"\\n-> Executing Iteration {i+1}/10...")
        try:
            res = run_iteration(subset_corpus, qa_pairs, subset_keys, it["chunk_size"], it["overlap"], it["k"])
            results.append(res)
        except Exception as e:
            print(f"Failed iteration {i}: {e}")
            continue
            
    if not results:
        print("All iterations failed.")
        exit(1)
    
    best_config = sorted(results, key=lambda x: (x["mrr"], x["hit_rate"]), reverse=True)[0]
    print(f"\\n🏆 Optimization Selected Best Config: {best_config}")
    
    append_to_notebook(best_config, results)
