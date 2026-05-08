import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
import time

# Initialize LLM
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

# Create Prompt
system_prompt = """You are an expert technical editor. Your job is to rewrite complex sentences into simple, single-fact atomic sentences.
You will be provided with an "Expected Answer" from an evaluation dataset. 

RULES:
1. Break down the answer into extremely short, simple sentences containing exactly ONE fact each.
2. If the sentence contains a list of items (e.g. "Team X contains A, B, and C"), split it so each item gets its own sentence (e.g. "Team X contains A. Team X contains B. Team X contains C.").
3. Do NOT change the meaning or drop any factual information. Do NOT hallucinate new information.
4. If the Expected Answer is indicating that there is no information (e.g., "The handbook does not publicly disclose...", "No exact monetary budget is listed..."), ignore the original text entirely and simply return the exact string: "No information found."

Example 1:
Original: "The Database Excellence Stage is composed of the Database Architecture Team, Database Automation Team, and Database Health Team."
Output: "The Database Excellence Stage includes the Database Architecture Team. The Database Excellence Stage includes the Database Automation Team. The Database Excellence Stage includes the Database Health Team."

Example 2:
Original: "No exact monetary budget for SEO campaigns is publicly listed in the handbook. However, the handbook does detail the overall SEO strategy framework."
Output: "No information found."

Example 3:
Original: "A key strategy outlined for CSMs is Helping Customers Decide to Migrate to SaaS."
Output: "A key strategy outlined for CSMs is Helping Customers Decide to Migrate to SaaS."
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "Original Expected Answer: {expected_answer}")
])

chain = prompt | llm

def process_dataset():
    input_path = "data/Evaluation Dataset (Answer) V0.5.csv"
    output_path = "data/Evaluation Dataset (Answer) V0.6_Atomic.csv"
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Processing {len(df)} rows...")
    new_answers = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        expected_answer = str(row["Expected Answer"])
        
        # We can handle empty or extremely short ones, but LLM is robust
        if pd.isna(row["Expected Answer"]):
            new_answers.append("No information found.")
            continue
            
        try:
            result = chain.invoke({"expected_answer": expected_answer}).content
            # Clean up the output if it has newlines or bullet points
            result = result.replace('\n', ' ').strip()
            # If the LLM outputted bullet points, replace them with spaces
            result = result.replace('- ', '').replace('* ', '')
            # Clean up double spaces
            while '  ' in result:
                result = result.replace('  ', ' ')
            new_answers.append(result)
            
            # Simple rate limiting protection
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            new_answers.append(expected_answer) # fallback
            
    df["Expected Answer"] = new_answers
    
    df.to_csv(output_path, index=False)
    print(f"Saved atomic dataset to {output_path}")

if __name__ == "__main__":
    process_dataset()
