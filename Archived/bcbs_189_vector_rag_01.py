# Updated integration: Replacing YouTube transcript with PDF vectorization for Basel III test cases

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from typing import List
from pydantic import BaseModel, Field
import pandas as pd
import re

# --- Environment Setup ---
load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Load PDF ---
pdf_path = r"C:\Users\Arnav\Documents\Python\langchain-models\AJ_GenAI_solution\bcbs189.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# --- Chunking ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# --- Embedding & Vector Store ---
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# --- RAG Chain ---
prompt = PromptTemplate(
    template="""
    You are a Basel III regulatory assistant.
    Answer ONLY using the context from the document provided below.
    If the context does not support the answer, say "Insufficient context."

    {context}
    Question: {question}
    """,
    input_variables=['context', 'question']
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

rag_chain = parallel_chain | prompt | ChatOpenAI(model="gpt-4", temperature=0.2) | StrOutputParser()

# --- Basel Test Case Generation ---
class BaselTestCase(BaseModel):
    test_title: str
    test_case_title: str
    test_case_description: str
    test_steps: List[str]
    test_data: List[str]
    expected_results: List[str]

class BaselTestCaseList(BaseModel):
    test_cases: List[BaselTestCase]

structured_model = ChatOpenAI(model="gpt-4", temperature=0).with_structured_output(BaselTestCaseList)

# --- Requirement-Based Prompt ---
def build_testcase_prompt(requirement: str, max_cases: int = 2):
    return f"""
You are a Basel 3.1 compliance testing expert.
Generate {max_cases} test cases for the following regulatory requirement:
\"\"\"{requirement}\"\"\"
Each test case must include:
- Test title
- Test case title and description
- At least 3 test steps
- Test data
- Expected results
Return in JSON under key 'test_cases'. Cover different branches or edge cases.
"""

# --- Requirements (Manual or CSV-driven) ---

# code to read requirements from a CSV file can be added here if needed. in case of manual input, we can define them directly


csv_path = r"C:\Users\Arnav\Documents\Python\langchain-models\AJ_GenAI_solution\basel_requirements.csv"
requirements = None

# Try loading requirements from CSV
if os.path.exists(csv_path):
    df_req = pd.read_csv(csv_path)
    if 'requirement' in df_req.columns and not df_req['requirement'].dropna().empty:
        requirements = df_req['requirement'].dropna().tolist()
        print(f"Loaded {len(requirements)} requirements from CSV.")
# Fallback to manual list
if requirements is None:
    manual_requirements = [
        "Risk weights for real estate exposures must be assigned based on LTV thresholds: <=80% = 35%, >80% = 75%.",
        "Retail SME exposures must be < EUR 1 million and part of a diversified portfolio to qualify for 75% RW."
    ]
    if manual_requirements:
        requirements = manual_requirements
        print(f"Using {len(requirements)} manual requirements.")
# If neither found, raise error
if not requirements:
    raise ValueError("No requirements found: Please provide a CSV file or a manual requirements list.")

all_rows = []
for req in requirements:
    prompt = build_testcase_prompt(req)
    results = structured_model.invoke(prompt)
    for tc in results.test_cases:
        d = tc.dict()
        max_len = max(len(d['test_steps']), len(d['test_data']), len(d['expected_results']))
        for i in range(max_len):
            row = {
                'requirement': req,
                'test_title': d['test_title'],
                'test_case_title': d['test_case_title'],
                'test_case_description': d['test_case_description'],
                'step_number': f"Step {i+1}",
                'test_step': re.sub(r"^Step\s*\d+\s*:\s*", "", d['test_steps'][i]) if i < len(d['test_steps']) else "",
                'test_data': d['test_data'][i] if i < len(d['test_data']) else "",
                'expected_result': d['expected_results'][i] if i < len(d['expected_results']) else ""
            }
            all_rows.append(row)

# --- Export ---
df = pd.DataFrame(all_rows)
print(f"✅ Generated {len(df)} test cases from {len(requirements)} requirements.")
# --- Save to CSV ---
print(df.head())  # Display first few rows for verification
df.to_csv(r"C:\Users\Arnav\Documents\Python\langchain-models\AJ_GenAI_solution\basel_testcases_from_bcbs189.csv", index=False)
print("✅ Test cases written to: basel_testcases_from_bcbs189.csv")
