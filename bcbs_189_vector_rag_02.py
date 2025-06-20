import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
import pandas as pd
import re
import tiktoken  # ✅ Token estimator

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION ---
TEST_MODE = True
load_dotenv()

pdf_path = r"C:\Users\Arnav\Documents\Python\langchain-models\AJ_GenAI_solution\AJ_GenAI_solution\bcbs189.pdf"
csv_path = r"C:\Users\Arnav\Documents\Python\langchain-models\AJ_GenAI_solution\AJ_GenAI_solution\basel_requirements.csv"
output_path = r"C:\Users\Arnav\Documents\Python\langchain-models\AJ_GenAI_solution\AJ_GenAI_solution\basel_testcases_from_bcbs189.csv"

# --- Load and Truncate PDF ---
loader = PyPDFLoader(pdf_path)
documents = loader.load()

if TEST_MODE:
    documents = documents[:2]
    for doc in documents:
        doc.page_content = doc.page_content[:1000]

# --- Split and Vectorize ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# --- LangChain RAG Chain ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = PromptTemplate(
    template="""
You are a Basel III regulatory assistant.
Answer ONLY using the context below.
If context is insufficient, say "Insufficient context."

{context}
Question: {question}
""",
    input_variables=["context", "question"]
)

model_name = "gpt-3.5-turbo" if TEST_MODE else "gpt-4"

rag_chain = (
    RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    | rag_prompt
    | ChatOpenAI(model=model_name, temperature=0.2, max_tokens=512)
    | StrOutputParser()
)

# --- Structured Output Model ---
class BaselTestCase(BaseModel):
    test_title: str
    test_case_title: str
    test_case_description: str
    test_steps: List[str]
    test_data: List[str]
    expected_results: List[str]

class BaselTestCaseList(BaseModel):
    test_cases: List[BaselTestCase]

llm_for_structured = ChatOpenAI(
    model=model_name,
    temperature=0,
    max_tokens=1024
).with_structured_output(BaselTestCaseList, method="function_calling")

def build_testcase_prompt(rag_answer: str, max_cases: int = 2) -> str:
    return f"""
You are a Basel 3.1 compliance testing expert.
Based on the following document-derived context, generate {max_cases} test cases:

\"\"\"{rag_answer}\"\"\"

Each test case must include:
- Test title
- Test case title and description
- At least 3 test steps
- Test data
- Expected results

Return only JSON under key 'test_cases'.
"""

# --- Token Estimation ---
def estimate_tokens(text: str, model_name: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# --- Load Requirements ---
requirements = None
if os.path.exists(csv_path):
    df_req = pd.read_csv(csv_path)
    if 'requirement' in df_req.columns and not df_req['requirement'].dropna().empty:
        requirements = df_req['requirement'].dropna().tolist()

if not requirements:
    requirements = [
        "Total RWAs (post-floor) ≥ 72.5% of RWAs calculated under standardized approaches"
        "Risk weights for real estate exposures must be assigned based on LTV thresholds: <=80% = 35%, >80% = 75%.",
        "Retail SME exposures must be < EUR 1 million and part of a diversified portfolio to qualify for 75% RW."
    ]

if TEST_MODE:
    requirements = requirements[:1]

# --- Main Execution ---
all_rows = []
total_estimated_tokens = 0

for req in requirements:
    print(f"\n🔍 Requirement: {req}")

    try:
        rag_answer = rag_chain.invoke(req)
        print(f"📘 RAG Answer Preview: {rag_answer[:300]}...")
    except Exception as e:
        print(f"❌ RAG Chain Error: {e}")
        continue

    try:
        test_prompt = build_testcase_prompt(rag_answer)
        structured_result = llm_for_structured.invoke(test_prompt)
    except Exception as e:
        print(f"❌ Test Case Generation Error: {e}")
        continue

    for tc in structured_result.test_cases:
        d = tc.dict()
        max_len = max(len(d['test_steps']), len(d['test_data']), len(d['expected_results']))
        for i in range(max_len):
            all_rows.append({
                'requirement': req,
                'test_title': d['test_title'],
                'test_case_title': d['test_case_title'],
                'test_case_description': d['test_case_description'],
                'step_number': f"Step {i+1}",
                'test_step': re.sub(r"^Step\s*\d+\s*:\s*", "", d['test_steps'][i]) if i < len(d['test_steps']) else "",
                'test_data': d['test_data'][i] if i < len(d['test_data']) else "",
                'expected_result': d['expected_results'][i] if i < len(d['expected_results']) else ""
            })

    # --- Estimate token usage
    rag_token_est = estimate_tokens(rag_answer, model_name)
    struct_token_est = estimate_tokens(test_prompt, model_name)
    total_estimated_tokens += (rag_token_est + struct_token_est)

    print(f"🔢 Estimated tokens: RAG={rag_token_est}, Prompt={struct_token_est}, Total={rag_token_est + struct_token_est}")

# --- Save and Preview Output ---
df = pd.DataFrame(all_rows)
df.to_csv(output_path, index=False)

print("\n✅ Test case generation complete.")
print(f"📄 Output file saved: {output_path}")
print(f"📊 Total estimated tokens used: {total_estimated_tokens}")
print("\n📋 Sample Output Preview:")
print(df.head())

def generate_test_cases_for_requirement(requirement: str):
    try:
        rag_answer = rag_chain.invoke(requirement)
        test_prompt = build_testcase_prompt(rag_answer)
        structured_result = llm_for_structured.invoke(test_prompt)

        result_rows = []
        for tc in structured_result.test_cases:
            d = tc.dict()
            max_len = max(len(d['test_steps']), len(d['test_data']), len(d['expected_results']))
            for i in range(max_len):
                result_rows.append({
                    'requirement': requirement,
                    'test_title': d['test_title'],
                    'test_case_title': d['test_case_title'],
                    'test_case_description': d['test_case_description'],
                    'step_number': f"Step {i+1}",
                    'test_step': re.sub(r"^Step\s*\d+\s*:\s*", "", d['test_steps'][i]) if i < len(d['test_steps']) else "",
                    'test_data': d['test_data'][i] if i < len(d['test_data']) else "",
                    'expected_result': d['expected_results'][i] if i < len(d['expected_results']) else ""
                })
        return pd.DataFrame(result_rows)
    except Exception as e:
        return f"Error: {e}"

def init_rag_chain(test_mode=False):
    global rag_chain, llm_for_structured, model_name

    # Load and truncate PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    if test_mode:
        documents = documents[:2]
        for doc in documents:
            doc.page_content = doc.page_content[:1000]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    model_name = "gpt-3.5-turbo" if test_mode else "gpt-4"

    rag_chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | rag_prompt
        | ChatOpenAI(model=model_name, temperature=0.2, max_tokens=512)
        | StrOutputParser()
    )

    llm_for_structured = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=1024
    ).with_structured_output(BaselTestCaseList, method="function_calling")

    return model_name  # for use in token estimation


def generate_test_cases_for_requirement(requirement: str):
    try:
        rag_answer = rag_chain.invoke(requirement)
        test_prompt = build_testcase_prompt(rag_answer)
        structured_result = llm_for_structured.invoke(test_prompt)

        rag_tokens = estimate_tokens(rag_answer, model_name)
        prompt_tokens = estimate_tokens(test_prompt, model_name)

        result_rows = []
        for tc in structured_result.test_cases:
            d = tc.dict()
            max_len = max(len(d['test_steps']), len(d['test_data']), len(d['expected_results']))
            for i in range(max_len):
                result_rows.append({
                    'requirement': requirement,
                    'test_title': d['test_title'],
                    'test_case_title': d['test_case_title'],
                    'test_case_description': d['test_case_description'],
                    'step_number': f"Step {i+1}",
                    'test_step': re.sub(r"^Step\s*\d+\s*:\s*", "", d['test_steps'][i]) if i < len(d['test_steps']) else "",
                    'test_data': d['test_data'][i] if i < len(d['test_data']) else "",
                    'expected_result': d['expected_results'][i] if i < len(d['expected_results']) else ""
                })

        return pd.DataFrame(result_rows), rag_tokens, prompt_tokens
    except Exception as e:
        return f"Error: {e}", 0, 0
