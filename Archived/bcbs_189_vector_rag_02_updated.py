
import os
import pandas as pd
import re
from typing import List
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import tiktoken

# --- CONFIGURATION ---
TEST_MODE = True

# --- Folder-based File Scanner ---
def scan_folder_for_files(folder_path):
    supported = ('.pdf', '.csv', '.xlsx', '.xls', '.txt', '.doc', '.docx')
    found_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith(supported):
            found_files.append(os.path.join(folder_path, file))
    return found_files

# --- LangChain RAG Setup ---
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

class BaselTestCase(BaseModel):
    test_title: str
    test_case_title: str
    test_case_description: str
    test_steps: List[str]
    test_data: List[str]
    expected_results: List[str]

class BaselTestCaseList(BaseModel):
    test_cases: List[BaselTestCase]

def estimate_tokens(text: str, model_name: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def build_testcase_prompt(rag_answer: str, max_cases: int = 2) -> str:
    return f"""
You are a Basel 3.1 compliance testing expert.
Based on the following document-derived context, generate {max_cases} test cases:

"""{rag_answer}"""

Each test case must include:
- Test title
- Test case title and description
- At least 3 test steps
- Test data
- Expected results

Return only JSON under key 'test_cases'.
"""

def init_rag_chain_from_pdf(pdf_path: str, test_mode: bool = True):
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

    return rag_chain, llm_for_structured, model_name

def generate_test_cases_for_requirement(requirement: str, rag_chain, llm_for_structured, model_name: str):
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
