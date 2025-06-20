
import streamlit as st
import pandas as pd
from bcbs_189_vector_rag_02_updated import generate_test_cases_for_requirement, init_rag_chain

st.set_page_config(page_title="Basel 3.1 Test Case Generator", layout="wide")

st.markdown("""
    <style>
    .dataframe td {
        white-space: pre-wrap;
        word-wrap: break-word;
        vertical-align: top;
    }
    .stDataFrame div[data-testid="stVerticalBlock"] {
        overflow-x: auto;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧪 Basel 3.1 RAG-based Test Case Generator")
st.markdown("This app uses LLM + Vector Search to generate structured test cases from Basel 3.1 regulatory requirements.")

# --- Test Mode Toggle ---
test_mode = st.checkbox("🧪 Enable Test Mode (truncated PDF + GPT-3.5)", value=True)
model_name = None  # Will initialize later after file upload


# --- Step 1: Test Case Requirement Input ---
st.header("📌 Step 1: Enter or Upload Basel Requirements")

input_mode = st.radio("Select Input Method for Requirements:", ["Manual Entry", "Upload CSV File"], horizontal=True)

manual_reqs, csv_reqs = [], []

if input_mode == "Manual Entry":
    multiline_input = st.text_area(
        "✍️ Enter one or more Basel 3.1 requirements (each on a new line):",
        height=200,
        placeholder="e.g.\n1. Real estate exposures must be risk-weighted based on LTV.\n2. Output floor must be applied at 72.5%..."
    )
    if multiline_input.strip():
        manual_reqs = [line.strip() for line in multiline_input.strip().splitlines() if line.strip()]

elif input_mode == "Upload CSV File":
    uploaded_file = st.file_uploader("📄 Upload CSV file with a 'requirement' column:", type=['csv'])
    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        if 'requirement' in df_uploaded.columns:
            csv_reqs = df_uploaded['requirement'].dropna().tolist()
            st.success(f"✅ {len(csv_reqs)} requirements loaded from CSV.")
        else:
            st.error("❌ Uploaded file must contain a column named 'requirement'.")

all_reqs = manual_reqs or csv_reqs

# --- Step 2: Supporting File Upload ---
st.header("📂 Step 2: (Optional) Upload Supporting Files")

folder_reqs = []
uploaded_files = st.file_uploader(
    "📁 Upload files for context extraction (txt, pdf, docx, xlsx):",
    type=["csv", "xls", "xlsx", "txt", "docx", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} files uploaded.")
    model_name = init_rag_chain(test_mode, uploaded_files)  # ✅ Now safe

    for file in uploaded_files:
        st.markdown(f"- {file.name}")
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
                if 'requirement' in df.columns:
                    folder_reqs.extend(df['requirement'].dropna().tolist())
            elif file.name.endswith((".xls", ".xlsx")):
                df = pd.read_excel(file)
                if 'requirement' in df.columns:
                    folder_reqs.extend(df['requirement'].dropna().tolist())
            elif file.name.endswith(".txt"):
                content = file.read().decode("utf-8")
                folder_reqs.extend([line.strip() for line in content.splitlines() if line.strip()])
            elif file.name.endswith(".docx"):
                from docx import Document
                doc = Document(file)
                folder_reqs.extend([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            elif file.name.endswith(".pdf"):
                from PyPDF2 import PdfReader
                reader = PdfReader(file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        folder_reqs.extend([line.strip() for line in text.splitlines() if line.strip()])
        except Exception as e:
            st.warning(f"❌ Failed to process {file.name}: {e}")
else:
    st.info("You can skip this step if you have no supporting files.")

# --- Step 3: Generate Test Cases ---
final_reqs = all_reqs or folder_reqs
if final_reqs:
    st.header("🧪 Step 3: Generate Test Cases")

    if st.button("🚀 Generate Test Cases"):
        all_results, total_tokens = [], 0
        with st.spinner("Processing each requirement..."):
            for i, req in enumerate(final_reqs, 1):
                st.markdown(f"---\n### 🔍 Requirement {i} of {len(final_reqs)}:")
                st.code(req, language='text')

                result_df, rag_tok, struct_tok = generate_test_cases_for_requirement(req)
                if isinstance(result_df, pd.DataFrame):
                    st.dataframe(result_df, use_container_width=True)
                    all_results.append(result_df)
                    total_tokens += (rag_tok + struct_tok)
                    st.caption(f"🔢 Tokens used for this requirement: RAG = {rag_tok}, Prompt = {struct_tok}, Total = {rag_tok + struct_tok}")
                else:
                    st.error(result_df)

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download All Test Cases (CSV)", csv, "basel_test_cases.csv", "text/csv")
            st.success(f"📊 Total Tokens Used: {total_tokens}")
else:
    st.info("Please input Basel requirements to proceed.")
