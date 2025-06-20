
import streamlit as st
import pandas as pd
import os
from pathlib import Path
from Archived.bcbs_189_vector_rag_mocked_updated import generate_test_cases_for_requirement, init_rag_chain

st.set_page_config(page_title="Basel 3.1 Test Case Generator", layout="wide")

# Custom CSS to handle long text in DataFrame
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

st.markdown("This app uses dummy logic to simulate structured test case generation from Basel 3.1 regulatory requirements.")

# --- Test Mode Toggle ---
test_mode = st.checkbox("🧪 Enable Test Mode (simulated)", value=True)
model_name = init_rag_chain(test_mode)

# --- Input Method ---
input_mode = st.radio("📥 Select Input Method:", ["Manual Entry", "Upload CSV File", "Load from Folder"], horizontal=True)

manual_reqs, csv_reqs, folder_reqs = [], [], []

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

else:
    folder_path = st.text_input("📁 Enter the folder path where files are located:")
    if folder_path and Path(folder_path).exists():
        st.success("✅ Folder found. Reading files...")
        allowed_ext = [".csv", ".xls", ".xlsx", ".txt", ".docx", ".pdf"]
        all_files = [f for f in Path(folder_path).iterdir() if f.suffix.lower() in allowed_ext]
        if not all_files:
            st.warning("⚠️ No supported files found in the folder.")
        else:
            st.markdown("### 📂 Files to be processed:")
            for file in all_files:
                st.markdown(f"- {file.name}")
                try:
                    if file.suffix == ".csv":
                        df = pd.read_csv(file)
                        if 'requirement' in df.columns:
                            folder_reqs.extend(df['requirement'].dropna().tolist())
                    elif file.suffix in [".xls", ".xlsx"]:
                        df = pd.read_excel(file)
                        if 'requirement' in df.columns:
                            folder_reqs.extend(df['requirement'].dropna().tolist())
                    elif file.suffix == ".txt":
                        with open(file, 'r', encoding='utf-8') as f:
                            folder_reqs.extend([line.strip() for line in f if line.strip()])
                    elif file.suffix == ".docx":
                        from docx import Document
                        doc = Document(file)
                        folder_reqs.extend([para.text.strip() for para in doc.paragraphs if para.text.strip()])
                    elif file.suffix == ".pdf":
                        from PyPDF2 import PdfReader
                        reader = PdfReader(file)
                        for page in reader.pages:
                            text = page.extract_text()
                            if text:
                                folder_reqs.extend([line.strip() for line in text.splitlines() if line.strip()])
                except Exception as e:
                    st.warning(f"❌ Failed to process {file.name}: {e}")

# --- Run and Show Results ---
all_reqs = manual_reqs or csv_reqs or folder_reqs
if all_reqs:
    if st.button("🚀 Generate Test Cases"):
        all_results, total_tokens = [], 0
        with st.spinner("Processing each requirement..."):
            for i, req in enumerate(all_reqs, 1):
                st.markdown(f"---\n### 🔍 Requirement {i} of {len(all_reqs)}:")
                st.code(req, language='text')

                result_df, rag_tok, struct_tok = generate_test_cases_for_requirement(req)
                if isinstance(result_df, pd.DataFrame):
                    st.dataframe(result_df, use_container_width=True)
                    all_results.append(result_df)
                    total_tokens += (rag_tok + struct_tok)
                    st.caption(f"🔢 Tokens used for this requirement: Simulated = {rag_tok + struct_tok}")
                else:
                    st.error(result_df)

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            csv = final_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download All Test Cases (CSV)", csv, "basel_test_cases.csv", "text/csv")
            st.success(f"📊 Total Simulated Tokens Used: {total_tokens}")
else:
    st.info("Please enter or load requirements.")
