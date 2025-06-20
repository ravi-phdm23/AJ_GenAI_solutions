
import streamlit as st
import os
from datetime import datetime
from fpdf import FPDF
from bcbs_189_vector_rag_02_updated import scan_folder_for_files, init_rag_chain_from_pdf, generate_test_cases_for_requirement

st.set_page_config(page_title="Basel 3.1 PDF Processor", layout="centered")
st.title("📘 Basel 3.1 Test Case Generator (from PDF)")

# --- Folder Path Input ---
folder_path = st.text_input("📁 Enter folder path containing PDFs")

# --- Consent Section ---
consent_given = st.checkbox("✅ I confirm these documents are reviewed and approved for test case generation.")

# --- Save Consent to PDF ---
def save_consent_pdf(folder_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Document Consent Log", ln=1, align="C")
    pdf.ln(10)
    text = (
        "The user has confirmed that the files in the folder have been reviewed "
        "and are approved for generating test cases.

"
        f"Folder Path: {folder_path}
"
        f"Timestamp: {timestamp}"
    )
    pdf.multi_cell(0, 10, text)
    pdf_path = "llm_consent_log.pdf"
    pdf.output(pdf_path)
    return pdf_path

# --- Main Execution ---
if folder_path and consent_given:
    if os.path.isdir(folder_path):
        st.success("✅ Folder found. Ready to process.")
        if st.button("🚀 Generate Test Cases from PDFs"):
            try:
                pdf_path = save_consent_pdf(folder_path)
                st.session_state["pdf_path"] = pdf_path

                files = scan_folder_for_files(folder_path)
                pdf_files = [f for f in files if f.lower().endswith(".pdf")]
                st.info(f"{len(pdf_files)} PDF files found.")

                for file in pdf_files:
                    st.write(f"📄 {os.path.basename(file)}")

                if pdf_files:
                    rag_chain, llm_structured, model = init_rag_chain_from_pdf(pdf_files[0], test_mode=True)
                    sample_req = "Risk weights for real estate exposures must be assigned based on LTV thresholds."
                    df, _, _ = generate_test_cases_for_requirement(sample_req, rag_chain, llm_structured, model)
                    st.subheader("📋 Sample Test Cases from PDF")
                    st.dataframe(df)
                else:
                    st.warning("No PDF files found to process.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.error("❌ Invalid folder path.")

# --- PDF Download Button ---
if "pdf_path" in st.session_state and os.path.exists(st.session_state["pdf_path"]):
    with open(st.session_state["pdf_path"], "rb") as f:
        st.download_button("📥 Download Consent Log (PDF)", f, file_name="llm_consent_log.pdf")
