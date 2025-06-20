
import streamlit as st
import os
from datetime import datetime
from fpdf import FPDF
from Archived.bcbs_189_vector_rag_02_mocked_updated import scan_folder_for_files, generate_test_cases_for_requirement

st.set_page_config(page_title="Test Case Folder Processor", layout="centered")
st.title("📂 Folder-Based Test Case Processor")

# --- Folder Path Input ---
folder_path = st.text_input("📁 Enter path to folder containing input files:")

# --- Consent Section ---
consent_given = st.checkbox("✅ I confirm the documents are reviewed and ready to process.")

# --- Save Consent to PDF ---
def save_consent_pdf(folder_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Document Processing Consent", ln=1, align="C")
    pdf.ln(10)
    text = (
        "The user has confirmed that the uploaded files have been reviewed "
        "and are ready for processing."
        f"Folder Path: {folder_path}"
        f"Timestamp: {timestamp}"
    )
    pdf.multi_cell(0, 10, text)
    pdf_path = "document_processing_consent_log.pdf"
    pdf.output(pdf_path)
    return pdf_path

# --- Execution Logic ---
if folder_path and consent_given:
    if os.path.isdir(folder_path):
        st.success("✅ Folder found. Ready to process.")
        if st.button("🚀 Process Files"):
            try:
                pdf_path = save_consent_pdf(folder_path)
                st.session_state["pdf_path"] = pdf_path

                files = scan_folder_for_files(folder_path)
                st.info(f"{len(files)} supported files found.")
                for file in files:
                    st.write(f"📄 {os.path.basename(file)}")

                # Mock test generation preview (for demonstration)
                st.divider()
                sample_req = "Risk weights for real estate exposures must be assigned based on LTV thresholds."
                df, _, _ = generate_test_cases_for_requirement(sample_req)
                st.subheader("📋 Sample Generated Test Case Table")
                st.dataframe(df)

            except Exception as e:
                st.error(f"❌ Error: {e}")
    else:
        st.error("❌ Invalid folder path.")

# --- PDF Download Button ---
if "pdf_path" in st.session_state and os.path.exists(st.session_state["pdf_path"]):
    with open(st.session_state["pdf_path"], "rb") as f:
        st.download_button("📥 Download Consent PDF", f, file_name="document_processing_consent_log.pdf")
