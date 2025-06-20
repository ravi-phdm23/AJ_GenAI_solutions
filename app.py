# app.py

import streamlit as st
import pandas as pd
from bcbs_189_vector_rag_02 import generate_test_cases_for_requirement

st.set_page_config(page_title="Basel 3.1 Test Case Generator", layout="wide")
st.title("🧪 Basel 3.1 RAG-based Test Case Generator")

with st.form(key='requirement_form'):
    user_input = st.text_area("📌 Enter a Basel 3.1 Requirement:", height=150,
                              placeholder="e.g., Total RWAs (post-floor) ≥ 72.5% of standardized RWA...")
    submit_button = st.form_submit_button(label='Generate Test Cases')

if submit_button and user_input:
    st.info("🔍 Generating test cases... Please wait.")
    result = generate_test_cases_for_requirement(user_input)

    if isinstance(result, pd.DataFrame):
        st.success("✅ Test cases generated successfully.")
        st.dataframe(result, use_container_width=True)
        csv = result.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "basel_test_cases.csv", "text/csv")
    else:
        st.error(f"❌ {result}")
