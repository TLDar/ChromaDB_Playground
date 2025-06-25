import tempfile
import streamlit as st
from backend import process_pdf, query_llm

st.title("Research paper RAG assistant")

pdf_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if pdf_file:
    # using tempfile to work with a path rather than a stream
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name
        
    with st.spinner("Extracting and chunking text..."):
        chunks = process_pdf(tmp_path)
        st.success(f"Loaded {len(chunks)} chunks.")

    query = st.text_input("Ask a question about the paper:")

    if query:
        with st.spinner("Querying LLM..."):
            response = query_llm(query, chunks)
            st.write("### Response")
            st.markdown(response)
