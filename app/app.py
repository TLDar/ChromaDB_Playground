import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from backend import process_pdf, query_llm, define_db_and_store_embeddings

@st.cache_resource
def load_embedding_model(model):
    return HuggingFaceEmbeddings(
        model_name=model
    )

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

    model_choice = st.selectbox("Choose embedding model", [
        "all-MiniLM-L6-v2",
        "multi-qa-MiniLM-L6-cos-v1",
        "BAAI/bge-base-en-v1.5"
    ])
    
    if st.button("Load Embedding Model"):
        with st.spinner("Loading model..."):
            embedding_model = load_embedding_model(model_choice)
            st.session_state.embedding_model = embedding_model
            st.success("Model loaded.")
     
    embedding_model = st.session_state.get("embedding_model", None)
    
    if "db" not in st.session_state and embedding_model:
        with st.spinner("Processing chunks to define embeddings..."):
            db = define_db_and_store_embeddings(chunks, embedding_model)
            st.session_state.db = db
            st.success("Embeddings defined.")
    
    query = st.text_input("Ask a question about the paper:")
    
    db = st.session_state.get("db", None)

    if db and query:
        with st.spinner("Querying LLM..."):
            response = query_llm(query, db)
            st.write("### Response")
            st.markdown(response)
