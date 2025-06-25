from turtle import st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
import requests
import pandas as pd
import re
#from openai import OpenAI  # Replace with actual API if different

API_URL_MISTRAL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
API_KEY = os.getenv("API_KEY")
headers = {
    "Authorization": f"Bearer {API_KEY}"
}

def process_pdf(pdf_file):
    loader = PyMuPDFLoader(pdf_file)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    return chunks

def define_db_and_store_embeddings(chunks, embedder):
    rag_collection = "rag_collection"
    
    db = Chroma.from_documents(
        chunks,
        embedding=embedder,
        persist_directory="./chroma_db",
        collection_name=rag_collection
    )
    
    return db

def define_context(query, db):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    relevant_chunks = retriever.invoke(query) # looking for the top 5 answers (ie with minimal distance to the query)
    context = "\n".join([doc.page_content for doc in relevant_chunks])
    return context

def query_llm(query, db):
    context = define_context(query, db)
    prompt = f"Answer the question based on this context:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = call_llm_api_mistral(prompt)

    if response is not None:
        # Regular expression pattern to extract answers
        pattern = r"Answer:\s*(.*)"
        # Find all matches
        answer = re.findall(pattern, response)
        return answer[0]
    else :
        return "There has been an error while querying the LLM."

def call_llm_api_mistral(prompt):
    response = requests.post(API_URL_MISTRAL, headers=headers, json={"inputs": prompt})
    try:
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif "error" in data:
            print("API Error:", data["error"])
            return None
        else:
            print("Unexpected response format:", data)
            return None
    except Exception as e:
        print("Exception during parsing:", e)
        print("Response text:", response.text)
        return None
    
