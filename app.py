import streamlit as st

st.write("Just trying out streamlit.write")

import streamlit as st
import chromadb

import chromadb
from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE

# Initialize a persistent ChromaDB client
import chromadb
from chromadb.config import Settings

# 2) Pass it under the `settings` argument
client = chromadb.Client()

# 3) (Re)create your collection
collection = client.get_or_create_collection(name="my_collection")
