import streamlit as st
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Core RAG Components
from src.helper import download_embeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

# 1. PAGE CONFIG & CACHING
st.set_page_config(page_title="Nexus Intelligence System", layout="wide")
load_dotenv()

@st.cache_resource
def get_embeddings():
    return download_embeddings()

# API Key handling
groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

# 2. SESSION STATE
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_db" not in st.session_state: 
    # Attempt to load existing index if it exists
    if os.path.exists("faiss_index"):
        st.session_state.vector_db = FAISS.load_local("faiss_index", get_embeddings(), allow_dangerous_deserialization=True)
    else:
        st.session_state.vector_db = None

# --- STREAMING FUNCTION ---
def get_streaming_response(prompt, current_k, current_temp, mode):
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=groq_key, 
        temperature=current_temp,
        streaming=True # Enable streaming in the LLM
    )
    
    system_instruction = "You are the Nexus Technical Agent. "
    if mode == "Deep Think":
        system_instruction += "Think step-by-step. Analyze all provided context for complex links."

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", f"{system_instruction}\n\nContext:\n{{context}}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": current_k})
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Stream the response
    full_response = ""
    container = st.empty() # Placeholder for the typing effect
    
    # We use stream for the retrieval chain
    for chunk in retrieval_chain.stream({"input": prompt, "chat_history": st.session_state.chat_history}):
        if "answer" in chunk:
            full_response += chunk["answer"]
            container.markdown(full_response + "▌") # Add a cursor effect
    
    container.markdown(full_response) # Final render without cursor
    return full_response

# ... [File Upload Logic stays the same, adding save_local] ...
# Inside your "Initialize Sync" button logic:
# st.session_state.vector_db = FAISS.from_documents(final_docs, get_embeddings())
# st.session_state.vector_db.save_local("faiss_index") # Persist the index