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

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Nexus Intelligence System", layout="wide")
load_dotenv()

# --- TECHNICAL UI STYLING ---
st.markdown("""
    <style>
    .stChatInputContainer { max-width: 95% !important; }
    .stChatMessage { border-radius: 2px !important; border: 1px solid #374151; margin-bottom: 8px; }
    [data-testid="stSidebar"] { border-right: 1px solid #374151; }
    .log-text { font-family: 'Courier New', monospace; font-size: 0.75rem; color: #9ca3af; }
    .metric-text { font-size: 0.85rem; font-weight: bold; color: #3b82f6; }
    </style>
    """, unsafe_allow_html=True)

# --- CLOUD-SAFE API KEY LOGIC ---
try:
    # Check Streamlit Cloud Secrets first
    groq_key = st.secrets["GROQ_API_KEY"]
except:
    # Fallback to local .env file
    groq_key = os.getenv("GROQ_API_KEY")

# 2. SESSION STATE INITIALIZATION
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "logs" not in st.session_state: st.session_state.logs = []
if "total_chunks" not in st.session_state: st.session_state.total_chunks = 0

def add_log(text):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {text}")

def load_single_file(uploaded_file):
    suffix = f".{uploaded_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    if uploaded_file.name.endswith(".pdf"): loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith(".docx"): loader = Docx2txtLoader(tmp_path)
    elif uploaded_file.name.endswith(".xlsx"): loader = UnstructuredExcelLoader(tmp_path)
    
    docs = loader.load()
    os.remove(tmp_path)
    return docs

# 3. SIDEBAR: TECHNICAL CONTROL
with st.sidebar:
    st.title("System Control")
    
    if st.button("Reset Session"):
        st.session_state.messages, st.session_state.chat_history, st.session_state.logs, st.session_state.total_chunks = [], [], [], 0
        st.rerun()

    st.divider()
    st.subheader("Data Ingestion")
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)
    
    if st.button("Initialize Sync"):
        if not groq_key:
            st.error("API Key Missing! Add GROQ_API_KEY to Secrets.")
        elif uploaded_files:
            start_time = time.time()
            add_log(f"Starting ingestion for {len(uploaded_files)} files.")
            
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_single_file, uploaded_files))
            
            all_docs = [doc for sublist in results for doc in sublist]
            