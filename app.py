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
            
            # Efficient Chunking Strategy
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            final_docs = text_splitter.split_documents(all_docs)
            st.session_state.total_chunks = len(final_docs)
            
            # Generate Embeddings & FAISS Index
            st.session_state.vector_db = FAISS.from_documents(final_docs, download_embeddings())
            
            duration = time.time() - start_time
            add_log(f"Neural sync complete in {duration:.2f}s")
            st.success(f"Sync Complete: {duration:.2f}s")

    # Executive Summary Tool
    if st.session_state.vector_db:
        st.divider()
        st.subheader("Executive Analysis")
        if st.button("Generate Summary Report"):
            with st.spinner("Synthesizing Report..."):
                llm_summary = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key)
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 8})
                sample_docs = retriever.invoke("Summarize the main themes and technical details of these documents.")
                context_text = "\n".join([doc.page_content for doc in sample_docs])
                
                report = llm_summary.invoke(f"Create a professional executive summary of the following content. Use bullet points:\n\n{context_text}")
                st.session_state.messages.append({"role": "assistant", "content": f"### 📄 Executive Summary\n\n{report.content}"})
                st.rerun()

    # System Metrics & Logs
    st.divider()
    st.markdown(f"**Nodes Active:** <span class='metric-text'>{st.session_state.total_chunks}</span>", unsafe_allow_html=True)
    st.subheader("System Logs")
    for log in st.session_state.logs[-5:]:
        st.markdown(f"<p class='log-text'>{log}</p>", unsafe_allow_html=True)

# 4. MAIN CHAT INTERFACE
st.title("Nexus Intelligence Agent")
st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. RETRIEVAL & RESPONSE LOGIC
if prompt := st.chat_input("Enter technical query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.vector_db is None:
            st.error("System Offline: Vector Database not initialized.")
        elif not groq_key:
            st.error("Invalid API Key configuration.")
        else:
            with st.spinner("Analyzing Signals..."):
                try:
                    llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0.1)
                    
                    prompt_template = ChatPromptTemplate.from_messages([
                        ("system", "You are the Nexus Technical Agent. Answer accurately using context. If unknown, say 'Data unavailable'.\n\nContext:\n{context}"),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                    ])

                    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
                    document_chain = create_stuff_documents_chain(llm, prompt_template)
                    retrieval_chain = create_retrieval_chain(retriever, document_chain)

                    response = retrieval_chain.invoke({
                        "input": prompt, 
                        "chat_history": st.session_state.chat_history
                    })

                    ans = response["answer"]
                    st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=ans)])
                    
                    if len(st.session_state.chat_history) > 10: 
                        st.session_state.chat_history = st.session_state.chat_history[-10:]

                    st.markdown(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    add_log(f"Error: {str(e)}")