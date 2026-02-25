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
    """Caches the embedding model to memory to speed up multiple syncs."""
    return download_embeddings()

# --- CLOUD-SAFE API KEY LOGIC ---
try:
    groq_key = st.secrets["GROQ_API_KEY"]
except:
    groq_key = os.getenv("GROQ_API_KEY")

# 2. SESSION STATE
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
    
    if uploaded_file.name.endswith(".pdf"): 
        loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith(".docx"): 
        loader = Docx2txtLoader(tmp_path)
    elif uploaded_file.name.endswith(".xlsx"): 
        loader = UnstructuredExcelLoader(tmp_path)
    
    docs = loader.load()
    # Adding metadata: filename
    for doc in docs:
        doc.metadata["source_file"] = uploaded_file.name
        
    os.remove(tmp_path)
    return docs

# 3. SIDEBAR
with st.sidebar:
    st.title("System Control")
    if groq_key: st.success(f"Nexus Online | API Connected")
    else: st.error("API Key Offline")

    if st.button("Purge Neural Cache"):
        st.session_state.messages, st.session_state.chat_history, st.session_state.logs, st.session_state.total_chunks, st.session_state.vector_db = [], [], [], 0, None
        st.rerun()

    st.divider()
    st.subheader("Data Ingestion")
    uploaded_files = st.file_uploader("Upload Tech Docs", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)
    
    if st.button("Initialize High-Speed Sync"):
        if uploaded_files:
            start_time = time.time()
            add_log("Starting parallel neural indexing...")
            
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_single_file, uploaded_files))
            
            all_docs = [doc for sublist in results for doc in sublist]
            
            # Optimized splitting for 2026 technical docs
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            final_docs = text_splitter.split_documents(all_docs)
            st.session_state.total_chunks = len(final_docs)
            
            # FAISS Indexing with Cached Embeddings
            st.session_state.vector_db = FAISS.from_documents(final_docs, get_embeddings())
            
            duration = time.time() - start_time
            add_log(f"Sync optimized: {duration:.2f}s for {len(final_docs)} nodes.")
            st.success("Sync Complete.")

    st.divider()
    st.subheader("System Logs")
    for log in st.session_state.logs[-5:]:
        st.markdown(f"<p style='font-size:10px; color:gray;'>{log}</p>", unsafe_allow_html=True)

# 4. MAIN CHAT INTERFACE
st.title("Nexus Intelligence Agent")
st.caption("Advanced RAG with Source Attribution & Agentic Memory")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. RETRIEVAL & STREAMING LOGIC
if prompt := st.chat_input("Query the knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("Nexus requires data. Please upload and sync documents.")
        else:
            try:
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=0.1)
                
                # UPDATED PROMPT: Instruct AI to cite sources
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "You are the Nexus Technical Agent. Use the provided context to answer. Always mention the source file and page number if available at the end of your answer.\n\nContext:\n{context}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                # Improved Retriever: Similarity Search with a higher k-value
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 6})
                
                document_chain = create_stuff_documents_chain(llm, prompt_template)
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                with st.spinner("Synthesizing response..."):
                    response = retrieval_chain.invoke({
                        "input": prompt, 
                        "chat_history": st.session_state.chat_history
                    })

                ans = response["answer"]
                
                # --- SOURCE ATTRIBUTION UI ---
                st.markdown(ans)
                with st.expander("🔍 View Source Evidence"):
                    for i, doc in enumerate(response["context"]):
                        source = doc.metadata.get("source_file", "Unknown")
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Source {i+1}:** {source} (Pg. {page})")
                        st.caption(f"{doc.page_content[:200]}...")
                
                st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=ans)])
                if len(st.session_state.chat_history) > 10: st.session_state.chat_history = st.session_state.chat_history[-10:]
                st.session_state.messages.append({"role": "assistant", "content": ans})

            except Exception as e:
                st.error(f"System Error: {e}")