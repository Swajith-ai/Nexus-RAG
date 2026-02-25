import streamlit as st
import os
import tempfile
import time
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# --- CORE STABLE RAG COMPONENTS ---
from src.helper import download_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

# 1. PAGE CONFIGURATION & CACHING
st.set_page_config(page_title="Nexus Intelligence System", layout="wide")
load_dotenv()

@st.cache_resource
def get_embeddings():
    """Caches the embedding model to RAM to prevent slow reloads."""
    return download_embeddings()

# --- CLOUD-SAFE API KEY LOGIC ---
try:
    groq_key = st.secrets["GROQ_API_KEY"]
except:
    groq_key = os.getenv("GROQ_API_KEY")

# 2. SESSION STATE INITIALIZATION
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "bm25_retriever" not in st.session_state: st.session_state.bm25_retriever = None
if "data_frames" not in st.session_state: st.session_state.data_frames = {}
if "logs" not in st.session_state: st.session_state.logs = []
if "total_chunks" not in st.session_state: st.session_state.total_chunks = 0
if "sync_time" not in st.session_state: st.session_state.sync_time = 0.0

def add_log(text):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {text}")

def load_single_file(uploaded_file):
    suffix = f".{uploaded_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    docs = []
    if uploaded_file.name.endswith(".pdf"): 
        loader = PyPDFLoader(tmp_path)
    elif uploaded_file.name.endswith(".docx"): 
        loader = Docx2txtLoader(tmp_path)
    elif uploaded_file.name.endswith((".xlsx", ".csv")):
        # Read for visualization
        df = pd.read_excel(tmp_path) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(tmp_path)
        st.session_state.data_frames[uploaded_file.name] = df
        loader = UnstructuredExcelLoader(tmp_path)
    
    docs = loader.load()
    for doc in docs: doc.metadata["source_file"] = uploaded_file.name
    os.remove(tmp_path)
    return docs

# 3. SIDEBAR: INTELLIGENCE MODES & SYSTEM CONTROLS
with st.sidebar:
    st.title("Nexus Control")
    
    st.subheader("🧠 Intelligence Mode")
    mode = st.radio(
        "Select Operation Mode:",
        ["Instant", "General", "Research", "Deep Think"],
        index=1
    )

    mode_data = {
        "Instant": {"temp": 0.7, "k": 3, "desc": "⚡ **Instant:** Strict Context. High speed."},
        "General": {"temp": 0.4, "k": 5, "desc": "⚖️ **General:** Strict Context. Balanced."},
        "Research": {"temp": 0.2, "k": 10, "desc": "🔍 **Research:** Deep retrieval. Analytical."},
        "Deep Think": {"temp": 0.1, "k": 15, "desc": "🧠 **Deep Think:** Maximum accuracy & cross-referencing."}
    }
    
    st.info(mode_data[mode]["desc"])
    
    if st.button("Purge Session"):
        st.session_state.messages, st.session_state.chat_history, st.session_state.logs, st.session_state.vector_db, st.session_state.data_frames = [], [], [], None, {}
        st.rerun()

    st.divider()
    st.subheader("📁 Data Ingestion")
    uploaded_files = st.file_uploader("Upload Tech Docs", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)
    
    if st.button("Initialize Sync"):
        if uploaded_files:
            start_time = time.time()
            add_log(f"Sync initiated for {len(uploaded_files)} files...")
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_single_file, uploaded_files))
            all_docs = [doc for sublist in results for doc in sublist]
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
            final_docs = text_splitter.split_documents(all_docs)
            
            # Hybrid Initialization
            st.session_state.vector_db = FAISS.from_documents(final_docs, get_embeddings())
            st.session_state.bm25_retriever = BM25Retriever.from_documents(final_docs)
            
            st.session_state.total_chunks = len(final_docs)
            st.session_state.sync_time = time.time() - start_time
            add_log(f"Sync successful: {st.session_state.sync_time:.2f}s")
            st.success(f"Sync Complete: {st.session_state.sync_time:.2f}s")

# 4. DATA VISUALIZER (Plotly)
if st.session_state.data_frames:
    with st.expander("📊 Data Visualizer"):
        fname = st.selectbox("Select Data Source:", list(st.session_state.data_frames.keys()))
        df = st.session_state.data_frames[fname]
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X-Axis", df.columns)
        y_col = c2.selectbox("Y-Axis", df.columns)
        ctype = c3.selectbox("Chart Type", ["Bar", "Line", "Scatter"])
        fig = getattr(px, ctype.lower())(df, x=x_col, y=y_col, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# 5. MAIN CHAT INTERFACE
st.title("Nexus Intelligence Agent")
st.caption(f"Status: **{mode} Mode** is engaged.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. RETRIEVAL & RESPONSE LOGIC
if prompt := st.chat_input("Query the knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("Nexus requires data. Please upload and sync documents.")
        else:
            try:
                analysis_start = time.time()
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=mode_data[mode]["temp"])
                
                # --- DYNAMIC STRICT PROMPT LOGIC ---
                if mode in ["Instant", "General"]:
                    sys_instr = (
                        "You are the Nexus Strict Agent. You MUST only answer using the provided context. "
                        "If the answer is not in the context, strictly state: "
                        "'I'm sorry, but that information is not available in the uploaded documents for this mode.'"
                    )
                else:
                    sys_instr = (
                        "You are the Nexus Research Agent. Deeply analyze the context. "
                        "If information is missing from the docs, you may use general knowledge but "
                        "MUST label it as 'General Knowledge / Theoretical'."
                    )
                
                if mode == "Deep Think":
                    sys_instr += " Cross-reference all files and look for logical contradictions."

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", f"{sys_instr}\n\nContext:\n{{context}}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                # --- ENSEMBLE RETRIEVER (Hybrid Search) ---
                faiss_ret = st.session_state.vector_db.as_retriever(search_kwargs={"k": mode_data[mode]["k"]})
                bm25_ret = st.session_state.bm25_retriever
                ensemble_ret = EnsembleRetriever(retrievers=[faiss_ret, bm25_ret], weights=[0.7, 0.3])

                doc_chain = create_stuff_documents_chain(llm, prompt_template)
                ret_chain = create_retrieval_chain(ensemble_ret, doc_chain)

                with st.spinner(f"Nexus {mode} is analyzing..."):
                    response = ret_chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})

                ans = response["answer"]
                st.markdown(ans)
                st.caption(f"⏱️ Analysis: {time.time()-analysis_start:.2f}s | Mode: {mode}")
                
                with st.expander("🔍 Intelligence Evidence"):
                    for i, doc in enumerate(response["context"]):
                        source = doc.metadata.get("source_file", "Unknown")
                        st.markdown(f"**{i+1}. {source}**")
                        st.caption(f"{doc.page_content[:150]}...")
                
                st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=ans)])
                if len(st.session_state.chat_history) > 10: st.session_state.chat_history = st.session_state.chat_history[-10:]
                st.session_state.messages.append({"role": "assistant", "content": ans})

            except Exception as e:
                st.error(f"Execution Error: {e}")