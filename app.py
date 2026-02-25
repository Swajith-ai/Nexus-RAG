import streamlit as st
import os
import tempfile
import time
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# --- UNIVERSAL SMART IMPORTER (2026 FIX) ---
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers import EnsembleRetriever

from src.helper import download_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

# --- 1. PAGE CONFIG & CACHING ---
st.set_page_config(page_title="Nexus Analytics Engine", layout="wide")
load_dotenv()

@st.cache_resource
def get_embeddings():
    return download_embeddings()

try:
    groq_key = st.secrets["GROQ_API_KEY"]
except:
    groq_key = os.getenv("GROQ_API_KEY")

# --- 2. SESSION STATE ---
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "bm25_retriever" not in st.session_state: st.session_state.bm25_retriever = None
if "data_frames" not in st.session_state: st.session_state.data_frames = {}
if "total_chunks" not in st.session_state: st.session_state.total_chunks = 0

def load_single_file(uploaded_file):
    suffix = f".{uploaded_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    docs = []
    try:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
            docs = loader.load()
        elif uploaded_file.name.endswith((".xlsx", ".csv")):
            df = pd.read_excel(tmp_path) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(tmp_path)
            st.session_state.data_frames[uploaded_file.name] = df
            loader = UnstructuredExcelLoader(tmp_path)
            docs = loader.load()
    except Exception as e:
        st.error(f"Error loading {uploaded_file.name}: {e}")
    
    for doc in docs: doc.metadata["source_file"] = uploaded_file.name
    os.remove(tmp_path)
    return docs

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("Nexus Control")
    mode = st.radio("Intelligence Mode:", ["Instant", "General", "Deep Think"], index=1)
    
    mode_config = {
        "Instant": {"temp": 0.7, "k": 3, "desc": "⚡ Optimized for speed."},
        "General": {"temp": 0.3, "k": 6, "desc": "⚖️ Balanced accuracy."},
        "Deep Think": {"temp": 0.1, "k": 12, "desc": "🧠 Maximum precision."}
    }
    st.info(mode_config[mode]["desc"])

    st.divider()
    uploaded_files = st.file_uploader("Upload Docs", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)
    
    if st.button("Initialize Hybrid Sync"):
        if uploaded_files:
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_single_file, uploaded_files))
            all_docs = [doc for sublist in results for doc in sublist]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            final_docs = splitter.split_documents(all_docs)
            
            st.session_state.vector_db = FAISS.from_documents(final_docs, get_embeddings())
            st.session_state.bm25_retriever = BM25Retriever.from_documents(final_docs)
            st.session_state.total_chunks = len(final_docs)
            st.success(f"Sync Complete: {time.time()-start_time:.2f}s")

# --- 4. DATA VISUALIZER ---
if st.session_state.data_frames:
    with st.expander("📊 Data Visualizer"):
        file_name = st.selectbox("Select file:", list(st.session_state.data_frames.keys()))
        df = st.session_state.data_frames[file_name]
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X-Axis", df.columns)
        y = c2.selectbox("Y-Axis", df.columns)
        ctype = c3.selectbox("Type", ["Bar", "Line", "Scatter"])
        
        fig = getattr(px, ctype.lower())(df, x=x, y=y, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# --- 5. CHAT INTERFACE ---
st.title("Nexus Intelligence Agent")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("Please sync documents first.")
        else:
            try:
                analysis_start = time.time()
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=mode_config[mode]["temp"])
                
                # Retrieval Logic
                faiss_ret = st.session_state.vector_db.as_retriever(search_kwargs={"k": mode_config[mode]["k"]})
                bm25_ret = st.session_state.bm25_retriever
                ensemble_ret = EnsembleRetriever(retrievers=[faiss_ret, bm25_ret], weights=[0.7, 0.3])

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "You are the Nexus Agent. Answer using context. Cite source files.\n\nContext:\n{context}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                chain = create_retrieval_chain(ensemble_ret, create_stuff_documents_chain(llm, prompt_template))
                response = chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})

                st.markdown(response["answer"])
                st.caption(f"⏱️ Analysis: {time.time()-analysis_start:.2f}s | Mode: {mode}")
                
                st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=response["answer"])])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            except Exception as e:
                st.error(f"Error: {e}")