import streamlit as st
import os
import tempfile
import time
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Core RAG Components
from src.helper import download_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
try:
    from langchain_community.retrievers import EnsembleRetriever
except ImportError:
    # Fallback for different environment structures
    from langchain.retrievers import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

# 1. PAGE CONFIG & CACHING
st.set_page_config(page_title="Nexus Analytics Engine", layout="wide")
load_dotenv()

@st.cache_resource
def get_embeddings():
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
if "bm25_retriever" not in st.session_state: st.session_state.bm25_retriever = None
if "data_frames" not in st.session_state: st.session_state.data_frames = {}
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
        add_log(f"Error loading {uploaded_file.name}: {e}")
    
    for doc in docs: doc.metadata["source_file"] = uploaded_file.name
    os.remove(tmp_path)
    return docs

# 3. SIDEBAR
with st.sidebar:
    st.title("Nexus Analytics")
    
    st.subheader("🧠 Intelligence Mode")
    mode = st.radio("Select Mode:", ["Instant", "General", "Deep Think"], index=1)
    
    mode_config = {
        "Instant": {"temp": 0.7, "k": 3, "desc": "⚡ Fast responses, shallow search."},
        "General": {"temp": 0.3, "k": 6, "desc": "⚖️ Balanced accuracy and speed."},
        "Deep Think": {"temp": 0.1, "k": 12, "desc": "🧠 High precision, deep cross-referencing."}
    }
    st.info(mode_config[mode]["desc"])

    st.divider()
    uploaded_files = st.file_uploader("Upload Docs (PDF/Excel)", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)
    
    if st.button("Initialize Hybrid Sync"):
        if uploaded_files:
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_single_file, uploaded_files))
            
            all_docs = [doc for sublist in results for doc in sublist]
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            final_docs = splitter.split_documents(all_docs)
            
            # Hybrid Search Components
            faiss_db = FAISS.from_documents(final_docs, get_embeddings())
            bm25 = BM25Retriever.from_documents(final_docs)
            
            st.session_state.vector_db = faiss_db
            st.session_state.bm25_retriever = bm25
            st.session_state.total_chunks = len(final_docs)
            
            sync_duration = time.time() - start_time
            st.success(f"Hybrid Sync Complete: {sync_duration:.2f}s")
            add_log(f"Synced {len(uploaded_files)} files in {sync_duration:.2f}s")

# 4. MAIN INTERFACE
st.title("Nexus Intelligence Agent")

# Data Visualizer
if st.session_state.data_frames:
    with st.expander("📊 Data Visualizer"):
        file_to_plot = st.selectbox("Select file to visualize:", list(st.session_state.data_frames.keys()))
        df = st.session_state.data_frames[file_to_plot]
        cols = df.columns.tolist()
        col1, col2, col3 = st.columns(3)
        with col1: x_axis = st.selectbox("X-Axis:", cols)
        with col2: y_axis = st.selectbox("Y-Axis:", cols)
        with col3: chart_type = st.selectbox("Chart Type:", ["Bar", "Line", "Scatter"])
        
        if chart_type == "Bar": fig = px.bar(df, x=x_axis, y=y_axis, template="plotly_dark")
        elif chart_type == "Line": fig = px.line(df, x=x_axis, y=y_axis, template="plotly_dark")
        else: fig = px.scatter(df, x=x_axis, y=y_axis, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

# 5. HYBRID RETRIEVAL & RESPONSE
if prompt := st.chat_input("Ask about your documents or data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("Please sync documents first.")
        else:
            try:
                analysis_start = time.time()
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=mode_config[mode]["temp"])
                
                # Hybrid Retrieval Logic
                faiss_retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": mode_config[mode]["k"]})
                bm25_retriever = st.session_state.bm25_retriever
                ensemble_retriever = EnsembleRetriever(retrievers=[faiss_retriever, bm25_retriever], weights=[0.7, 0.3])

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "You are the Nexus Technical Agent. Answer using context. Cite source files and pages.\n\nContext:\n{context}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                chain = create_retrieval_chain(ensemble_retriever, create_stuff_documents_chain(llm, prompt_template))
                response = chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})

                st.markdown(response["answer"])
                
                analysis_duration = time.time() - analysis_start
                st.caption(f"⏱️ Analysis: {analysis_duration:.2f}s | Mode: {mode}")
                
                with st.expander("🔍 Evidence & Sources"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Source {i+1}:** {doc.metadata.get('source_file')} | Page: {doc.metadata.get('page', 'N/A')}")
                        st.caption(f"{doc.page_content[:200]}...")
                
                st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=response["answer"])])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

            except Exception as e:
                st.error(f"Error: {e}")