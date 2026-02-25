import streamlit as st
import os
import tempfile
import time
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# --- FIXED IMPORTS FOR LANGCHAIN 0.3.0 ---
from src.helper import download_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain_groq import ChatGroq
# Direct paths for chains
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

# 1. PAGE CONFIG
st.set_page_config(page_title="Nexus Analytics Engine", layout="wide")
load_dotenv()

@st.cache_resource
def get_embeddings():
    return download_embeddings()

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
        st.error(f"Error: {e}")
    for doc in docs: doc.metadata["source_file"] = uploaded_file.name
    os.remove(tmp_path)
    return docs

# 3. SIDEBAR
with st.sidebar:
    st.title("Nexus Control")
    mode = st.radio("Mode:", ["Instant", "General", "Deep Think"], index=1)
    mode_config = {
        "Instant": {"temp": 0.7, "k": 3},
        "General": {"temp": 0.3, "k": 6},
        "Deep Think": {"temp": 0.1, "k": 12}
    }
    uploaded_files = st.file_uploader("Upload Docs", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)
    if st.button("Initialize Hybrid Sync"):
        if uploaded_files:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_single_file, uploaded_files))
            all_docs = [doc for sublist in results for doc in sublist]
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            final_docs = splitter.split_documents(all_docs)
            st.session_state.vector_db = FAISS.from_documents(final_docs, get_embeddings())
            st.session_state.bm25_retriever = BM25Retriever.from_documents(final_docs)
            st.success("Sync Complete")

# 4. DATA VISUALIZER
if st.session_state.data_frames:
    with st.expander("📊 Data Visualizer"):
        file_name = st.selectbox("File:", list(st.session_state.data_frames.keys()))
        df = st.session_state.data_frames[file_name]
        x, y = st.columns(2)
        chart = st.selectbox("Type", ["Bar", "Line", "Scatter"])
        fig = getattr(px, chart.lower())(df, x=x.selectbox("X", df.columns), y=y.selectbox("Y", df.columns), template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# 5. CHAT
st.title("Nexus Intelligence Agent")
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("Sync first.")
        else:
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=mode_config[mode]["temp"])
            faiss_ret = st.session_state.vector_db.as_retriever(search_kwargs={"k": mode_config[mode]["k"]})
            ensemble_ret = EnsembleRetriever(retrievers=[faiss_ret, st.session_state.bm25_retriever], weights=[0.7, 0.3])
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Use context: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])
            
            chain = create_retrieval_chain(ensemble_ret, create_stuff_documents_chain(llm, prompt_template))
            res = chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
            st.markdown(res["answer"])
            st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=res["answer"])])
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})