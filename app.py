import streamlit as st
import os
import tempfile
import time
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# --- STABLE CORE RAG COMPONENTS (Your Working Versions) ---
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
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Nexus Intelligence System", layout="wide")
load_dotenv()

st.markdown("""
    <style>
    .stChatInputContainer { max-width: 95% !important; }
    .stChatMessage { border-radius: 5px !important; border: 1px solid #374151; margin-bottom: 8px; background-color: #111827; }
    [data-testid="stSidebar"] { border-right: 1px solid #374151; }
    </style>
    """, unsafe_allow_html=True)

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
if "logs" not in st.session_state: st.session_state.logs = []

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
        df = pd.read_excel(tmp_path) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(tmp_path)
        st.session_state.data_frames[uploaded_file.name] = df
        loader = UnstructuredExcelLoader(tmp_path)
    
    docs = loader.load()
    for doc in docs: doc.metadata["source_file"] = uploaded_file.name
    os.remove(tmp_path)
    return docs

# 3. SIDEBAR
with st.sidebar:
    st.title("Nexus Control")
    mode = st.radio("Intelligence Mode:", ["Instant", "General", "Research", "Deep Think"], index=1)

    m_data = {
        "Instant": {"temp": 0.7, "k": 3, "desc": "⚡ Fast & Strict (Docs only)."},
        "General": {"temp": 0.4, "k": 5, "desc": "⚖️ Balanced & Strict (Docs only)."},
        "Research": {"temp": 0.2, "k": 10, "desc": "🔍 Deep retrieval & Analysis."},
        "Deep Think": {"temp": 0.1, "k": 15, "desc": "🧠 Logical synthesis."}
    }
    st.info(m_data[mode]["desc"])

    if st.button("Purge Session"):
        st.session_state.messages, st.session_state.chat_history, st.session_state.data_frames = [], [], {}
        st.rerun()

    uploaded_files = st.file_uploader("Upload Docs", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)
    
    if st.button("Initialize Sync"):
        if uploaded_files:
            start_time = time.time()
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_single_file, uploaded_files))
            all_docs = [doc for sublist in results for doc in sublist]
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            final_docs = splitter.split_documents(all_docs)
            
            st.session_state.vector_db = FAISS.from_documents(final_docs, get_embeddings())
            st.session_state.bm25_retriever = BM25Retriever.from_documents(final_docs)
            st.success("Sync Complete!")

# 4. DATA VISUALIZER
if st.session_state.data_frames:
    with st.expander("📊 Data Visualizer"):
        fname = st.selectbox("Select File", list(st.session_state.data_frames.keys()))
        df = st.session_state.data_frames[fname]
        c1, c2, c3 = st.columns(3)
        x_col = c1.selectbox("X-Axis", df.columns)
        y_col = c2.selectbox("Y-Axis", df.columns)
        ctype = c3.selectbox("Type", ["Bar", "Line", "Scatter"])
        fig = getattr(px, ctype.lower())(df, x=x_col, y=y_col, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# 5. CHAT
st.title("Nexus Intelligence Agent")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Query Nexus..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("Please sync documents first.")
        else:
            try:
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key, temperature=m_data[mode]["temp"])
                
                # --- STRICT MODE LOGIC ---
                if mode in ["Instant", "General"]:
                    sys_prompt = "Answer ONLY using context. If not found, say you don't know based on documents."
                else:
                    sys_prompt = "Analyze context. If not found, you may use general knowledge but label it as such."

                prompt_tpl = ChatPromptTemplate.from_messages([
                    ("system", f"{sys_prompt}\n\nContext:\n{{context}}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ])

                faiss_ret = st.session_state.vector_db.as_retriever(search_kwargs={"k": m_data[mode]["k"]})
                ensemble_ret = EnsembleRetriever(retrievers=[faiss_ret, st.session_state.bm25_retriever], weights=[0.7, 0.3])

                chain = create_retrieval_chain(ensemble_ret, create_stuff_documents_chain(llm, prompt_tpl))
                res = chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})

                st.markdown(res["answer"])
                st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=res["answer"])])
                st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
            except Exception as e:
                st.error(f"Error: {e}")