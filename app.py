import streamlit as st
import os
import tempfile
import time
import pandas as pd
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# --- THE FIX: MODERN LANGCHAIN IMPORTS ---
from src.helper import download_embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.retrievers import EnsembleRetriever 
from langchain_groq import ChatGroq

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredExcelLoader

# 1. SETUP
st.set_page_config(page_title="Nexus Analytics", layout="wide")
load_dotenv()

@st.cache_resource
def get_embeddings():
    return download_embeddings()

try:
    groq_key = st.secrets["GROQ_API_KEY"]
except:
    groq_key = os.getenv("GROQ_API_KEY")

# 2. STATE
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "bm25_retriever" not in st.session_state: st.session_state.bm25_retriever = None
if "data_frames" not in st.session_state: st.session_state.data_frames = {}

def load_file(file):
    suffix = f".{file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getvalue())
        path = tmp.name
    
    docs = []
    if file.name.endswith(".pdf"):
        docs = PyPDFLoader(path).load()
    elif file.name.endswith(".docx"):
        docs = Docx2txtLoader(path).load()
    elif file.name.endswith((".xlsx", ".csv")):
        df = pd.read_excel(path) if file.name.endswith(".xlsx") else pd.read_csv(path)
        st.session_state.data_frames[file.name] = df
        docs = UnstructuredExcelLoader(path).load()
    
    for d in docs: d.metadata["source"] = file.name
    os.remove(path)
    return docs

# 3. UI
with st.sidebar:
    st.title("Nexus Control")
    mode = st.radio("Mode", ["Instant", "General", "Deep Think"], index=1)
    k_val = {"Instant": 3, "General": 6, "Deep Think": 12}[mode]
    
    files = st.file_uploader("Upload", type=["pdf", "docx", "xlsx", "csv"], accept_multiple_files=True)
    if st.button("Sync Documents"):
        if files:
            with ThreadPoolExecutor() as ex:
                results = list(ex.map(load_file, files))
            all_docs = [d for sub in results for d in sub]
            splitdocs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(all_docs)
            
            st.session_state.vector_db = FAISS.from_documents(splitdocs, get_embeddings())
            st.session_state.bm25_retriever = BM25Retriever.from_documents(splitdocs)
            st.success("Sync Complete")

# 4. VIZ
if st.session_state.data_frames:
    with st.expander("📊 Charts"):
        fname = st.selectbox("File", list(st.session_state.data_frames.keys()))
        df = st.session_state.data_frames[fname]
        x_col = st.selectbox("X Axis", df.columns)
        y_col = st.selectbox("Y Axis", df.columns)
        fig = px.bar(df, x=x_col, y=y_col, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# 5. CHAT
st.title("Nexus AI")
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    if st.session_state.vector_db:
        llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)
        v_ret = st.session_state.vector_db.as_retriever(search_kwargs={"k": k_val})
        e_ret = EnsembleRetriever(retrievers=[v_ret, st.session_state.bm25_retriever], weights=[0.7, 0.3])
        
        tpl = ChatPromptTemplate.from_messages([
            ("system", "Answer using context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        chain = create_retrieval_chain(e_ret, create_stuff_documents_chain(llm, tpl))
        res = chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
        
        with st.chat_message("assistant"):
            st.markdown(res["answer"])
            st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=res["answer"])])
            st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
    else:
        st.warning("Upload docs first.")