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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)

# -------------------------
# PAGE CONFIG & ENV
# -------------------------
st.set_page_config(page_title="Nexus Intelligence System", layout="wide")
load_dotenv()

@st.cache_resource
def get_embeddings():
    return download_embeddings()

try:
    groq_key = st.secrets["GROQ_API_KEY"]
except:
    groq_key = os.getenv("GROQ_API_KEY")

# -------------------------
# SESSION STATE INIT
# -------------------------
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None
if "logs" not in st.session_state: st.session_state.logs = []
if "total_chunks" not in st.session_state: st.session_state.total_chunks = 0
if "sync_time" not in st.session_state: st.session_state.sync_time = 0.0

def add_log(text):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {text}")

# -------------------------
# FILE LOADER
# -------------------------
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
    else:
        return []

    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = uploaded_file.name

    os.remove(tmp_path)
    return docs

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.title("Nexus Control")

    st.subheader("🧠 Intelligence Mode")
    mode = st.radio(
        "Select Operation Mode:",
        ["Instant", "General", "Research", "Deep Think"],
        index=1
    )

    mode_data = {
        "Instant": {"temp": 0.7, "k": 3, "desc": "⚡ Instant Mode"},
        "General": {"temp": 0.4, "k": 5, "desc": "⚖️ General Mode"},
        "Research": {"temp": 0.2, "k": 10, "desc": "🔍 Research Mode"},
        "Deep Think": {"temp": 0.1, "k": 15, "desc": "🧠 Deep Think Mode"},
    }

    st.info(mode_data[mode]["desc"])

    current_temp = mode_data[mode]["temp"]
    current_k = mode_data[mode]["k"]

    st.divider()

    if st.button("Purge Session"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.logs = []
        st.session_state.total_chunks = 0
        st.session_state.vector_db = None
        st.rerun()

    st.divider()
    st.subheader("📁 Data Ingestion")

    uploaded_files = st.file_uploader(
        "Upload Tech Docs",
        type=["pdf", "docx", "xlsx"],
        accept_multiple_files=True
    )

    if st.button("Initialize Sync"):
        if uploaded_files:
            start_time = time.time()
            add_log(f"Sync initiated for {len(uploaded_files)} files...")

            with ThreadPoolExecutor() as executor:
                results = list(executor.map(load_single_file, uploaded_files))

            all_docs = [doc for sublist in results for doc in sublist]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=150
            )

            final_docs = splitter.split_documents(all_docs)

            st.session_state.total_chunks = len(final_docs)

            vector_db = FAISS.from_documents(final_docs, get_embeddings())
            st.session_state.vector_db = vector_db

            st.session_state.sync_time = time.time() - start_time
            add_log(f"Sync successful: {st.session_state.sync_time:.2f}s")
            st.success(f"Sync Complete: {st.session_state.sync_time:.2f}s")

    st.divider()
    st.markdown(
        f"**Last Sync:** `{st.session_state.sync_time:.2f}s` | "
        f"**Nodes:** `{st.session_state.total_chunks}`"
    )

    st.subheader("📟 System Logs")
    for log in st.session_state.logs[-5:]:
        st.markdown(
            f"<p style='font-size:11px; color:#888; margin:0;'>{log}</p>",
            unsafe_allow_html=True
        )

# -------------------------
# MAIN CHAT UI
# -------------------------
st.title("Nexus Intelligence Agent")
st.caption(f"Status: **{mode} Mode** is engaged.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------
# RAG LOGIC (LangChain 1.x)
# -------------------------
if prompt := st.chat_input("Query the knowledge base..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("Nexus requires data. Please upload and sync documents.")
        else:
            try:
                analysis_start = time.time()

                llm = ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    groq_api_key=groq_key,
                    temperature=current_temp
                )

                system_instruction = "You are the Nexus Technical Agent."
                if mode == "Deep Think":
                    system_instruction += " Think step-by-step and analyze deeply."

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", f"{system_instruction}\n\nContext:\n{{context}}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{question}")
                ])

                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": current_k}
                )

                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)

                rag_chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                        "chat_history": lambda x: st.session_state.chat_history,
                    }
                    | prompt_template
                    | llm
                )

                with st.spinner(f"Nexus {mode} is analyzing..."):
                    response = rag_chain.invoke(prompt)

                ans = response.content
                analysis_duration = time.time() - analysis_start

                st.markdown(ans)
                st.caption(
                    f"⏱️ Analysis Time: **{analysis_duration:.2f}s** | "
                    f"Mode: **{mode}**"
                )

                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=ans)
                ])

                if len(st.session_state.chat_history) > 10:
                    st.session_state.chat_history = \
                        st.session_state.chat_history[-10:]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ans
                })

            except Exception as e:
                st.error(f"Execution Error: {e}")