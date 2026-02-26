import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from src.helper import load_pdf, text_split, download_embeddings

# ----------------------------
# 🔐 Load Environment
# ----------------------------
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

if not groq_key:
    st.error("Groq API Key not found. Please configure it.")
    st.stop()

# ----------------------------
# 🎨 UI CONFIG
# ----------------------------
st.set_page_config(page_title="Nexus Control", layout="wide")
st.title("Nexus Control")
st.subheader("🧠 Intelligence Mode")

# ----------------------------
# 🧠 Mode Selection
# ----------------------------
mode = st.radio(
    "Select Operation Mode:",
    ["Instant", "General", "Research", "Deep Think"],
    horizontal=True
)

mode_settings = {
    "Instant": {"k": 2, "temp": 0.2},
    "General": {"k": 5, "temp": 0.5},
    "Research": {"k": 8, "temp": 0.7},
    "Deep Think": {"k": 12, "temp": 0.9},
}

current_k = mode_settings[mode]["k"]
current_temp = mode_settings[mode]["temp"]

st.info(f"⚖️ {mode} Mode engaged. Using {current_k} chunks.")

# ----------------------------
# 📂 Session Init
# ----------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# 📁 Data Ingestion
# ----------------------------
st.subheader("📁 Data Ingestion")

uploaded_files = st.file_uploader(
    "Upload Tech Docs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("🔄 Sync Knowledge Base"):
    if uploaded_files:
        with st.spinner("Processing documents..."):
            os.makedirs("data", exist_ok=True)

            for file in uploaded_files:
                with open(os.path.join("data", file.name), "wb") as f:
                    f.write(file.read())

            raw_docs = load_pdf("data")
            chunks = text_split(raw_docs)
            embeddings = download_embeddings()

            vector_db = FAISS.from_documents(chunks, embeddings)
            vector_db.save_local("nexus_faiss")

            st.session_state.vector_db = vector_db

        st.success(f"Synced {len(chunks)} chunks successfully!")
    else:
        st.warning("Please upload at least one PDF.")

# ----------------------------
# 📦 Load Existing FAISS
# ----------------------------
if st.session_state.vector_db is None:
    if os.path.exists("nexus_faiss"):
        embeddings = download_embeddings()
        st.session_state.vector_db = FAISS.load_local(
            "nexus_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )

# ----------------------------
# 💬 Chat Section
# ----------------------------
st.subheader("Nexus Intelligence Agent")

query = st.text_input("Query the knowledge base...")

if query:
    if st.session_state.vector_db is None:
        st.warning("Please sync documents first.")
    else:
        start_time = time.time()

        retriever = st.session_state.vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": current_k,
                "fetch_k": current_k * 3,
                "lambda_mult": 0.5,
            }
        )

        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.3-70b-versatile",
            temperature=current_temp
        )

        prompt = ChatPromptTemplate.from_template(
            """
            You are Nexus Intelligence Agent.
            Answer the question using ONLY the provided context.

            Context:
            {context}

            Question:
            {question}
            """
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # LCEL pipeline
        rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        response = rag_chain.invoke(query)

        end_time = time.time()

        st.session_state.chat_history.append((query, response.content))

        st.markdown("### 🤖 Response")
        st.write(response.content)

        st.caption(f"⏱ Analysis Time: {round(end_time - start_time, 2)}s")