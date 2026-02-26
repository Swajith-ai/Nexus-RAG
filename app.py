import os
import time
import streamlit as st
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document

from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

# -----------------------
# Load Environment
# -----------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------
# Logging Setup
# -----------------------
logging.basicConfig(level=logging.INFO)

# -----------------------
# Streamlit Config
# -----------------------
st.set_page_config(page_title="Nexus RAG Enterprise", layout="wide")
st.title("🚀 Nexus Enterprise RAG")

# -----------------------
# Embeddings (Cached)
# -----------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -----------------------
# Cross Encoder Re-ranker
# -----------------------
@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()

# -----------------------
# Document Loader
# -----------------------
def load_documents(uploaded_files):
    documents = []

    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file.name

        documents.extend(docs)

    return documents


# -----------------------
# Semantic Chunking
# -----------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(documents)


# -----------------------
# Persistent Vector Store
# -----------------------
VECTOR_PATH = "vectorstore"

def build_or_load_vectorstore(docs):
    if os.path.exists(VECTOR_PATH):
        logging.info("Loading existing FAISS index...")
        return FAISS.load_local(VECTOR_PATH, embeddings, allow_dangerous_deserialization=True)

    logging.info("Building new FAISS index...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(VECTOR_PATH)
    return vectorstore


# -----------------------
# Hybrid Retrieval
# -----------------------
def hybrid_retrieve(query, vectorstore, docs, k=10):
    # Dense Search
    dense_docs = vectorstore.similarity_search(query, k=k)

    # BM25 Search
    tokenized_corpus = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:k]
    sparse_docs = [docs[i] for i in top_n]

    # Combine & Deduplicate
    combined = list({doc.page_content: doc for doc in dense_docs + sparse_docs}.values())

    return combined


# -----------------------
# Re-Ranking
# -----------------------
def rerank(query, docs, top_k=5):
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]


# -----------------------
# LLM Setup
# -----------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",
    temperature=0.2,
)

# -----------------------
# Streaming Answer
# -----------------------
def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
    You are an enterprise AI assistant.
    Answer ONLY from context.
    
    Context:
    {context}
    
    Question:
    {query}
    """

    response = llm.stream(prompt)

    full_answer = ""
    for chunk in response:
        full_answer += chunk.content
        yield full_answer


# -----------------------
# Streamlit UI
# -----------------------
uploaded_files = st.file_uploader(
    "Upload PDF files", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:

    with st.spinner("Processing documents..."):
        documents = load_documents(uploaded_files)
        split_docs = split_documents(documents)
        vectorstore = build_or_load_vectorstore(split_docs)

    st.success("Documents Ready!")

    query = st.text_input("Ask a question")

    if query:
        start_time = time.time()

        retrieved = hybrid_retrieve(query, vectorstore, split_docs, k=10)
        reranked = rerank(query, retrieved, top_k=5)

        st.subheader("Answer")

        answer_placeholder = st.empty()
        for partial in generate_answer(query, reranked):
            answer_placeholder.markdown(partial)

        end_time = time.time()

        st.write("---")
        st.subheader("Sources")

        for doc in reranked:
            st.write(f"📄 {doc.metadata.get('source')}")

        st.write(f"⏱ Response Time: {round(end_time - start_time, 2)} sec")