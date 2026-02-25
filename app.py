import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- CORE STABLE IMPORTS ---
from src.helper import download_embeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# 1. PAGE CONFIG
st.set_page_config(page_title="Nexus RAG - Basic", layout="wide")
load_dotenv()

@st.cache_resource
def get_embeddings():
    return download_embeddings()

# API Key Handling
try:
    groq_key = st.secrets["GROQ_API_KEY"]
except:
    groq_key = os.getenv("GROQ_API_KEY")

# 2. SESSION STATE
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vector_db" not in st.session_state: st.session_state.vector_db = None

def load_file(uploaded_file):
    suffix = f".{uploaded_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = Docx2txtLoader(tmp_path)
    
    docs = loader.load()
    os.remove(tmp_path)
    return docs

# 3. SIDEBAR
with st.sidebar:
    st.title("Nexus Control")
    uploaded_files = st.file_uploader("Upload PDF/Docx", type=["pdf", "docx"], accept_multiple_files=True)
    
    if st.button("Sync Documents"):
        if uploaded_files:
            all_docs = []
            for f in uploaded_files:
                all_docs.extend(load_file(f))
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            final_docs = splitter.split_documents(all_docs)
            
            st.session_state.vector_db = FAISS.from_documents(final_docs, get_embeddings())
            st.success("Sync Complete!")

# 4. CHAT INTERFACE
st.title("Nexus Intelligence Agent")



for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.vector_db:
            st.warning("Please upload and sync documents first.")
        else:
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_key)
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Answer the user's question based on the context: {context}"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ])

            retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})
            combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
            chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            response = chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
            
            st.markdown(response["answer"])
            st.session_state.chat_history.extend([HumanMessage(content=prompt), AIMessage(content=response["answer"])])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})