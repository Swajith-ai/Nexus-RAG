import streamlit as st
import os
import time

# --- 1. KEY CHECK ---
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
elif os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()
    
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ API Key is missing. Check .env or secrets.toml")
    st.stop()

# --- 2. IMPORTS ---
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_classic.chains.retrieval import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

st.title("🤖 Nexus Intelligence Agent")

# --- 3. LOADING FAISS ---
@st.cache_resource
def load_vectorstore():
    try:
        embeddings = OpenAIEmbeddings()
        index_path = "faiss_index"
        if os.path.exists(index_path):
            # We add a small success toast to see it working
            vstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            return vstore
        else:
            st.error(f"❌ Folder '{index_path}' not found in {os.getcwd()}")
            return None
    except Exception as e:
        st.error(f"❌ FAISS Error: {e}")
        return None

with st.status("Loading knowledge base...", expanded=False) as status:
    vectorstore = load_vectorstore()
    if vectorstore:
        status.update(label="✅ Knowledge base loaded!", state="complete")
    else:
        status.update(label="❌ Loading failed.", state="error")

# --- 4. CHAT LOGIC ---
if vectorstore:
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template("Answer based on: {context}\n\nQuestion: {input}")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display History
    for msg in st.session_state.chat_history:
        st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant").write(msg.content)

    # Input Box
    if user_query := st.chat_input("Ask a question..."):
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            res = retrieval_chain.invoke({"input": user_query})
            st.write(res["answer"])
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=res["answer"]))