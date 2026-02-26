import streamlit as st
import time
import os
from dotenv import load_dotenv

# NEW MODULAR IMPORTS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------
# Configuration & API Check
# -----------------------------
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    st.error("Missing OPENAI_API_KEY in .env file.")
    st.stop()

st.set_page_config(page_title="Nexus Intelligence Agent", page_icon="🤖")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Resource Loading
# -----------------------------
@st.cache_resource
def load_resources():
    # FAISS requires the embedding model to search
    embeddings = OpenAIEmbeddings()
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return None

vectorstore = load_resources()

# -----------------------------
# Main Application Logic
# -----------------------------
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Prompt Template
    prompt = ChatPromptTemplate.from_template("""
    You are an intelligent assistant. Answer strictly based on the provided context.
    If the answer is not in the context, say: "I could not find this information in the provided documents."

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Build Chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # UI Header
    st.title("Nexus Intelligence Agent")
    st.markdown("---")

    # Display Chat History using Native UI
    for message in st.session_state.chat_history:
        with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
            st.markdown(message.content)

    # Chat Input
    if user_input := st.chat_input("Query the knowledge base..."):
        # Show User Message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                start_time = time.time()
                
                # Logic Execution
                result = retrieval_chain.invoke({"input": user_input})
                answer = result["answer"]
                
                # Performance Metric
                elapsed = round(time.time() - start_time, 2)
                st.markdown(answer)
                st.caption(f"⏱ Response time: {elapsed}s")

        # Save to History
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))
else:
    st.warning("⚠️ 'faiss_index' not found. Please ensure your vector store is in the root directory.")