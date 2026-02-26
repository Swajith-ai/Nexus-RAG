import streamlit as st
import time
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# Check for API Key
if not os.getenv("OPENAI_API_KEY"):
    st.error("Error: OPENAI_API_KEY not found in environment variables.")
    st.stop()

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Nexus Intelligence Agent", layout="centered")

# Initialize Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Load Vectorstore
# -----------------------------
@st.cache_resource
def load_vectorstore():
    # IMPORTANT: You must use the same embedding model used to create the index
    embeddings = OpenAIEmbeddings() 
    try:
        return FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

vectorstore = load_vectorstore()

if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # -----------------------------
    # LLM & Chain Setup
    # -----------------------------
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
    You are a professional Nexus Intelligence Assistant. 
    Answer the user's question based ONLY on the provided context.
    If the answer isn't in the context, say: "I could not find this information in the provided documents."

    <context>
    {context}
    </context>

    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # -----------------------------
    # UI - Chat Interface
    # -----------------------------
    st.title("🤖 Nexus Intelligence Agent")
    st.divider()

    # Display existing chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    # User Input
    user_question = st.chat_input("Query the knowledge base...")

    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Process retrieval and generation
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                start_time = time.time()
                
                response = retrieval_chain.invoke({
                    "input": user_question,
                    "chat_history": st.session_state.chat_history
                })
                
                end_time = time.time()
                response_time = round(end_time - start_time, 2)
                
                answer = response["answer"]
                st.markdown(answer)
                st.caption(f"⏱ Response time: {response_time}s")

        # Update Session State
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=answer))
else:
    st.warning("Please ensure the 'faiss_index' folder exists in the root directory.")