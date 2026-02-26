import streamlit as st
import time
import os
from dotenv import load_dotenv

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="Nexus Intelligence Agent")

# -----------------------------
# 🔥 IMPORTANT FIX
# Initialize Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Load Vectorstore
# -----------------------------
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local("faiss_index", embeddings=None, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
You are an intelligent assistant.
Answer strictly based on the provided context.
If the answer is not in the context, say:
"I could not find this information in the provided documents."

<context>
{context}
</context>

Question: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# -----------------------------
# UI - Nexus Control
# -----------------------------
st.title("Nexus Intelligence Agent")

user_question = st.chat_input("Query the knowledge base...")

if user_question:
    start_time = time.time()

    response = retrieval_chain.invoke({
        "input": user_question,
        "chat_history": st.session_state.chat_history
    })

    end_time = time.time()
    response_time = round(end_time - start_time, 2)

    answer = response["answer"]

    # Store chat history
    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("assistant", answer))

    # Display Answer
    st.write(answer)
    st.caption(f"⏱ Response time: {response_time}s")