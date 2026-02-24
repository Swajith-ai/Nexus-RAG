# Nexus Intelligence Agent: Advanced Agentic RAG System

Nexus is a professional-grade Retrieval-Augmented Generation (RAG) assistant designed for high-speed technical document analysis. Built with Python 3.14 and Llama 3.3, it transcends basic chatbots by utilizing "Agentic Memory" and parallel processing.

## 🚀 Key Features

- **Agentic Memory:** Unlike standard RAG, Nexus maintains a message buffer to understand follow-up questions (e.g., "Tell me more about *that*").
- **Parallel Document Ingestion:** Uses `ThreadPoolExecutor` to load multiple PDF, DOCX, and XLSX files simultaneously, significantly reducing sync time.
- **Optimized Neural Indexing:** Uses FAISS and HuggingFace embeddings with an optimized chunking strategy (1000 tokens) for faster vectorization.
- **Executive UI:** A clean, monochromatic Streamlit interface with high-breadth input fields, technical logs, and automatic light/dark mode support.
- **Automated Executive Summary:** A dedicated agentic tool that synthesizes long-form documents into concise reports.

## 🛠️ Technical Stack

- **LLM:** Llama 3.3 (70B-Versatile via Groq)
- **Framework:** LangChain (Modular 2026 Distribution)
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Frontend:** Streamlit
- **Embeddings:** HuggingFace (Sentence-Transformers)

## 📁 Project Structure & Components

- `app.py`: The main engine handling the UI, Agentic logic, and Retrieval chains.
- `src/helper.py`: Contains the logic for initializing the embedding model.
- `requirements.txt`: List of dependencies required for cloud deployment.
- `.env`: (Private) Stores the Groq API Key.
- `faiss_index/`: Local storage for the mathematical representation of your documents.

## ⚙️ How It Works (Step-by-Step)

1. **Document Ingestion:** The system reads multiple file formats in parallel to save time.
2. **Chunking & Embedding:** Documents are split into segments and converted into 384-dimensional vectors.
3. **Neural Sync:** These vectors are stored in FAISS for near-instant retrieval.
4. **Agentic Retrieval:** When a query is entered, the agent retrieves the top 5 most relevant segments while considering previous chat context.
5. **Contextual Generation:** The LLM receives the prompt, the context, and the memory to generate a precise, verifiable answer.

---
*Created as a part of the AI/ML B.Tech curriculum at Sathyabama Institute of Science & Technology.*