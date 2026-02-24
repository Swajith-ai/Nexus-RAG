import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def load_pdf(data_path):
    # We are adding glob="*.pdf" to make sure it only looks for PDFs
    loader = PyPDFDirectoryLoader(data_path, glob="*.pdf")
    return loader.load()

def text_split(extracted_data):
    # We lowered the chunk size slightly for better accuracy
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)

def download_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")