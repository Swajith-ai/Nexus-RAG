from src.helper import load_pdf, text_split, download_embeddings
from langchain_community.vectorstores import FAISS
import os

# 1. Check if folder is empty
files = os.listdir("data/")
print(f"Files found in data folder: {files}")

print("Step 1: Loading PDFs...")
extracted_data = load_pdf("data/")
print(f"Total pages extracted: {len(extracted_data)}")

if len(extracted_data) == 0:
    print("!!! ERROR: No text could be read from your PDF. Try a different PDF file. !!!")
else:
    print("Step 2: Splitting text...")
    text_chunks = text_split(extracted_data)
    print(f"Total chunks created: {len(text_chunks)}")

    print("Step 3: Downloading Embeddings...")
    embeddings = download_embeddings()

    print("Step 4: Building FAISS database...")
    vector_db = FAISS.from_documents(text_chunks, embeddings)
    vector_db.save_local("faiss_index")
    print("\nSUCCESS! Your database is ready.")