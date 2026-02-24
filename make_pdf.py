from reportlab.pdfgen import canvas
import os

# Ensure data folder exists
if not os.path.exists("data"):
    os.makedirs("data")

c = canvas.Canvas("data/test_document.pdf")
c.drawString(100, 750, "Hello! This is a test document for my RAG project.")
c.drawString(100, 730, "The capital of France is Paris.")
c.drawString(100, 710, "Python 3.14 is the version I am currently using.")
c.drawString(100, 690, "FAISS is a powerful vector database for searching text.")
c.save()

print("Created data/test_document.pdf successfully!")