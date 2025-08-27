# # ingest.py (shortened placeholder)
# print("Ingest pipeline here — loads PDFs/TXT and builds FAISS index")
"""
Ingest documents in `data/` into a FAISS index (vectorstore/faiss_index).
- Supports PDF and TXT files.
- Uses GoogleGenerativeAIEmbeddings (models/embedding-001)
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "data"
VS_DIR = "vectorstore/faiss_index"
EMBEDDING_MODEL = "models/embedding-001"

def get_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY missing. Put it in .env or environment variable.")
    return api_key

def load_documents(data_dir: str = DATA_DIR):
    docs = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} not found. Place your PDFs/TXT inside it.")
    # PDFs: load all PDFs in directory
    pdf_loader = PyPDFDirectoryLoader(data_dir)
    docs.extend(pdf_loader.load())

    # Plain text files
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                txt_path = os.path.join(root, f)
                docs.extend(TextLoader(txt_path, encoding="utf-8").load())
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

def build_index(chunks, vs_dir: str = VS_DIR):
    os.makedirs(os.path.dirname(vs_dir), exist_ok=True)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(vs_dir):
        db = FAISS.load_local(vs_dir, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(chunks)
    else:
        db = FAISS.from_documents(chunks, embeddings)
    db.save_local(vs_dir)
    print(f"✅ Saved FAISS index to {vs_dir}")

if __name__ == "__main__":
    get_api_key()  # fail fast if missing
    print("Loading documents...")
    docs = load_documents()
    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks. Building index...")
    build_index(chunks)
    print("Done.")
