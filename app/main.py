# app/main.py
from fastapi import FastAPI, UploadFile, File
import shutil
import os

from app.utils.loader import load_pdf
from app.utils.chunking import chunk_documents
from app.services.embeddings import get_embeddings
from app.services.vector_store import create_vector_store, load_vector_store
from app.services.retriever import get_retriever
from app.services.rag_chain import build_rag_chain

# ----------------------------
# App Setup
# ----------------------------
app = FastAPI(title="Finance AI Document Assistant")

# Paths for uploads and FAISS index
UPLOAD_DIR = "data"
FAISS_INDEX_DIR = "faiss_index"

# Ensure folders exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# ----------------------------
# Health Check
# ----------------------------
@app.get("/status")
def health():
    return {"status": "running"}

# ----------------------------
# Upload Endpoint
# ----------------------------
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load and chunk document
    documents = load_pdf(file_path)
    chunks = chunk_documents(documents)

    # Create embeddings and persistent FAISS index
    embeddings = get_embeddings()
    create_vector_store(chunks, embeddings, index_path=FAISS_INDEX_DIR)

    return {"message": f"{file.filename} processed and saved to persistent FAISS index"}

# ----------------------------
# Query Endpoint
# ----------------------------
@app.post("/query")
async def query_document(query: str):

    embeddings = get_embeddings()
    # Load persistent FAISS index
    vectorstore = load_vector_store(embeddings, index_path=FAISS_INDEX_DIR)
    retriever = get_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)

    # Run query through RAG chain
    result = rag_chain.invoke(query)

    return {"answer": result}
