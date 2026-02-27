from langchain_community.vectorstores import FAISS

def create_vector_store(chunks, embeddings, index_path="faiss_index"):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_path)
    return vectorstore

def load_vector_store(embeddings, index_path="faiss_index"):
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
