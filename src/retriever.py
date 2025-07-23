# src/retriever.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS DB location
DB_FAISS_PATH = "vectorstore/db_faiss"

# Load embeddings 
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS DB & expose Retriever
def get_retriever():
    """
    Load vector database and return retriever object.
    It performs semantic search over embedded document chunks.
    """
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})  # Top 3 relevant chunks
    return retriever
