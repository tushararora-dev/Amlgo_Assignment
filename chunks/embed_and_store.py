import os
import json
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Define Paths
CHUNK_PATH = "chunks/chunks.json"
DB_FAISS_PATH = "vectordb/db_faiss"

# Load chunked documents from JSON file
def load_chunks(path=CHUNK_PATH):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in data]

# Load pre-trained embedding model from Hugging Face
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create embeddings from chunks and store them in a FAISS vector database
def embed_and_store():
    docs = load_chunks()                    # Load sentence-aware chunks
    embedding_model = get_embedding_model() # Load embedding model

    db = FAISS.from_documents(docs, embedding_model)  # Create FAISS index from docs
    db.save_local(DB_FAISS_PATH)                      # Save FAISS DB locally
    print(f"✅ Embeddings saved to FAISS at: {DB_FAISS_PATH}")

# Run when script is executed directly
if __name__ == "__main__":
    embed_and_store()
