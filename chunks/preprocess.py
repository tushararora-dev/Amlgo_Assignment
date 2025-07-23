import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv, find_dotenv

# Load environment variables 
load_dotenv(find_dotenv())

# Your single PDF file path
PDF_FILE = "data/AI_Training_Document.pdf"  

# Where to save the output chunks
CHUNK_PATH = "chunks/chunks.json"

# Chunk settings
CHUNK_SIZE = 300        # words per chunk
CHUNK_OVERLAP = 50      # overlapping words between chunks

# Load the PDF 
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

# Clean unwanted characters from text
def clean_text(text):
    return text.replace("\n", " ").replace("\t", " ").strip()

# Split into sentence-aware chunks
def create_chunks(text):
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    length = 0

    for sentence in sentences:
        words = sentence.split()
        if length + len(words) > CHUNK_SIZE:
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP:]  # overlap
            length = len(current)
        current.extend(words)
        length += len(words)

    if current:
        chunks.append(" ".join(current))

    return chunks

# Main function
def main():
    pages = load_pdf(PDF_FILE)
    print("✅ PDF loaded.")

    full_text = " ".join([clean_text(p.page_content) for p in pages])
    print("✅ Text cleaned.")

    split_chunks = create_chunks(full_text)
    print(f"✅ Created {len(split_chunks)} chunks.")

    # Wrap into LangChain Document format
    docs = [Document(page_content=chunk, metadata={"source": PDF_FILE}) for chunk in split_chunks]

    # Save chunks to JSON
    os.makedirs("chunks", exist_ok=True)
    with open(CHUNK_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            f,
            ensure_ascii=False,
            indent=2
        )

    print("✅ Chunks saved to:", CHUNK_PATH)

if __name__ == "__main__":
    main()
