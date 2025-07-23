import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv(find_dotenv())
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Load LLM from Groq
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=512
    )

# Custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Load vectorstore
@st.cache_resource
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
    return db

# Setup QA Chain
@st.cache_resource
def setup_qa_chain():
    vectorstore = load_vectorstore()
    llm = load_llm()
    prompt = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Streamlit UI
st.set_page_config(page_title="RAG PDF Chatbot", layout="centered")
st.title("📄 Chat with your PDF (RAG + Groq + FAISS)")

qa_chain = setup_qa_chain()

query = st.text_input("Ask a question based on your PDF:")
if query:
    with st.spinner("Generating answer..."):
        response = qa_chain.invoke({"query": query})
        st.markdown("### 🧠 Answer")
        st.success(response["result"])

        with st.expander("📚 Source Documents"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Document {i+1}:**")
                st.code(doc.page_content.strip()[:1000])  # limit length for readability
