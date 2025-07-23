# app.py

import streamlit as st
from src.pipeline import build_qa_chain
from src.retriever import DB_FAISS_PATH
import os

# Build LangChain QA chain
qa_chain = build_qa_chain()

# Streamlit setup
st.set_page_config(page_title="Legal RAG Chatbot ⚖️", layout="centered")
st.title("⚖️ Legal Chatbot for Terms, Privacy & Contracts")
st.markdown("**Amlgo Labs — Junior AI Engineer Assignment**")

# Sidebar info
with st.sidebar:
    st.header("📝 Project Info.")
    st.markdown("**Model in Use:** LLaMA 3 - 8B")
    
    # Show number of chunks loaded
    chunk_count = 0
    if os.path.exists(DB_FAISS_PATH):
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            db = FAISS.load_local(
                DB_FAISS_PATH,
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                allow_dangerous_deserialization=True
            )
            chunk_count = len(db.docstore._dict)
        except:
            pass
    st.markdown(f"**📄 Indexed Chunks:** {chunk_count}")

    # Reset button
    if st.button("♻️ Reset Chat"):
        st.session_state.messages = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (chat-style)
if user_input := st.chat_input("🔍 Ask about the legal documents..."):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔎 Analyzing your query..."):
            response = qa_chain.stream({"query": user_input})  # 🔁 Streaming response
            
            # Stream token-by-token
            full_response = ""
            response_box = st.empty()
            for chunk in response:
                full_response += chunk.get("result", "")
                response_box.markdown(full_response + "▌")

            response_box.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        # 📚 Sources
        with st.expander("📄 View Source Chunks"):
            for i, doc in enumerate(chunk.get("source_documents", [])):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.caption(f"📁 Source: {doc.metadata.get('source', 'N/A')}")
