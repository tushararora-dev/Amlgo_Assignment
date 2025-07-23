# src/pipeline.py

from langchain.chains import RetrievalQA
from src.retriever import get_retriever
from src.generator import load_llm, get_prompt

def build_qa_chain():
    """
    Combine retriever and LLM with the custom prompt into a RetrievalQA chain.
    """
    llm = load_llm()
    retriever = get_retriever()
    prompt = get_prompt()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Injects all chunks into the prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Quick CLI test
if __name__ == "__main__":
    chain = build_qa_chain()
    query = input("💬 Enter your query: ")
    result = chain.invoke({"query": query})

    print("\n🧠 Answer:\n", result["result"])
    print("\n📚 Source Documents:\n", result["source_documents"])
