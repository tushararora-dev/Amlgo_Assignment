import os 
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(find_dotenv())

# Load Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Step 1: Load LLM using Groq with LLama3
def load_llm():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=512
    )
    return llm

# Step 2: Define the custom prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Load vectorstore (FAISS)
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Step 4: Connect LLM + Prompt + Retriever into a QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Step 5: Get input from user
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

# Output result
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
