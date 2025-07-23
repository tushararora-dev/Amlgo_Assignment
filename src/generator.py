# src/generator.py

import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

#  Load Groq API key from .env
load_dotenv(find_dotenv())
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ✅ Load LLM (Groq + LLama3)
def load_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-8b-8192",
        temperature=0.3,
        max_tokens=512
    )

#  Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """
You are a legal document assistant designed to help users understand Terms & Conditions, Privacy Policies, and Legal Contracts. Your responses must be accurate, helpful, and based strictly on the provided documentation.

## Instructions:
1. **Source-based responses**: Answer ONLY using information from the provided context below
2. **Accuracy**: Never invent, assume, or extrapolate information not explicitly stated
3. **Clarity**: Explain legal concepts in plain language while maintaining precision
4. **Citations**: Reference specific sections or clauses when possible
5. **Limitations**: If information is insufficient or unclear, acknowledge this honestly

## Context Documents:
{context}

## User Question:
{question}

## Response Guidelines:
- If the answer is found in the context: Provide a clear, comprehensive answer
- If partially answered: Explain what you can determine and what remains unclear
- If not answered: State "I don't have information about this in the provided documents"
- For legal interpretations: Clarify that this is based on the document language, not legal advice
- For ambiguous terms: Explain multiple possible interpretations if present in the context

## Your Response:
"""


# ✅ Return LangChain prompt object
def get_prompt():
    """
    Return the custom prompt used to instruct the LLM.
    It ensures that the answer remains factual and grounded in retrieved docs.
    """
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
