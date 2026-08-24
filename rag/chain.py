import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from rag.retriever import load_retriever

load_dotenv()


def build_prompt():
    template = """You are a helpful research assistant. Use the following context from research papers to answer the question.
If the answer is not in the context, say "I couldn't find relevant information in the provided papers."

Context:
{context}

Question: {question}

Answer:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )


def build_chain():
    llm = ChatGroq(
        model=os.getenv("MODEL_NAME"),
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
        streaming=True,
    )

    retriever = load_retriever()
    prompt = build_prompt()

    return {"llm": llm, "retriever": retriever, "prompt": prompt}