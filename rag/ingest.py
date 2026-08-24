import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

load_dotenv()

PAPERS_DIR = "data/research_papers"
CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_papers():
    docs = []
    for file in os.listdir(PAPERS_DIR):
        if file.endswith(".pdf"):
            path = os.path.join(PAPERS_DIR, file)
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
            print(f"Loaded: {file}")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")
    return chunks


def save_to_chromadb(chunks):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print("Saved to ChromaDB!")
    return db

def add_single_pdf(file_path):
    """Ingest a single PDF and append it to the existing ChromaDB (no re-processing of old papers)"""
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    if not docs:
        return 0

    chunks = split_documents(docs)

    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    db.add_documents(chunks)
    db.persist()

    print(f"Added {len(chunks)} chunks from {os.path.basename(file_path)}")
    return len(chunks)

def ingest():
    print("Starting ingestion...")
    docs = load_papers()
    if not docs:
        print("No PDFs found in data/research_papers/")
        return
    chunks = split_documents(docs)
    save_to_chromadb(chunks)
    print("Ingestion complete!")


if __name__ == "__main__":
    ingest()