from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_retriever():
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    semantic_retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 9},
    )

    # Pull the same chunks out of Chroma to build a keyword (BM25) retriever
    stored = db.get(include=["documents", "metadatas"])
    bm25_docs = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(stored["documents"], stored["metadatas"])
    ]

    if not bm25_docs:
        # No chunks yet — fall back to semantic-only
        return semantic_retriever

    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 9

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6],
    )

    return hybrid_retriever