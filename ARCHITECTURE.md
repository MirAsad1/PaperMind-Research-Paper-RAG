# Architecture

This document explains how PaperMind is structured internally: how data flows from a raw PDF to a streamed answer, and what each file in the `rag/` package is responsible for.

---

## System Overview

```
                    ┌─────────────────────┐
                    │   PDF (uploaded or  │
                    │  placed in folder)  │
                    └──────────┬──────────┘
                               │
                      PyMuPDF extraction
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Raw text per page │
                    └──────────┬──────────┘
                               │
                  RecursiveCharacterTextSplitter
                  (chunk_size=1000, overlap=100)
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Text chunks      │
                    └──────────┬──────────┘
                               │
              SentenceTransformer (all-MiniLM-L6-v2)
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ChromaDB (local)  │
                    │  persisted vectors  │
                    └──────────┬──────────┘
                               │
                     ═══ at query time ═══
                               │
                               ▼
              ┌────────────────────────────────┐
              │        User question           │
              └────────────────┬───────────────┘
                               │
              ┌────────────────┴──────────────────┐
              ▼                                   ▼
    ┌─────────────────────┐                ┌────────────────────┐
    │  Semantic search    │                │   BM25 keyword     │
    │  (Chroma similarity)│                │   search           │
    └──────────┬──────────┘                └──────────┬─────────┘
               │                                      │
               └──────────────────┬───────────────────┘
                                  ▼
                         Ensemble Retriever
                       (Reciprocal Rank Fusion,
                    weights: 0.4 BM25 / 0.6 semantic)
                                   │
                                   ▼
                         Top matching chunks
                                   │
                                   ▼
                      Prompt assembled with context
                                   │
                                   ▼
                     Groq LLM (openai/gpt-oss-20b)
                            streaming=True
                                   │
                                   ▼
              Answer streamed token by token to the UI
              (with source paper + page shown alongside)
```

---

## Component Breakdown

### `rag/ingest.py`

Handles turning PDFs into searchable vectors.

- `load_papers()`: loads every PDF in `data/research_papers/` using `PyMuPDFLoader`. Used for the initial bulk ingestion via `python rag/ingest.py`.
- `split_documents()`: splits loaded documents into overlapping chunks (1000 characters, 100 overlap) using `RecursiveCharacterTextSplitter`. Overlap helps avoid losing context at chunk boundaries.
- `save_to_chromadb()`: embeds chunks with a local sentence-transformer model and writes them into a persisted ChromaDB collection.
- `add_single_pdf()`: the incremental path used by the in-app file uploader. Loads and chunks one PDF, then appends it to the existing ChromaDB collection instead of rebuilding the whole index. This keeps uploads fast and avoids reprocessing papers that are already indexed.

### `rag/retriever.py`

Builds the hybrid retriever used at query time.

- Loads the persisted ChromaDB collection and wraps it as a semantic retriever (`search_type="similarity"`, `k=9`).
- Pulls the same stored chunks out of Chroma to build a `BM25Retriever`, which scores chunks by keyword overlap rather than embedding similarity.
- Combines both retrievers with `EnsembleRetriever`, which merges their rankings using Reciprocal Rank Fusion. Semantic search is weighted slightly higher (0.6 vs 0.4) since it generally produces more relevant results, but BM25 helps surface chunks containing exact terms, acronyms, or names that embeddings can miss.
- Rebuilt fresh on every chain load, so newly uploaded PDFs are automatically included in both the semantic and keyword indexes without any extra steps.

### `rag/chain.py`

Wires together the LLM, retriever, and prompt template.

- Defines the prompt template that instructs the model to answer only from the provided context, and to say so explicitly if the answer is not present in the retrieved chunks. This is the main defense against hallucination.
- Configures `ChatGroq` with `streaming=True`, so tokens can be consumed incrementally rather than waiting for the full response.
- Returns the LLM, retriever, and prompt as separate objects (not a single packaged chain), so `app.py` can control retrieval and generation as two distinct steps. This is what makes streaming possible: the retrieval step runs first and finishes, then the generation step streams its output separately.

### `app.py`

The Streamlit interface and orchestration layer.

- Renders the sidebar (model info, loaded papers, file uploader) and the main chat interface.
- On file upload, saves the PDF to `data/research_papers/`, calls `add_single_pdf()` to index it, clears the cached chain (`st.cache_resource`), and reruns the app so the new paper is immediately queryable.
- On each user question: retrieves chunks, formats the prompt, and streams the LLM's response using `st.write_stream()`.
- Displays deduplicated source citations (paper name and page number) for every answer, pulled directly from chunk metadata.

---

## Key Design Decisions

**Why hybrid retrieval instead of semantic search alone?**
Semantic search is good at matching meaning but can miss chunks containing exact keywords, acronyms, or numeric values that do not embed distinctively. Testing showed hybrid retrieval consistently surfaced more complete answers, for example capturing specific statistics that semantic-only search missed. See the Evaluation section in the README for a concrete before and after example.

**Why incremental ingestion for uploads instead of full reprocessing?**
Rebuilding the entire vector store on every upload would be slow and wasteful, and risks duplicate entries if the same papers are reprocessed. `add_single_pdf()` only processes the new file and appends it, keeping uploads fast regardless of how many papers are already indexed.

**Why separate the LLM, retriever, and prompt instead of using a packaged chain (like `RetrievalQA`)?**
Packaged chains are convenient but treat retrieval and generation as one opaque step, which makes streaming and fine-grained control difficult. Splitting them out trades a small amount of boilerplate for the ability to stream responses and inspect retrieved chunks independently.

**Why a local vector store (ChromaDB) instead of a hosted one?**
For a project of this scale, a local persisted ChromaDB instance avoids external costs and infrastructure, while still supporting incremental updates. This keeps the project runnable by anyone with just a Groq API key and no additional accounts.