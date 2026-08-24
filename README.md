# 📄 PaperMind - Research Paper Q&A — RAG System

A Retrieval-Augmented Generation (RAG) application that lets you ask questions about research papers in natural language. Built with LangChain, ChromaDB, and Groq LLM. 
Featuring hybrid retrieval (semantic + keyword search) and real-time streaming answers.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Demo

Ask questions like:
- *"What is this research paper about?"*
- *"Summarize the key findings"*
- *"What does the paper say about Impacts of Covid-19?"*
- *"What methodology was used?"*

Upload your own PDF directly from the sidebar, and the system indexes it on the fly — no manual setup needed. Answers stream in token-by-token, with sources (paper + page number) shown for every response.

---

## 🏗️ Architecture

```
PDFs → PyMuPDF → Text Chunks → Embeddings → ChromaDB
                                                 ↓
User Query → Hybrid Retrieval (BM25 + Semantic Search) → Top Chunks
                                                 ↓
                                    Groq LLM (openai/gpt-oss-20b) → Streamed Answer
```

Retrieval combines two methods for better accuracy:
- **Semantic search** — finds chunks by meaning, using sentence-transformer embeddings
- **BM25 keyword search** — catches exact terms, acronyms, and names that embeddings can miss

Results from both are merged using an Ensemble Retriever (Reciprocal Rank Fusion), giving more robust results than semantic search alone.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for a detailed breakdown of the system design.

---

## 📁 Project Structure

```
research-rag/
├── rag/
│   ├── __init__.py       # Package initialization
│   ├── ingest.py         # PDF loading, chunking, embedding + single-PDF ingestion for uploads
│   ├── retriever.py      # Hybrid retriever (BM25 + ChromaDB semantic search)
│   └── chain.py          # LLM, retriever, and prompt setup (streaming-enabled)
├── data/
│   ├── research_papers/  # Place your PDF files here
│   └── chroma_db/        # Auto-generated vector store
├── app.py                # Streamlit web interface
├── requirements.txt      # Project dependencies
├── .env.example           # Environment variables template
├── ARCHITECTURE.md        # Detailed system design & data flow
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/MirAsad1/papermind-research-paper-rag.git
cd research-rag
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
```
Open `.env` and add your Groq API key and model name:
```
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=openai/gpt-oss-20b
```
Get a free API key at [console.groq.com](https://console.groq.com).

> **Note:** Groq periodically deprecates and updates its supported models. If you get a `model_not_found` error, check [console.groq.com/docs/models](https://console.groq.com/docs/models) for the current list of active models and update `MODEL_NAME` accordingly.

### 5. Add your research papers
Place your PDF files inside the `data/research_papers/` folder — or skip this and upload PDFs directly from the app's sidebar after launch.

### 6. Ingest the papers
```bash
python rag/ingest.py
```

### 7. Run the app
```bash
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| LangChain | RAG pipeline framework |
| ChromaDB | Local vector database |
| Sentence Transformers | Free text embeddings (`all-MiniLM-L6-v2`) |
| BM25 (rank_bm25) | Keyword-based retrieval, combined with semantic search |
| Groq (`openai/gpt-oss-20b`) | LLM for streamed answer generation |
| PyMuPDF | PDF text extraction |
| Streamlit | Web UI |

---

## 💡 Key Features

- **Hybrid Retrieval** — Combines semantic (embedding) search with BM25 keyword search for more accurate, robust results
- **Streaming Answers** — Responses generate token-by-token in real time, instead of waiting for the full answer
- **In-App PDF Upload** — Add new papers directly from the sidebar; they're chunked, embedded, and made queryable immediately, without restarting the app
- **Source Attribution** — Shows which paper and page each answer came from
- **Chat History** — Maintains conversation context within a session
- **Local Vector Store** — No external database cost, runs fully on your machine
- **Multi-paper Support** — Add multiple PDFs and query across all of them

---

## 📊 Evaluation

Retrieval quality was manually compared before and after adding hybrid search (BM25 + semantic), using the same question against the same paper.

**Example — "What are the main impacts of COVID-19 on students' academic experience?"**

| | Semantic-only retrieval | Hybrid retrieval (BM25 + semantic) |
|---|---|---|
| Coverage | Missed GPA impact and online-course enrollment shifts | Captured GPA drop (0.17 pts) and reduced online-course enrollment (-4 pts) in addition to graduation delays and study-time changes |
| Result | Partial answer | More complete, better-grounded answer |

This reflects a common RAG failure mode: pure semantic search can miss chunks containing specific terms or numeric findings that a keyword-aware retriever catches. Adding BM25 alongside semantic search noticeably improved answer completeness across several test questions.

*(A fuller automated evaluation — using an LLM-as-judge to score faithfulness and relevance per question — is a natural next step; see Future Improvements.)*

---

## 📌 Future Improvements

- [ ] Automated LLM-as-judge evaluation (faithfulness & relevance scoring)
- [ ] Cross-encoder re-ranking of retrieved chunks
- [ ] Support for multi-paper comparison queries
- [ ] Add metadata filters (search by author, year)
- [ ] Export chat history as PDF
- [ ] Live paper ingestion from arXiv API

---

## 📄 License

MIT License — feel free to use and modify.