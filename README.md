# 📄 PaperMind - Research Paper Q&A — RAG System

A Retrieval-Augmented Generation (RAG) application that lets you ask questions about research papers in natural language. Built with LangChain, ChromaDB, and Groq LLM.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2.16-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Demo

Ask questions like:
- *"What is this research paper about?"*
- *"Summarize the key findings"*
- *"What does the paper say about HEC?"*
- *"What methodology was used?"*

The system retrieves relevant chunks from the papers and generates accurate, context-aware answers.

---

## 🏗️ Architecture

```
PDFs → PyMuPDF → Text Chunks → Embeddings → ChromaDB
                                                 ↓
User Query → Embed Query → Similarity Search → Top 5 Chunks
                                                 ↓
                              Groq LLM (Llama 3) → Answer
```

---

## 📁 Project Structure

```
research-rag/
├── rag/
│   ├── __init__.py       # Package initialization
│   ├── ingest.py         # PDF loading, chunking, embedding
│   ├── retriever.py      # ChromaDB similarity search
│   └── chain.py          # LangChain QA chain with Groq
├── data/
│   ├── papers/           # Place your PDF files here
│   └── chroma_db/        # Auto-generated vector store
├── app.py                # Streamlit web interface
├── requirements.txt      # Project dependencies
├── .env.example          # Environment variables template
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/MirAsad1/PaperMind-Research-Paper-RAG.git
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
Open `.env` and add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key at [console.groq.com](https://console.groq.com)

### 5. Add your research papers
Place your PDF files inside the `data/papers/` folder.

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
| Groq (Llama 3) | LLM for answer generation |
| PyMuPDF | PDF text extraction |
| Streamlit | Web UI |

---

## 💡 Key Features

- **Semantic Search** — Finds relevant content by meaning, not just keywords
- **Source Attribution** — Shows which paper and page each answer came from
- **Chat History** — Maintains conversation context within a session
- **Local Vector Store** — No external database cost, runs fully on your machine
- **Multi-paper Support** — Add multiple PDFs and query across all of them

---

## 📌 Future Improvements

- [ ] Support for multi-paper comparison queries
- [ ] Add metadata filters (search by author, year)
- [ ] Export chat history as PDF
- [ ] Live paper ingestion from arXiv API

---

## 📄 License

MIT License — feel free to use and modify.