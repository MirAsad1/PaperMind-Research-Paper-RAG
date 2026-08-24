import os
import streamlit as st
from rag.chain import build_chain

st.set_page_config(
    page_title="PaperMind",
    page_icon="📄",
    layout="wide",
)

@st.cache_resource
def load_chain():
    return build_chain()

chain = load_chain()

# --- Sidebar ---
with st.sidebar:
    st.title("📄 PaperMind - Research Paper Q&A Assistant")
    st.markdown("---")

    st.markdown("### 🤖 Model Info")
    st.markdown("- **LLM:** - ")
    st.markdown("- **Provider:** Groq")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector Store:** ChromaDB")

    st.markdown("---")

    st.markdown("### 📁 Papers Loaded")
    papers_dir = "data/research_papers"
    papers = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    if papers:
        for paper in papers:
            st.markdown(f"- 📄 {paper}")
    else:
        st.warning("No papers found in data/research_papers/")

    st.markdown("---")
    st.markdown("### ⬆️ Upload a New Paper")
    uploaded_file = st.file_uploader("Add a PDF to the knowledge base", type=["pdf"])

    if uploaded_file is not None:
        save_path = os.path.join(papers_dir, uploaded_file.name)

        if os.path.exists(save_path):
            st.info(f"'{uploaded_file.name}' is already in the knowledge base.")
        else:
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Processing {uploaded_file.name}..."):
                from rag.ingest import add_single_pdf
                num_chunks = add_single_pdf(save_path)

            st.success(f"Added {uploaded_file.name} ({num_chunks} chunks)")
            load_chain.clear()  # force chain to rebuild with new data
            st.rerun()

    st.markdown("---")
    st.markdown("### 💡 How to Use")
    st.markdown("""
1. Papers are already ingested
2. Type your questions
3. AI will semantically search relevant chunks
4. Sources will be shown below each answer
""")

    st.markdown("---")
    st.caption("Built with LangChain, Groq & Streamlit")

# --- Main Area ---
st.title("Ask About Your Research Papers")
st.caption("Powered by RAG — Retrieval Augmented Generation")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the research papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            docs = chain["retriever"].invoke(prompt)
            context = "\n\n".join(doc.page_content for doc in docs)
            formatted_prompt = chain["prompt"].format(context=context, question=prompt)

        def stream_response():
            for chunk in chain["llm"].stream(formatted_prompt):
                yield chunk.content

        answer = st.write_stream(stream_response)

        with st.expander("📚 Sources"):
            seen = set()
            for doc in docs:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                key = f"{source}-{page}"
                if key not in seen:
                    seen.add(key)
                    st.markdown(f"- **{source}** — Page {page}")

    st.session_state.messages.append({"role": "assistant", "content": answer})