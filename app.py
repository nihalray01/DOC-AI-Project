import streamlit as st

from src.pdf_processor import process_uploaded_pdfs
from src.rag_engine import RAGEngine
from src.vector_store import VectorStore

st.set_page_config(
    page_title="DocuMind AI - Document RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.main-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    background: linear-gradient(90deg, #1E88E5 0%, #42A5F5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-subtitle {
    text-align: center;
    font-size: 1.05rem;
    opacity: 0.8;
    margin-bottom: 1.5rem;
}

.status-card {
    background-color: rgba(30, 136, 229, 0.05);
    padding: 0.8rem 1rem;
    border-radius: 8px;
    border-left: 4px solid #1E88E5;
    margin-bottom: 1rem;
}

.source-badge {
    background-color: rgba(0, 0, 0, 0.05);
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 600;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_vector_store() -> VectorStore:
    """
    Load embedding model and initialize FAISS vector store.
    """
    return VectorStore()


@st.cache_resource
def load_rag_engine() -> RAGEngine:
    """
    Initialize RAG Engine (auto-detects Groq or Gemini API key).
    """
    return RAGEngine()


def initialize_session_state() -> None:
    """
    Initialize Streamlit session state keys.
    """
    defaults = {
        "messages": [],
        "documents_processed": False,
        "processed_file_names": [],
        "total_chunks": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_project() -> None:
    """
    Reset application state and clearing cached vector store.
    """
    st.cache_resource.clear()
    st.session_state.messages = []
    st.session_state.documents_processed = False
    st.session_state.processed_file_names = []
    st.session_state.total_chunks = 0
    st.rerun()


initialize_session_state()

vector_store = load_vector_store()
rag_engine = load_rag_engine()
provider_info = rag_engine.get_provider_info()

st.markdown(
    '<div class="main-title">📚 DocuMind AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">Intelligent PDF Document Assistant using Retrieval-Augmented Generation (RAG)</div>',
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("📄 Document Control")

    uploaded_files = st.file_uploader(
        "Upload PDF document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload text-based PDF files to index and query.",
    )

    st.subheader("⚙️ Chunking Settings")

    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Maximum characters per document snippet.",
    )

    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=50,
        max_value=400,
        value=200,
        step=50,
        help="Overlapping text between consecutive chunks to maintain context.",
    )

    process_button = st.button(
        "🚀 Process & Index Documents",
        type="primary",
        use_container_width=True,
    )

    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one PDF file before processing.")
        elif chunk_overlap >= chunk_size:
            st.error("Chunk overlap must be strictly less than chunk size.")
        else:
            try:
                with st.spinner("Extracting text and generating FAISS embeddings..."):
                    chunks, errors = process_uploaded_pdfs(
                        uploaded_files=uploaded_files,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                    if not chunks:
                        st.error(
                            "No readable text extracted. Scanned PDFs require OCR."
                        )
                    else:
                        vector_store.build_index(chunks)
                        st.session_state.documents_processed = True
                        st.session_state.processed_file_names = [
                            f.name for f in uploaded_files
                        ]
                        st.session_state.total_chunks = len(chunks)
                        st.session_state.messages = []
                        st.success(
                            f"Successfully indexed {len(chunks)} chunks from {len(uploaded_files)} document(s)."
                        )

                    for err in errors:
                        st.warning(err)

            except Exception as error:
                st.error(f"Processing failed: {error}")

    st.divider()

    st.subheader("📊 System Status")

    # API Connection Card
    if rag_engine.is_configured():
        st.success(
            f"**{provider_info['provider']}** Connected\n\n"
            f"*Model*: `{provider_info['model']}`"
        )
    else:
        st.warning(
            "API key not set in `.env`.\n\n"
            "Running in **Retrieval-Only Fallback Mode**."
        )

    # Vector DB Card
    if st.session_state.documents_processed:
        st.info(
            f"**Vector Store**: Ready\n\n"
            f"• **Files**: {len(st.session_state.processed_file_names)}\n"
            f"• **Chunks**: {st.session_state.total_chunks}"
        )
        with st.expander("📁 Indexed Files"):
            for name in st.session_state.processed_file_names:
                st.caption(f"• {name}")
    else:
        st.caption("Vector store is empty.")

    st.divider()

    if st.button("🗑️ Reset All Data", use_container_width=True):
        reset_project()


# Welcome guide if no documents processed
if not st.session_state.documents_processed:
    st.info("👈 Upload PDF documents in the sidebar and click **Process & Index Documents** to get started.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1. 📄 Extract & Chunk")
        st.write("Reads text page-by-page from PDFs and splits into overlapping context windows.")

    with col2:
        st.markdown("### 2. 🧠 Embed & Index")
        st.write("Converts snippets into 384-d embeddings using `all-MiniLM-L6-v2` and indexes with FAISS.")

    with col3:
        st.markdown("### 3. 🎯 Grounded Answer")
        st.write("Retrieves top matching context and generates accurate LLM answers with citation metadata.")


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(message["sources"], 1):
                    st.markdown(
                        f"**Source {idx}**: `{src['file_name']}` — Page {src['page_number']} "
                        f"(Similarity Score: `{src['similarity_score']:.3f}`)"
                    )
                    st.caption(src["text"])
                    st.divider()


# Chat input box
question = st.chat_input(
    "Ask a question about your uploaded documents...",
    disabled=not st.session_state.documents_processed,
)


if question:
    cleaned_q = question.strip()

    if cleaned_q:
        st.session_state.messages.append({"role": "user", "content": cleaned_q})

        with st.chat_message("user"):
            st.markdown(cleaned_q)

        try:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving document context and generating answer..."):
                    retrieved_chunks = vector_store.search(
                        question=cleaned_q,
                        top_k=5,
                        minimum_score=0.15,
                    )

                    answer = rag_engine.generate_answer(
                        question=cleaned_q,
                        retrieved_chunks=retrieved_chunks,
                    )

                    st.markdown(answer)

                    if retrieved_chunks:
                        with st.expander("🔍 View Retrieved Sources"):
                            for idx, src in enumerate(retrieved_chunks, 1):
                                st.markdown(
                                    f"**Source {idx}**: `{src['file_name']}` — Page {src['page_number']} "
                                    f"(Similarity Score: `{src['similarity_score']:.3f}`)"
                                )
                                st.caption(src["text"])
                                st.divider()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": retrieved_chunks,
                }
            )

        except Exception as error:
            error_msg = f"An error occurred while answering: {error}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )