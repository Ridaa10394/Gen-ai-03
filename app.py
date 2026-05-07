import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient

from rag.chat import answer
from rag.embeddings import get_embedder
from rag.ingest import ingest_pdf
from rag.retrieve import search

load_dotenv(override=True)


def get_secret(key: str) -> str:
    val = os.getenv(key, "")
    if val:
        return val
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


GROQ_API_KEY = get_secret("GROQ_API_KEY")
QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")

st.set_page_config(page_title="NotebookLM RAG", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a document and ask questions grounded in its contents.")

if not (GROQ_API_KEY and QDRANT_URL and QDRANT_API_KEY):
    st.error(
        "Missing API keys. Set `GROQ_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` "
        "in a local `.env` file or in Streamlit / Hugging Face Space secrets."
    )
    st.stop()


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@st.cache_resource
def get_groq_client() -> Groq:
    return Groq(api_key=GROQ_API_KEY)


@st.cache_resource(show_spinner="Loading embedding model...")
def warm_embedder():
    return get_embedder()


qdrant = get_qdrant_client()
groq_client = get_groq_client()
warm_embedder()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:8]
if "collection" not in st.session_state:
    st.session_state.collection = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded and uploaded.name != st.session_state.doc_name:
        with st.spinner(f"Indexing {uploaded.name}..."):
            collection_name = f"doc_{st.session_state.session_id}"
            try:
                qdrant.delete_collection(collection_name)
            except Exception:
                pass
            n_chunks = ingest_pdf(uploaded, qdrant, collection_name)
            st.session_state.collection = collection_name
            st.session_state.doc_name = uploaded.name
            st.session_state.messages = []
        if n_chunks:
            st.success(f"Indexed {n_chunks} chunks from {uploaded.name}")
        else:
            st.warning("No extractable text found in this PDF.")

    if st.session_state.doc_name:
        st.caption(f"Active document: **{st.session_state.doc_name}**")
    else:
        st.info("Upload a PDF to begin.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question about your document...")
if query:
    if not st.session_state.collection:
        st.warning("Please upload a PDF first.")
        st.stop()
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            chunks = search(qdrant, st.session_state.collection, query, k=5)
            response = answer(groq_client, query, chunks)
        st.markdown(response)
        if chunks:
            with st.expander("Sources"):
                for c in chunks:
                    st.markdown(f"**Page {c['page']}** — score `{c['score']:.3f}`")
                    snippet = c["text"][:300].replace("\n", " ")
                    st.caption(snippet + ("..." if len(c["text"]) > 300 else ""))
    st.session_state.messages.append({"role": "assistant", "content": response})
