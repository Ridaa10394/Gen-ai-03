---
title: NotebookLM RAG
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# NotebookLM RAG

Chat with any PDF using Retrieval-Augmented Generation. Built for **Assignment 03 — Google NotebookLM RAG**.

## Live demo
_(deploy to Hugging Face Spaces and paste the link here)_

## Stack

| Layer | Choice | Tier |
|---|---|---|
| UI | Streamlit | OSS |
| LLM | Groq `llama-3.3-70b-versatile` | Free |
| Embeddings | fastembed (`BAAI/bge-small-en-v1.5`, 384d, runs locally) | Free / no API |
| Vector DB | Qdrant Cloud | Free (1 GB) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | OSS |
| PDF parsing | `pypdf` | OSS |

## How it works

1. User uploads a PDF in the sidebar.
2. The PDF is parsed page-by-page and split into chunks (1000 chars, 200 overlap).
3. Each chunk is embedded locally with fastembed and upserted to a per-session Qdrant collection.
4. When the user asks a question, the query is embedded, the top-k similar chunks are retrieved, and Groq is prompted with **strict instructions to answer only from the retrieved context**.
5. Page-numbered sources are shown in an expandable panel below each answer.

## Get free API keys

- **Groq**: https://console.groq.com/keys → "Create API Key" (Google login; no card)
- **Qdrant Cloud**: https://cloud.qdrant.io → create free cluster → copy the cluster URL and API key

## Local setup

```bash
git clone <this-repo>
cd Gen-ai-03

python -m venv venv
venv\Scripts\activate         # Windows PowerShell
# source venv/bin/activate    # macOS / Linux

pip install -r requirements.txt

copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
# then fill in the three keys

streamlit run app.py
```

App opens at http://localhost:8501.

## Deploy to Hugging Face Spaces (free)

1. Create a Space at https://huggingface.co/new-space → SDK: **Streamlit**.
2. Push this repo to the Space's git remote.
3. In **Settings → Variables and secrets**, add three secrets:
   - `GROQ_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
4. The Space auto-builds and serves at `https://huggingface.co/spaces/<you>/<name>`.

## Project layout

```
app.py                Streamlit UI + chat loop
rag/
  embeddings.py       Local fastembed wrapper
  ingest.py           PDF load, chunk, embed, upsert to Qdrant
  retrieve.py         Query embed + Qdrant similarity search
  chat.py             Prompt template + Groq call
requirements.txt
.env.example
```
