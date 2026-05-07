import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.embeddings import EMBED_DIM, embed_documents

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_pdf_pages(file) -> list[tuple[int, str]]:
    reader = PdfReader(file)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[dict] = []
    for page_num, text in pages:
        if not text.strip():
            continue
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "page": page_num})
    return chunks


def ensure_collection(client: QdrantClient, name: str) -> None:
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


def ingest_pdf(file, qdrant: QdrantClient, collection_name: str) -> int:
    pages = extract_pdf_pages(file)
    chunks = chunk_pages(pages)
    if not chunks:
        return 0
    vectors = embed_documents([c["text"] for c in chunks])
    ensure_collection(qdrant, collection_name)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": chunk["text"], "page": chunk["page"]},
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    qdrant.upsert(collection_name=collection_name, points=points)
    return len(points)
