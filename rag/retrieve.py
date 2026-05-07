from qdrant_client import QdrantClient

from rag.embeddings import embed_query


def search(
    qdrant: QdrantClient,
    collection_name: str,
    query: str,
    k: int = 5,
) -> list[dict]:
    vector = embed_query(query)
    result = qdrant.query_points(
        collection_name=collection_name,
        query=vector,
        limit=k,
        with_payload=True,
    )
    return [
        {
            "text": h.payload["text"],
            "page": h.payload["page"],
            "score": h.score,
        }
        for h in result.points
    ]
