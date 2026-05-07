from fastembed import TextEmbedding

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


_embedder: TextEmbedding | None = None


def get_embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _embedder


def embed_documents(texts: list[str]) -> list[list[float]]:
    embedder = get_embedder()
    return [vec.tolist() for vec in embedder.embed(texts)]


def embed_query(text: str) -> list[float]:
    embedder = get_embedder()
    return list(next(embedder.query_embed(text)).tolist())
