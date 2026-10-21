from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load the embedding model once when this module is imported.
model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts: list[str]):
    """Convert a list of text strings into normalized vectors."""
    if not texts:
        raise ValueError("texts must not be empty.")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings


def create_query_embedding(query: str):
    """Convert a single user question into a normalized vector."""
    if not query or not query.strip():
        raise ValueError("query must not be empty.")

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return query_embedding
