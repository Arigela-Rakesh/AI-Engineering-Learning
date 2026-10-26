import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. Load the embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


# 2. Create embeddings for document chunks
def create_embeddings(chunks: list[str]) -> np.ndarray:
    """
    Convert document chunks into normalized embedding vectors.
    """

    if not chunks:
        raise ValueError("Chunks cannot be empty.")

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


# 3. Build a FAISS index
def build_faiss_index(
    chunks: list[str],
) -> tuple[faiss.IndexFlatIP, list[str]]:
    """
    Create a FAISS index from document chunks.
    """

    embeddings = create_embeddings(chunks)

    # Create an index using inner product similarity
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    # Add document embeddings to the index
    index.add(embeddings)

    return index, chunks


# 4. Search for relevant document chunks
def search_documents(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: list[str],
    top_k: int = 3,
) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks for a query.
    """

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not chunks:
        return []

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # Convert the query into an embedding
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Search the FAISS index
    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks)),
    )

    results = []

    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index == -1:
            continue

        results.append(
            {
                "chunk": chunks[chunk_index],
                "score": float(score),
                "chunk_index": int(chunk_index),
            }
        )

    return results


# 5. Save the FAISS index and chunks
def save_vector_store(
    index: faiss.IndexFlatIP,
    chunks: list[str],
    directory: str = "rag/vector_store",
) -> None:
    """
    Save the FAISS index and its associated text chunks.
    """

    store_path = Path(directory)

    store_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    faiss.write_index(
        index,
        str(store_path / "index.faiss"),
    )

    with open(
        store_path / "chunks.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )


# 6. Load the saved vector store
def load_vector_store(
    directory: str = "rag/vector_store",
) -> tuple[faiss.IndexFlatIP, list[str]]:
    """
    Load the saved FAISS index and text chunks.
    """

    store_path = Path(directory)

    index = faiss.read_index(
        str(store_path / "index.faiss"),
    )

    with open(
        store_path / "chunks.json",
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    return index, chunks