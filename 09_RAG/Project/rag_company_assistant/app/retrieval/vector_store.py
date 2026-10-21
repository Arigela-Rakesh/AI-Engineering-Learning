import faiss
import numpy as np

from app.embeddings.embedder import create_embeddings


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def add_documents(self, documents: list[dict]) -> None:
        if not documents:
            raise ValueError("documents must not be empty.")

        texts = [document["text"] for document in documents]
        embeddings = create_embeddings(texts)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        self.chunks = [dict(document) for document in documents]

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Add documents before searching.")

        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if min_score is not None and not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1 and 1.")

        from app.embeddings.embedder import create_query_embedding
        import numpy as np

        query_embedding = np.asarray(
            create_query_embedding(query),
            dtype=np.float32,
        )

        result_count = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, result_count)

        results = []

        for position in range(result_count):
            chunk_index = int(indices[0][position])

            if chunk_index == -1:
                continue

            score = float(scores[0][position])

            if min_score is not None and score < min_score:
                continue

            result = dict(self.chunks[chunk_index])
            result["score"] = round(score, 4)
            results.append(result)

        return results
