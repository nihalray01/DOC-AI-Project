from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class VectorStore:
    """
    A simple FAISS-based vector database for document retrieval.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: Any = None
        self.chunks: list[dict] = []

    def _encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Convert document texts into normalized embeddings.
        """
        if hasattr(self.model, "encode_document"):
            embeddings = self.model.encode_document(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        return np.asarray(embeddings, dtype="float32")

    def _encode_query(
        self,
        question: str,
    ) -> np.ndarray:
        """
        Convert a user question into a normalized embedding.
        """
        if hasattr(self.model, "encode_query"):
            embedding = self.model.encode_query(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            embedding = self.model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        return np.asarray(embedding, dtype="float32")

    def build_index(
        self,
        chunks: list[dict],
    ) -> None:
        """
        Create a FAISS index from document chunks.
        """
        if not chunks:
            raise ValueError(
                "No document chunks were provided."
            )

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self._encode_documents(texts)

        if embeddings.ndim != 2:
            raise ValueError(
                "Invalid embedding shape received."
            )

        embedding_dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            embedding_dimension
        )
        self.index.add(embeddings)
        self.chunks = chunks

    def search(
        self,
        question: str,
        top_k: int = 5,
        minimum_score: float = 0.15,
    ) -> list[dict]:
        """
        Retrieve document chunks relevant to a question.
        """
        if self.index is None or not self.chunks:
            raise ValueError(
                "Vector index has not been created."
            )

        if not question.strip():
            return []

        valid_top_k = min(
            max(top_k, 1),
            len(self.chunks),
        )

        query_embedding = self._encode_query(
            question.strip()
        )

        scores, indices = self.index.search(
            query_embedding,
            valid_top_k,
        )

        results: list[dict] = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            numeric_score = float(score)

            if numeric_score < minimum_score:
                continue

            chunk = self.chunks[index_position].copy()
            chunk["similarity_score"] = numeric_score
            results.append(chunk)

        return results

    def is_ready(self) -> bool:
        """
        Check whether documents have been indexed.
        """
        return (
            self.index is not None
            and len(self.chunks) > 0
        )