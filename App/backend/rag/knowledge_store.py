from __future__ import annotations

from typing import Callable

import faiss
import numpy as np

from App.backend.models.domain import INTEREST_CATEGORY_MAP


class KnowledgeStore:
    def __init__(
        self,
        documents: list[dict],
        embedding_fn: Callable[[str], list[float]],
        embedding_dim: int = 1536,
    ) -> None:
        self.documents = documents
        self.embedding_fn = embedding_fn
        self.embedding_dim = embedding_dim
        self.metadata: dict[int, dict] = {}
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedding_dim))
        self._build_index()

    def _build_index(self) -> None:
        embeddings: list[list[float]] = []
        ids: list[int] = []

        for offset, doc in enumerate(self.documents, start=1):
            self.metadata[offset] = {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "category": doc["category"],
                "text": doc["text"],
            }
            embeddings.append(self.embedding_fn(f"{doc['title']}. {doc['text']}"))
            ids.append(offset)

        if not embeddings:
            return

        emb_array = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(emb_array)
        self.index.add_with_ids(emb_array, np.array(ids, dtype=np.int64))

    def search(
        self,
        query: str,
        top_k: int = 3,
        category_filter: list[str] | None = None,
    ) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        query_vec = np.array([self.embedding_fn(query)], dtype=np.float32)
        faiss.normalize_L2(query_vec)
        scores, ids = self.index.search(query_vec, self.index.ntotal)

        results: list[dict] = []
        for score, int_id in zip(scores[0], ids[0]):
            if int_id == -1:
                continue
            metadata = self.metadata[int(int_id)]
            if category_filter and metadata["category"] not in category_filter:
                continue
            results.append({**metadata, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def search_personalized(self, query: str, interests: list[str], top_k: int = 3) -> list[dict]:
        categories = sorted(
            {
                INTEREST_CATEGORY_MAP[interest.strip().lower()]
                for interest in interests
                if interest.strip().lower() in INTEREST_CATEGORY_MAP
            }
        )
        if not categories:
            return self.search(query, top_k=top_k)
        return self.search(query, top_k=top_k, category_filter=categories)
