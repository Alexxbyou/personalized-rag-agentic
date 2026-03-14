from __future__ import annotations

from App.backend.models.domain import Query
from App.backend.rag.knowledge_store import KnowledgeStore


def retrieval_planner_agent(
    query_obj: Query,
    knowledge_store: KnowledgeStore,
    interests: list[str],
    top_k: int = 3,
) -> list[dict]:
    if interests:
        personalized_results = knowledge_store.search_personalized(query_obj.query, interests, top_k=top_k)
        generic_results = knowledge_store.search(query_obj.query, top_k=top_k)

        deduped: list[dict] = []
        seen_doc_ids: set[str] = set()
        for result in personalized_results + generic_results:
            if result["doc_id"] in seen_doc_ids:
                continue
            seen_doc_ids.add(result["doc_id"])
            deduped.append(result)
        deduped.sort(key=lambda item: item["score"], reverse=True)
        return deduped

    return knowledge_store.search(query_obj.query, top_k=top_k)
