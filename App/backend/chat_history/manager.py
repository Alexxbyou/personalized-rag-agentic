from __future__ import annotations

from mem0 import Memory as Mem0Memory


class ChatHistoryManager:
    def __init__(self, api_key: str, llm_model: str = "gpt-5-mini", embedding_model: str = "text-embedding-3-small") -> None:
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": llm_model,
                    "api_key": api_key,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": embedding_model,
                    "api_key": api_key,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "chat_history",
                    "embedding_model_dims": 1536,
                },
            },
            "version": "v1.1",
        }
        self.memory = Mem0Memory.from_config(config)

    def add_conversation(self, user_id: str, event_id: str, question: str, answer: str) -> None:
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        self.memory.add(conversation, user_id=user_id, metadata={"event_id": event_id})

    def search_history(self, query: str, user_id: str, limit: int = 3, threshold: float = 0.5) -> list[str]:
        results = self.memory.search(query=query, user_id=user_id, limit=limit)

        if isinstance(results, dict):
            entries = results.get("results", [])
        elif isinstance(results, list):
            entries = results
        else:
            entries = []

        matched: list[str] = []
        for entry in entries:
            score = entry.get("score", 0)
            if score >= threshold:
                matched.append(entry.get("memory", str(entry)))
        return matched
