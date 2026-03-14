from __future__ import annotations

from App.backend.models.domain import Memory


def memory_update_agent(user_id: str, memory_store: list[Memory], top_k: int = 3) -> list[Memory]:
    user_memories = [memory for memory in memory_store if memory.user_id == user_id]
    active_memories = [memory for memory in user_memories if not memory.is_expired()]
    active_memories.sort(key=lambda memory: memory.priority_score(), reverse=True)
    return active_memories[:top_k]
