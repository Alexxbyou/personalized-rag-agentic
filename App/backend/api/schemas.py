from __future__ import annotations

from pydantic import BaseModel, Field

from App.backend.models.domain import ChatProfile, Memory


class MemoryPayload(BaseModel):
    memory_id: str
    user_id: str
    memory_type: str
    memory_text: str
    confidence: float
    expiry_days: int
    last_update: str

    def to_domain(self) -> Memory:
        return Memory.from_dict(self.model_dump())

    @classmethod
    def from_domain(cls, memory: Memory) -> "MemoryPayload":
        return cls(**memory.to_dict())


class ChatProfilePayload(BaseModel):
    user_id: str
    preferred_style: str = "helpful"
    segment: str = "unknown"
    interests: list[str] = Field(default_factory=list)
    channel: str = ""
    consent_personalization: bool = False
    profile_summary: str = ""
    clicked_docs: list[str] = Field(default_factory=list)
    interaction_history: list[dict] = Field(default_factory=list)
    memories: list[MemoryPayload] = Field(default_factory=list)

    def to_domain(self) -> ChatProfile:
        return ChatProfile(
            user_id=self.user_id,
            preferred_style=self.preferred_style,
            segment=self.segment,
            interests=list(self.interests),
            channel=self.channel,
            consent_personalization=self.consent_personalization,
            profile_summary=self.profile_summary,
            clicked_docs=list(self.clicked_docs),
            interaction_history=list(self.interaction_history),
            memories=[memory.to_domain() for memory in self.memories],
        )

    @classmethod
    def from_domain(cls, profile: ChatProfile) -> "ChatProfilePayload":
        return cls(
            user_id=profile.user_id,
            preferred_style=profile.preferred_style,
            segment=profile.segment,
            interests=profile.interests,
            channel=profile.channel,
            consent_personalization=profile.consent_personalization,
            profile_summary=profile.profile_summary,
            clicked_docs=profile.clicked_docs,
            interaction_history=profile.interaction_history,
            memories=[MemoryPayload.from_domain(memory) for memory in profile.memories],
        )


class QueryRequest(BaseModel):
    chat_profile: ChatProfilePayload
    query: str


class QueryResponse(BaseModel):
    user_id: str
    answer: str
    safety_result: dict | None = None
    final_prompt: str = ""
    knowledge_doc_ids: list[str] = Field(default_factory=list)
