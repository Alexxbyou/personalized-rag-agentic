from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta


INTEREST_CATEGORY_MAP = {
    "wealth management": "wealth",
    "etf": "wealth",
    "savings": "retail_banking",
    "budgeting": "retail_banking",
    "liquidity": "treasury",
    "cash management": "treasury",
}


@dataclass
class Memory:
    memory_id: str
    user_id: str
    memory_type: str
    memory_text: str
    confidence: float
    expiry_days: int
    last_update: str

    def is_expired(self, today: date | None = None) -> bool:
        reference_date = today or datetime.now().date()
        last = datetime.strptime(self.last_update, "%Y-%m-%d").date()
        return reference_date > last + timedelta(days=self.expiry_days)

    def priority_score(self) -> float:
        type_weight = 2.0 if self.memory_type == "episodic" else 1.0
        return type_weight * self.confidence

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "memory_text": self.memory_text,
            "confidence": self.confidence,
            "expiry_days": self.expiry_days,
            "last_update": self.last_update,
        }

    def with_text(self, memory_text: str) -> "Memory":
        payload = self.to_dict()
        payload["memory_text"] = memory_text
        return Memory.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict) -> "Memory":
        return cls(
            memory_id=payload["memory_id"],
            user_id=payload["user_id"],
            memory_type=payload["memory_type"],
            memory_text=payload["memory_text"],
            confidence=float(payload["confidence"]),
            expiry_days=int(payload["expiry_days"]),
            last_update=payload["last_update"],
        )


@dataclass
class ChatProfile:
    user_id: str
    preferred_style: str = "helpful"
    segment: str = "unknown"
    interests: list[str] = field(default_factory=list)
    channel: str = ""
    consent_personalization: bool = False
    profile_summary: str = ""
    clicked_docs: list[str] = field(default_factory=list)
    interaction_history: list[dict] = field(default_factory=list)
    memories: list[Memory] = field(default_factory=list)

    def get_interest_categories(self) -> list[str]:
        categories = set()
        for interest in self.interests:
            mapped = INTEREST_CATEGORY_MAP.get(interest.strip().lower())
            if mapped:
                categories.add(mapped)
        return sorted(categories)

    def has_consent(self) -> bool:
        return self.consent_personalization


@dataclass
class Query:
    query: str
    user_id: str
    preferred_style: str = "helpful"
    knowledge_context: list[dict] = field(default_factory=list)
    past_relevant_conversation_context: list[str] = field(default_factory=list)
    memory: list[Memory] = field(default_factory=list)
    final_prompt: str = ""
