from __future__ import annotations

import dspy

from App.backend.models.domain import ChatProfile, Query


PERSONA_INSTRUCTIONS = {
    "concise_technical": "Respond in a concise, technical style. Use bullet points, comparisons, and precise financial terminology. Avoid marketing language.",
    "empathetic_simple": "Respond in a warm, empathetic tone. Use simple language, step-by-step guidance, and encouraging words. Avoid jargon.",
    "executive_brief": "Respond with a short executive summary. Lead with risk flags, then key metrics, then recommendations. Be direct and brief.",
    "helpful": "Respond as a helpful financial assistant. Be clear, accurate, and professional.",
}


class ContextRelevanceJudge(dspy.Signature):
    query: str = dspy.InputField(desc="The user's financial question")
    contexts: str = dspy.InputField(desc="Numbered contexts prefixed with [K<i>] or [H<i>]")
    relevant_indices: str = dspy.OutputField(desc="Comma-separated labels like K0, K2, H0 or none")


class ContextRelevanceFilter(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.judge = dspy.ChainOfThought(ContextRelevanceJudge)

    def forward(
        self,
        query: str,
        knowledge_contexts: list[dict],
        history_contexts: list[str],
    ) -> tuple[list[dict], list[str]]:
        numbered: list[str] = []
        for index, doc in enumerate(knowledge_contexts):
            numbered.append(f"[K{index}] {doc['title']}: {doc['text'][:300]}")
        for index, ctx in enumerate(history_contexts):
            numbered.append(f"[H{index}] {ctx[:300]}")

        if not numbered:
            return knowledge_contexts, history_contexts

        result = self.judge(query=query, contexts="\n".join(numbered))
        indices_str = result.relevant_indices.strip().lower()
        if indices_str == "none" or not indices_str:
            return [], []

        filtered_knowledge: list[dict] = []
        filtered_history: list[str] = []
        for token in indices_str.replace(",", " ").split():
            token = token.strip().strip("[]")
            if token.startswith("k") and token[1:].isdigit():
                idx = int(token[1:])
                if 0 <= idx < len(knowledge_contexts):
                    filtered_knowledge.append(knowledge_contexts[idx])
            elif token.startswith("h") and token[1:].isdigit():
                idx = int(token[1:])
                if 0 <= idx < len(history_contexts):
                    filtered_history.append(history_contexts[idx])
        return filtered_knowledge, filtered_history


class PersonalizedFinancialQA(dspy.Signature):
    persona_style: str = dspy.InputField(desc="Communication style")
    user_context: str = dspy.InputField(desc="User profile summary and preferences")
    knowledge_context: str = dspy.InputField(desc="Retrieved financial knowledge documents")
    query: str = dspy.InputField(desc="The user's financial question")
    memory_context: str = dspy.InputField(desc="Relevant user memories and prior conversations")
    answer: str = dspy.OutputField(desc="Personalized financial answer grounded in the knowledge context")


class PromptConstructor(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cot = dspy.ChainOfThought(PersonalizedFinancialQA)

    def forward(self, query_obj: Query, profile: ChatProfile) -> str:
        knowledge_text = "\n".join(
            f"[{doc['doc_id']}] {doc['title']}: {doc['text']}" for doc in query_obj.knowledge_context
        ) or "No relevant knowledge documents found."

        memory_parts = [f"[{memory.memory_type}] {memory.memory_text}" for memory in query_obj.memory]
        memory_parts.extend(f"[chat_history] {ctx}" for ctx in query_obj.past_relevant_conversation_context)
        memory_text = "\n".join(memory_parts) if memory_parts else "No prior memory context."

        user_context = profile.profile_summary if profile.has_consent() else "General user, no personalization."
        result = self.cot(
            persona_style=profile.preferred_style if profile.has_consent() else "helpful",
            user_context=user_context,
            knowledge_context=knowledge_text,
            query=query_obj.query,
            memory_context=memory_text,
        )
        return result.answer


def build_final_prompt(query_obj: Query, profile: ChatProfile, max_features: int = 6) -> str:
    style = profile.preferred_style if profile.has_consent() else "helpful"
    persona_instruction = PERSONA_INSTRUCTIONS.get(style, PERSONA_INSTRUCTIONS["helpful"])

    system_parts = [f"You are a personalized financial assistant.\n\n**Style**: {persona_instruction}"]
    if profile.has_consent() and profile.profile_summary:
        system_parts.append(f"\n**User Profile**: {profile.profile_summary}")

    if query_obj.knowledge_context:
        knowledge_lines = [
            f"- [{doc['doc_id']}] {doc['title']}: {doc['text']}"
            for doc in query_obj.knowledge_context[:max_features]
        ]
        system_parts.append("\n**Knowledge Context**:\n" + "\n".join(knowledge_lines))

    memory_lines = [
        f"- [{memory.memory_type}] {memory.memory_text}"
        for memory in query_obj.memory[:max_features]
    ]
    memory_lines.extend(
        f"- [past_conversation] {ctx}"
        for ctx in query_obj.past_relevant_conversation_context[:max_features]
    )
    if memory_lines:
        system_parts.append("\n**Memory Context**:\n" + "\n".join(memory_lines))

    return "\n".join(system_parts)
