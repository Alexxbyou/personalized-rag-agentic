from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from App.backend.agents.memory import memory_update_agent
from App.backend.agents.profile import profile_agent
from App.backend.agents.prompt import ContextRelevanceFilter, PromptConstructor, build_final_prompt
from App.backend.agents.retrieval import retrieval_planner_agent
from App.backend.agents.safety import context_safety_agent, query_safety_agent
from App.backend.chat_history.manager import ChatHistoryManager
from App.backend.data_loader.loaders import (
    DEFAULT_DATA_DIR,
    load_config,
    load_interactions_df,
    load_knowledge_documents,
    load_memory_store,
    load_profiles_df,
)
from App.backend.llm.openai_client import OpenAIService
from App.backend.models.domain import ChatProfile, Memory, Query
from App.backend.rag.knowledge_store import KnowledgeStore


class RAGState(TypedDict):
    user_id: str
    query_text: str
    chat_profile: Optional[ChatProfile]
    query_obj: Optional[Query]
    knowledge_context: list
    safety_result: Optional[dict]
    final_prompt: str
    answer: str
    chat_history_context: list


class BackendRuntime:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.config = load_config(self.data_dir)
        self.llm_service = OpenAIService()
        self.llm_service.configure_dspy()

        self.profiles_df = load_profiles_df(self.data_dir)
        self.interactions_df = load_interactions_df(self.data_dir)
        self.memory_store = load_memory_store(self.data_dir)
        self.knowledge_store = KnowledgeStore(
            documents=load_knowledge_documents(self.data_dir),
            embedding_fn=self.llm_service.get_embedding,
        )
        self.chat_history_manager = ChatHistoryManager(api_key=self.llm_service.api_key)
        self.context_filter = ContextRelevanceFilter()
        self.prompt_constructor = PromptConstructor()
        self.graph = self._build_rag_graph()
        self.app = self.graph.compile()

    def start_session(self, user_id: str) -> ChatProfile:
        top_k_memories = self.config.get("profile_retrieval", {}).get("top_k_profile_memories", 3)
        profile = profile_agent(user_id, self.profiles_df, self.interactions_df)
        if profile.has_consent():
            profile.memories = memory_update_agent(user_id, self.memory_store, top_k=top_k_memories)
        return profile

    def _init_query_node(self, state: RAGState) -> dict:
        profile = state["chat_profile"] or ChatProfile(user_id=state["user_id"])
        query_obj = Query(
            query=state["query_text"],
            user_id=state["user_id"],
            preferred_style=profile.preferred_style,
            memory=list(profile.memories),
        )
        return {"query_obj": query_obj}

    def _history_search_node(self, state: RAGState) -> dict:
        try:
            history = self.chat_history_manager.search_history(
                state["query_text"],
                state["user_id"],
                limit=3,
            )
        except Exception:
            history = []

        query_obj = state["query_obj"]
        assert query_obj is not None
        query_obj.past_relevant_conversation_context = history
        return {"chat_history_context": history, "query_obj": query_obj}

    def _retrieval_plan_node(self, state: RAGState) -> dict:
        profile = state["chat_profile"] or ChatProfile(user_id=state["user_id"])
        query_obj = state["query_obj"]
        assert query_obj is not None
        interests = profile.interests if profile.has_consent() else []
        knowledge = retrieval_planner_agent(query_obj, self.knowledge_store, interests)
        query_obj.knowledge_context = knowledge
        return {"knowledge_context": knowledge, "query_obj": query_obj}

    def _safety_check_node(self, state: RAGState) -> dict:
        return {"safety_result": query_safety_agent(state["query_text"], self.llm_service)}

    def _context_filter_node(self, state: RAGState) -> dict:
        query_obj = state["query_obj"]
        assert query_obj is not None

        filtered_knowledge, filtered_history = self.context_filter(
            query=query_obj.query,
            knowledge_contexts=query_obj.knowledge_context,
            history_contexts=query_obj.past_relevant_conversation_context,
        )

        query_obj.knowledge_context = [
            {**doc, "text": context_safety_agent(doc["text"])}
            for doc in filtered_knowledge
        ]
        query_obj.past_relevant_conversation_context = filtered_history
        query_obj.memory = [
            memory.with_text(context_safety_agent(memory.memory_text))
            for memory in query_obj.memory
        ]
        return {"knowledge_context": query_obj.knowledge_context, "query_obj": query_obj}

    def _prompt_build_node(self, state: RAGState) -> dict:
        profile = state["chat_profile"] or ChatProfile(user_id=state["user_id"])
        query_obj = state["query_obj"]
        assert query_obj is not None
        max_features = self.config.get("prompting", {}).get("max_profile_features_in_prompt", 6)
        final_prompt = build_final_prompt(query_obj, profile, max_features=max_features)
        query_obj.final_prompt = final_prompt
        return {"final_prompt": final_prompt, "query_obj": query_obj}

    def _generate_node(self, state: RAGState) -> dict:
        answer = self.llm_service.call_llm(
            [
                {"role": "system", "content": state["final_prompt"]},
                {"role": "user", "content": state["query_text"]},
            ]
        )
        return {"answer": answer}

    def _memory_store_node(self, state: RAGState) -> dict:
        try:
            event_id = f"E{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.chat_history_manager.add_conversation(
                state["user_id"],
                event_id,
                state["query_text"],
                state["answer"],
            )
        except Exception:
            pass
        return {}

    @staticmethod
    def _safety_router(state: RAGState) -> str:
        safety = state.get("safety_result", {}) or {}
        return "safe" if safety.get("is_safe", False) else "unsafe"

    def _build_rag_graph(self) -> StateGraph:
        graph = StateGraph(RAGState)
        graph.add_node("init_query", self._init_query_node)
        graph.add_node("history_search", self._history_search_node)
        graph.add_node("retrieval_plan", self._retrieval_plan_node)
        graph.add_node("safety_check", self._safety_check_node)
        graph.add_node("context_filter", self._context_filter_node)
        graph.add_node("prompt_build", self._prompt_build_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("memory_store", self._memory_store_node)

        graph.add_edge(START, "init_query")
        graph.add_edge("init_query", "history_search")
        graph.add_edge("history_search", "retrieval_plan")
        graph.add_edge("retrieval_plan", "safety_check")
        graph.add_conditional_edges(
            "safety_check",
            self._safety_router,
            {"safe": "context_filter", "unsafe": END},
        )
        graph.add_edge("context_filter", "prompt_build")
        graph.add_edge("prompt_build", "generate")
        graph.add_edge("generate", "memory_store")
        graph.add_edge("memory_store", END)
        return graph

    def invoke_query(self, chat_profile: ChatProfile, query_text: str) -> dict:
        initial_state: RAGState = {
            "user_id": chat_profile.user_id,
            "query_text": query_text,
            "chat_profile": chat_profile,
            "query_obj": None,
            "knowledge_context": [],
            "safety_result": None,
            "final_prompt": "",
            "answer": "",
            "chat_history_context": [],
        }
        final_state = self.app.invoke(initial_state)

        if not final_state.get("answer"):
            safety = final_state.get("safety_result", {}) or {}
            final_state["answer"] = (
                "I'm sorry, I can only help with financial topics. "
                f"{safety.get('reason', '')}"
            ).strip()
        return final_state

    def run_query(self, chat_profile: ChatProfile, query_text: str) -> str:
        return self.invoke_query(chat_profile, query_text)["answer"]
