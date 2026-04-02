from __future__ import annotations

from fastapi import Depends, FastAPI

from App.backend.api.dependencies import get_runtime
from App.backend.api.schemas import QueryRequest, QueryResponse
from App.backend.orchestration.pipeline import BackendRuntime


app = FastAPI(title="Personalized RAG Backend", version="0.1.0")


@app.get("/healthz")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(
    request: QueryRequest,
    runtime: BackendRuntime = Depends(get_runtime),
) -> QueryResponse:
    profile = request.chat_profile.to_domain()
    result = runtime.invoke_query(profile, request.query)
    query_obj = result.get("query_obj")
    knowledge_context = query_obj.knowledge_context if query_obj else []

    safety = result.get("safety_result") or {}
    return QueryResponse(
        user_id=profile.user_id,
        answer=result["answer"],
        safety_result=result.get("safety_result"),
        rejection_type=safety.get("rejection_type"),
        final_prompt=result.get("final_prompt", ""),
        knowledge_doc_ids=[doc["doc_id"] for doc in knowledge_context],
    )
