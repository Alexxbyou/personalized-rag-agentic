from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from App.backend.api.dependencies import get_runtime
from App.backend.api.main import app
from App.backend.api.schemas import ChatProfilePayload
from App.backend.orchestration.pipeline import BackendRuntime


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is required for live integration test")
def test_query_endpoint_live_openai() -> None:
    runtime = BackendRuntime()
    profile = runtime.start_session("U001")
    payload = {
        "chat_profile": ChatProfilePayload.from_domain(profile).model_dump(),
        "query": "Compare money market funds and short-duration bond funds",
    }

    app.dependency_overrides[get_runtime] = lambda: runtime
    try:
        with TestClient(app) as client:
            response = client.post("/query", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "U001"
    assert body["answer"]
    assert body["safety_result"]["is_safe"] is True
    assert body["rejection_type"] is None
    assert any(doc_id in body["knowledge_doc_ids"] for doc_id in ["DOC001", "DOC002"])


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is required for live integration test")
def test_query_rejected_irrelevant() -> None:
    """Off-topic query should be rejected with rejection_type='irrelevant'."""
    runtime = BackendRuntime()
    profile = runtime.start_session("U001")
    payload = {
        "chat_profile": ChatProfilePayload.from_domain(profile).model_dump(),
        "query": "What is the whether like in Singapore today?",
    }

    app.dependency_overrides[get_runtime] = lambda: runtime
    try:
        with TestClient(app) as client:
            response = client.post("/query", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["safety_result"]["is_safe"] is False
    assert body["rejection_type"] == "irrelevant"
    assert "financial" in body["answer"].lower()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY is required for live integration test")
def test_query_rejected_sensitive() -> None:
    """Query containing PII should be rejected with rejection_type='sensitive'."""
    runtime = BackendRuntime()
    profile = runtime.start_session("U001")
    payload = {
        "chat_profile": ChatProfilePayload.from_domain(profile).model_dump(),
        "query": "Transfer $5000 from my account 1234567890 to john.doe@email.com",
    }

    app.dependency_overrides[get_runtime] = lambda: runtime
    try:
        with TestClient(app) as client:
            response = client.post("/query", json=payload)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["safety_result"]["is_safe"] is False
    assert body["rejection_type"] == "sensitive"
    assert "sensitive" in body["answer"].lower() or "PII" in body["answer"]
