from __future__ import annotations

import json
import re

from App.backend.llm.openai_client import OpenAIService


PII_PATTERNS = [
    (r"\b[STFGstfg]\d{7}[A-Za-z]\b", "[NRIC_REDACTED]"),
    (r"\b\d{10,16}\b", "[ACCOUNT_REDACTED]"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_REDACTED]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_REDACTED]"),
    (r"\$\d+[\d,.]*\s*(in|from|to)\s+account", "[MONETARY_REF_REDACTED]"),
]


def query_safety_agent(query: str, llm_service: OpenAIService) -> dict:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a safety classifier. Determine if the user query is related to "
                "financial topics (banking, investments, savings, budgeting, treasury, "
                "risk management, insurance, loans, funds, markets). "
                'Respond with ONLY a JSON object: {"is_safe": true/false, "reason": "..."}'
            ),
        },
        {"role": "user", "content": query},
    ]
    response = llm_service.call_llm(messages)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        if "true" in response.lower():
            return {"is_safe": True, "reason": "Query appears financial-domain relevant."}
        return {"is_safe": False, "reason": response}


def context_safety_agent(context_text: str) -> str:
    cleaned = context_text
    for pattern, replacement in PII_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned
