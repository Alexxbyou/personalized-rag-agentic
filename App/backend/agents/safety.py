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


SAFETY_REJECTION_MESSAGES = {
    "irrelevant": "I'm sorry, your query does not appear to be related to financial topics. I can only assist with financial questions.",
    "harmful": "I'm sorry, your query contains harmful or inappropriate content. I am unable to process this request.",
    "sensitive": "I'm sorry, your query contains sensitive or personally identifiable information. Please remove any PII and try again.",
}


def query_safety_agent(query: str, llm_service: OpenAIService) -> dict:
    """Use LLM to check whether the query is financial-domain relevant,
    free of PII/sensitive data, and free of harmful content.

    Returns dict with:
        is_safe (bool): True if the query passes all checks.
        rejection_type (str|None): "irrelevant", "harmful", or "sensitive" when unsafe.
        reason (str): Explanation.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a safety classifier. Evaluate the user query on THREE dimensions:\n"
                "1. RELEVANCE — Is the query related to financial topics "
                "(banking, investments, savings, budgeting, treasury, "
                "risk management, insurance, loans, funds, markets)?\n"
                "2. HARMFUL — Does the query contain harmful or inappropriate content?\n"
                "3. SENSITIVE — Does the query contain personally identifiable information "
                "(PII) or other sensitive data (e.g. account numbers, NRIC, emails)?\n\n"
                "If the query fails ANY dimension, set is_safe to false and set "
                "rejection_type to one of: \"irrelevant\", \"harmful\", or \"sensitive\".\n"
                "If multiple issues exist, pick the most severe: harmful > sensitive > irrelevant.\n\n"
                "Respond with ONLY a JSON object:\n"
                "{\"is_safe\": true/false, \"rejection_type\": null or \"irrelevant\"/\"harmful\"/\"sensitive\", \"reason\": \"...\"}"
            ),
        },
        {"role": "user", "content": query},
    ]
    response = llm_service.call_llm(messages)
    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        if "true" in response.lower():
            result = {"is_safe": True, "rejection_type": None, "reason": "Query appears financial-domain relevant."}
        else:
            result = {"is_safe": False, "rejection_type": "irrelevant", "reason": response}

    # Ensure rejection_type key always exists
    result.setdefault("rejection_type", None if result.get("is_safe") else "irrelevant")
    return result


def safety_rejection_message(safety_result: dict) -> str:
    """Return a user-facing rejection message based on the safety rejection type."""
    rejection_type = safety_result.get("rejection_type", "irrelevant")
    message = SAFETY_REJECTION_MESSAGES.get(rejection_type, SAFETY_REJECTION_MESSAGES["irrelevant"])
    reason = safety_result.get("reason", "")
    return f"{message} (Details: {reason})" if reason else message


def context_safety_agent(context_text: str) -> str:
    cleaned = context_text
    for pattern, replacement in PII_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned
