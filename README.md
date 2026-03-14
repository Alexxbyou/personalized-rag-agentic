# Personalized RAG


## Overview


## How to run
1. Activate the prepared environment and make sure `.env` contains `OPENAI_API_KEY`.

```bash
conda create -n prag_env python=3.11 -y
conda activate prag_env
pip install -r requriements
```

2. Start the FastAPI service from the project root.

```bash
uvicorn App.backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Send a test request to `POST /query`.

This API expects a serialized `ChatProfile` plus the user query. The example below matches the notebook-style `U001` session payload.

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "chat_profile": {
      "user_id": "U001",
      "preferred_style": "concise_technical",
      "segment": "mass_affluent",
      "interests": ["wealth management", "ETF"],
      "channel": "mobile",
      "consent_personalization": true,
      "profile_summary": "Prefers direct technical answers with product comparisons.",
      "clicked_docs": ["DOC001", "DOC002"],
      "interaction_history": [
        {
          "event_id": "E001",
          "user_id": "U001",
          "query": "Compare money market funds and short-duration bond funds",
          "clicked_docs": "DOC001|DOC002",
          "dwell_seconds": 140,
          "feedback": "too generic",
          "timestamp": "2026-03-01T10:15:00"
        }
      ],
      "memories": [
        {
          "memory_id": "M001",
          "user_id": "U001",
          "memory_type": "long_term_preference",
          "memory_text": "Dislikes marketing language and prefers bullet points.",
          "confidence": 0.91,
          "expiry_days": 180,
          "last_update": "2025-06-01"
        }
      ]
    },
    "query": "Compare money market funds and short-duration bond funds"
  }'
```

4. Optional health check:

```bash
curl "http://127.0.0.1:8000/healthz"
```


## Folder Structure
