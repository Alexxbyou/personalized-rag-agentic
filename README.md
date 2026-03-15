# Personalized RAG


## Overview
This project implements a **Personalized RAG Financial Assistant** — an agentic chatbot that retrieves relevant financial knowledge and adapts its communication style to each user's profile. The system combines Retrieval-Augmented Generation with user memory, profile-driven persona selection, and safety guardrails to deliver contextually grounded, personalized financial guidance.

The project has 3 main code deliverables:
* **Runnable Notebook**: a runnable `Jupyter Notebook` with Project background, analysis and key results [link](Notebook/LLM-Chatbot.ipynb)
* **Analytic Report**: the report section of the runnable Notebook is saved as pdf file in [file](Notebook/Analytic%20Report.pdf)
* **Backend App**: a code base for backend app, which host a uvicorn fastapi server that receive stateless chat API call. [link](App/backend)


## How to run
1. Activate the prepared environment and make sure `.env` contains `OPENAI_API_KEY`.

2. Prepare conda environment, this can be used for `Notebook/LLM-Chatbot.ipynb` running as well.
```bash
conda create -n prag_env python=3.11 -y
conda activate prag_env
pip install -r requriements
```

3. Start the FastAPI service from the project root.

```bash
uvicorn App.backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

4. Optional health check:

```bash
curl "http://127.0.0.1:8000/healthz"
```

5. Send a test request to `POST /query`.

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



## Folder Structure
```text
.
├── App/                          # application source root
│   └── backend/                  # modular FastAPI backend converted from the notebook MVP
│       ├── agents/               # profile, memory, retrieval, safety, and prompt agents
│       ├── api/                  # FastAPI app, dependency wiring, and request/response schemas
│       ├── chat_history/         # mem0-based chat history storage and retrieval
│       ├── data_loader/          # loaders for CSV, JSON, and JSONL seed data
│       ├── llm/                  # OpenAI client wrapper and DSPy setup
│       ├── models/               # core domain models such as Memory, ChatProfile, and Query
│       ├── orchestration/        # LangGraph pipeline orchestration
│       ├── rag/                  # FAISS-backed knowledge retrieval layer
│       └── tests/                # backend integration tests
├── Data/                         # seed data for profiles, memories, logs, corpus, and config
├── Docs/                         # assignment instructions and rubric
└── Notebook/                     # original notebook MVP and exploratory implementation
```
