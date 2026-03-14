from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from App.backend.models.domain import Memory


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"


def load_config(data_dir: Path = DEFAULT_DATA_DIR) -> dict:
    return json.loads((data_dir / "config_template.json").read_text())


def load_profiles_df(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "user_profiles.csv")


def load_interactions_df(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "interaction_log.csv")


def load_memory_store(data_dir: Path = DEFAULT_DATA_DIR) -> list[Memory]:
    payload = json.loads((data_dir / "memory_store_seed.json").read_text())
    return [Memory.from_dict(item) for item in payload]


def load_knowledge_documents(data_dir: Path = DEFAULT_DATA_DIR) -> list[dict]:
    documents: list[dict] = []
    with (data_dir / "knowledge_corpus.jsonl").open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    return documents
