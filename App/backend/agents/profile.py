from __future__ import annotations

import pandas as pd

from App.backend.models.domain import ChatProfile


def profile_agent(user_id: str, profiles_df: pd.DataFrame, interactions_df: pd.DataFrame) -> ChatProfile:
    row = profiles_df[profiles_df["user_id"] == user_id]
    if row.empty:
        return ChatProfile(user_id=user_id)

    row = row.iloc[0]
    consent = str(row.get("consent_personalization", "N")).strip().upper() == "Y"

    user_interactions = interactions_df[interactions_df["user_id"] == user_id]
    interaction_history = user_interactions.to_dict("records")

    clicked_docs: list[str] = []
    for _, interaction in user_interactions.iterrows():
        docs = str(interaction.get("clicked_docs", "")).split("|")
        clicked_docs.extend([doc.strip() for doc in docs if doc.strip()])

    interests = [item.strip() for item in str(row.get("interests", "")).split(";") if item.strip()]

    return ChatProfile(
        user_id=user_id,
        preferred_style=row.get("preferred_style", "helpful") if consent else "helpful",
        segment=row.get("segment", "unknown"),
        interests=interests if consent else [],
        channel=row.get("channel", ""),
        consent_personalization=consent,
        profile_summary=row.get("profile_summary", "") if consent else "",
        clicked_docs=clicked_docs if consent else [],
        interaction_history=interaction_history,
    )
