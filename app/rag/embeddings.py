from __future__ import annotations

from collections.abc import Sequence

from app.core.config import settings
from app.rag.client import get_openai_client
from app.rag.retry import openai_retry


@openai_retry
def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed a batch of texts via OpenAI embeddings (model + dimension set in config)."""
    if not texts:
        return []
    client = get_openai_client()
    result = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=list(texts),
    )
    return [item.embedding for item in result.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
