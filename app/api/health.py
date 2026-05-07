"""Liveness probe. Mounted outside the API version prefix on purpose — health is platform, not API."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
