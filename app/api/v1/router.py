"""Aggregator for v1 routes. main.py mounts this under /api/v1."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import chat, session, upload

api_router = APIRouter()
api_router.include_router(session.router)
api_router.include_router(upload.router)
api_router.include_router(chat.router)
