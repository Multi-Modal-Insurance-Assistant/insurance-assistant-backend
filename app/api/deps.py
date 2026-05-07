from __future__ import annotations

from fastapi import Cookie, Depends, Response

from app.core.config import settings
from app.core.session_store import SessionStore, get_session_store
from app.models.session import Session


def get_session(
    response: Response,
    session_cookie: str | None = Cookie(default=None, alias=settings.session_cookie_name),
    store: SessionStore = Depends(get_session_store),
) -> Session:
    """Resolve the per-request Session from the cookie. Issues a new cookie when one is missing."""
    session = store.get_or_create(session_cookie)
    if session.session_id != session_cookie:
        response.set_cookie(
            key=settings.session_cookie_name,
            value=session.session_id,
            httponly=True,
            samesite="lax",
            max_age=settings.session_ttl_seconds,
            path="/",
        )
    return session
