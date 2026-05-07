"""Thread-safe in-process session store. Replace with Redis for multi-instance deployments."""
from __future__ import annotations

import threading
import time
import uuid

from app.models.session import Session


class SessionStore:
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            now = time.time()
            if session_id and session_id in self._sessions:
                existing = self._sessions[session_id]
                existing.last_seen_at = now
                return existing
            new_id = session_id or uuid.uuid4().hex
            created = Session(session_id=new_id, created_at=now, last_seen_at=now)
            self._sessions[new_id] = created
            return created

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_seen_at = time.time()
            return session

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def purge_expired(self) -> list[str]:
        now = time.time()
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if now - s.last_seen_at > self._ttl]
            for sid in expired:
                del self._sessions[sid]
            return expired


_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    global _store
    if _store is None:
        from app.core.config import settings

        _store = SessionStore(settings.session_ttl_seconds)
    return _store
