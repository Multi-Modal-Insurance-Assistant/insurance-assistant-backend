from __future__ import annotations

from app.core.session_store import SessionStore
from app.models.session import ChatTurn


def test_session_store_creates_new_when_no_id() -> None:
    store = SessionStore(ttl_seconds=3600)
    s1 = store.get_or_create(None)
    s2 = store.get_or_create(None)
    assert s1.session_id != s2.session_id


def test_session_store_returns_existing() -> None:
    store = SessionStore(ttl_seconds=3600)
    s1 = store.get_or_create(None)
    s2 = store.get_or_create(s1.session_id)
    assert s1 is s2


def test_session_store_isolation() -> None:
    store = SessionStore(ttl_seconds=3600)
    a = store.get_or_create(None)
    b = store.get_or_create(None)
    a.history.append(ChatTurn(role="user", content="hello"))
    assert len(b.history) == 0
