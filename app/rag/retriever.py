from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from app.models.session import Session
from app.rag.constants import BM25_WEIGHT, RRF_K, SEMANTIC_WEIGHT, TOP_K
from app.rag.embeddings import embed_query
from app.rag.store import collection_for


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any]


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "")]


def rebuild_bm25(session: Session, ids: list[str], texts: list[str]) -> None:
    if not texts:
        session.bm25 = None
        session.bm25_doc_ids = []
        return
    tokenized = [tokenize(t) for t in texts]
    session.bm25 = BM25Okapi(tokenized)
    session.bm25_doc_ids = list(ids)


def _ensure_bm25_loaded(session: Session) -> None:
    """Rebuild BM25 from persisted Chroma data if it was lost (e.g. server restart)."""
    if session.bm25 is not None:
        return
    coll = collection_for(session.session_id)
    if coll.count() == 0:
        return
    data = coll.get(include=["documents"])
    rebuild_bm25(session, ids=list(data.get("ids", [])), texts=list(data.get("documents", [])))


def _semantic_search(session_id: str, query: str, k: int) -> list[Hit]:
    coll = collection_for(session_id)
    total = coll.count()
    if total == 0:
        return []
    qvec = embed_query(query)
    res = coll.query(
        query_embeddings=[qvec],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
    )
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    hits: list[Hit] = []
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
        sim = max(0.0, 1.0 - float(dist))
        hits.append(Hit(chunk_id=chunk_id, text=doc, score=sim, metadata=meta or {}))
    return hits


def _bm25_search(session: Session, query: str, k: int) -> list[Hit]:
    _ensure_bm25_loaded(session)
    if not session.bm25 or not session.bm25_doc_ids:
        return []
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = session.bm25.get_scores(tokens)
    if not len(scores):
        return []
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    selected_ids = [session.bm25_doc_ids[i] for i in ranked if scores[i] > 0]
    if not selected_ids:
        return []
    coll = collection_for(session.session_id)
    fetched = coll.get(ids=selected_ids, include=["documents", "metadatas"])
    by_id = {
        fid: (doc, meta)
        for fid, doc, meta in zip(
            fetched.get("ids", []),
            fetched.get("documents", []),
            fetched.get("metadatas", []),
            strict=False,
        )
    }
    max_score = max(scores) or 1.0
    hits: list[Hit] = []
    for i in ranked:
        chunk_id = session.bm25_doc_ids[i]
        if scores[i] <= 0 or chunk_id not in by_id:
            continue
        doc, meta = by_id[chunk_id]
        hits.append(
            Hit(chunk_id=chunk_id, text=doc, score=float(scores[i]) / max_score, metadata=meta or {})
        )
    return hits


def hybrid_search(session: Session, query: str, k: int | None = None) -> list[Hit]:
    """Hybrid search via Reciprocal Rank Fusion of cosine semantic + BM25 keyword."""
    k = k or TOP_K
    semantic = _semantic_search(session.session_id, query, k=k * 2)
    keyword = _bm25_search(session, query, k=k * 2)

    fused: dict[str, dict[str, Any]] = {}

    for rank, hit in enumerate(semantic, start=1):
        slot = fused.setdefault(hit.chunk_id, {"hit": hit, "score": 0.0})
        slot["score"] += SEMANTIC_WEIGHT / (RRF_K + rank)
        slot["hit"] = hit

    for rank, hit in enumerate(keyword, start=1):
        slot = fused.setdefault(hit.chunk_id, {"hit": hit, "score": 0.0})
        slot["score"] += BM25_WEIGHT / (RRF_K + rank)

    ordered = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:k]
    out: list[Hit] = []
    for entry in ordered:
        original: Hit = entry["hit"]
        out.append(
            Hit(
                chunk_id=original.chunk_id,
                text=original.text,
                score=float(entry["score"]),
                metadata=original.metadata,
            )
        )
    return out
