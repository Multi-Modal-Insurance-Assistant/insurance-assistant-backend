from __future__ import annotations

import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.v1.router import api_router as v1_router
from app.core.config import settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging

logger = logging.getLogger("app")


def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="Insurance Assistant API",
        version="0.1.0",
        description=(
            "Multi-modal RAG backend for the Secure Insurance Assistant challenge. "
            "Per-session isolation via HttpOnly cookie + per-session Chroma collection; "
            "PDF / DOCX / Image ingestion (Tesseract OCR with eng+vie, pre-OCR upscale, "
            "--psm 6); hybrid retrieval (BM25 + cosine via Reciprocal Rank Fusion, top-20); "
            "OpenAI gpt-4.1-mini for grounded, citation-tagged answers; YAML-versioned "
            "prompt with explicit anti-hallucination rules (no category-to-named-entity "
            "transfer, no negative claims about source documents)."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["set-cookie"],
    )

    @app.exception_handler(AppError)
    async def _app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "code": exc.code},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    app.include_router(health_router)
    app.include_router(v1_router, prefix="/api/v1")

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
