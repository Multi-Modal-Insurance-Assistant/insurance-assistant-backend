"""Application settings.

Holds **deployment** and **secret** values only:
- Secrets (API keys)
- Environment-dependent values (URLs, hosts, ports, persist paths, CORS)
- Operational toggles (log level, reload, OCR language packs)
- Business limits required by the product spec (file size / count / PDF pages)

Algorithmic hyperparameters (chunk size, top-k, BM25 / semantic weights, RRF k,
chat history depth) live in module-level `constants.py` files so that changing
them goes through code review rather than ops.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # --- OpenAI ---
    openai_api_key: str = Field(..., description="OpenAI API key (sk-...)")
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-large"

    # --- Chroma ---
    chroma_persist_dir: Path = Path("data/chroma")

    # --- Session ---
    session_cookie_name: str = "iasid"
    session_ttl_seconds: int = 60 * 60 * 24

    # --- OCR ---
    ocr_languages: str = "eng+vie"

    # --- Business limits (spec-fixed, env-tunable as escape hatch) ---
    max_files_per_session: int = 2
    max_file_size_bytes: int = 5 * 1024 * 1024
    max_pdf_pages: int = 20

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return stripped  # let pydantic JSON-parse
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
