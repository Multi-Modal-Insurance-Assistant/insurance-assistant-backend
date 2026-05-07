"""Prompt loader. Each prompt lives in a YAML file in this package and is
parsed once at startup into a Pydantic model so we get type-checked access
plus a single source of truth for prompt text.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_PROMPTS_DIR = Path(__file__).parent


class AnswerPrompt(BaseModel):
    version: int = Field(..., ge=1)
    system_instruction: str
    fallback_no_context: str


def _load_yaml(filename: str) -> dict:
    path = _PROMPTS_DIR / filename
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def load_answer_prompt() -> AnswerPrompt:
    return AnswerPrompt(**_load_yaml("answer.yaml"))


__all__ = ["AnswerPrompt", "load_answer_prompt"]
