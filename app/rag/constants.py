"""Retrieval / answering hyperparameters. Behavior-affecting; change via code review, not env."""
from __future__ import annotations

# Hybrid retrieval
TOP_K: int = 20
BM25_WEIGHT: float = 0.4
SEMANTIC_WEIGHT: float = 0.6
RRF_K: int = 60

# Chat history kept per session and forwarded to the LLM
MAX_HISTORY_TURNS: int = 4  # both user + assistant pairs
HISTORY_LIMIT: int = 20  # hard cap on stored turns

# LLM generation
LLM_TEMPERATURE: float = 0.1
LLM_MAX_OUTPUT_TOKENS: int = 1024
