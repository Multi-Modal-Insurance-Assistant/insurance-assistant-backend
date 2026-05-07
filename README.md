# Insurance Assistant — Backend

FastAPI backend for the **Secure Multi-Modal Insurance Assistant** challenge. Handles ingestion of PDF / DOCX / Image (with OCR), per-session vector storage in Chroma, and grounded answers via OpenAI `gpt-4.1-mini` with inline citations.

## Stack

| Concern      | Choice                                                                                                               |
| ------------ | -------------------------------------------------------------------------------------------------------------------- |
| Runtime      | Python 3.11 + FastAPI + Uvicorn                                                                                      |
| Pkg manager  | [`uv`](https://docs.astral.sh/uv/)                                                                                   |
| Vector DB    | Chroma (`PersistentClient`, cosine, **one collection per session**)                                                  |
| Embeddings   | OpenAI `text-embedding-3-large` (multilingual, 3072d)                                                                |
| LLM          | OpenAI `gpt-4.1-mini` — non-streaming                                                                                |
| PDF parsing  | PyMuPDF (`pymupdf`) + Tesseract fallback for scanned pages                                                           |
| DOCX parsing | `python-docx` (paragraphs + tables walked in body order; citations anchor on the nearest heading text — Ctrl-F-able in Word) |
| Image OCR    | `pytesseract` with `eng+vie` language packs                                                                          |
| Chunker      | Pack consecutive blocks up to ~800 chars; **flush at heading boundaries** so a chunk never spans two sections        |
| Retrieval    | **Hybrid**: BM25 (`rank-bm25`) + cosine, fused via Reciprocal Rank Fusion (top-20, BM25=0.4 / cos=0.6)               |
| Prompt       | YAML-versioned ([`app/rag/prompts/answer.yaml`](app/rag/prompts/answer.yaml), currently **v4**) — bump `version` when semantics change |
| Retry        | `tenacity` only on transient failures (5xx, 429, network) — fail-fast on auth/quota                                  |
| Session      | HTTP-only cookie `iasid` → in-memory store + per-session Chroma collection                                           |
| OCR tuning   | Pre-OCR Lanczos upscale to ≥2000px long edge + Tesseract `--psm 6` (uniform block) for posters / table cells        |

## Project layout

```
insurance-assistant-backend/
├── pyproject.toml                # uv-managed deps + console script (`backend`)
├── .env.example                  # secrets + deployment vars only
├── app/
│   ├── main.py                   # FastAPI app factory + uvicorn entrypoint
│   ├── api/
│   │   ├── deps.py               # session cookie dependency
│   │   ├── health.py             # /health (unversioned)
│   │   └── v1/
│   │       ├── router.py         # aggregates /api/v1 routes
│   │       └── routes/
│   │           ├── session.py    # /api/v1/session
│   │           ├── upload.py     # /api/v1/upload
│   │           └── chat.py       # /api/v1/chat
│   ├── core/                     # cross-cutting infra
│   │   ├── config.py             # pydantic-settings (env-driven)
│   │   ├── exceptions.py         # AppError hierarchy → mapped to HTTP envelope
│   │   ├── logging.py
│   │   └── session_store.py      # in-memory, thread-safe
│   ├── models/                   # plain domain dataclasses
│   │   └── session.py            # Session / FileRecord / ChatTurn
│   ├── schemas/                  # Pydantic request/response IO
│   │   └── session.py / upload.py / chat.py
│   ├── services/                 # business orchestration (router → service → domain)
│   │   ├── upload_service.py     # validate → extract → chunk → embed → persist → BM25
│   │   └── chat_service.py       # checks → hybrid retrieve → LLM → history append
│   ├── ingestion/                # PDF / DOCX / Image extraction + chunking
│   │   ├── constants.py          # CHUNK_SIZE, CHUNK_OVERLAP, OCR_MIN_LONG_EDGE_PX, OCR_PSM, PDF_OCR_DPI
│   │   └── pdf.py / docx.py / image.py / extractor.py / chunker.py
│   └── rag/                      # retrieval + LLM
│       ├── constants.py          # TOP_K, BM25/SEMANTIC weights, RRF_K, history limits
│       ├── client.py             # OpenAI client singleton (max_retries=0; we own retry)
│       ├── retry.py              # `openai_retry` — only retries 5xx + 429 + network
│       ├── embeddings.py
│       ├── store.py              # Chroma client + per-session collection
│       ├── retriever.py          # hybrid BM25 + cosine, RRF fusion
│       ├── llm.py                # context format + chat.completions.create
│       └── prompts/
│           ├── __init__.py       # cached loader (Pydantic-validated)
│           └── answer.yaml       # versioned system prompt + fallback line
└── tests/                        # unit tests (chunker, extractor, session)
```

### Layering rules

```
api → services → (models, ingestion, rag) → core
```

- **api** does HTTP I/O only (parse request, validate cookies, serialise response).
- **services** orchestrate the workflow (validate, call ingestion, call rag, mutate session).
- **ingestion / rag** know about formats, embeddings, vector DB. No HTTP.
- **core** is cross-cutting: config, exceptions, logging, session store. No business logic.
- **models** are plain dataclasses, dependency-free.

### Where do I tune X?

| Knob                                               | Lives in                                       | Change via                             |
| -------------------------------------------------- | ---------------------------------------------- | -------------------------------------- |
| `OPENAI_API_KEY`, model names                      | `.env` → `app/core/config.py`                  | env / deploy                           |
| Server host / port / log level / CORS              | `.env` → `app/core/config.py`                  | env / deploy                           |
| Session TTL, cookie name, OCR languages            | `.env` → `app/core/config.py`                  | env / deploy                           |
| Max files / size / PDF pages (spec)                | `app/core/config.py` defaults                  | code (env-overridable as escape hatch) |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, OCR DPI / upscale / PSM | `app/ingestion/constants.py`               | **code review (PR)**                   |
| `TOP_K`, `BM25_WEIGHT`, `SEMANTIC_WEIGHT`, `RRF_K` | `app/rag/constants.py`                         | **code review (PR)**                   |
| LLM temperature / max tokens / history depth       | `app/rag/constants.py`                         | **code review (PR)**                   |
| System prompt + fallback wording                   | `app/rag/prompts/answer.yaml` (bump `version`) | **code review (PR)**                   |

Rationale: deployment values rotate by environment; algorithmic hyperparameters and prompt wording change retrieval/answer quality and should travel with code + tests.

## Setup

### 1. System packages (one-time)

OCR depends on the Tesseract binary plus language packs:

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-vie
```

### 2. Install `uv` (skip if you already have it)

Check first:

```bash
uv --version
```

If `uv: command not found`, install it (no sudo needed — drops a binary in `~/.local/bin`):

```bash
# macOS / Linux / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# or via pipx / pip if you prefer
pipx install uv     # or:  pip install --user uv
```

Open a new shell (or `source ~/.bashrc` / `~/.zshrc`) so `uv` is on `PATH`, then re-run `uv --version` to confirm. Full docs: <https://docs.astral.sh/uv/getting-started/installation/>.

### 3. Python environment

```bash
cd insurance-assistant-backend
uv sync
```

`uv` reads `.python-version` (3.11), downloads that interpreter if needed, creates `.venv/`, and installs everything pinned in `uv.lock`. No separate `python -m venv` step required.

### 4. Environment variables

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
```

Get a key from https://platform.openai.com/api-keys.

### 5. Run

```bash
# Console script (defined in pyproject.toml [project.scripts]):
uv run backend
# or directly via uvicorn:
uv run uvicorn app.main:app --reload --port 8000
# or as a module:
uv run python -m app.main
```

Swagger UI: <http://localhost:8000/docs>

### 6. Run tests

```bash
uv run pytest
```

Tests cover the chunker, format detection, and session store. They do not call OpenAI — no API key required.

## API

All `/api/v1/*` endpoints accept/issue an `iasid` cookie. The Next.js client must send `credentials: "include"` and the backend's `CORS_ORIGINS` must include the frontend origin.

| Method | Path              | Purpose                                                            |
| ------ | ----------------- | ------------------------------------------------------------------ |
| GET    | `/health`         | Liveness                                                           |
| GET    | `/api/v1/session` | Get current session id + uploaded files (creates one if missing)   |
| DELETE | `/api/v1/session` | Wipe this session: drop Chroma collection + clear cookie           |
| POST   | `/api/v1/upload`  | `multipart/form-data` with one or more `files` → indexed file info |
| POST   | `/api/v1/chat`    | `{ "question": "..." }` → `{ answer, citations[] }`                |

### Error envelope

Domain errors return JSON with both a human `detail` and a machine `code`:

```json
{ "detail": "File 'big.pdf' exceeds the 5MB limit.", "code": "file_too_large" }
```

See [`app/core/exceptions.py`](app/core/exceptions.py) for the full code list (`unsupported_file`, `file_too_large`, `pdf_too_long`, `too_many_files`, `no_extractable_text`, `no_documents`, `embedding_service_unavailable`, `llm_service_unavailable`, …).

### Citation alignment

Each `CitationOut` carries an `index` (1-based) that matches the inline `[#N]` markers in the answer text. The frontend uses this to render clickable chips.

```json
{
  "session_id": "ab12...",
  "answer": "Theft of personal items is covered up to USD 2,000 [#1]. Minimum charter capital for life insurers is VND 600B [#2].",
  "citations": [
    {
      "index": 1,
      "file_id": "f0...",
      "filename": "policy.pdf",
      "location": "Page 4",
      "chunk_id": "f0..._3"
    },
    {
      "index": 2,
      "file_id": "f1...",
      "filename": "01_VN_LuatKinhDoanhBaoHiem.docx",
      "location": "Điều 44 – Vốn điều lệ tối thiểu",
      "chunk_id": "f1..._12"
    }
  ]
}
```

PDFs cite by page; DOCX cite by nearest heading text (Ctrl-F-able in Word); images cite by filename + the OCR-derived snippet anchor. The frontend chip renders `<filename> — <location>` directly.

Backend filters citations to only those actually referenced in the LLM's answer (so unused retrieved chunks don't leak into the response).

### Upload

```bash
curl -i -c jar.txt -b jar.txt \
  -F "files=@policy.pdf" \
  -F "files=@claim_form.png" \
  http://localhost:8000/api/v1/upload
```

### Chat

```bash
curl -s -c jar.txt -b jar.txt \
  -H "Content-Type: application/json" \
  -d '{"question":"What is covered for theft?"}' \
  http://localhost:8000/api/v1/chat
```

## How the bar requirements are met

| Requirement                                           | Where                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------------------ |
| PDF / DOCX / Image                                    | `app/ingestion/{pdf,docx,image}.py`                                      |
| OCR for images (eng + vie)                            | `pytesseract`, configured via `OCR_LANGUAGES`                            |
| Chunk metadata: filename + page/section + upload date | `app/services/upload_service.py` (Chroma `metadatas`)                    |
| Context-only answers + citations                      | `app/rag/llm.py` + system prompt **v4** enforces `[#N]` markers          |
| Anti-hallucination on named entities / specific cases | Prompt rule 7 — never transfer general category facts (waiting-period tables, definition lists) onto a specifically-named plan/article/person/case. v4 adds a worked example for case-specific facts (a patient's diagnosis comes from the claim form, not from a general waiting-period list) |
| Anti-"document lacks X" claims                        | Prompt rule 8 — context is a retrieval subset, never assert the source file lacks a topic |
| "I don't know" handling                               | Prompt fallback line in EN/VI; backend short-circuits on empty retrieval |
| Session isolation                                     | `app/api/deps.py` cookie + `app/rag/store.py` per-session collection     |
| Loading state (no streaming)                          | Single response per `/api/v1/chat`; frontend renders spinner             |
| File limits (2 / 5MB / 20 pages)                      | `app/services/upload_service.py` + `app/ingestion/pdf.py`                |
| Hybrid search (Senior tie-breaker)                    | `app/rag/retriever.py` (BM25 + cosine, RRF fusion, top-20)               |
| Type hints (Senior tie-breaker)                       | All modules typed; Pydantic v2 schemas; strict ruff config               |

## Evaluation

Three manually-judged test suites cover the bar requirements end-to-end across distinct document domains. Each suite runs across multiple isolated sessions including dedicated **KB** (knowledge-boundary / hallucination) and **ISO** (cross-session leakage) checks.

| Suite                                | Docs                                                       | PASS | PARTIAL | FAIL | Strict % | Lenient % (P + ½·Partial) |
| ------------------------------------ | ---------------------------------------------------------- | ---- | ------- | ---- | -------- | -------------------------- |
| v1 — claim docs                      | health/auto policy, claim forms, ID cards (VN + EN)        | 55   | 4       | 0    | 93.2%    | 96.6%                      |
| v2 — legal / regulatory / cross-lang | Insurance Business Law, FAQ, regulatory guide, glossary    | 50   | 9       | 0    | 84.7%    | 93.2%                      |
| v3 — operational / specialised       | ILP/UL guide, market stats, claims SOP, UW guidelines, fraud infographic, risk framework | 43   | 7       | 0    | 86.0%    | 93.0%                      |
| **Combined (168 Qs · 28 sessions)**  |                                                            | **148** | **20** | **0** | **88.1%** | **94.0%**           |

- **0 failures** and **0 hallucinations** across all 168 questions on entirely unseen suite-v3 fixtures (no overlap with what tuning was done against)
- KB (anti-hallucination) checks: **0/13 hallucinations** including the `Diamond Plan` / `VF-Growth 2023` named-entity traps
- ISO (session-isolation) checks: **9/9** — empty session correctly returns `HTTP 400 no_documents` on any question
- The 20 PARTIAL cases hit the main fact correctly but miss a secondary detail in the reference answer (borderline on strict grading); none represent an answer-quality regression

The journey there — `gpt-4o-mini` + emb-3-small + TOP_K=6 + prompt v1 → 78 % with hallucinations — to the current config is logged in `scripts/test-report-final.md` (gitignored alongside the runners). Each iteration's win is attributable: TOP_K=20 fixed retrieval recall on multi-fact questions, prompt rules 7+8 fixed the named-entity hallucinations and the "document lacks X" false negatives, the section-aware chunker fixed cross-section citation drift (a chunk whose body was mostly *Điều 22* used to label itself *Điều 17*), and OCR upscale + PSM=6 fixed phone-number / table-cell misreads.

## Trade-offs and next steps

- **In-memory session store.** Sessions live in process memory; chat history is lost on restart, though Chroma vectors persist on disk and BM25 is rebuilt lazily on first query. For multi-instance deployments, swap `SessionStore` for Redis.
- **Sync OpenAI calls run in a threadpool.** `chat` uses `run_in_threadpool` so the event loop stays responsive. For high concurrency, switch to `AsyncOpenAI`.
- **OCR is line-by-line Tesseract.** Good for printed forms; handwriting is out of scope per the spec. For scanned PDFs we render at 200 DPI; for raw images we Lanczos-upscale to ≥2000px on the long edge and pass `--psm 6` (uniform block) — both empirically lift accuracy on posters / ID-card photos / dense table cells where small glyphs default-segment poorly.
- **DOCX citations anchor on the nearest heading text** (e.g. `policy.docx — "Điều 44 – Vốn điều lệ tối thiểu"`) instead of `Page` or `Section <n>`. DOCX pagination is unstable across renderers and bare ordinals are unactionable; the heading text is something the user can paste into Word's Find dialog and jump straight to. For content that sits before any heading we fall back to a 60-char snippet of the paragraph itself — also Ctrl-F-able. Tables inherit the heading of the section they sit in (no "Table N" labels). The spec accepts this in lieu of stable pagination.
- **Section-aware chunking.** The chunker flushes its buffer the moment it sees a block belonging to a different section, so a chunk never spans two distinct headings. Without this, a short heading + intro could pack onto the next heading's content and the chunk would inherit the wrong head section — a chunk whose body is mostly "Điều 22" would cite "Điều 17". PDFs (where every block has `section=None`) are unaffected; the comparison is `None != None` which is False, so packing behaviour is unchanged.
- **Token optimization.** We embed and store ~800-char chunks rather than full documents. We do _not_ pre-summarize before indexing — that would lose citation fidelity. Per-chat input is capped via `TOP_K=20` and `MAX_HISTORY_TURNS=4`. Total per-chat cost ≈ **$0.003** with current models (≈1,600 chats per $5 OpenAI credit).
- **Retry policy.** Centralised in `app/rag/retry.py`: 5 attempts × exponential jitter (1s → 16s) **only** on transient errors (5xx, 429, network). Auth/quota/bad-request errors fail fast — no looping.
- **Prompt versioning.** [`answer.yaml`](app/rag/prompts/answer.yaml) is at **version 4** and carries a `version` field. v3 added two anti-hallucination rules on top of v2's thoroughness rules: rule 7 (named-entity isolation — don't transfer category facts to a specifically-named plan/article) and rule 8 (no negative claims — never assert the source document "lacks" a topic just because retrieval missed it). v4 generalises rule 7 with a worked cross-document example: **case-specific facts come from case-specific documents**. The patient's diagnosis is in the claim form; a waiting-period table listing "90 days for cancer" tells you about general policy structure, not about any specific patient's actual condition. This was the difference between a v3 hallucination (`patient diagnosed with cancer`) and the correct v4 answer (`J18.9 viêm phổi` from the claim form chunk) on the same fixtures.
- **Logging.** Best-effort PII redaction lives in `app/core/logging.py` (`RedactingFormatter`): emails, phone numbers, and 9+ digit IDs are masked in the formatted output before any handler sees them. We never intentionally log prompt or response bodies — only filenames and exception traces — so the redactor is defense in depth, not the primary control. Set `LOG_LEVEL=warning` in production to suppress info-level lines entirely.

## Security & privacy notes

- Session isolation is enforced at the Chroma collection level — User A's vectors are never visible to User B (verified via the ISO test suite).
- Cookies are `HttpOnly` + `SameSite=Lax`. For production, terminate at HTTPS and add `Secure`.
- No real customer data is bundled. `.env` is gitignored; `.env.example` ships placeholder keys only.
- Log lines pass through `RedactingFormatter` (`app/core/logging.py`) which masks emails, phone numbers, and 9+ digit IDs in the final formatted string — covers msg, args and exception tracebacks in one pass. Best-effort regex, not a substitute for not logging sensitive content.
