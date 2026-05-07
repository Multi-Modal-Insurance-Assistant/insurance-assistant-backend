from __future__ import annotations

import logging
import sys

from app.core.logging import RedactingFormatter, _redact


def test_redact_email() -> None:
    assert _redact("contact alice@example.com today") == "contact [EMAIL] today"


def test_redact_phone_with_separators() -> None:
    assert _redact("call +84 90 123 4567 now") == "call [PHONE] now"


def test_redact_phone_dashes() -> None:
    assert _redact("ring 090-123-4567") == "ring [PHONE]"


def test_redact_long_digit_id() -> None:
    out = _redact("CCCD 012345678901")
    assert "012345678901" not in out


def test_keeps_short_numbers() -> None:
    assert _redact("Page 5 of 12") == "Page 5 of 12"


def test_keeps_dates() -> None:
    assert _redact("date 2025-05-07") == "date 2025-05-07"


def test_keeps_versions() -> None:
    assert _redact("version 1.2.3") == "version 1.2.3"


def test_no_pii_unchanged() -> None:
    assert _redact("Extraction failed for policy.pdf") == "Extraction failed for policy.pdf"


def test_formatter_redacts_exception_traceback() -> None:
    """The formatter must mask PII inside formatted exception text, not just msg."""
    fmt = RedactingFormatter("%(message)s")
    try:
        raise ValueError("invalid email alice@example.com")
    except ValueError:
        record = logging.LogRecord(
            name="t",
            level=logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg="boom",
            args=(),
            exc_info=sys.exc_info(),
        )
    out = fmt.format(record)
    assert "alice@example.com" not in out
    assert "[EMAIL]" in out
