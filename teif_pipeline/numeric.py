# -*- coding: utf-8 -*-
"""Shared number/date parsing. Previously duplicated three times (in
src/mapping/xml_builder.py, src/extraction/ner_extractor.py, and
src/extraction/evaluate_ner.py) with the same non-trivial reasoning behind
each copy -- exactly the kind of "same raw string interpreted twice, in two
different files, with two different fallback policies" problem the audit
flagged as motivation for the typed Invoice boundary. This module is the
one place that logic lives now; the TEIF builder formats Decimal/date
values back to TEIF's string conventions (3-decimal amounts, ddMMyy dates)
at serialization time, not here.
"""

import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional

_FR_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11,
    "décembre": 12, "decembre": 12,
}


def parse_decimal(raw) -> Optional[Decimal]:
    """Whichever of '.'/',' appears LAST in the string is the decimal
    separator; earlier ones are thousands separators to strip. Anchoring on
    "exactly 3 digits after the separator" instead breaks on TND amounts,
    which conventionally use exactly 3 decimals (millimes) -- "303.847"
    would misparse as "303847" under that rule.

    Refuses (returns None) on text containing an embedded newline: a
    genuine amount/quantity is always a single inline token; an embedded
    newline means the span crosses multiple separately-extracted text
    lines glued together -- the signature of a rotated/multi-line PDF
    region (any template can produce this), not a normal amount.
    """
    if raw is None or raw == "":
        return None
    text = str(raw)
    if "\n" in text or "\r" in text:
        return None

    s = re.sub(r"[^\d.,\-]", "", text)
    if not s:
        return None
    negative = s.startswith("-")
    s = s.lstrip("-")
    sep_pos = max(s.rfind("."), s.rfind(","))
    if sep_pos == -1:
        integer_part, decimal_part = s, ""
    else:
        integer_part = re.sub(r"[.,]", "", s[:sep_pos])
        decimal_part = s[sep_pos + 1:]
    if not integer_part and not decimal_part:
        return None
    value_text = (integer_part or "0") + ("." + decimal_part if decimal_part else "")
    try:
        value = Decimal(value_text)
    except InvalidOperation:
        return None
    return -value if negative else value


def parse_date(raw) -> Optional[date]:
    """Accepts whatever format the source invoice used -- already-compact
    ddMMyy digits (the TTN template extracts DUE_DATE that way), dd/mm/yyyy,
    dd-mm-yyyy, dd.mm.yyyy, 2-digit years, or a spelled-out French date
    ("20 février 2026"). Returns None (never a guess) if none of those
    match.
    """
    if not raw:
        return None
    s = str(raw).strip()

    m = re.fullmatch(r"(\d{2})(\d{2})(\d{2})", s)
    if m:
        d, mo, y = (int(g) for g in m.groups())
        return _safe_date(2000 + y, mo, d)

    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)  # ISO 8601 -- unambiguous, checked before dd-mm-yyyy
    if m:
        y, mo, d = m.groups()
        return _safe_date(int(y), int(mo), int(d))

    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", s)
    if m:
        d, mo, y = m.groups()
        return _safe_date(int(y), int(mo), int(d))

    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})", s)
    if m:
        d, mo, y = (int(g) for g in m.groups())
        return _safe_date(2000 + y, mo, d)

    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-zàâäéèêëîïôöùûüÿ]+)\s+(\d{4})", s, re.IGNORECASE)
    if m:
        d, month_name, y = m.groups()
        mo = _FR_MONTHS.get(month_name.lower())
        if mo:
            return _safe_date(int(y), mo, int(d))

    return None


def _safe_date(year, month, day) -> Optional[date]:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def format_ddmmyy(d: Optional[date]) -> Optional[str]:
    """The TEIF wire format for a date (Dtm/DateText, format="ddMMyy")."""
    return d.strftime("%d%m%y") if d else None


def format_amount(value: Optional[Decimal]) -> Optional[str]:
    """TND amounts are conventionally expressed to 3 decimals (millimes),
    matching exemple_elfatoora.xml's "2.000" / "0.240" style."""
    if value is None:
        return None
    return f"{value:.3f}"


def format_quantity(value: Optional[Decimal]) -> Optional[str]:
    """Less strict than format_amount: quantities aren't a currency, so
    trailing zeros are trimmed (12.900 -> 12.9) rather than forced to 3dp."""
    if value is None:
        return None
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"
