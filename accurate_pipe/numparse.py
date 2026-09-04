# -*- coding: utf-8 -*-
"""
Parsing helpers for the messy number / date strings that come out of an
invoice ("117.025", "1 234,560", "27/08/2026", "450 000").

We do NOT re-implement this - the existing teif_pipeline.numeric module
already handles Tunisian conventions (comma-decimal, space- or
period-thousands, dd/mm/yyyy) and is tested. This file just re-exports it
under names the accurate_pipe code uses, with a note on what each does, so
there's a single obvious import site.

    to_decimal("1 234,560")  -> Decimal("1234.560")   or None if unparseable
    to_date("27/08/2026")    -> datetime.date(2026, 8, 27)   or None
"""

from teif_pipeline.numeric import parse_date as to_date       # noqa: F401
from teif_pipeline.numeric import parse_decimal as to_decimal  # noqa: F401

__all__ = ["to_decimal", "to_date"]
