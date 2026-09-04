# -*- coding: utf-8 -*-
from datetime import date
from decimal import Decimal

from teif_pipeline.numeric import format_amount, format_ddmmyy, parse_date, parse_decimal


def test_parse_decimal_disambiguates_millimes_correctly():
    # the bug found and fixed earlier this session -- regression guard
    assert parse_decimal("303.847") == Decimal("303.847")
    assert parse_decimal("12628.512") == Decimal("12628.512")


def test_parse_decimal_handles_thousands_and_comma_decimal():
    assert parse_decimal("1 414,579") == Decimal("1414.579")


def test_parse_decimal_rejects_embedded_newline():
    assert parse_decimal("/TD\n000\n000\n2") is None


def test_parse_date_handles_iso_slash_dash_and_compact():
    assert parse_date("2025-03-14") == date(2025, 3, 14)
    assert parse_date("14/03/2025") == date(2025, 3, 14)
    assert parse_date("14-03-2025") == date(2025, 3, 14)
    assert parse_date("140325") == date(2025, 3, 14)


def test_parse_date_handles_french_month_name():
    assert parse_date("20 février 2026") == date(2026, 2, 20)


def test_format_roundtrip():
    assert format_amount(Decimal("50")) == "50.000"
    assert format_ddmmyy(date(2025, 3, 14)) == "140325"
