# -*- coding: utf-8 -*-
"""
Maps an extracted field's VALUE back to where it sits on the PDF page, so
the review UI can draw a highlight box in the same color as that field's
row in the fields panel.

Two tiers of precision, matched to what's actually knowable -- never
overstated:

  EXACT (precise=True): the value is found in the page's own text layer
  via pdfplumber's Page.search(), which returns real character-level
  bounding boxes. This covers a "text" page's whole content and a "mixed"
  page's embedded body (invoice number, dates, amounts, line items --
  the values that matter most for review).

  APPROXIMATE (precise=False): the value came from the vision model
  reading a rasterized band (a mixed page's letterhead/footer) or a fully
  scanned page. The vision model returns text, not per-word coordinates,
  so there is no exact box to draw. The honest fallback is the band's own
  bounding box -- "the value came from roughly here" -- returned with
  precise=False so the UI renders it differently (dashed, lower opacity)
  instead of implying a precision it doesn't have.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pdfplumber

from accurate_pipe.ingest import raster_text_bands


@dataclass
class Rect:
    page: int
    x0: float
    top: float
    x1: float
    bottom: float
    precise: bool

    def to_dict(self) -> dict:
        return asdict(self)


def find_rects(pdf_path: str, value, *, max_matches: int = 4) -> list[Rect]:
    """Every place `value` can be located on the PDF, exact where a text
    layer has it, otherwise the rasterized band(s) it might have come
    from. Non-string / empty values return no rects."""
    if not isinstance(value, str) or not value.strip():
        return []

    rects: list[Rect] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_no, page in enumerate(pdf.pages):
                try:
                    matches = page.search(value, case=False)
                except Exception:
                    matches = []
                for m in matches[:max_matches]:
                    rects.append(Rect(page_no, m["x0"], m["top"], m["x1"], m["bottom"], True))

            if rects:
                return rects[:max_matches]

            # nothing in any page's text layer -- offer the rasterized
            # band(s) as an approximate source, clearly marked as such
            for page_no, page in enumerate(pdf.pages):
                for b in raster_text_bands(page):
                    rects.append(Rect(page_no, b[0], b[1], b[2], b[3], False))
    except Exception:
        return []

    return rects[:max_matches]


def rects_for_fields(pdf_path: str, fields: dict) -> dict[str, list[dict]]:
    """{flat_field_name: [Rect.to_dict(), ...]} for every scalar field
    with a string value, plus one entry per line item sub-value under
    keys like "line_items[0].description" (so each row can get its own
    color in the UI while still being found precisely on the page)."""
    out: dict[str, list[dict]] = {}

    for key, value in fields.items():
        if key.startswith("_") or key == "line_items":
            continue
        if isinstance(value, str):
            rects = find_rects(pdf_path, value)
            if rects:
                out[key] = [r.to_dict() for r in rects]

    for i, item in enumerate(fields.get("line_items") or []):
        if not isinstance(item, dict):
            continue
        for sub_key, sub_value in item.items():
            if not isinstance(sub_value, str) or not sub_value.strip():
                continue
            rects = find_rects(pdf_path, sub_value)
            if rects:
                out[f"line_items[{i}].{sub_key}"] = [r.to_dict() for r in rects]

    return out
