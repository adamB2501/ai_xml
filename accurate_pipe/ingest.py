# -*- coding: utf-8 -*-
"""
STEP 1 of the pipeline:  PDF file  ->  one block of source text.

The guiding principle: use the *most reliable* source available for each
part of the page.

  - A real embedded text layer is exact (every digit right). Keep it.
  - A rasterized band (letterhead / footer) or a scanned page is only
    pixels - the vision model reads those.

So we first CLASSIFY the PDF, then gather text accordingly:

    classification   how we get the text
    --------------   -------------------------------------------------
    "text"           pdfplumber only. Fast path, no model calls.
    "mixed"          pdfplumber for the body  +  vision model for each
                     rasterized band, spliced in at the right position.
    "scanned"        vision model transcribes every page image.
                     (also used when a page is one big image with a
                      poor-quality OCR text layer behind it)

Everything here is deterministic except the vision calls. If Ollama is
down, "text" PDFs still work; "mixed"/"scanned" raise OllamaError.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pdfplumber

from . import config, llm
from .prompts import VISION_TRANSCRIBE_PROMPT


# ---------------------------------------------------------------------------
# result object
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    text: str                        # the gathered source text
    classification: str              # "text" | "mixed" | "scanned"
    low_confidence_source: bool = False   # True for scans / OCR-layer pages
    notes: list[str] = field(default_factory=list)
    vision_calls: list[llm.LLMResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# geometry helpers (pure pdfplumber, no OCR)
# ---------------------------------------------------------------------------

def _bbox_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _chars_in_bbox(chars, b, pad: float = 1.5) -> int:
    x0, y0, x1, y1 = b
    return sum(
        1 for c in chars
        if c["x0"] >= x0 - pad and c["x1"] <= x1 + pad
        and c["top"] >= y0 - pad and c["bottom"] <= y1 + pad
    )


def _rects_touch(a, b, pad: float = 2.0) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                or a[3] + pad < b[1] or b[3] + pad < a[1])


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def raster_text_bands(page) -> list[tuple[float, float, float, float]]:
    """Image regions on `page` that look like a rasterized text band
    (a letterhead / footer / stamp) rather than a logo or a full-page
    background. Overlapping images are merged first so a 'logo + company
    details' letterhead counts as one band.

    Returns bboxes in PDF points (x0, top, x1, bottom).
    """
    page_area = float(page.width) * float(page.height)
    if page_area <= 0:
        return []

    boxes = [
        (float(im["x0"]), float(im["top"]), float(im["x1"]), float(im["bottom"]))
        for im in page.images
    ]
    # A full-page image is a background/scan, not a band - drop it before merging
    # (it would otherwise swallow every other box).
    boxes = [b for b in boxes
             if _bbox_area(b) / page_area <= config.REGION_MAX_AREA_FRAC]

    # merge overlapping / adjacent boxes
    merged: list = []
    for box in boxes:
        for i, m in enumerate(merged):
            if _rects_touch(box, m):
                merged[i] = _union(m, box)
                break
        else:
            merged.append(box)
    # a merge can connect two existing boxes - collapse until stable
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if _rects_touch(merged[i], merged[j]):
                    merged[i] = _union(merged[i], merged[j])
                    del merged[j]
                    changed = True
                    break
            if changed:
                break

    chars = page.chars
    out = []
    for b in merged:
        area_frac = _bbox_area(b) / page_area
        width_frac = (b[2] - b[0]) / float(page.width)
        if area_frac > config.REGION_MAX_AREA_FRAC:
            continue
        if area_frac < config.REGION_MIN_AREA_FRAC and width_frac < config.REGION_MIN_WIDTH_FRAC:
            continue
        if _chars_in_bbox(chars, b) > config.MAX_CHARS_OVER_REGION:
            continue  # text is already in the layer - not a raster band
        out.append(b)
    out.sort(key=lambda b: (b[1], b[0]))          # top-to-bottom
    return out


def _has_full_page_image(page) -> bool:
    page_area = float(page.width) * float(page.height)
    return any(
        _bbox_area((im["x0"], im["top"], im["x1"], im["bottom"])) / page_area
        >= config.FULL_PAGE_IMAGE_FRAC
        for im in page.images
    ) if page_area else False


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def classify(pdf_path: str) -> str:
    """"text" | "mixed" | "scanned" - see module docstring."""
    with pdfplumber.open(pdf_path) as pdf:
        total_chars = sum(len(p.chars) for p in pdf.pages)
        if total_chars < config.SCANNED_CHAR_THRESHOLD:
            return "scanned"
        for page in pdf.pages:
            if _has_full_page_image(page) and len(page.chars) > 0:
                # image behind a text layer -> scanned origin, OCR-layer text.
                # Treat as scanned so the vision model re-reads it cleanly.
                return "scanned"
            if raster_text_bands(page):
                return "mixed"
    return "text"


# ---------------------------------------------------------------------------
# text gathering per classification
# ---------------------------------------------------------------------------

def _ocr_band(page, bbox) -> tuple[str, llm.LLMResult]:
    """Render one region of the page and transcribe it with the vision
    model. Returns (text, call-record)."""
    crop_image = page.crop(bbox).to_image(resolution=config.RENDER_DPI).original
    result = llm.vision(VISION_TRANSCRIBE_PROMPT, crop_image)
    return result.text.strip(), result


def _gather_mixed(pdf) -> IngestResult:
    res = IngestResult(text="", classification="mixed")
    page_texts: list[str] = []

    for pi, page in enumerate(pdf.pages):
        bands = raster_text_bands(page)
        body = (page.extract_text() or "").strip()

        if not bands:
            if body:
                page_texts.append(body)
            continue

        # where does the embedded body sit vertically?
        chars = page.chars
        body_top = min((c["top"] for c in chars), default=None)
        body_bottom = max((c["bottom"] for c in chars), default=None)

        above, below, overlapping = [], [], []
        for b in bands:
            if body_top is None:
                above.append(b)
            elif b[3] <= body_top:
                above.append(b)
            elif b[1] >= body_bottom:
                below.append(b)
            else:
                overlapping.append(b)

        parts: list[str] = []
        for b in above:
            txt, call = _ocr_band(page, b)
            res.vision_calls.append(call)
            if txt:
                parts.append(txt)
        if body:
            parts.append(body)
        for b in overlapping:
            # can't cleanly place it - append with a marker rather than lose it
            txt, call = _ocr_band(page, b)
            res.vision_calls.append(call)
            if txt:
                parts.append(f"[overlapping region]\n{txt}")
                res.notes.append(
                    f"page {pi}: a raster band overlapped the body text; its "
                    "position in the merged text is approximate."
                )
        for b in below:
            txt, call = _ocr_band(page, b)
            res.vision_calls.append(call)
            if txt:
                parts.append(txt)

        page_texts.append("\n".join(parts))
        res.notes.append(
            f"page {pi}: body from text layer + {len(bands)} rasterized "
            f"band(s) read by {config.VISION_MODEL}."
        )

    res.text = "\n".join(page_texts).strip()
    return res


def _gather_scanned(pdf) -> IngestResult:
    res = IngestResult(text="", classification="scanned", low_confidence_source=True)
    page_texts = []
    for pi, page in enumerate(pdf.pages):
        page_image = page.to_image(resolution=config.RENDER_DPI).original
        call = llm.vision(VISION_TRANSCRIBE_PROMPT, page_image)
        res.vision_calls.append(call)
        page_texts.append(call.text.strip())
        res.notes.append(f"page {pi}: full page transcribed by {config.VISION_MODEL} "
                         "(scanned / image source - treat values as lower confidence).")
    res.text = "\n".join(page_texts).strip()
    return res


def _gather_text(pdf) -> IngestResult:
    parts = [(p.extract_text() or "").strip() for p in pdf.pages]
    return IngestResult(
        text="\n".join(p for p in parts if p).strip(),
        classification="text",
        notes=["all text came from the embedded text layer; no model calls."],
    )


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def ingest(pdf_path: str) -> IngestResult:
    kind = classify(pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        if kind == "text":
            return _gather_text(pdf)
        if kind == "mixed":
            return _gather_mixed(pdf)
        return _gather_scanned(pdf)
