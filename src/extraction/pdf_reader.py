import re
import unicodedata

import pdfplumber

# --- Cleaning building blocks -------------------------------------------------

_ARABIC_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
]

_CID_PATTERN = re.compile(r"\(cid:\d+\)")

_SPACED_TOKEN_RUN = re.compile(r"(?:[\d/.\-]\s){3,}[\d/.\-]")

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")

_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")

# Only used now to identify lines that are *truly empty after whitespace
# normalization* (nothing but whitespace) -- not lines that merely look
# like punctuation, since a bare "-" or ":" can be a genuine table rule,
# separator, or a partially-extracted value worth keeping for the model
# to see in context. Removing content on a "looks like junk" heuristic is
# exactly the kind of strict condition that silently drops real data.
_BLANK_LINE = re.compile(r"^\s*$")


def _is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp <= end for start, end in _ARABIC_RANGES)


def strip_arabic(text: str) -> str:
    """Remove Arabic-script characters.

    NOT applied by default anymore. The old assumption -- "Arabic text in
    this document set is always a duplicate of the adjacent French label/
    value, so it carries no unique information" -- doesn't hold across all
    documents, and silently deleting a whole script is a strict, blanket
    condition that can erase real, non-duplicate data (e.g. an Arabic-only
    company name, address, or a bilingual invoice where the two languages
    aren't actually redundant). Kept as an explicit opt-in utility so a
    caller who has actually verified duplication for their specific
    document set can still use it -- see `clean_text(..., strip_arabic_script=False)`.
    """
    return "".join(ch for ch in text if not _is_arabic_char(ch))


def strip_cid_artifacts(text: str, placeholder: str = "") -> str:
    """Replace (cid:N) tokens. These represent genuinely unrecoverable
    glyphs (the PDF's font has no usable Unicode mapping), so there's no
    real data being lost here -- but we still let the caller keep a visible
    placeholder instead of silently vanishing whitespace, so it's obvious
    in the output that something was there and couldn't be read, rather
    than looking like clean text."""
    return _CID_PATTERN.sub(placeholder, text)


def collapse_spaced_tokens(text: str) -> str:
    """Collapse character-by-character spaced sequences like dates/numbers
    that got split into individual text objects in the PDF
    ("0 2 / 0 1 / 2 0 1 5" -> "02/01/2015"). Purely additive/joining, does
    not remove any characters, so it's not a data-loss risk."""

    def _collapse(match: re.Match) -> str:
        return re.sub(r"\s+", "", match.group(0))

    return _SPACED_TOKEN_RUN.sub(_collapse, text)


def unwrap_markdown_links(text: str) -> str:
    """Replace [label](url) with "label (url)" instead of dropping the
    label entirely. The old version kept only the URL and discarded the
    label text -- fine when label == domain, but a real information loss
    whenever the label differs from the URL (e.g. a display name, a
    tracking-id-bearing link, or descriptive anchor text)."""

    def _replace(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        if label.strip() == url.strip() or url.endswith(label.strip()):
            return url
        return f"{label} ({url})"

    return _MARKDOWN_LINK.sub(_replace, text)


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")  # non-breaking space
    lines = text.split("\n")
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in lines]
    # Only drop lines that are now genuinely empty -- not lines that merely
    # consist of punctuation/short tokens, which used to be discarded by
    # _JUNK_LINE and could contain real (if terse) extracted content.
    lines = [line for line in lines if not _BLANK_LINE.match(line)]
    text = "\n".join(lines)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def clean_text(
    text: str,
    strip_arabic_script: bool = True,
    cid_placeholder: str = "",
) -> str:
    """Full cleaning pipeline.

    - strip_arabic_script defaults to True: confirmed for this document
      set that Arabic content is always a duplicate of the adjacent
      French label/value, so stripping it loses nothing. Set to False
      for a document set where that hasn't been verified.
    - CID artifacts are replaced with `cid_placeholder` (default: removed,
      same as before, since there's no recoverable content there) but the
      knob exists if you'd rather keep a visible marker.
    """
    text = strip_cid_artifacts(text, placeholder=cid_placeholder)
    if strip_arabic_script:
        text = strip_arabic(text)
    text = collapse_spaced_tokens(text)
    text = unwrap_markdown_links(text)
    text = normalize_whitespace(text)
    return text


# --- Extraction ---------------------------------------------------------------


def extract_text(pdf_path, clean: bool = True, **clean_kwargs):
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    raw = "\n".join(full_text)
    return clean_text(raw, **clean_kwargs) if clean else raw


def extract_words_with_positions(pdf_path, drop_cid_only: bool = True, drop_arabic: bool = True):
    """Extract words with their bounding boxes.

    drop_arabic defaults to True: confirmed duplicate-label content for
    this document set, so dropping pure-Arabic words loses no unique
    signal. Set to False if using this on a document set where that
    hasn't been verified -- unverified duplication assumptions are what
    caused the original silent data loss.
    """
    all_words = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            words = page.extract_words()
            for w in words:
                text = w["text"]
                if drop_cid_only and _CID_PATTERN.fullmatch(text):
                    continue
                if drop_arabic and strip_arabic(text) == "":
                    continue
                all_words.append(
                    {
                        "text": text,
                        "x0": w["x0"],
                        "x1": w["x1"],
                        "top": w["top"],
                        "bottom": w["bottom"],
                        "page": page_number,
                    }
                )
    return all_words


# --- Table extraction ----------------------------------------------------------


def extract_tables(pdf_path, clean: bool = True, strategy: str = "lines", **clean_kwargs):
    """First-choice table extraction, with automatic fallback.

    The old docstring claimed a lines -> text -> position fallback chain
    but the code never actually implemented it -- a page that returned []
    with "lines" just silently returned no table, which is itself a
    strict condition that drops the whole table's data. Now genuinely
    falls back to strategy="text" per page when "lines" finds nothing, and
    the caller can still reach for extract_table_by_position() if both
    geometric strategies fail (e.g. no rulings and irregular whitespace).
    """
    settings_primary = {"vertical_strategy": strategy, "horizontal_strategy": strategy}
    settings_fallback = {"vertical_strategy": "text", "horizontal_strategy": "text"}

    all_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables(table_settings=settings_primary)
            if not tables and strategy != "text":
                tables = page.extract_tables(table_settings=settings_fallback)
            if clean:
                tables = [
                    [[clean_text(cell, **clean_kwargs) if cell else cell for cell in row] for row in table]
                    for table in tables
                ]
            all_pages.append(tables)
    return all_pages


def _cluster_1d(values: list, tolerance: float):
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= tolerance:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def extract_table_by_position(
    pdf_path,
    page_number: int,
    row_tolerance: float = 3.0,
    col_tolerance: float = 5.0,
    min_rows_for_column: int = 3,
    clean: bool = True,
    **clean_kwargs,
):
    """Fully automatic, purely geometric fallback for tables with no
    rulings and inconsistent whitespace.

    Changed: this used to return [] outright whenever fewer than
    `min_rows_for_column` rows shared a column, or fewer than 2 supported
    columns were found -- discarding every word on the page even though
    real text had already been located and positioned. Now it degrades
    gracefully instead of discarding:
      - if row clustering succeeds but no reliable columns are found,
        each row is returned as a single-cell row (still real content,
        just not split into columns) instead of an empty list;
      - only truly empty input (no words at all) returns [].
    """
    words = [w for w in extract_words_with_positions(pdf_path) if w["page"] == page_number]
    if not words:
        return []

    # 1. Cluster into rows by y-position.
    words_sorted = sorted(words, key=lambda w: w["top"])
    rows, current_row, current_top = [], [], None
    for w in words_sorted:
        if current_top is None or abs(w["top"] - current_top) <= row_tolerance:
            current_row.append(w)
            current_top = w["top"] if current_top is None else current_top
        else:
            rows.append(current_row)
            current_row, current_top = [w], w["top"]
    if current_row:
        rows.append(current_row)

    def _row_as_single_cell(row):
        text = " ".join(w["text"] for w in sorted(row, key=lambda w: w["x0"]))
        return [clean_text(text, **clean_kwargs)] if clean else [text]

    if len(rows) < min_rows_for_column:
        # Not enough rows to trust column detection -- but the rows
        # themselves are still real, positioned text. Return them
        # ungrouped rather than throwing them away.
        return [_row_as_single_cell(r) for r in rows]

    # 2. Cluster x0 positions per row first (dedupe within a row), then
    # across all rows, keeping only clusters that appear in enough rows.
    per_row_x0 = [sorted({w["x0"] for w in row}) for row in rows]
    all_x0 = [x for row_x0 in per_row_x0 for x in row_x0]
    candidate_cols = _cluster_1d(all_x0, col_tolerance)

    col_support = []
    for col_x in candidate_cols:
        support = sum(1 for row_x0 in per_row_x0 if any(abs(x - col_x) <= col_tolerance for x in row_x0))
        if support >= min_rows_for_column:
            col_support.append(col_x)
    col_support.sort()

    if len(col_support) < 2:
        # No reliable recurring columns -- fall back to single-cell rows
        # instead of dropping the whole region.
        return [_row_as_single_cell(r) for r in rows]

    # Column boundaries: midpoint between consecutive detected columns.
    bounds = []
    for i, cx in enumerate(col_support):
        left = -float("inf") if i == 0 else (col_support[i - 1] + cx) / 2
        right = float("inf") if i == len(col_support) - 1 else (cx + col_support[i + 1]) / 2
        bounds.append((left, right))

    # 3. Assign words to columns, build output rows.
    result = []
    for row in rows:
        cells = ["" for _ in bounds]
        row_words = sorted(row, key=lambda w: w["x0"])
        for w in row_words:
            placed = False
            for i, (left, right) in enumerate(bounds):
                if left <= w["x0"] < right:
                    cells[i] = (cells[i] + " " + w["text"]).strip()
                    placed = True
                    break
            if not placed:
                # Word fell outside every detected column boundary (can
                # happen at the far edges). Previously this word was
                # silently dropped; now it's appended to the nearest edge
                # column instead of vanishing.
                nearest = min(range(len(bounds)), key=lambda i: abs(w["x0"] - col_support[i]))
                cells[nearest] = (cells[nearest] + " " + w["text"]).strip()
        if clean:
            cells = [clean_text(c, **clean_kwargs) for c in cells]
        if any(cells):
            result.append(cells)
    return result


def is_scanned_pdf(pdf_path):
    text = extract_text(pdf_path)
    return len(text.strip()) < 20


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    if is_scanned_pdf(path):
        print("This PDF looks scanned/image-based. You'll need OCR, not this script.")
    else:
        print(extract_text(path))
