import re
import unicodedata

import pdfplumber

_ARABIC_RANGES = [
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
]

_CID_PATTERN = re.compile(r"\(cid:\d+\)")
_SPACED_TOKEN_RUN = re.compile(r"(?:[\d/.\-]\s){3,}[\d/.\-]")
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_LINES = re.compile(r"\n{3,}")
_BLANK_LINE = re.compile(r"^\s*$")


def _is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp <= end for start, end in _ARABIC_RANGES)


def strip_arabic(text: str) -> str:
    return "".join(ch for ch in text if not _is_arabic_char(ch))


def strip_cid_artifacts(text: str, placeholder: str = "") -> str:
    return _CID_PATTERN.sub(placeholder, text)


def collapse_spaced_tokens(text: str) -> str:
    def _collapse(match: re.Match) -> str:
        return re.sub(r"\s+", "", match.group(0))

    return _SPACED_TOKEN_RUN.sub(_collapse, text)


def unwrap_markdown_links(text: str) -> str:
    def _replace(m: re.Match) -> str:
        label, url = m.group(1), m.group(2)
        if label.strip() == url.strip() or url.endswith(label.strip()):
            return url
        return f"{label} ({url})"

    return _MARKDOWN_LINK.sub(_replace, text)


def normalize_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ")
    lines = text.split("\n")
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in lines]
    lines = [line for line in lines if not _BLANK_LINE.match(line)]
    text = "\n".join(lines)
    text = _MULTI_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def clean_text(
    text: str,
    strip_arabic_script: bool = True,
    cid_placeholder: str = "",
) -> str:
    text = strip_cid_artifacts(text, placeholder=cid_placeholder)
    if strip_arabic_script:
        text = strip_arabic(text)
    text = collapse_spaced_tokens(text)
    text = unwrap_markdown_links(text)
    text = normalize_whitespace(text)
    return text


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


def extract_tables(pdf_path, clean: bool = True, strategy: str = "lines", **clean_kwargs):
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
    words = [w for w in extract_words_with_positions(pdf_path) if w["page"] == page_number]
    if not words:
        return []

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
        return [_row_as_single_cell(r) for r in rows]

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
        return [_row_as_single_cell(r) for r in rows]

    bounds = []
    for i, cx in enumerate(col_support):
        left = -float("inf") if i == 0 else (col_support[i - 1] + cx) / 2
        right = float("inf") if i == len(col_support) - 1 else (cx + col_support[i + 1]) / 2
        bounds.append((left, right))

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