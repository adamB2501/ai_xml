import pdfplumber


def extract_text(pdf_path):
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    return "\n".join(full_text)


def extract_words_with_positions(pdf_path):
    all_words = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            words = page.extract_words()
            for w in words:
                all_words.append(
                    {
                        "text": w["text"],
                        "x0": w["x0"],
                        "x1": w["x1"],
                        "top": w["top"],
                        "bottom": w["bottom"],
                        "page": page_number,
                    }
                )
    return all_words


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
