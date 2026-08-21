import pytesseract
from pdf2image import convert_from_path


def extract_text_ocr(pdf_path, dpi=300):
    """OCR-based text extraction for scanned PDFs."""
    full_text = []
    pages = convert_from_path(pdf_path, dpi=dpi)
    for page_image in pages:
        text = pytesseract.image_to_string(page_image)
        if text:
            full_text.append(text)
    return "\n".join(full_text)


def extract_words_with_positions_ocr(pdf_path, dpi=300):
    """OCR-based word extraction with bounding boxes, mirroring
    extract_words_with_positions() from the text-based extractor."""
    all_words = []
    pages = convert_from_path(pdf_path, dpi=dpi)

    for page_number, page_image in enumerate(pages):
        data = pytesseract.image_to_data(page_image, output_type=pytesseract.Output.DICT)

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            word = data["text"][i].strip()
            if not word:
                continue

            x0 = data["left"][i]
            y0 = data["top"][i]
            x1 = x0 + data["width"][i]
            y1 = y0 + data["height"][i]

            all_words.append(
                {
                    "text": word,
                    "x0": x0,
                    "x1": x1,
                    "top": y0,
                    "bottom": y1,
                    "page": page_number,
                    "confidence": data["conf"][i],  # OCR-specific: worth keeping
                }
            )

    return all_words


if __name__ == "__main__":
    import sys

    path = sys.argv[1]
    print(extract_text_ocr(path))
