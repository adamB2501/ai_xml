import json
import os
import sys

from extraction.ner_extractor import extract_fields
from extraction.pdf_reader import extract_text, is_scanned_pdf


def process_invoice(pdf_path, output_dir="data/output"):
    if is_scanned_pdf(pdf_path):
        raise ValueError(f"{pdf_path} looks scanned/image-based. OCR step needed, not supported yet.")

    text = extract_text(pdf_path)
    fields = extract_fields(text)

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{filename}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, ensure_ascii=False)

    return fields, output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_invoice.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    fields, output_path = process_invoice(pdf_path)

    print(json.dumps(fields, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output_path}")
