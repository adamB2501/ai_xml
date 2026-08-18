import spacy

nlp = spacy.load("ner_model")

LABEL_TO_FIELD = {
    "INVOICE_NUMBER": "invoice_number",
    "DATE": "issue_date",
    "SELLER_NAME": "seller_name",
    "BUYER_NAME": "buyer_name",
    "TOTAL_TTC": "total_ttc",
    "TOTAL_HT": "total_ht",
    "TOTAL_TVA": "total_tva",
}


def extract_fields(invoice_text):
    doc = nlp(invoice_text)
    fields = {v: "" for v in LABEL_TO_FIELD.values()}
    for ent in doc.ents:
        field_name = LABEL_TO_FIELD.get(ent.label_)
        if field_name:
            fields[field_name] = ent.text
    return fields


if __name__ == "__main__":
    import json
    import sys

    from pdf_reader import extract_text

    pdf_path = sys.argv[1]
    text = extract_text(pdf_path)
    fields = extract_fields(text)
    print(json.dumps(fields, indent=2, ensure_ascii=False))
