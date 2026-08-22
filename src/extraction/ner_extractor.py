import re

import spacy

nlp = spacy.load("ner_model")

# --- Full label set (from training_data.py) -------------------------------
# Every one of the 39 trained labels now has an explicit handler below --
# nothing falls into other_fields unless the loaded model emits a label
# this script has genuinely never seen (schema drift, not a design gap).

SINGLETON_LABELS = {
    "INVOICE_NUMBER": "invoice_number",
    "SELLER_NAME": "seller_name",
    "BUYER_NAME": "buyer_name",
    "TOTAL_HT": "total_ht",
    "TVA_AMOUNT": "total_tva",
    "TOTAL_TTC": "total_ttc",
    "PAYMENT_TERMS": "payment_terms",
    "RC_NUMBER": "rc_number",
    "CAPITAL_SOCIAL": "capital_social",
    "PO_NUMBER": "po_number",
    "DELIVERY_NOTE_NUMBER": "delivery_note_number",
    "STAMP_DUTY": "stamp_duty",
    "WITHHOLDING_TAX": "withholding_tax",
    "ADVANCE_PAYMENT": "advance_payment",
    "AMOUNT_PAID": "amount_paid",
    "BALANCE_DUE": "balance_due",
    "CURRENCY": "currency",
    "BANK_NAME": "bank_name",
    "RIB": "rib",
    "IBAN": "iban",
    "SWIFT_BIC": "swift_bic",
    "TAX_EXEMPTION_REASON": "tax_exemption_reason",
}

# DUE_DATE is deliberately NOT in SINGLETON_LABELS above -- it needs its
# own reconciliation step against the positional DATE-based guess (see
# Fix #1 below), rather than being overwritten by generic singleton logic.
DUE_DATE_LABEL = "DUE_DATE"
DUE_DATE_KEY = "due_date_from_model"

# Fix #3: schema mismatch protection.
# If the model was trained on a different naming convention (e.g. English
# SUBTOTAL/VAT_TOTAL/TOTAL instead of French HT/TVA/TTC), SINGLETON_LABELS
# would silently never match and those fields would always come back empty.
# ALIASES lets multiple raw model labels resolve to the same canonical field.
# Add to this dict rather than assuming the hardcoded label set is exhaustive.
ALIASES = {
    "SUBTOTAL": "TOTAL_HT",
    "NET_TOTAL": "TOTAL_HT",
    "VAT_TOTAL": "TVA_AMOUNT",
    "TAX_TOTAL": "TVA_AMOUNT",
    "TOTAL": "TOTAL_TTC",
    "TOTAL_DUE": "TOTAL_TTC",
    "GRAND_TOTAL": "TOTAL_TTC",
}

# Fields that can legitimately appear more than once per invoice --
# multiple VAT rates across line items, address fragments split apart by
# label/value declustering (see the TTN template), repeated contact info
# in header + footer, etc. Collected as lists rather than forced into a
# single value with the same "keep longer, log the rest" conflict logic
# used for true singletons.
REPEATABLE_LABELS = {
    "DATE": "dates",
    "SELLER_TAX_ID": "seller_tax_ids",
    "BUYER_TAX_ID": "buyer_tax_ids",
    "SELLER_ADDRESS": "seller_addresses",
    "BUYER_ADDRESS": "buyer_addresses",
    "DELIVERY_ADDRESS": "delivery_addresses",
    "PHONE": "phones",
    "EMAIL": "emails",
    "WEBSITE": "websites",
    "DISCOUNT_RATE": "discount_rates",
    "DISCOUNT_AMOUNT": "discount_amounts",
    "VAT_RATE": "vat_rates",
}

ITEM_LABELS = {
    "ITEM_DESC": "description",
    "ITEM_QTY": "quantity",
    "ITEM_UNIT_PRICE": "unit_price",
    "ITEM_LINE_TOTAL": "line_total",
}


def _normalize(text):
    """Collapse whitespace/case so formatting differences don't look like
    a genuine conflict (Fix #4)."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def _check_label_schema():
    """Fix #3: warn loudly at import time if the loaded model's label set
    doesn't line up with what this script expects, instead of failing
    silently field-by-field at extraction time."""
    try:
        model_labels = set(nlp.get_pipe("ner").labels)
    except Exception:
        return  # pipeline doesn't expose labels this way; skip the check

    known = (
        set(SINGLETON_LABELS)
        | set(ALIASES)
        | set(REPEATABLE_LABELS)
        | set(ITEM_LABELS)
        | {DUE_DATE_LABEL}
    )
    unrecognized = model_labels - known
    if unrecognized:
        print(
            "[schema warning] ner_model emits labels this script doesn't "
            f"handle: {sorted(unrecognized)}. Their values will be returned "
            "under fields['other_fields'] instead of a named field. Add them "
            "to SINGLETON_LABELS / ALIASES / REPEATABLE_LABELS / ITEM_LABELS "
            "if they should be treated as first-class fields."
        )


_check_label_schema()


def _group_line_items(item_ents):
    lines = []
    current = None

    for ent in item_ents:
        field = ITEM_LABELS[ent.label_]

        if ent.label_ == "ITEM_DESC":
            if current is not None:
                lines.append(current)
            current = {
                "description": ent.text,
                "quantity": None,
                "unit_price": None,
                "line_total": None,
                "_span": (ent.start_char, ent.end_char),
            }
        else:
            if current is None:
                current = {
                    "description": None,
                    "quantity": None,
                    "unit_price": None,
                    "line_total": None,
                    "_span": (ent.start_char, ent.end_char),
                }
            if current.get(field) is not None:
                lines.append(current)
                current = {
                    "description": None,
                    "quantity": None,
                    "unit_price": None,
                    "line_total": None,
                    "_span": (ent.start_char, ent.end_char),
                }
            current[field] = ent.text
            current["_span"] = (current["_span"][0], ent.end_char)

    if current is not None:
        lines.append(current)

    return lines


def extract_fields(invoice_text, debug=False):
    doc = nlp(invoice_text)

    fields = {v: "" for v in SINGLETON_LABELS.values()}
    fields.update({v: [] for v in REPEATABLE_LABELS.values()})
    fields["line_items"] = []
    fields["other_fields"] = {}  # Fix #2: unmapped labels land here, always

    conflicts = []
    date_ents = []  # (ent) kept separately so we can filter by position later
    item_ents = []
    due_date_ents = []  # model's own DUE_DATE predictions, reconciled below

    for ent in doc.ents:
        if debug:
            print(ent.text, ent.label_)

        raw_label = ent.label_
        label = ALIASES.get(raw_label, raw_label)  # Fix #3: resolve aliases first

        if raw_label == DUE_DATE_LABEL:
            due_date_ents.append(ent)

        elif label in SINGLETON_LABELS:
            key = SINGLETON_LABELS[label]
            existing = fields[key]
            if existing:
                # Fix #4: only a real conflict if the normalized text differs.
                # Same value repeated (e.g. seller name in header + footer)
                # is expected structurally, not model noise.
                if _normalize(existing) != _normalize(ent.text):
                    # keep the longer/more complete-looking value, log the rest
                    if len(ent.text) > len(existing):
                        conflicts.append((label, existing, ent.text))
                        fields[key] = ent.text
                    else:
                        conflicts.append((label, existing, ent.text))
            else:
                fields[key] = ent.text

        elif label == "DATE":
            date_ents.append(ent)

        elif label in REPEATABLE_LABELS:
            key = REPEATABLE_LABELS[label]
            fields[key].append(ent.text)

        elif label in ITEM_LABELS:
            item_ents.append(ent)

        else:
            # Fix #2: never silently drop data. Group by label so multiple
            # unmapped entities of the same type don't clobber each other.
            fields["other_fields"].setdefault(label, []).append(ent.text)

    fields["line_items"] = [{k: v for k, v in line.items() if k != "_span"} for line in _group_line_items(item_ents)]

    # --- Fix #1: date disambiguation, position-aware ---
    # A DATE entity that falls inside the character span covered by the
    # line-item block is almost certainly a per-line date (e.g. a "Date"
    # column), not the invoice issue/due date — even though the model
    # tags both with the same generic DATE label. Exclude those from
    # header-date assignment instead of assuming pure document order.
    item_spans = [line["_span"] for line in _group_line_items(item_ents)] if item_ents else []
    item_region = None
    if item_spans:
        item_region = (min(s for s, _ in item_spans), max(e for _, e in item_spans))

    header_dates = []
    item_region_dates = []
    for ent in date_ents:
        if item_region and item_region[0] <= ent.start_char <= item_region[1]:
            item_region_dates.append(ent.text)
        else:
            header_dates.append(ent.text)

    fields["issue_date"] = header_dates[0] if len(header_dates) > 0 else ""
    if len(header_dates) > 2:
        fields["extra_dates"] = header_dates[2:]
    if item_region_dates:
        # Not lost, just not guessed at — surfaced separately since we
        # can't safely assume which line each one belongs to without a
        # dedicated ITEM_DATE label in the model.
        fields["other_fields"].setdefault("ITEM_DATE_CANDIDATES", []).extend(item_region_dates)

    # --- Fix #1b: DUE_DATE reconciliation ---
    # Previously this field was populated ONLY from the positional guess
    # ("second generic DATE found outside the item region"), while the
    # model's own explicit DUE_DATE predictions were silently discarded --
    # they landed in other_fields["DUE_DATE"] and were never read back out.
    # If the model tagged a DUE_DATE explicitly, that's a direct label and
    # should win over a positional guess. The positional guess (second
    # header date) is now only a fallback for when the model didn't emit
    # DUE_DATE at all.
    due_date_conflicts = []
    due_date_value = ""
    for ent in due_date_ents:
        if due_date_value and _normalize(due_date_value) != _normalize(ent.text):
            due_date_conflicts.append((due_date_value, ent.text))
            if len(ent.text) > len(due_date_value):
                due_date_value = ent.text
        else:
            due_date_value = due_date_value or ent.text

    if due_date_value:
        fields["due_date"] = due_date_value
        fields["due_date_source"] = "model_label"
    elif len(header_dates) > 1:
        fields["due_date"] = header_dates[1]
        fields["due_date_source"] = "positional_fallback"
    else:
        fields["due_date"] = ""
        fields["due_date_source"] = "none"

    if debug:
        if conflicts:
            print("\n[conflicts] singleton field saw >1 differing value:")
            for label, kept, dropped in conflicts:
                print(f"  {label}: kept {kept!r}, dropped {dropped!r}")
        if due_date_conflicts:
            print("\n[conflicts] DUE_DATE saw >1 differing model prediction:")
            for kept, dropped in due_date_conflicts:
                print(f"  kept {kept!r}, dropped {dropped!r}")
        if fields["other_fields"]:
            print("\n[unmapped] labels with no dedicated handler (see fields['other_fields']):")
            for label, texts in fields["other_fields"].items():
                print(f"  {label}: {texts}")

    return fields


if __name__ == "__main__":
    import json
    import sys

    from pdf_reader import extract_text

    pdf_path = sys.argv[1]
    debug = "--debug" in sys.argv
    text = extract_text(pdf_path)
    fields = extract_fields(text, debug=debug)
    print(json.dumps(fields, indent=2, ensure_ascii=False))