# -*- coding: utf-8 -*-
"""
STEP 4a of the pipeline:  fields dict  ->  canonical Invoice object.

The Invoice model, the TEIF XML builder, and the TEIF validator all live
in teif_pipeline and are already correct - we reuse them unchanged. This
file is only the adapter: it maps our flat fields dict (the shape the LLM
returns, defined in prompts.FIELD_SPEC) onto teif_pipeline.models.Invoice.

Rules kept consistent with the rest of the project:
  - a value that won't parse as a number/date becomes None, never a
    plausible-looking default
  - nothing is invented; a missing field stays missing and the TEIF
    builder will emit its own "mandatory field absent" warning
  - the raw fields dict and our processing notes ride along on the Invoice
    (source_text / extraction_notes / field_provenance) for the audit trail
"""

from __future__ import annotations

from decimal import Decimal

from teif_pipeline.models import (
    Address,
    Invoice,
    LineItem,
    Party,
    PaymentTerms,
    TaxLine,
)

from .numparse import to_date, to_decimal


def _vat_rate_from_items(fields: dict) -> Decimal | None:
    """Most Tunisian invoices have one VAT rate for the whole document.
    Take the first non-null line rate we see."""
    for it in fields.get("line_items") or []:
        r = to_decimal(it.get("vat_rate_percent"))
        if r is not None:
            return r
    return None


def _line_items(fields: dict, doc_vat_rate: Decimal | None) -> list[LineItem]:
    out = []
    for it in fields.get("line_items") or []:
        out.append(LineItem(
            item_code=(it.get("code") or None),
            description=(it.get("description") or None),
            quantity=to_decimal(it.get("quantity")),
            unit_price_ht=to_decimal(it.get("unit_price_ht")),
            line_total_ht=to_decimal(it.get("line_total_ht")),
            vat_rate_percent=to_decimal(it.get("vat_rate_percent")) or doc_vat_rate,
        ))
    return out


def _taxes(fields: dict, total_ht: Decimal | None, vat_rate: Decimal | None) -> list[TaxLine]:
    """One TaxLine per tax type actually present - matches how
    teif_pipeline builds InvoiceTax."""
    taxes: list[TaxLine] = []

    stamp = to_decimal(fields.get("stamp_duty"))
    if stamp is not None:
        taxes.append(TaxLine(
            tax_type_code="I-1601", tax_type_label="droit de timbre",
            rate_percent=Decimal(0), tax_amount=stamp,
        ))

    tva = to_decimal(fields.get("tva_amount"))
    if tva is not None or vat_rate is not None:
        taxes.append(TaxLine(
            tax_type_code="I-1602", tax_type_label="TVA",
            rate_percent=vat_rate, base_amount=total_ht, tax_amount=tva,
        ))

    return taxes


def _payment(fields: dict) -> PaymentTerms | None:
    """Footer bank details, when the vision model read any."""
    keys = ("bank_name", "rib", "iban", "swift_bic", "payment_terms")
    if not any(fields.get(k) for k in keys):
        return None
    return PaymentTerms(
        description=fields.get("payment_terms") or None,
        bank_name=fields.get("bank_name") or None,
        rib=fields.get("rib") or None,
        iban=fields.get("iban") or None,
        swift_bic=fields.get("swift_bic") or None,
    )


def to_invoice(fields: dict, *, source_text: str, model_used: str,
               notes: list[str] | None = None) -> Invoice:
    total_ht = to_decimal(fields.get("total_ht"))
    vat_rate = _vat_rate_from_items(fields)

    other_refs = fields.get("other_references") or []
    unmapped = {"other_references": other_refs} if other_refs else {}

    return Invoice(
        invoice_number=fields.get("invoice_number") or None,
        issue_date=to_date(fields.get("issue_date")),

        seller=Party(
            name=fields.get("seller_name") or None,
            tax_id=fields.get("seller_tax_id") or None,
            rc_number=fields.get("seller_rc_number") or None,
            address=Address(raw_text=fields.get("seller_address") or None,
                            description=fields.get("seller_address") or None),
        ),
        buyer=Party(
            name=fields.get("buyer_name") or None,
            tax_id=fields.get("buyer_tax_id") or None,
            address=Address(raw_text=fields.get("buyer_address") or None,
                            description=fields.get("buyer_address") or None),
        ),

        line_items=_line_items(fields, vat_rate),

        currency=(fields.get("currency") or "TND"),
        total_ht=total_ht,
        tva_amount=to_decimal(fields.get("tva_amount")),
        total_ttc=to_decimal(fields.get("total_ttc")),
        stamp_duty=to_decimal(fields.get("stamp_duty")),
        capital_social=to_decimal(fields.get("seller_capital")),
        taxes=_taxes(fields, total_ht, vat_rate),

        payment=_payment(fields),

        source_text=source_text,
        extraction_backend=f"accurate_pipe:{model_used}",
        field_provenance={k: f"accurate_pipe:{model_used}"
                          for k, v in fields.items() if v and not k.startswith("_")},
        unmapped_entities=unmapped,
        extraction_notes=list(notes or []),
    )
