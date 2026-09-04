# -*- coding: utf-8 -*-
"""The canonical Invoice model -- the one contract every ingestion adapter,
extraction backend, validation gate, and the TEIF builder all speak.

Design decisions made here (see docs/phase0_audit.md §8 for the open
questions this resolves):

  - Almost everything is Optional. Per the brief: "Nothing reaches the
    builder without passing validation gates" -- that statement only makes
    sense if the model itself doesn't enforce TEIF-mandatory-ness, the
    gates do. An Invoice needs to be constructible even when badly
    incomplete, so a failed extraction can still flow into the review
    queue (Phase 6) as a first-class object instead of an exception.
  - Provenance (`name_alternates`, `field_provenance`, `unmapped_entities`)
    lives ON the model, not in a side-channel. That's what makes the
    review queue's "surface uncertain spans next to the PDF" workable
    against one object instead of a join.
  - `LineItem.vat_rate_percent` exists even though no backend fills it
    today (every source assumes one invoice-level rate) -- the model is a
    superset of what's extractable now, not just what's extracted today.
"""

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PartnerIdType(str, Enum):
    """Referentiel I-0 (Partner Identifier Type), from the TTN guide."""

    MATRICULE_FISCAL_TN = "I-01"
    CIN = "I-02"
    CARTE_SEJOUR = "I-03"
    MATRICULE_FISCAL_ETRANGER = "I-04"


class Address(BaseModel):
    description: Optional[str] = None  # AdressDescription -- freeform remainder
    street: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    country: str = "TN"  # ISO 3166-1
    raw_text: Optional[str] = None  # original extracted span, kept for audit


class Party(BaseModel):
    tax_id: Optional[str] = None
    tax_id_type: PartnerIdType = PartnerIdType.MATRICULE_FISCAL_TN
    name: Optional[str] = None
    name_alternates: list[str] = Field(default_factory=list)  # conflicting values a gate discarded
    address: Address = Field(default_factory=Address)
    rc_number: Optional[str] = None  # registre de commerce -- seller, in practice
    phone: Optional[str] = None
    fax: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None


class LineItem(BaseModel):
    item_code: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[Decimal] = None
    unit_of_measure: str = "UNIT"  # no extractor supplies this today
    unit_price_ht: Optional[Decimal] = None
    line_total_ht: Optional[Decimal] = None
    vat_rate_percent: Optional[Decimal] = None  # per-line; falls back to Invoice-level rate if unset
    inferred_fields: list[str] = Field(default_factory=list)  # arithmetically backfilled, not extracted


class TaxLine(BaseModel):
    tax_type_code: str  # I-1601 droit de timbre / I-1602 TVA / I-1604 retenue a la source / ...
    tax_type_label: str
    rate_percent: Optional[Decimal] = None
    base_amount: Optional[Decimal] = None  # I-177
    tax_amount: Optional[Decimal] = None  # I-178


class PaymentTerms(BaseModel):
    description: Optional[str] = None
    bank_name: Optional[str] = None
    rib: Optional[str] = None
    iban: Optional[str] = None
    swift_bic: Optional[str] = None


class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    document_type_code: str = "I-11"  # Facture; not extracted today, always assumed

    issue_date: Optional[date] = None
    due_date: Optional[date] = None
    period_start: Optional[date] = None  # no source label today; TEIF (I-36) supports it
    period_end: Optional[date] = None

    seller: Party = Field(default_factory=Party)
    buyer: Party = Field(default_factory=Party)

    line_items: list[LineItem] = Field(default_factory=list)

    currency: str = "TND"
    total_ht: Optional[Decimal] = None
    tva_amount: Optional[Decimal] = None
    total_ttc: Optional[Decimal] = None
    stamp_duty: Optional[Decimal] = None
    capital_social: Optional[Decimal] = None
    taxes: list[TaxLine] = Field(default_factory=list)

    payment: Optional[PaymentTerms] = None

    unique_reference: Optional[str] = None  # TTN reference -- no NER label maps to it today

    # --- provenance / audit trail -- not part of TEIF, needed by the pipeline itself
    source_text: Optional[str] = None  # raw extracted text this record was built from
    extraction_backend: Optional[str] = None  # "spacy_ner" | "llm:<model>" | "hybrid" | "database"
    field_provenance: dict[str, str] = Field(default_factory=dict)  # per top-level field: which backend supplied it
    unmapped_entities: dict[str, list[str]] = Field(default_factory=dict)  # never silently dropped
    extraction_notes: list[str] = Field(default_factory=list)  # e.g. "currency X wasn't a recognized code, defaulted to TND"
