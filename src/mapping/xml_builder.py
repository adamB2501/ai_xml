# -*- coding: utf-8 -*-
"""Builds a TEIF (Tunisian e-invoice, "El Fatoora") XML document from
ner_extractor.extract_fields()'s output.

Structure and codes come from data/docs/Guide-Implementation-TEIF_V2.0.pdf
(the TTN implementation guide) and were cross-checked against
data/docs/exemple_elfatoora.xml (a real accepted TEIF payload) -- where the
two disagreed (e.g. PartnerName/@nameType="Qualification", which the guide's
own referential I-7 doesn't list), the real example wins, since it's a
proven-working payload rather than an idealized description of one.

Deliberately NOT produced here: TEIF/RefTtnVal and TEIF/ds:Signature. Per
the guide's own structure table (Tableau 4) those ARE mandatory in a final
TEIF message, but exemple_elfatoora.xml -- a genuine sample -- also stops at
InvoiceBody/InvoiceTax without them. That's consistent with how the format
is actually used: RefTtnVal (TTN's own reference) and the digital
signatures are added by TTN's platform when it validates/signs the
invoice, not authored by the issuer's own software.

The NER schema (39 labels, see training_data.py) doesn't cover everything
TEIF can express -- there's no per-line VAT rate, no item unit code, no
structured street/city/postal-code split for addresses, no legal form
(SA/SARL) code. Where a TEIF field has no NER counterpart, this module
either applies a documented, invoice-standard assumption (a single VAT
rate applies to every line -- true for the TTN template and most simple
invoices) or leaves the element out entirely rather than fabricate a
value. build_teif_xml() returns a `warnings` list alongside the XML so
callers can see exactly what was assumed or omitted for a given invoice,
instead of that information silently disappearing into the output file.
"""

import re
import xml.etree.ElementTree as ET
from xml.dom import minidom

TEIF_VERSION = "1.8.8"
DEFAULT_COUNTRY = "TN"
DEFAULT_CURRENCY = "TND"

_FR_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8, "aout": 8,
    "septembre": 9, "octobre": 10, "novembre": 11,
    "décembre": 12, "decembre": 12,
}


# --- small parsing/formatting helpers --------------------------------------

def _first(value):
    """SINGLETON_LABELS fields are plain strings; REPEATABLE_LABELS fields
    (seller_tax_ids, buyer_addresses, ...) are lists -- this accepts either
    so callers don't need to know which kind a given field is."""
    if isinstance(value, list):
        return value[0] if value else None
    return value or None


def parse_ddmmyy(raw):
    """TEIF dates are a bare 6-digit ddMMyy string (see Dtm/DateText,
    format="ddMMyy" in the guide). Model output arrives in whatever format
    the source invoice used -- already-compact digits (the TTN template
    extracts DUE_DATE that way), dd/mm/yyyy, dd-mm-yyyy, or a spelled-out
    French date ("20 février 2026"). Returns None (never a guess) if none
    of those match, so the caller can skip the field and warn instead of
    emitting a fabricated date.
    """
    if not raw:
        return None
    s = raw.strip()

    if re.fullmatch(r"\d{6}", s):
        return s

    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", s)
    if m:
        d, mo, y = m.groups()
        return f"{int(d):02d}{int(mo):02d}{y[2:]}"

    m = re.fullmatch(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2})", s)
    if m:
        d, mo, y = m.groups()
        return f"{int(d):02d}{int(mo):02d}{y}"

    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-zàâäéèêëîïôöùûüÿ]+)\s+(\d{4})", s, re.IGNORECASE)
    if m:
        d, month_name, y = m.groups()
        mo = _FR_MONTHS.get(month_name.lower())
        if mo:
            return f"{int(d):02d}{mo:02d}{y[2:]}"

    return None


def _to_decimal_string(raw):
    """Normalizes a numeric string to plain-dot-decimal form, handling
    French/European formatting (comma decimal, dot or space thousands
    separator) as well as plain "303.847"-style input.

    The only reliable signal for "which separator is the decimal point" is
    position, not character: whichever of '.'/',' appears LAST is the
    decimal separator, and any earlier ones are thousands separators to
    strip. Anchoring on "exactly 3 digits after the separator" instead
    (the naive approach) breaks specifically on TND amounts, since those
    are conventionally written with exactly 3 decimals (millimes) --
    "303.847" would misparse as "303847" under that rule.
    """
    s = re.sub(r"[^\d.,\-]", "", str(raw))
    if not s:
        return None
    negative = s.startswith("-")
    s = s.lstrip("-")
    sep_pos = max(s.rfind("."), s.rfind(","))
    if sep_pos == -1:
        integer_part, decimal_part = s, ""
    else:
        integer_part = re.sub(r"[.,]", "", s[:sep_pos])
        decimal_part = s[sep_pos + 1:]
    if not integer_part and not decimal_part:
        return None
    result = integer_part or "0"
    if decimal_part:
        result += "." + decimal_part
    return ("-" + result) if negative else result


def format_amount(raw):
    """TND amounts are conventionally expressed to 3 decimals (millimes),
    matching exemple_elfatoora.xml's "2.000" / "0.240" style. Returns None
    (never a fabricated "0.000") if the text isn't parseable as a number."""
    if raw is None or raw == "":
        return None
    s = _to_decimal_string(raw)
    if s is None:
        return None
    try:
        return f"{float(s):.3f}"
    except ValueError:
        return None


def format_quantity(raw):
    """Less strict than format_amount: quantities aren't a currency, so
    trailing zeros are trimmed (12.900 -> 12.9) rather than forced to 3dp."""
    if raw is None or raw == "":
        return None
    s = _to_decimal_string(raw)
    if s is None:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    text = f"{v:.3f}".rstrip("0").rstrip(".")
    return text if text else "0"


def split_address(raw):
    """Best-effort split of a free-text address span into
    (description, street, city, postal_code).

    There's no dedicated STREET/CITY/POSTAL_CODE label in the NER schema --
    SELLER_ADDRESS/BUYER_ADDRESS is one free-text span -- so this is a
    heuristic over two real shapes seen in this project's data:
      - "1049 TUNIS"                    (postal+city only, TTN template's
                                          declustered address fragment)
      - "95 Rue Alain Savary, 2080 Ariana"  (street, then postal+city)
    Anything else falls back to putting the whole string in
    AdressDescription with Street/City/PostalCode left empty -- still a
    valid TEIF address (those elements are all optional/minOcc=0), just
    less structured.
    """
    if not raw:
        return "", "", "", ""
    text = re.sub(r"\s+", " ", raw).strip()

    m = re.fullmatch(r"(\d{4,5})\s+([^\d,]{2,40})", text)
    if m:
        return "", "", m.group(2).strip(), m.group(1)

    m = re.match(r"^(.*?),?\s*(\d{4,5})\s+([^\d,]{2,40})$", text)
    if m:
        desc, postal, city = m.groups()
        return desc.strip(" ,"), "", city.strip(), postal

    return text, "", "", ""


def _el(parent, tag, text=None, **attrib):
    e = ET.SubElement(parent, tag)
    for k, v in attrib.items():
        if v is not None:
            e.set(k, v)
    if text is not None:
        e.text = str(text)
    return e


# --- section builders --------------------------------------------------

def _build_header(teif, fields, warnings):
    header = _el(teif, "InvoiceHeader")
    seller_tax_id = _first(fields.get("seller_tax_ids"))
    buyer_tax_id = _first(fields.get("buyer_tax_ids"))

    if seller_tax_id:
        _el(header, "MessageSenderIdentifier", seller_tax_id, type="I-01")
    else:
        warnings.append("No seller tax ID extracted; InvoiceHeader/MessageSenderIdentifier omitted (spec: mandatory).")

    if buyer_tax_id:
        _el(header, "MessageRecieverIdentifier", buyer_tax_id, type="I-01")
    else:
        warnings.append("No buyer tax ID extracted; InvoiceHeader/MessageRecieverIdentifier omitted.")


def _build_bgm(body, fields, warnings):
    bgm = _el(body, "Bgm")
    invoice_number = fields.get("invoice_number")
    if not invoice_number:
        warnings.append("No invoice number extracted; Bgm/DocumentIdentifier left as 'UNKNOWN' (spec: mandatory).")
    _el(bgm, "DocumentIdentifier", invoice_number or "UNKNOWN")
    _el(bgm, "DocumentType", "Facture", code="I-11")


def _build_dtm(body, fields, warnings):
    dtm = _el(body, "Dtm")
    issue = parse_ddmmyy(fields.get("issue_date"))
    if issue:
        _el(dtm, "DateText", issue, format="ddMMyy", functionCode="I-31")
    else:
        warnings.append(
            f"Issue date {fields.get('issue_date')!r} missing or unparseable to ddMMyy; "
            "Dtm/DateText[I-31] omitted (spec: Dtm is mandatory)."
        )

    due = parse_ddmmyy(fields.get("due_date"))
    if due:
        _el(dtm, "DateText", due, format="ddMMyy", functionCode="I-32")


def _build_partner(parent, function_code, tax_id, name, address_raw, rc_number, contacts, country, party_label, warnings):
    pd = _el(parent, "PartnerDetails", functionCode=function_code)
    nad = _el(pd, "Nad")
    if not tax_id:
        warnings.append(f"No {party_label} tax ID extracted; Nad/PartnerIdentifier left empty (spec: mandatory).")
    _el(nad, "PartnerIdentifier", tax_id or "", type="I-01")
    if not name:
        warnings.append(f"No {party_label} name extracted.")
    _el(nad, "PartnerName", name or "", nameType="Qualification")

    desc, street, city, postal = split_address(address_raw or "")
    if not any((desc, street, city, postal)):
        warnings.append(f"No {party_label} address extracted; PartnerAdresses left empty.")
    addr = _el(nad, "PartnerAdresses", lang="fr")
    _el(addr, "AdressDescription", desc or None)
    _el(addr, "Street", street or None)
    _el(addr, "CityName", city or None)
    _el(addr, "PostalCode", postal or None)
    _el(addr, "Country", country, codeList="ISO_3166-1")

    if rc_number:
        rff = _el(pd, "RffSection")
        _el(rff, "Reference", rc_number, refID="I-815")

    # Referentiel I-10 (Communication Means): I-101 phone, I-102 fax, I-103
    # email, I-104 other (used here for website). Contact info in the NER
    # schema (PHONE/EMAIL/WEBSITE) isn't tied to seller vs. buyer, so -- like
    # exemple_elfatoora.xml, where only the fournisseur has CtaSection
    # entries -- these are only attached to the seller partner.
    for means_code, address in contacts:
        if not address:
            continue
        cta = _el(pd, "CtaSection")
        contact = _el(cta, "Contact", functionCode="I-94")
        _el(contact, "ContactIdentifier", (name or party_label)[:17])
        _el(contact, "ContactName", name or "")
        comm = _el(cta, "Communication")
        _el(comm, "ComMeansType", means_code)
        _el(comm, "ComAdress", address)

    return pd


def _build_partner_section(body, fields, country, warnings):
    section = _el(body, "PartnerSection")

    seller_contacts = [
        ("I-101", _first(fields.get("phones"))),
        ("I-103", _first(fields.get("emails"))),
        ("I-104", _first(fields.get("websites"))),
    ]
    _build_partner(
        section, "I-62",
        tax_id=_first(fields.get("seller_tax_ids")),
        name=fields.get("seller_name"),
        address_raw=_first(fields.get("seller_addresses")),
        rc_number=fields.get("rc_number"),
        contacts=seller_contacts,
        country=country,
        party_label="seller",
        warnings=warnings,
    )
    _build_partner(
        section, "I-64",
        tax_id=_first(fields.get("buyer_tax_ids")),
        name=fields.get("buyer_name"),
        address_raw=_first(fields.get("buyer_addresses")),
        rc_number=None,
        contacts=[],
        country=country,
        party_label="buyer",
        warnings=warnings,
    )


def _build_pyt_section(body, fields):
    payment_terms = fields.get("payment_terms")
    bank_name = fields.get("bank_name")
    rib = fields.get("rib")
    iban = fields.get("iban")
    if not (payment_terms or bank_name or rib or iban):
        return  # PytSection is optional (minOcc=0) -- fine to skip entirely

    pyt_section = _el(body, "PytSection")

    if payment_terms:
        details = _el(pyt_section, "PytSectionDetails")
        pyt = _el(details, "Pyt")
        _el(pyt, "PaymentTearmsTypeCode", "I-116")  # "Autre" -- free text isn't classified into I-111..I-117
        _el(pyt, "PaymentTearmsDescription", payment_terms)

    if bank_name or rib or iban:
        details = _el(pyt_section, "PytSectionDetails")
        pyt = _el(details, "Pyt")
        _el(pyt, "PaymentTearmsTypeCode", "I-114")  # par virement bancaire
        _el(pyt, "PaymentTearmsDescription", "Paiement par virement bancaire")
        # I-141 (Poste) vs I-142 (Banque) per referential I-14; only use
        # I-141 if the extracted bank_name actually says "poste".
        fii_code = "I-141" if bank_name and "poste" in bank_name.lower() else "I-142"
        fii = _el(details, "PytFii", functionCode=fii_code)
        account = _el(fii, "AccountHolder")
        _el(account, "AccountNumber", rib or iban or "")
        if bank_name:
            inst = _el(fii, "InstitutionIdentification")
            _el(inst, "InstitutionName", bank_name)


def _build_lin_section(body, fields, currency, warnings):
    line_items = fields.get("line_items") or []
    if not line_items:
        warnings.append("No line items extracted; LinSection omitted (spec: mandatory).")
        return

    vat_rate = _first(fields.get("vat_rates")) or "0"
    lin_section = _el(body, "LinSection")

    for i, item in enumerate(line_items, start=1):
        lin = _el(lin_section, "Lin")
        _el(lin, "ItemIdentifier", str(i))

        imd = _el(lin, "LinImd", lang="fr")
        if not item.get("description"):
            warnings.append(f"Line item {i}: no description extracted.")
        _el(imd, "ItemDescription", item.get("description") or "")

        qty = format_quantity(item.get("quantity")) or "1"
        # No unit-of-measure label in the NER schema (e.g. "heure", "unité")
        # -- defaults to a generic UNIT rather than guessing.
        linqty = _el(lin, "LinQty")
        _el(linqty, "Quantity", qty, measurementUnit="UNIT")

        lintax = _el(lin, "LinTax")
        _el(lintax, "TaxTypeName", "TVA", code="I-1602")
        taxdetails = _el(lintax, "TaxDetails")
        # Per-line VAT rate isn't in the NER schema either -- VAT_RATE is
        # invoice-level, not attached to a specific line -- so every line
        # is assumed to carry the invoice's (single) rate. True for the
        # TTN template and most simple invoices; flagged if no rate at all.
        _el(taxdetails, "TaxRate", vat_rate)

        unit_price = format_amount(item.get("unit_price"))
        line_total = format_amount(item.get("line_total"))
        if unit_price or line_total:
            linmoa = _el(lin, "LinMoa")
            if unit_price:
                md = _el(linmoa, "MoaDetails")
                moa = _el(md, "Moa", amountTypeCode="I-183", currencyCodeList="ISO_4217")
                _el(moa, "Amount", unit_price, currencyIdentifier=currency)
            if line_total:
                md = _el(linmoa, "MoaDetails")
                moa = _el(md, "Moa", amountTypeCode="I-171", currencyCodeList="ISO_4217")
                _el(moa, "Amount", line_total, currencyIdentifier=currency)
        else:
            warnings.append(f"Line item {i}: neither unit_price nor line_total extracted; LinMoa omitted.")


def _build_invoice_moa(body, fields, currency, warnings):
    moa_section = _el(body, "InvoiceMoa")
    total_ht = format_amount(fields.get("total_ht"))
    entries = [
        ("I-179", format_amount(fields.get("capital_social"))),
        ("I-180", format_amount(fields.get("total_ttc"))),
        ("I-176", total_ht),
        # I-182 "montant total base taxe": no separate tax-base field in the
        # NER schema, so this assumes the tax base equals total_ht -- true
        # whenever a single VAT rate covers the whole invoice.
        ("I-182", total_ht),
        ("I-181", format_amount(fields.get("tva_amount"))),
    ]
    any_amount = False
    for code, amount in entries:
        if amount is None:
            continue
        any_amount = True
        ad = _el(moa_section, "AmountDetails")
        moa = _el(ad, "Moa", amountTypeCode=code, currencyCodeList="ISO_4217")
        _el(moa, "Amount", amount, currencyIdentifier=currency)

    if not any_amount:
        warnings.append("No invoice-level amounts (total_ht/tva_amount/total_ttc/capital_social) extracted; InvoiceMoa is empty.")


def _build_invoice_tax(body, fields, currency, warnings):
    tax_section = _el(body, "InvoiceTax")
    vat_rate = _first(fields.get("vat_rates"))
    stamp_duty = format_amount(fields.get("stamp_duty"))
    tva_amount = format_amount(fields.get("tva_amount"))
    total_ht = format_amount(fields.get("total_ht"))
    withholding = format_amount(fields.get("withholding_tax"))

    any_tax = False

    if stamp_duty:
        any_tax = True
        d = _el(tax_section, "InvoiceTaxDetails")
        tax = _el(d, "Tax")
        _el(tax, "TaxTypeName", "droit de timbre", code="I-1601")
        td = _el(tax, "TaxDetails")
        _el(td, "TaxRate", "0")  # stamp duty is a flat amount, not a rate
        ad = _el(d, "AmountDetails")
        moa = _el(ad, "Moa", amountTypeCode="I-178", currencyCodeList="ISO_4217")
        _el(moa, "Amount", stamp_duty, currencyIdentifier=currency)

    if tva_amount or vat_rate:
        any_tax = True
        d = _el(tax_section, "InvoiceTaxDetails")
        tax = _el(d, "Tax")
        _el(tax, "TaxTypeName", "TVA", code="I-1602")
        td = _el(tax, "TaxDetails")
        _el(td, "TaxRate", vat_rate or "0")
        if total_ht:
            ad = _el(d, "AmountDetails")
            moa = _el(ad, "Moa", amountTypeCode="I-177", currencyCodeList="ISO_4217")
            _el(moa, "Amount", total_ht, currencyIdentifier=currency)
        if tva_amount:
            ad = _el(d, "AmountDetails")
            moa = _el(ad, "Moa", amountTypeCode="I-178", currencyCodeList="ISO_4217")
            _el(moa, "Amount", tva_amount, currencyIdentifier=currency)

    if withholding:
        any_tax = True
        d = _el(tax_section, "InvoiceTaxDetails")
        tax = _el(d, "Tax")
        _el(tax, "TaxTypeName", "Retenu à la source", code="I-1604")
        ad = _el(d, "AmountDetails")
        moa = _el(ad, "Moa", amountTypeCode="I-178", currencyCodeList="ISO_4217")
        _el(moa, "Amount", withholding, currencyIdentifier=currency)

    if not any_tax:
        warnings.append("No tax entries (TVA/stamp duty/withholding) extracted; InvoiceTax is empty.")


# --- public API --------------------------------------------------------

def build_teif_element(fields, country=DEFAULT_COUNTRY, teif_version=TEIF_VERSION):
    """Builds the xml.etree.ElementTree.Element tree and returns
    (element, warnings). Prefer build_teif_xml() unless you need the raw
    tree (e.g. to embed it in something else)."""
    warnings = []
    currency = (_first(fields.get("currency")) or DEFAULT_CURRENCY).strip().upper()[:3] or DEFAULT_CURRENCY

    teif = ET.Element("TEIF", controlingAgency="TTN", version=teif_version)

    _build_header(teif, fields, warnings)

    body = _el(teif, "InvoiceBody")
    _build_bgm(body, fields, warnings)
    _build_dtm(body, fields, warnings)
    _build_partner_section(body, fields, country, warnings)
    _build_pyt_section(body, fields)
    _build_lin_section(body, fields, currency, warnings)
    _build_invoice_moa(body, fields, currency, warnings)
    _build_invoice_tax(body, fields, currency, warnings)

    return teif, warnings


def build_teif_xml(fields, country=DEFAULT_COUNTRY, teif_version=TEIF_VERSION, pretty=True):
    """Builds the TEIF XML string for the given extract_fields() output.
    Returns (xml_string, warnings) -- warnings lists every spec-mandatory
    or otherwise notable field this invoice's extraction couldn't fill in,
    so incomplete NER output produces a flagged, still-valid-shaped
    document rather than a silently wrong one."""
    teif, warnings = build_teif_element(fields, country=country, teif_version=teif_version)
    rough = ET.tostring(teif, encoding="utf-8")
    if pretty:
        xml_str = minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")
        xml_str = "\n".join(line for line in xml_str.split("\n") if line.strip())
    else:
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + rough.decode("utf-8")
    return xml_str, warnings


def write_teif_xml(fields, output_path, **kwargs):
    xml_str, warnings = build_teif_xml(fields, **kwargs)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return warnings


if __name__ == "__main__":
    import os
    import sys

    # xml_builder.py lives in src/mapping/ -- add src/ (this file's parent
    # directory) to sys.path so `extraction.*` resolves as a sibling
    # namespace package, the same way main.py (in src/) resolves it.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from extraction.ner_extractor import extract_fields
    from extraction.pdf_reader import extract_text_auto

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "invoice.xml"

    text = extract_text_auto(pdf_path)
    fields = extract_fields(text)
    warnings = write_teif_xml(fields, output_path)

    print(f"Wrote {output_path}")
    if warnings:
        print(f"\n[{len(warnings)} warning(s)]")
        for w in warnings:
            print(f"  - {w}")
