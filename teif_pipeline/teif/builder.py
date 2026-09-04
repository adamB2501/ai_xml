# -*- coding: utf-8 -*-
"""Builds TEIF XML from a canonical Invoice. Refactored from
src/mapping/xml_builder.py: XML construction (tags, attributes, referential
codes, structure) is unchanged from that file for every case the
characterization tests cover -- only the input source changed, from a
`fields.get(...)` dict lookup to reading a typed Invoice/Party/LineItem.

One deliberate exception, disclosed rather than silent: _build_invoice_tax
now iterates `invoice.taxes` (a generic list) instead of three separate
hardcoded if-blocks for stamp duty / TVA / withholding. The old dict had no
place to put a fourth tax type without a fourth hardcoded block; the model
does (see models.py's TaxLine), so keeping the hardcoded structure here
would mean not actually using the model's generality, or maintaining two
parallel representations of the same data. The XML each currently-supported
tax type emits is byte-identical -- see tests/characterization.

Still NOT produced here, per the original file's own reasoning: TEIF/RefTtnVal
and TEIF/ds:Signature -- added by TTN's platform when it validates/signs the
invoice, not authored by the issuer's own software.
"""

import xml.etree.ElementTree as ET
from decimal import Decimal
from xml.dom import minidom

from ..models import Invoice, Party
from ..numeric import format_amount, format_ddmmyy, format_quantity
from .codes import (
    DEFAULT_COUNTRY,
    TEIF_VERSION,
)

DOCUMENT_TYPE_LABELS = {"I-11": "Facture", "I-12": "Facture d'avoir"}


def _el(parent, tag, text=None, **attrib):
    e = ET.SubElement(parent, tag)
    for k, v in attrib.items():
        if v is not None:
            e.set(k, v)
    if text is not None:
        e.text = str(text)
    return e


def _amount_str(value: Decimal | None) -> str | None:
    return format_amount(value)


# --- section builders --------------------------------------------------

def _build_header(teif, invoice: Invoice, warnings):
    header = _el(teif, "InvoiceHeader")
    if invoice.seller.tax_id:
        _el(header, "MessageSenderIdentifier", invoice.seller.tax_id, type=invoice.seller.tax_id_type.value)
    else:
        warnings.append("No seller tax ID; InvoiceHeader/MessageSenderIdentifier omitted (spec: mandatory).")

    if invoice.buyer.tax_id:
        _el(header, "MessageRecieverIdentifier", invoice.buyer.tax_id, type=invoice.buyer.tax_id_type.value)
    else:
        warnings.append("No buyer tax ID; InvoiceHeader/MessageRecieverIdentifier omitted.")


def _build_bgm(body, invoice: Invoice, warnings):
    bgm = _el(body, "Bgm")
    if not invoice.invoice_number:
        warnings.append("No invoice number; Bgm/DocumentIdentifier left as 'UNKNOWN' (spec: mandatory).")
    _el(bgm, "DocumentIdentifier", invoice.invoice_number or "UNKNOWN")
    label = DOCUMENT_TYPE_LABELS.get(invoice.document_type_code, "Facture")
    _el(bgm, "DocumentType", label, code=invoice.document_type_code)


def _build_dtm(body, invoice: Invoice, warnings):
    dtm = _el(body, "Dtm")
    issue = format_ddmmyy(invoice.issue_date)
    if issue:
        _el(dtm, "DateText", issue, format="ddMMyy", functionCode="I-31")
    else:
        warnings.append("Issue date missing or unparseable; Dtm/DateText[I-31] omitted (spec: Dtm is mandatory).")

    due = format_ddmmyy(invoice.due_date)
    if due:
        _el(dtm, "DateText", due, format="ddMMyy", functionCode="I-32")

    period_start = format_ddmmyy(invoice.period_start)
    period_end = format_ddmmyy(invoice.period_end)
    if period_start and period_end:
        _el(dtm, "DateText", f"{period_start}-{period_end}", format="ddMMyy-ddMMyy", functionCode="I-36")


def _build_partner(parent, function_code: str, party: Party, country: str, party_label: str, warnings, include_contacts: bool):
    pd = _el(parent, "PartnerDetails", functionCode=function_code)
    nad = _el(pd, "Nad")
    if not party.tax_id:
        warnings.append(f"No {party_label} tax ID; Nad/PartnerIdentifier left empty (spec: mandatory).")
    _el(nad, "PartnerIdentifier", party.tax_id or "", type=party.tax_id_type.value)
    if not party.name:
        warnings.append(f"No {party_label} name extracted.")
    _el(nad, "PartnerName", party.name or "", nameType="Qualification")

    addr = party.address
    if not any((addr.description, addr.street, addr.city, addr.postal_code)):
        warnings.append(f"No {party_label} address extracted; PartnerAdresses left empty.")
    adresses = _el(nad, "PartnerAdresses", lang="fr")
    _el(adresses, "AdressDescription", addr.description)
    _el(adresses, "Street", addr.street)
    _el(adresses, "CityName", addr.city)
    _el(adresses, "PostalCode", addr.postal_code)
    _el(adresses, "Country", country or addr.country, codeList="ISO_3166-1")

    if party.rc_number:
        rff = _el(pd, "RffSection")
        _el(rff, "Reference", party.rc_number, refID="I-815")

    # Referentiel I-9/I-10: contact function + communication means. Like
    # exemple_elfatoora.xml, where only the fournisseur has CtaSection
    # entries, contacts are only emitted for the party the caller marks
    # include_contacts=True for (the seller, in the current bridge).
    if include_contacts:
        for means_code, value in (("I-101", party.phone), ("I-102", party.fax), ("I-103", party.email), ("I-104", party.website)):
            if not value:
                continue
            cta = _el(pd, "CtaSection")
            contact = _el(cta, "Contact", functionCode="I-94")
            _el(contact, "ContactIdentifier", (party.name or party_label)[:17])
            _el(contact, "ContactName", party.name or "")
            comm = _el(cta, "Communication")
            _el(comm, "ComMeansType", means_code)
            _el(comm, "ComAdress", value)

    return pd


def _build_partner_section(body, invoice: Invoice, country: str, warnings):
    section = _el(body, "PartnerSection")
    _build_partner(section, "I-62", invoice.seller, country, "seller", warnings, include_contacts=True)
    _build_partner(section, "I-64", invoice.buyer, country, "buyer", warnings, include_contacts=False)


def _build_pyt_section(body, invoice: Invoice):
    payment = invoice.payment
    if payment is None:
        return  # PytSection is optional (minOcc=0)

    pyt_section = _el(body, "PytSection")

    if payment.description:
        details = _el(pyt_section, "PytSectionDetails")
        pyt = _el(details, "Pyt")
        _el(pyt, "PaymentTearmsTypeCode", "I-116")  # "Autre" -- free text isn't classified into I-111..I-117
        _el(pyt, "PaymentTearmsDescription", payment.description)

    if payment.bank_name or payment.rib or payment.iban:
        details = _el(pyt_section, "PytSectionDetails")
        pyt = _el(details, "Pyt")
        _el(pyt, "PaymentTearmsTypeCode", "I-114")  # par virement bancaire
        _el(pyt, "PaymentTearmsDescription", "Paiement par virement bancaire")
        fii_code = "I-141" if payment.bank_name and "poste" in payment.bank_name.lower() else "I-142"
        fii = _el(details, "PytFii", functionCode=fii_code)
        account = _el(fii, "AccountHolder")
        _el(account, "AccountNumber", payment.rib or payment.iban or "")
        if payment.bank_name:
            inst = _el(fii, "InstitutionIdentification")
            _el(inst, "InstitutionName", payment.bank_name)


def _build_lin_section(body, invoice: Invoice, currency: str, warnings):
    if not invoice.line_items:
        warnings.append("No line items; LinSection omitted (spec: mandatory).")
        return

    lin_section = _el(body, "LinSection")
    for i, item in enumerate(invoice.line_items, start=1):
        lin = _el(lin_section, "Lin")
        _el(lin, "ItemIdentifier", str(i))

        imd = _el(lin, "LinImd", lang="fr")
        if not item.description:
            warnings.append(f"Line item {i}: no description extracted.")
        _el(imd, "ItemDescription", item.description or "")

        qty = format_quantity(item.quantity) or "1"
        linqty = _el(lin, "LinQty")
        _el(linqty, "Quantity", qty, measurementUnit=item.unit_of_measure)

        lintax = _el(lin, "LinTax")
        _el(lintax, "TaxTypeName", "TVA", code="I-1602")
        taxdetails = _el(lintax, "TaxDetails")
        # Not routed through format_amount()'s 3dp formatting -- a tax RATE
        # (e.g. "12.0") isn't a millimes-precision currency amount, and the
        # original always wrote the raw percentage text as-is.
        _el(taxdetails, "TaxRate", str(item.vat_rate_percent) if item.vat_rate_percent is not None else "0")

        unit_price = _amount_str(item.unit_price_ht)
        line_total = _amount_str(item.line_total_ht)
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
            warnings.append(f"Line item {i}: neither unit_price_ht nor line_total_ht extracted; LinMoa omitted.")


def _build_invoice_moa(body, invoice: Invoice, currency: str, warnings):
    moa_section = _el(body, "InvoiceMoa")
    total_ht = _amount_str(invoice.total_ht)
    entries = [
        ("I-179", _amount_str(invoice.capital_social)),
        ("I-180", _amount_str(invoice.total_ttc)),
        ("I-176", total_ht),
        # I-182 "montant total base taxe": assumes the tax base equals
        # total_ht -- true whenever a single VAT rate covers the whole
        # invoice (see LineItem.vat_rate_percent's docstring).
        ("I-182", total_ht),
        ("I-181", _amount_str(invoice.tva_amount)),
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
        warnings.append("No invoice-level amounts (total_ht/tva_amount/total_ttc/capital_social); InvoiceMoa is empty.")


def _build_invoice_tax(body, invoice: Invoice, currency: str, warnings):
    tax_section = _el(body, "InvoiceTax")
    if not invoice.taxes:
        warnings.append("No tax entries (TVA/stamp duty/withholding); InvoiceTax is empty.")
        return

    for tax in invoice.taxes:
        d = _el(tax_section, "InvoiceTaxDetails")
        t = _el(d, "Tax")
        _el(t, "TaxTypeName", tax.tax_type_label, code=tax.tax_type_code)
        td = _el(t, "TaxDetails")
        _el(td, "TaxRate", str(tax.rate_percent) if tax.rate_percent is not None else "0")

        if tax.base_amount is not None:
            ad = _el(d, "AmountDetails")
            moa = _el(ad, "Moa", amountTypeCode="I-177", currencyCodeList="ISO_4217")
            _el(moa, "Amount", _amount_str(tax.base_amount), currencyIdentifier=currency)
        if tax.tax_amount is not None:
            ad = _el(d, "AmountDetails")
            moa = _el(ad, "Moa", amountTypeCode="I-178", currencyCodeList="ISO_4217")
            _el(moa, "Amount", _amount_str(tax.tax_amount), currencyIdentifier=currency)


# --- public API --------------------------------------------------------

def build_teif_element(invoice: Invoice, country: str = DEFAULT_COUNTRY, teif_version: str = TEIF_VERSION):
    """Returns (xml.etree.ElementTree.Element, warnings)."""
    warnings = []
    currency = (invoice.currency or "TND").strip().upper()[:3] or "TND"

    teif = ET.Element("TEIF", controlingAgency="TTN", version=teif_version)

    _build_header(teif, invoice, warnings)

    body = _el(teif, "InvoiceBody")
    _build_bgm(body, invoice, warnings)
    _build_dtm(body, invoice, warnings)
    _build_partner_section(body, invoice, country, warnings)
    _build_pyt_section(body, invoice)
    _build_lin_section(body, invoice, currency, warnings)
    _build_invoice_moa(body, invoice, currency, warnings)
    _build_invoice_tax(body, invoice, currency, warnings)

    return teif, warnings


def build_teif_xml(invoice: Invoice, country: str = DEFAULT_COUNTRY, teif_version: str = TEIF_VERSION, pretty: bool = True):
    """Returns (xml_string, warnings)."""
    teif, warnings = build_teif_element(invoice, country=country, teif_version=teif_version)
    rough = ET.tostring(teif, encoding="utf-8")
    if pretty:
        xml_str = minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")
        xml_str = "\n".join(line for line in xml_str.split("\n") if line.strip())
    else:
        xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + rough.decode("utf-8")
    return xml_str, warnings


def write_teif_xml(invoice: Invoice, output_path: str, **kwargs):
    xml_str, warnings = build_teif_xml(invoice, **kwargs)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    return warnings
