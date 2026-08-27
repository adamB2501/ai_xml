# -*- coding: utf-8 -*-
"""Validates a TEIF (Tunisian e-invoice) XML document two independent ways:

  1. Structural: are the elements Tableau 4 (the TTN guide's structure
     table) marks Mandatory actually present, and do coded attributes
     (functionCode, amountTypeCode, TaxTypeName/@code, ...) use values from
     the guide's referentials (Annexe A)? This catches a malformed or
     incomplete document independent of xml_builder.py -- e.g. a TEIF file
     from somewhere else, or one hand-edited after generation.

  2. Business logic: do the numbers in the document actually add up --
     line items sum to the invoice HT total, HT + taxes sum to TTC? A
     document can be perfectly well-structured and still be wrong (a
     transcription error, or -- if an LLM ever sits in this pipeline, see
     the earlier discussion of that architecture doc -- a hallucinated
     value). This is the same check that doc's "Pydantic validation" layer
     described; it doesn't need Pydantic, just arithmetic.

Same shape as ner_extractor.py's schema check and xml_builder.py's
warnings list: findings are collected and returned rather than raising on
the first problem, so one run tells you everything wrong at once, and the
caller decides what to do about it (main.py prints them; nothing here
refuses to proceed).
"""

import xml.etree.ElementTree as ET

AAMOUNT_TOLERANCE = 0.02  # TND; absorbs rounding in millimes-level amounts

VALID_CONTROLLING_AGENCY = {"TTN", "Tunisie TradeNet"}
VALID_PARTNER_ID_TYPE = {"I-01", "I-02", "I-03", "I-04"}
VALID_DATE_FUNCTION = {f"I-3{n}" for n in range(1, 9)}
VALID_PARTNER_FUNCTION = {f"I-6{n}" for n in range(1, 10)}
SELLER_FUNCTIONS = {"I-62", "I-63", "I-66"}  # Fournisseur / Vendeur / Emetteur
BUYER_FUNCTIONS = {"I-61", "I-64", "I-65"}  # Acheteur / Client / Receveur
VALID_COMM_MEANS = {"I-101", "I-102", "I-103", "I-104"}
VALID_TAX_TYPE_CODE = {f"I-16{n}" for n in range(1, 10)} | {"I-160", "I-1601", "I-1602", "I-1603", "I-1604"}
VALID_AMOUNT_TYPE_CODE = {f"I-17{n}" for n in range(1, 10)} | {f"I-18{n}" for n in range(0, 10)}


class Finding:
    __slots__ = ("severity", "path", "message")

    def __init__(self, severity, path, message):
        self.severity = severity  # "error" (spec violation) | "warning" (business-logic / advisory)
        self.path = path
        self.message = message

    def __repr__(self):
        return f"[{self.severity.upper()}] {self.path}: {self.message}"


def _load(xml_source):
    """Accepts a file path, an XML string, or an already-parsed Element."""
    if isinstance(xml_source, ET.Element):
        return xml_source
    if isinstance(xml_source, str) and xml_source.lstrip().startswith("<"):
        return ET.fromstring(xml_source)
    tree = ET.parse(xml_source)
    return tree.getroot()


def _text(el):
    return (el.text or "").strip() if el is not None else ""


def _amount(el):
    amt = el.find("Amount") if el is not None else None
    text = _text(amt)
    try:
        return float(text)
    except ValueError:
        return None


# --- structural checks ---------------------------------------------------

def _check_root(root, findings):
    if root.tag != "TEIF":
        findings.append(Finding("error", "/", f"root element is <{root.tag}>, expected <TEIF>"))
        return
    agency = root.get("controlingAgency")
    if agency not in VALID_CONTROLLING_AGENCY:
        findings.append(Finding("error", "/TEIF@controlingAgency", f"{agency!r} not in {VALID_CONTROLLING_AGENCY}"))
    if not root.get("version"):
        findings.append(Finding("error", "/TEIF@version", "missing (mandatory)"))


def _check_header(root, findings):
    header = root.find("InvoiceHeader")
    if header is None:
        findings.append(Finding("error", "/TEIF/InvoiceHeader", "missing (mandatory)"))
        return
    sender = header.find("MessageSenderIdentifier")
    if sender is None or not _text(sender):
        findings.append(Finding("error", "/TEIF/InvoiceHeader/MessageSenderIdentifier", "missing or empty (mandatory)"))
    elif sender.get("type") not in VALID_PARTNER_ID_TYPE:
        findings.append(Finding("warning", "/TEIF/InvoiceHeader/MessageSenderIdentifier@type", f"{sender.get('type')!r} not a known I-0 code"))


def _check_bgm(body, findings):
    bgm = body.find("Bgm")
    if bgm is None:
        findings.append(Finding("error", "InvoiceBody/Bgm", "missing (mandatory)"))
        return
    doc_id = bgm.find("DocumentIdentifier")
    if doc_id is None or not _text(doc_id):
        findings.append(Finding("error", "InvoiceBody/Bgm/DocumentIdentifier", "missing or empty (mandatory)"))
    doc_type = bgm.find("DocumentType")
    if doc_type is None or not doc_type.get("code"):
        findings.append(Finding("warning", "InvoiceBody/Bgm/DocumentType@code", "missing"))


def _check_dtm(body, findings):
    dtm = body.find("Dtm")
    if dtm is None or not dtm.findall("DateText"):
        findings.append(Finding("error", "InvoiceBody/Dtm", "missing or has no DateText (spec: mandatory, minOcc=1)"))
        return
    for date_el in dtm.findall("DateText"):
        func = date_el.get("functionCode")
        if func not in VALID_DATE_FUNCTION:
            findings.append(Finding("warning", "InvoiceBody/Dtm/DateText@functionCode", f"{func!r} not a known I-3 code"))
        if not _text(date_el):
            findings.append(Finding("error", "InvoiceBody/Dtm/DateText", "empty"))


def _check_partner_section(body, findings):
    section = body.find("PartnerSection")
    if section is None:
        findings.append(Finding("error", "InvoiceBody/PartnerSection", "missing (mandatory)"))
        return

    details = section.findall("PartnerDetails")
    if not details:
        findings.append(Finding("error", "InvoiceBody/PartnerSection", "no PartnerDetails (mandatory, minOcc=1)"))

    functions_seen = set()
    for pd in details:
        func = pd.get("functionCode")
        functions_seen.add(func)
        if func not in VALID_PARTNER_FUNCTION:
            findings.append(Finding("warning", "PartnerDetails@functionCode", f"{func!r} not a known I-6 code"))

        nad = pd.find("Nad")
        if nad is None:
            findings.append(Finding("error", f"PartnerDetails[{func}]/Nad", "missing (mandatory)"))
            continue
        pid = nad.find("PartnerIdentifier")
        if pid is None or not _text(pid):
            findings.append(Finding("error", f"PartnerDetails[{func}]/Nad/PartnerIdentifier", "missing or empty (mandatory)"))
        elif pid.get("type") not in VALID_PARTNER_ID_TYPE:
            findings.append(Finding("warning", f"PartnerDetails[{func}]/Nad/PartnerIdentifier@type", f"{pid.get('type')!r} not a known I-0 code"))

        for cta in pd.findall("CtaSection"):
            comm = cta.find("Communication")
            if comm is not None:
                means = _text(comm.find("ComMeansType"))
                if means not in VALID_COMM_MEANS:
                    findings.append(Finding("warning", f"PartnerDetails[{func}]/CtaSection/Communication/ComMeansType", f"{means!r} not a known I-10 code"))

    if not (functions_seen & SELLER_FUNCTIONS):
        findings.append(Finding("warning", "InvoiceBody/PartnerSection", "no seller-role PartnerDetails (I-62/I-63/I-66)"))
    if not (functions_seen & BUYER_FUNCTIONS):
        findings.append(Finding("warning", "InvoiceBody/PartnerSection", "no buyer-role PartnerDetails (I-61/I-64/I-65)"))


def _check_lin_section(body, findings):
    section = body.find("LinSection")
    if section is None or not section.findall("Lin"):
        findings.append(Finding("error", "InvoiceBody/LinSection", "missing or has no Lin (spec: mandatory)"))
        return None

    lines = []
    for i, lin in enumerate(section.findall("Lin"), start=1):
        path = f"LinSection/Lin[{i}]"
        imd = lin.find("LinImd")
        desc = _text(imd.find("ItemDescription")) if imd is not None else ""
        if not desc:
            findings.append(Finding("warning", f"{path}/LinImd/ItemDescription", "empty"))

        qty_el = lin.find("LinQty/Quantity")
        if qty_el is None or not _text(qty_el):
            findings.append(Finding("warning", f"{path}/LinQty/Quantity", "missing"))

        tax_rate_el = lin.find("LinTax/TaxDetails/TaxRate")
        if tax_rate_el is None:
            findings.append(Finding("warning", f"{path}/LinTax/TaxDetails/TaxRate", "missing"))

        line_total = None
        for moa in lin.findall("LinMoa/MoaDetails/Moa"):
            if moa.get("amountTypeCode") == "I-171":
                line_total = _amount(moa)
        if line_total is None:
            findings.append(Finding("warning", f"{path}/LinMoa", "no I-171 (line total) amount"))
        lines.append(line_total)

    return lines


def _invoice_moa_amounts(body):
    amounts = {}
    for moa in body.findall("InvoiceMoa/AmountDetails/Moa"):
        code = moa.get("amountTypeCode")
        val = _amount(moa)
        if code and val is not None:
            amounts[code] = val
    return amounts


def _check_invoice_moa(body, findings):
    moa_section = body.find("InvoiceMoa")
    if moa_section is None or not moa_section.findall("AmountDetails"):
        findings.append(Finding("error", "InvoiceBody/InvoiceMoa", "missing or empty (spec: mandatory)"))
        return {}
    amounts = _invoice_moa_amounts(body)
    for code in amounts:
        if code not in VALID_AMOUNT_TYPE_CODE:
            findings.append(Finding("warning", "InvoiceMoa/AmountDetails/Moa@amountTypeCode", f"{code!r} not a known I-17/I-18 code"))
    if "I-180" not in amounts:
        findings.append(Finding("warning", "InvoiceMoa", "no I-180 (Montant Total TTC facture)"))
    if "I-176" not in amounts:
        findings.append(Finding("warning", "InvoiceMoa", "no I-176 (Montant total HT facture)"))
    return amounts


def _check_invoice_tax(body, findings):
    tax_section = body.find("InvoiceTax")
    if tax_section is None or not tax_section.findall("InvoiceTaxDetails"):
        findings.append(Finding("error", "InvoiceBody/InvoiceTax", "missing or empty (spec: mandatory)"))
        return []

    entries = []
    for i, details in enumerate(tax_section.findall("InvoiceTaxDetails"), start=1):
        path = f"InvoiceTax/InvoiceTaxDetails[{i}]"
        tax = details.find("Tax")
        code = None
        if tax is not None:
            type_name = tax.find("TaxTypeName")
            code = type_name.get("code") if type_name is not None else None
            if code not in VALID_TAX_TYPE_CODE:
                findings.append(Finding("warning", f"{path}/Tax/TaxTypeName@code", f"{code!r} not a known I-16 code"))
        else:
            findings.append(Finding("error", f"{path}/Tax", "missing"))

        tax_amount = None
        for moa in details.findall("AmountDetails/Moa"):
            if moa.get("amountTypeCode") == "I-178":
                tax_amount = _amount(moa)
        entries.append((code, tax_amount))
    return entries


# --- business-logic (arithmetic) checks -----------------------------------

def _check_totals_consistency(line_totals, invoice_moa, tax_entries, findings):
    known_line_totals = [v for v in (line_totals or []) if v is not None]
    total_ht = invoice_moa.get("I-176")
    total_ttc = invoice_moa.get("I-180")

    if known_line_totals and total_ht is not None:
        lines_sum = sum(known_line_totals)
        if len(known_line_totals) == len(line_totals) and abs(lines_sum - total_ht) > AAMOUNT_TOLERANCE:
            findings.append(Finding(
                "warning", "InvoiceMoa/I-176",
                f"sum of line totals ({lines_sum:.3f}) != declared total HT ({total_ht:.3f})",
            ))

    tax_sum = sum(amount for _, amount in tax_entries if amount is not None)
    if total_ht is not None and total_ttc is not None and tax_sum:
        expected_ttc = total_ht + tax_sum
        if abs(expected_ttc - total_ttc) > AAMOUNT_TOLERANCE:
            findings.append(Finding(
                "warning", "InvoiceMoa/I-180",
                f"total HT ({total_ht:.3f}) + sum of taxes ({tax_sum:.3f}) = {expected_ttc:.3f}, "
                f"!= declared total TTC ({total_ttc:.3f})",
            ))

    tva_total_moa = invoice_moa.get("I-181")
    tva_from_tax = sum(amount for code, amount in tax_entries if code == "I-1602" and amount is not None)
    if tva_total_moa is not None and tva_from_tax and abs(tva_total_moa - tva_from_tax) > AAMOUNT_TOLERANCE:
        findings.append(Finding(
            "warning", "InvoiceMoa/I-181",
            f"declared TVA total ({tva_total_moa:.3f}) != sum of InvoiceTax TVA entries ({tva_from_tax:.3f})",
        ))


# --- public API ------------------------------------------------------------

def validate_teif_xml(xml_source):
    """Returns a list of Finding objects. Empty list means no problems
    found by either check. Never raises on a malformed *document* (only on
    genuinely unparseable input, e.g. a nonexistent path) -- validation
    failures are findings, not exceptions."""
    findings = []
    try:
        root = _load(xml_source)
    except ET.ParseError as exc:
        return [Finding("error", "/", f"not well-formed XML: {exc}")]

    _check_root(root, findings)
    _check_header(root, findings)

    body = root.find("InvoiceBody")
    if body is None:
        findings.append(Finding("error", "/TEIF/InvoiceBody", "missing (mandatory)"))
        return findings

    _check_bgm(body, findings)
    _check_dtm(body, findings)
    _check_partner_section(body, findings)
    line_totals = _check_lin_section(body, findings)
    invoice_moa = _check_invoice_moa(body, findings)
    tax_entries = _check_invoice_tax(body, findings)

    _check_totals_consistency(line_totals, invoice_moa, tax_entries, findings)

    return findings


def print_report(findings):
    if not findings:
        print("[validate_xml] no issues found")
        return
    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    print(f"[validate_xml] {len(errors)} error(s), {len(warnings)} warning(s)")
    for f in findings:
        print(f"  {f}")


if __name__ == "__main__":
    import sys

    findings = validate_teif_xml(sys.argv[1])
    print_report(findings)
    sys.exit(1 if any(f.severity == "error" for f in findings) else 0)
