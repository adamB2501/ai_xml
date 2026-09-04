# -*- coding: utf-8 -*-
"""
Every instruction sent to a model lives here, as a plain string, with a
comment above it explaining WHY each rule is in the prompt. If the
pipeline ever extracts something wrong, this is the first file to look at:
the fix is almost always "the prompt didn't forbid that mistake clearly
enough", not a code change.

Two models are prompted:

  * the VISION model      -> VISION_TRANSCRIBE_PROMPT
  * the TEXT (field) model -> build_extraction_messages(...)

The text model is asked to work in TWO passes in a single response:
  PASS 1  extract every field
  PASS 2  re-read the source and check its own answer (occurrence check +
          arithmetic). Anything it can't verify becomes null, and it
          writes down what it checked in a "_verification" block.
This "make it grade its own homework" step is cheap (a few hundred extra
tokens) and catches a large share of confident-but-wrong values.
"""

from __future__ import annotations

import json


# ===========================================================================
# VISION MODEL  --  image -> plain text
# ===========================================================================
# We deliberately ask ONLY for transcription here, not field extraction.
# Reasons:
#   - The vision model's strength is reading pixels. Field disambiguation
#     (which number is the VAT vs the total) is done better by the text
#     model over the full document, where all the context is.
#   - A transcription is easy for a human to eyeball against the image when
#     debugging. A JSON blob straight from the VLM is not.
#   - For a MIXED pdf the VLM only sees a small band (letterhead/footer);
#     it has no idea what the rest of the invoice says, so it *can't*
#     sensibly extract invoice-level fields anyway.
#
# Rules, and why:
#   "exactly as written"      -> stop it 'helpfully' fixing OCR-looking text
#   "do not translate"        -> Tunisian invoices mix FR/AR; keep source
#   "reading order"           -> so the text model sees label next to value
#   "one row per line, | ..." -> preserve table structure for the text model
#   "if unreadable, write [illegible]" -> an explicit gap beats a guess
VISION_TRANSCRIBE_PROMPT = (
    "You are transcribing text from an image of an invoice (or a cropped "
    "part of one, such as the top letterhead or bottom footer).\n"
    "\n"
    "Transcribe EVERY piece of visible text, exactly as written. Rules:\n"
    "- Do not translate. Do not summarise. Do not correct spelling or "
    "numbers. Do not add anything that is not visibly printed.\n"
    "- Keep the natural reading order (top to bottom, left to right).\n"
    "- For a table, put each row on its own line with cells separated by "
    "' | ' in column order.\n"
    "- Preserve digits and separators exactly (e.g. 1 234,560 stays "
    "'1 234,560').\n"
    "- If a piece of text is genuinely unreadable, write [illegible] in "
    "its place rather than guessing.\n"
    "\n"
    "Output only the transcription, no commentary."
)


# ===========================================================================
# TEXT MODEL  --  gathered text -> structured fields (+ self-check)
# ===========================================================================

# The exact field list we want back. Keeping it here (rather than inline in
# a giant f-string) makes it easy to see at a glance what the pipeline
# knows how to extract. Each entry: (json_key, human description shown to
# the model). Order is the order the model sees them.
FIELD_SPEC: list[tuple[str, str]] = [
    ("invoice_number",
     "the invoice number / numero de facture (e.g. after 'FACTURE N°')"),
    ("issue_date",
     "the issue date, usually after 'Le :' or 'Date'. Return it exactly as "
     "printed (e.g. '27/08/2026')."),
    ("seller_name",
     "the company ISSUING the invoice - normally the name in the top "
     "letterhead / logo area, or repeated in the footer."),
    ("seller_tax_id",
     "the SELLER's Matricule Fiscal / Code TVA. Shape: 6-7 digits, then 3 "
     "letters, then 3 digits (e.g. '0736202XAM000'), sometimes written "
     "with spaces or slashes. Often in the footer near 'Code TVA' or 'MF'."),
    ("seller_address", "the seller's postal address."),
    ("seller_rc_number",
     "the seller's Registre de Commerce number, usually after 'RC' or "
     "'R.C :' (e.g. 'B15880 1996')."),
    ("seller_capital",
     "the seller's share capital, usually after 'capital de' / 'au capital "
     "de' (e.g. '450 000')."),
    ("buyer_name",
     "the party being BILLED - the one under a 'DOIT', 'Client', 'Nom "
     "Client' or 'Destinataire' heading. This is NOT a department name "
     "like 'Direction Commerciale'."),
    ("buyer_tax_id",
     "the BUYER's Matricule Fiscal (same shape as the seller's, but a "
     "DIFFERENT value). Near 'Matricule Fiscal' inside the client block. "
     "If the client block shows no tax id, return null."),
    ("buyer_address", "the buyer's postal address."),
    ("currency",
     "the ISO currency code if printed. If only words like 'dinars' / "
     "'millimes' appear, return 'TND'. Otherwise null."),
    ("total_ht",
     "total before tax - 'Total HT' / 'Total Hors TVA' / 'Base'."),
    ("tva_amount",
     "the total VAT amount - 'TVA', 'Montant TVA', 'Cumul TVA'. This is "
     "NOT the VAT rate (19, 7, ...); it is a money amount."),
    ("stamp_duty",
     "the fiscal stamp - 'Droit de Timbre' / 'Timbre Fiscal' (a small "
     "amount, often 0.600 or 1.000)."),
    ("total_ttc",
     "the final amount payable - 'NET A PAYER' / 'Total TTC' / 'TOTAL "
     "FACTURE'."),
    ("delivery_note_number",
     "delivery-note reference - after 'BL' or 'B.L.' (may look like "
     "'260078:24/08/2026' - keep the whole string)."),
    ("other_references",
     "a list of any other document references printed (e.g. lines starting "
     "'FI', 'BC', 'Ref commande'). Empty list if none."),
]

# The line-item sub-fields, same idea.
LINE_ITEM_SPEC: list[tuple[str, str]] = [
    ("code", "the article code / reference in the first column (may be "
             "'**' or '000' or a real SKU - keep whatever is there)."),
    ("description", "the article label / description."),
    ("unit_price_ht", "unit price before tax ('Prix U.HT'). null if blank."),
    ("quantity", "the quantity ('Quantité' / 'Qte'). null if blank."),
    ("vat_rate_percent", "the VAT RATE percent for this line (e.g. 19). "
                         "null if blank."),
    ("line_total_ht", "the line amount before tax ('Montant H.TVA'). null "
                      "if blank."),
]


def _numbered(spec: list[tuple[str, str]], indent: str = "") -> str:
    """Render a (key, description) list as 'key: description' bullet lines."""
    return "\n".join(f"{indent}- {key}: {desc}" for key, desc in spec)


# --- the system message: role + hard rules + the two-pass process ----------
#
# Written as separate labelled sections so it's easy to see what each part
# is doing. Every rule maps to a real mistake models make on these invoices:
#
#   rule 1  (verbatim)          -> models 'normalise' 1.000 -> 1, dd/mm/yyyy
#                                  -> yyyy-mm-dd, etc. TEIF wants the raw value.
#   rule 2  (null not guess)    -> the single most important rule. A guessed
#                                  total on a tax document is a liability.
#   rule 3  (columns scrambled) -> pdfplumber interleaves columns; the model
#                                  must not assume the word after a label is
#                                  its value.
#   rule 4  (seller != buyer)   -> the classic failure: both parties have a
#                                  Matricule Fiscal; models swap them.
#   rule 5  (amount vs rate)    -> '19' next to 'TVA' is the rate, not the
#                                  VAT amount. Different fields.
#   rule 6  (no math in pass 1) -> keep extraction and reasoning separate so
#                                  a wrong sum doesn't corrupt a good read.
SYSTEM_MESSAGE = f"""\
You convert raw Tunisian invoice text into structured fields for e-invoicing (TEIF).

The text you receive is raw extraction output. Expect: interleaved columns,
labels separated from their values, mangled French accents, duplicated
header/footer text, and occasional OCR errors. Read carefully.

HARD RULES
1. VERBATIM. Copy each value exactly as printed - same digits, same
   separators, same date format. Never reformat, round, translate, or
   "clean up" a value.
2. NEVER GUESS. If a field is not clearly present in the text, its value is
   null. A wrong value is far worse than a null one here.
3. COLUMNS MAY BE SCRAMBLED. The word immediately after a label is not
   necessarily that label's value. Use meaning, not just position.
4. SELLER vs BUYER. The seller issues the invoice (letterhead/footer). The
   buyer is under the 'DOIT' / 'Client' heading. They each have their own
   Matricule Fiscal and address - never swap them. seller_tax_id and
   buyer_tax_id must be different strings.
5. AMOUNT vs RATE. A VAT amount is money (e.g. 18.525). A VAT rate is a
   percent (e.g. 19). Do not put a rate where an amount is asked for.
6. In PASS 1 do not do arithmetic - just read.

PROCESS (produce both passes in your single JSON answer)

PASS 1 - EXTRACT
  Fill every field in the schema by reading the text.

PASS 2 - RE-CHECK YOUR OWN ANSWER
  Re-read the SOURCE TEXT and verify:
    (a) OCCURRENCE: every non-null value you wrote appears, character for
        character, somewhere in the source text. If you cannot find it,
        set that field to null.
    (b) TOTALS: total_ht + tva_amount + stamp_duty should equal total_ttc
        (allow tiny rounding). If it doesn't, at least one of those four is
        read from the wrong place - find the right value or null the ones
        you cannot confirm.
    (c) LINE SUM: the line_total_ht values should add up to total_ht. If
        not, re-check the line items.
    (d) TAX IDS: seller_tax_id != buyer_tax_id, and each matches the
        6-7 digits / 3 letters / 3 digits shape (ignoring spaces/slashes).
  Put the outcome in "_verification":
    {{"checks_done": [short strings], "problems_found": [short strings]}}
  If PASS 2 made you change or null a value, keep the corrected version in
  the main fields (do not revert to the PASS 1 value).

OUTPUT
  Return ONLY the JSON object matching the schema. No prose, no markdown
  fences, no explanation outside the JSON.
"""


def build_user_message(source_text: str) -> str:
    """The user turn: what to extract, from what text."""
    return (
        "Extract the following fields.\n\n"
        "TOP-LEVEL FIELDS:\n"
        f"{_numbered(FIELD_SPEC)}\n\n"
        "line_items: a list; for each row on the invoice's items table, an "
        "object with:\n"
        f"{_numbered(LINE_ITEM_SPEC, indent='  ')}\n\n"
        "_verification: object with 'checks_done' and 'problems_found' "
        "string lists, filled during PASS 2.\n\n"
        "SOURCE TEXT:\n"
        "----------------------------------------\n"
        f"{source_text}\n"
        "----------------------------------------\n"
    )


def build_extraction_messages(source_text: str) -> list[dict]:
    """The full /api/chat `messages` array for one extraction call."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": build_user_message(source_text)},
    ]


# --- JSON schema handed to Ollama's `format` parameter ---------------------
# Ollama constrains the model's output to match this. It does NOT replace
# the prompt (the model still needs to be told what each field means) - it
# just guarantees the response parses as JSON with these keys and types, so
# extract.py never has to deal with "the model wrapped it in ```json" or
# "it returned a sentence".
_STR_OR_NULL = {"type": ["string", "null"]}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        **{key: _STR_OR_NULL for key, _ in FIELD_SPEC if key != "other_references"},
        "other_references": {"type": "array", "items": {"type": "string"}},
        "line_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {key: _STR_OR_NULL for key, _ in LINE_ITEM_SPEC},
            },
        },
        "_verification": {
            "type": "object",
            "properties": {
                "checks_done": {"type": "array", "items": {"type": "string"}},
                "problems_found": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["checks_done", "problems_found"],
        },
    },
    "required": [
        "invoice_number", "issue_date", "seller_name", "seller_tax_id",
        "buyer_name", "buyer_tax_id", "total_ht", "tva_amount", "total_ttc",
        "line_items", "_verification",
    ],
}


def response_schema_json() -> str:
    """The schema as a compact string (for logging / the report)."""
    return json.dumps(RESPONSE_SCHEMA, separators=(",", ":"))


# ===========================================================================
# TARGETED RE-ASK  --  ask again for only the fields that came back bad
# ===========================================================================
# Used by reask.py. The philosophy: the first pass fails a field either
# because the model didn't bother to emit it (long field list, it lost
# focus) or because it needs a fact it can now be handed. So the re-ask
#   - names ONLY the deficient fields (short list -> the model can focus)
#   - passes back what IS known, as a disambiguator
#   - passes any arithmetically-implied value (e.g. the stamp duty MUST be
#     total_ttc - total_ht - tva_amount)
#   - keeps the same "verbatim, null if truly absent, never guess" rules
#
# The answer is still schema-constrained (a small schema built on the fly
# from just the requested keys) so it always parses.

_DESC = dict(FIELD_SPEC)
_LINE_DESC = dict(LINE_ITEM_SPEC)

REASK_SYSTEM_MESSAGE = """\
You already extracted this invoice once. Some fields came back missing or
failed a consistency check. Find ONLY those fields now, reading the same
source text.

RULES (unchanged)
- VERBATIM: copy each value exactly as printed.
- NEVER GUESS: if a field is genuinely not in the text, return null.
- Use the HINTS below - they tell you what is already known and, where the
  arithmetic fixes a value, what number to expect.
- Return ONLY a JSON object with exactly the requested keys.
"""


def build_reask_messages(
    source_text: str,
    fields_wanted: list[str],
    known: dict,
    hints: dict[str, str] | None = None,
) -> list[dict]:
    """
    fields_wanted : the deficient top-level keys (may include "line_items")
    known         : the fields we DO have, shown to the model for context
    hints         : {field_name: "extra instruction"} - e.g. an implied value
    """
    hints = hints or {}

    lines = ["Fields to find:"]
    for key in fields_wanted:
        if key == "line_items":
            lines.append("  - line_items: the full items table. Its columns, "
                         "in order, are: Code Article | Article | Prix U.HT | "
                         "Quantité | TVA | Montant H.TVA. Split each row on "
                         "those columns; do NOT merge the code with the "
                         "description. Include every row, even ones priced 0.")
            for k, d in LINE_ITEM_SPEC:
                lines.append(f"      {k}: {d}")
        else:
            lines.append(f"  - {key}: {_DESC.get(key, key)}")
        if key in hints:
            lines.append(f"      HINT: {hints[key]}")

    known_shown = {k: v for k, v in known.items()
                   if v and not k.startswith("_") and k != "line_items"}
    context = json.dumps(known_shown, ensure_ascii=False, indent=2)

    user = (
        "\n".join(lines)
        + "\n\nAlready known (for context / disambiguation - do not change these):\n"
        + context
        + "\n\nSOURCE TEXT:\n"
        "----------------------------------------\n"
        f"{source_text}\n"
        "----------------------------------------\n"
    )
    return [
        {"role": "system", "content": REASK_SYSTEM_MESSAGE},
        {"role": "user", "content": user},
    ]


def build_reask_schema(fields_wanted: list[str]) -> dict:
    """A minimal JSON schema for just the re-asked keys, so Ollama's
    `format` still forces a parseable reply."""
    props: dict = {}
    for key in fields_wanted:
        if key == "line_items":
            props["line_items"] = RESPONSE_SCHEMA["properties"]["line_items"]
        elif key == "other_references":
            props["other_references"] = {"type": "array", "items": {"type": "string"}}
        else:
            props[key] = _STR_OR_NULL
    return {"type": "object", "properties": props, "required": list(props)}
