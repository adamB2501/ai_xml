# -*- coding: utf-8 -*-
"""
Template-specific synthetic generator for the TTN/TradeNet (ONP-style)
invoice template, reverse-engineered directly from a real raw pdfplumber
extraction of that template (not invented/generic noise).

Unlike training_data.py (generic French commercial invoice phrasing that
this template shares almost no structure with), every example produced
here reproduces the EXACT real structural artifacts of this one template:

  - the header account-info block, where buyer identity fragments
    (name/address) are interleaved out-of-order with unrelated
    label:value pairs (declustering, confirmed against the real extract)
  - the rotated "SA au capital de X DT / Code TVA Y / RC Z" stamp,
    which extracts with token order reversed AND each token's
    characters reversed (confirmed by reconstructing the real block
    and getting back exactly the pasted raw text)
  - line-item rows, which (for this template) extract INTACT per row --
    confirmed from the real sample, so item entities are NOT scrambled
    here, unlike the header/stamp blocks
  - the totals block, which extracts as a flattened mix of headers and
    values (confirmed from the real sample)

All entity offsets are computed programmatically (never hand-counted) by
building the text from labeled segments and tracking positions as they're
assembled, then independently re-verified by slicing.

Open schema question (see NOTE below): a few real fields in this template
-- "Code Client", the long TTN registration reference, and the CCP
payment-slip numbers -- don't map cleanly to any of the 39 existing
labels. They are deliberately left UNLABELED here rather than forced
into a wrong label (e.g. tagging "Code Client" as PO_NUMBER), consistent
with the project's "don't over-generalize / don't force a bad label"
principle. Flagging this explicitly so it's a conscious decision, not a
silent gap: if these recur often across your real 400-PDF set, they may
be worth adding as new labels (e.g. ACCOUNT_CODE, REGISTRATION_REF).
"""

import random


# --- French integer -> words, compact implementation for invoice totals ---
# Only needs to be plausible-looking (this text is deliberately left
# UNLABELED -- its only training purpose is teaching the model this line
# is NOT a line item / NOT an amount entity, since "CENT" etc. was
# previously misfired on as ITEM_QTY when this line was absent from
# training data entirely).
_UNITS = ["", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
          "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
          "dix-sept", "dix-huit", "dix-neuf"]
_TENS = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante",
         "soixante-dix", "quatre-vingt", "quatre-vingt-dix"]


def _french_below_100(n):
    if n < 20:
        return _UNITS[n]
    tens, rem = divmod(n, 10)
    if tens in (7, 9):
        tens -= 1
        rem += 10
    word = _TENS[tens]
    if rem:
        word += ("-et-" if rem == 1 and tens in (2, 3, 4, 5, 6) else "-") + _UNITS[rem]
    return word


def _french_below_1000(n):
    if n < 100:
        return _french_below_100(n)
    hundreds, rem = divmod(n, 100)
    word = ("cent" if hundreds == 1 else _UNITS[hundreds] + " cent")
    if rem:
        word += " " + _french_below_100(rem)
    return word


def french_number_to_words(n):
    """Good enough for typical invoice amounts (0-999999); not a complete
    French numeral implementation, but doesn't need to be -- this text is
    unlabeled boilerplate, only present so the model sees the pattern and
    learns not to tag fragments of it as entities."""
    n = int(n)
    if n == 0:
        return "zéro"
    thousands, rem = divmod(n, 1000)
    parts = []
    if thousands:
        parts.append(("mille" if thousands == 1 else _french_below_1000(thousands) + " mille"))
    if rem:
        parts.append(_french_below_1000(rem))
    return " ".join(parts)


BUYER_ORGS = [
    ("Office National des Postes", "Centre Informatique Complexe Hached", "Tunis"),
    ("Office National de l'Assainissement", "Direction Régionale Sfax", "Sfax"),
    ("Société Tunisienne de l'Électricité et du Gaz", "Centre de Gestion Nabeul", "Nabeul"),
    ("Ministère des Finances", "Direction Générale des Impôts", "Tunis"),
    ("Caisse Nationale de Sécurité Sociale", "Agence Régionale Sousse", "Sousse"),
    ("Office des Céréales", "Direction Logistique Bizerte", "Bizerte"),
]

CONNECTION_TYPES = ["SMTP", "EDI", "FTP", "VPN"]
PROFILES = ["Banques", "Assurances", "Administration", "Grande Distribution"]
RANGS = ["NP", "P", "N"]
ACCOUNT_NICKNAMES = ["ONPS", "STEGN", "CNSSA", "MFDGI", "OCLB"]

# Fixed, real seller identity -- same legal entity (Tunisie TradeNet) issues
# every invoice on this template, so these are constants, not randomized.
SELLER_ADDRESS_TEXT = (
    "Rue du Lac Malaren, Lotissement El Khalij Les Berges du\n"
    "Lac, 1053-Tunis,, Tunisie"
)
SELLER_PHONE_1 = "71 86 17 12"
SELLER_PHONE_2 = "71 86 11 41"
SELLER_WEBSITE = "www.tradenet.com.tn"

ITEM_CATALOG = [
    ("SMTP. P", "C. SMTP principal"),
    ("TCEAP", "Dossier TCEAP"),
    ("FDE", "Dossier FDE"),
    ("EDI. S", "Service EDI standard"),
    ("VPN. C", "Connexion VPN client"),
    ("ARCH", "Archivage électronique"),
]


def _rand_digits(n, rng):
    return "".join(rng.choice("0123456789") for _ in range(n))


def _rand_alnum(n, rng):
    return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(n))


def _fmt_amount(x):
    return f"{x:.3f}".replace(".", ".")


def build_reversed_stamp(rc_number, tva_code, capital_groups, currency="DT"):
    """
    Reproduces the real 'rotated stamp' extraction artifact.

    capital_groups: list of digit-group strings, e.g. ["2", "000", "000"]
    for a capital of "2 000 000 DT".

    Returns (block_text, rc_span_text, tax_id_span_text, capital_span_text)
    where the span texts are exactly what a NOISY-text slice at the
    correct offsets will contain (garbled/reversed, matching the real
    artifact -- NOT the clean value).
    """
    clean_tokens = (
        ["SA", "au", "capital", "de"]
        + list(capital_groups)
        + [f"{currency}/", "Code", "TVA", tva_code, "/", "RC", rc_number]
    )
    raw_tokens = list(reversed(clean_tokens))
    raw_lines = [t[::-1] for t in raw_tokens]
    block_text = "\n".join(raw_lines)

    n = len(clean_tokens)
    k = len(capital_groups)
    # derived directly from the index algebra worked out against the real sample
    rc_span = raw_lines[0]
    tax_id_span = raw_lines[3]
    capital_span = "\n".join(raw_lines[6 : 6 + k + 1])

    return block_text, rc_span, tax_id_span, capital_span


def generate_ttn_example(rng=random):
    buyer_org, buyer_dept, buyer_city = rng.choice(BUYER_ORGS)
    postal_code = rng.choice(["1049", "1002", "3000", "7000", "4000", "2000"])
    rang = rng.choice(RANGS)
    profil = rng.choice(PROFILES)
    code_client = _rand_digits(8, rng)
    buyer_tax_id = f"{_rand_digits(7, rng)}{rng.choice('ABCDEFGHJKLMNPQRSTVWXYZ')}PM000"
    due_date = f"{rng.randint(1,28):02d}{rng.randint(1,12):02d}{rng.randint(15,26):02d}"
    unique_ref = _rand_digits(26, rng)

    invoice_num = str(rng.randint(1, 999))
    date_str = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(2015,2026)}"
    period_start = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(2015,2026)}"
    period_end = f"{rng.randint(1,28):02d}/{rng.randint(1,12):02d}/{rng.randint(2015,2026)}"
    account_name = rng.choice(ACCOUNT_NICKNAMES)
    connection_mode = rng.choice(CONNECTION_TYPES)

    ccp_ref = _rand_digits(13, rng)
    ccp_code_org = _rand_digits(4, rng)
    ccp_cles = f"{rng.randint(1,9)}M"

    rc_number = f"B{_rand_digits(9, rng)}"
    tva_code = f"{_rand_digits(6, rng)}XAM000"
    capital_groups = rng.choice([["2", "000", "000"], ["1", "500", "000"], ["500", "000"]])

    n_items = rng.randint(2, 4)
    items = []
    total_ht = 0.0
    for _ in range(n_items):
        code, desig = rng.choice(ITEM_CATALOG)
        qty = round(rng.uniform(1, 20), 1)
        pu = round(rng.uniform(4, 500), 3)
        line_total = round(qty * pu, 3)
        total_ht += line_total
        items.append((code, desig, qty, pu, line_total))

    tva_rate = rng.choice([7, 12, 13, 19])
    montant_tva = round(total_ht * tva_rate / 100, 3)
    droit_timbre = 0.5
    total_ttc = round(total_ht + montant_tva + droit_timbre, 3)

    segments = []  # list of (literal_text,) or (entity_text, label)

    def lit(s):
        segments.append((s, None))

    def ent(s, label):
        segments.append((s, label))

    # --- full label cluster (all boilerplate labels, no values yet) ---
    # Confirmed structure: this template extracts EVERY label in one
    # contiguous run first, then every value in a separate run much later
    # -- a more thorough declustering than earlier modeled (previously
    # only the account-info labels were clustered; the seller contact
    # labels and invoice-number/period labels were entirely missing from
    # training, which is why phone-number and period-date fragments were
    # getting mis-tagged as VAT_RATE / item fields at inference).
    lit("Adresse :\nTélephone :\nTélecopie :\nSite Web :\n")
    lit("Nom Compte :\nMode de connexion :\nRang du compte :\nProfil :\n")
    lit("Code Client :\nMatricule Fiscal :\nDate Limite de paiement :\n")
    lit("Date (mirror)\n")  # Arabic mirror in the real doc; kept boilerplate
    lit("Facture N° Periode : du Au (mirror)\n")

    # --- seller identity values (fixed real company, constant across
    # every invoice on this template) ---
    ent(SELLER_ADDRESS_TEXT, "SELLER_ADDRESS")
    lit("\n")
    ent(SELLER_PHONE_1, "PHONE")
    lit("\n")
    ent(SELLER_PHONE_2, "PHONE")
    lit("\n")
    ent(SELLER_WEBSITE, "WEBSITE")
    lit("\n")
    lit("(adresse/téléphone/télécopie/site web label mirrors)\n")

    # --- invoice number / period values (previously entirely missing) ---
    ent(invoice_num, "INVOICE_NUMBER")
    lit(" ")
    ent(period_start, "DATE")
    lit(" ")
    ent(period_end, "DATE")
    lit("\n")

    # --- account-info values (declustered, matches confirmed real order) ---
    lit(f"{account_name}\n{connection_mode}\n{rang}\n{profil}\n{code_client}\n")
    ent(buyer_tax_id, "BUYER_TAX_ID")
    lit("\n")
    ent(due_date, "DUE_DATE")
    lit("\n")
    ent(date_str, "DATE")
    lit("\n")

    # --- buyer identity block (fragments interleaved, matches real sample) ---
    ent(buyer_org, "BUYER_NAME")
    lit("\n")
    lit(f"{account_name}\n")
    ent(f"{postal_code} {buyer_city.upper()}", "BUYER_ADDRESS")
    lit("\n")
    ent(buyer_dept, "BUYER_NAME")
    lit("\n")
    ent(buyer_city, "BUYER_ADDRESS")
    lit("\n")

    lit("Référence Unique ")
    lit(f"{unique_ref}\n")  # unique_ref left UNLABELED -- see module docstring
    lit("Copie de la facture électronique enregistrée chez TTN sous la référence : ")
    lit(f"{unique_ref}\n")

    # --- reversed stamp block ---
    stamp_block, rc_span, tax_id_span, capital_span = build_reversed_stamp(
        rc_number, tva_code, capital_groups
    )
    # stamp_block is a single joined string; we need to locate the three
    # spans' offsets *within* stamp_block to emit them as separate segments
    # in the right order (they appear in raw top-to-bottom order as:
    # rc_span first, tax_id_span a few lines down, capital_span after that)
    lines = stamp_block.split("\n")
    idx = 0
    for i, ln in enumerate(lines):
        if i == 0:
            ent(ln, "RC_NUMBER")
        elif i == 3:
            ent(ln, "SELLER_TAX_ID")
        elif 6 <= i <= 6 + len(capital_groups):
            # first line of the capital_span run -> emit whole run once
            if i == 6:
                ent(capital_span, "CAPITAL_SOCIAL")
            # remaining lines of the run already consumed by the ent() above;
            # skip re-emitting them as literals
            if i == 6 + len(capital_groups):
                lit("\n")
            continue
        else:
            lit(ln)
        if i != len(lines) - 1:
            lit("\n")
    lit("\n")

    # --- item table (extracts INTACT per row for this template) ---
    lit("Code Désignation Quantité T.V.A. % P.U.H.T.V.A. Total H.T.V.A.\n")
    for (code, desig, qty, pu, line_total) in items:
        lit(f"{code} ")
        ent(desig, "ITEM_DESC")
        lit(" ")
        ent(f"{qty}", "ITEM_QTY")
        lit(f" {tva_rate} ")
        ent(_fmt_amount(pu), "ITEM_UNIT_PRICE")
        lit(" ")
        ent(_fmt_amount(line_total), "ITEM_LINE_TOTAL")
        lit("\n")

    # --- totals block (flattened, matches real sample) ---
    lit("Taux (%) Base Montant TVA Total H.T.V.A. ")
    ent(_fmt_amount(total_ht), "TOTAL_HT")
    lit("\n")
    ent(str(tva_rate) + ".0", "VAT_RATE")
    lit(" ")
    lit(_fmt_amount(total_ht))
    lit(" ")
    ent(_fmt_amount(montant_tva), "TVA_AMOUNT")
    lit("\n")
    lit("Montant TVA ")
    ent(_fmt_amount(montant_tva), "TVA_AMOUNT")
    lit("\n")
    lit("Droit de Timbre ")
    ent(_fmt_amount(droit_timbre), "STAMP_DUTY")
    lit("\n")
    lit("Total ")
    lit(_fmt_amount(total_ht))
    lit(" ")
    lit(_fmt_amount(montant_tva))
    lit(" Montant T.T.C ")
    ent(_fmt_amount(total_ttc), "TOTAL_TTC")
    lit("\n")

    lit("Arrête la présente facture, sauf erreur ou omission de notre part, à la somme de :\n")

    # --- amount in words (unlabeled boilerplate) ---
    # Previously absent from training entirely -- this is exactly what
    # caused "CENT" (from "CENT CINQUANTE DEUX...") to get mis-tagged as
    # ITEM_QTY at inference. Included here purely so the model learns the
    # pattern is not an entity, not because the exact wording needs to be
    # perfectly correct French (it's never labeled).
    dinars = int(total_ttc)
    millimes = round((total_ttc - dinars) * 1000)
    words = f"{french_number_to_words(dinars).upper()} DINARS ET {french_number_to_words(millimes).upper()} MILLIMES"
    lit(f"{words}\n")

    lit("A Régler exclusivement au niveau des bureaux postaux sur présentation de la facture.\n")

    # --- CCP payment-slip boilerplate (unlabeled) ---
    # Previously absent entirely -- caused CCP reference/amount fragments
    # to get mis-tagged as BUYER_ADDRESS at inference. The montant here
    # is set to match total_ttc since that's semantically what it really
    # is; the org/reference/cles codes are structural filler, not
    # meaningful identifiers worth labeling (see module docstring).
    lit(
        "Poste Coupon de Versement CCP Poste Bulletin de Versement CCP\n"
        "(coupon/bulletin Arabic mirrors)\n"
        f"Code org Montant Référence Cles Code org Montant Référence Cles\n"
        f"{ccp_code_org} {_fmt_amount(total_ttc)} {ccp_ref} {ccp_cles} "
        f"{ccp_code_org} {_fmt_amount(total_ttc)} {ccp_ref} {ccp_cles}\n"
    )

    # assemble
    text_parts = []
    entities = []
    cur = 0
    for (s, lab) in segments:
        text_parts.append(s)
        if lab is not None:
            entities.append((cur, cur + len(s), lab))
        cur += len(s)

    text = "".join(text_parts)
    return text, {"entities": entities}


def generate_ttn_dataset(n=50, seed=2026):
    rng = random.Random(seed)
    return [generate_ttn_example(rng) for _ in range(n)]


TTN_TRAIN_DATA = generate_ttn_dataset(n=200)


def write_dataset_py(dataset, path, var_name="TTN_TRAIN_DATA"):
    """Write a list of (text, {'entities': [(start, end, label), ...]})
    tuples out as an importable .py file, in the same literal-tuple shape
    as training_data.py."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Auto-generated by ttn_template_data.generate_ttn_dataset()\n")
        f.write(f"# {len(dataset)} examples, same (text, {{'entities': [...]}}) shape as training_data.py\n\n")
        f.write(f"{var_name} = [\n")
        for text, ann in dataset:
            f.write("    (\n")
            f.write(f"        {text!r},\n")
            f.write("        {\n")
            f.write('            "entities": [\n')
            for (s, e, lab) in ann["entities"]:
                f.write(f"                ({s}, {e}, {lab!r}),\n")
            f.write("            ]\n")
            f.write("        },\n")
            f.write("    ),\n")
        f.write("]\n")


if __name__ == "__main__":
    # quick self-verification: slice every entity and print alongside
    for i, (text, ann) in enumerate(TTN_TRAIN_DATA[:2]):
        print(f"===== TTN EXAMPLE {i} =====")
        print(text)
        print("--- entities ---")
        for (s, e, lab) in ann["entities"]:
            print(f"  {lab}: {text[s:e]!r}")
        print()

    out_path = "ttn_train_data.py"
    write_dataset_py(TTN_TRAIN_DATA, out_path)
    print(f"Wrote {len(TTN_TRAIN_DATA)} examples to {out_path}")