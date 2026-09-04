# -*- coding: utf-8 -*-
"""TEIF referential codes (Annexe A of the TTN implementation guide) --
extracted here so builder.py and validate.py share exactly one copy
instead of two independently-typed-out lists silently drifting apart.
"""

TEIF_VERSION = "1.8.8"
DEFAULT_COUNTRY = "TN"
DEFAULT_CURRENCY = "TND"

VALID_CONTROLLING_AGENCY = {"TTN", "Tunisie TradeNet"}

# Referentiel I-0: Partner Identifier Type
VALID_PARTNER_ID_TYPE = {"I-01", "I-02", "I-03", "I-04"}

# Referentiel I-3: Date Function
VALID_DATE_FUNCTION = {f"I-3{n}" for n in range(1, 9)}

# Referentiel I-6: Partner Function
VALID_PARTNER_FUNCTION = {f"I-6{n}" for n in range(1, 10)}
SELLER_FUNCTIONS = {"I-62", "I-63", "I-66"}  # Fournisseur / Vendeur / Emetteur
BUYER_FUNCTIONS = {"I-61", "I-64", "I-65"}  # Acheteur / Client / Receveur

# Referentiel I-10: Communication Means
VALID_COMM_MEANS = {"I-101", "I-102", "I-103", "I-104"}

# Referentiel I-16: Tax type
VALID_TAX_TYPE_CODE = {f"I-16{n}" for n in range(1, 10)} | {"I-160", "I-1601", "I-1602", "I-1603", "I-1604"}

# Referentiel I-17: Amount type
VALID_AMOUNT_TYPE_CODE = {f"I-17{n}" for n in range(1, 10)} | {f"I-18{n}" for n in range(0, 10)}
