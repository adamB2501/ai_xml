# -*- coding: utf-8 -*-
"""
STEP 3b of the pipeline:  the targeted re-ask loop.

verify.py (STEP 3) tells us which fields are missing or failed a check.
This module asks the text model again for ONLY those fields, hands it the
context it needs (what's already known, any arithmetically-implied value),
and folds an answer back in ONLY if it survives the same checks.

Loop shape
----------
    while there are still-deficient fields that haven't hit their retry cap
      and we're under the global round ceiling:
        - ask the model for those fields (one call, all of them)
        - for each returned value: accept it iff
              (a) it appears verbatim in the source text, AND
              (b) folding it in does not increase the number of ERROR-level
                  problems (and, if that field had an error, the error is gone)
        - re-run verify.py
        - bump each asked field's attempt counter

Anything still deficient when the loop ends is listed by name in the
result; the pipeline turns that into a "needs human review" verdict with
those specific fields called out.

Config:
    REASK_MAX_ATTEMPTS_PER_FIELD   retries allowed for one field
    REASK_MAX_ROUNDS               hard ceiling on model calls
    REASK_MODEL                    which model (None -> config.TEXT_MODEL)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import config, llm, prompts, verify
from .numparse import to_decimal


# fields we actively try to recover if absent (superset of TEIF-mandatory)
_EXPECTED = [
    "invoice_number", "issue_date",
    "seller_name", "seller_tax_id",
    "buyer_name", "buyer_tax_id",
    "total_ht", "tva_amount", "stamp_duty", "total_ttc",
]

# verify.Problem.field values that are not a single JSON key -> the keys
# they actually implicate
_GROUP_FIELDS = {
    "totals": ["total_ht", "tva_amount", "stamp_duty", "total_ttc"],
    "line_items": ["line_items"],
}
_IGNORE_PROBLEM_FIELDS = {"_verification"}  # model self-report, not directly actionable


@dataclass
class RoundLog:
    round_no: int
    asked: list[str]
    accepted: list[str]
    rejected: list[str]          # returned a value but it failed the accept test
    model_seconds: float


@dataclass
class ReaskResult:
    fields: dict                          # possibly-updated fields dict
    problems: list[verify.Problem]        # verify() result after the loop
    still_deficient: list[str]            # fields we could not fix
    attempts: dict                        # field -> how many times re-asked
    rounds: list[RoundLog] = field(default_factory=list)
    llm_calls: list[llm.LLMResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return any(r.accepted for r in self.rounds)


# ---------------------------------------------------------------------------
# which fields are deficient right now
# ---------------------------------------------------------------------------

def blocking_fields(fields: dict, problems: list[verify.Problem]) -> list[str]:
    """The subset of deficient fields whose failure should force human
    review if the loop can't fix it: a mandatory field that is still null,
    or a field named by an ERROR-level problem. A field that only ever had
    a WARNING (e.g. "value not found verbatim" on a lightly-reformatted
    number) is worth a re-ask but does NOT block on its own."""
    mandatory_null = [k for k in ("invoice_number", "issue_date", "seller_tax_id",
                                  "buyer_tax_id", "total_ttc")
                      if not fields.get(k)]
    error_fields: list[str] = []
    for p in problems:
        if p.severity != "error":
            continue
        for key in _GROUP_FIELDS.get(p.field, [p.field]):
            if key not in error_fields:
                error_fields.append(key)
    out: list[str] = []
    for k in mandatory_null + error_fields:
        if k not in out:
            out.append(k)
    return out


def deficient_fields(fields: dict, problems: list[verify.Problem]) -> list[str]:
    """Ordered, de-duplicated list of top-level keys worth re-asking
    (includes warning-flagged fields - a re-ask is cheap and might help)."""
    out: list[str] = []

    def want(key: str):
        if key not in out:
            out.append(key)

    # 1. expected fields that are null
    for key in _EXPECTED:
        if not fields.get(key):
            want(key)

    # 2. fields named by a verify problem
    for p in problems:
        if p.field in _IGNORE_PROBLEM_FIELDS:
            continue
        if p.field in _GROUP_FIELDS:
            # only re-ask the members that are null, unless none are (then all,
            # because one of the present values must be misread)
            members = _GROUP_FIELDS[p.field]
            missing = [m for m in members if m != "line_items" and not fields.get(m)]
            if p.field == "line_items":
                want("line_items")
            elif missing:
                for m in missing:
                    want(m)
            else:
                for m in members:
                    want(m)
        else:
            want(p.field)

    return out


# ---------------------------------------------------------------------------
# hints handed to the model
# ---------------------------------------------------------------------------

def _fmt(d) -> str:
    return f"{d:.3f}".rstrip("0").rstrip(".") if d is not None else "?"


def build_hints(fields: dict) -> dict[str, str]:
    hints: dict[str, str] = {}

    ht = to_decimal(fields.get("total_ht"))
    tva = to_decimal(fields.get("tva_amount"))
    ttc = to_decimal(fields.get("total_ttc"))
    stamp = to_decimal(fields.get("stamp_duty"))

    # stamp duty is fixed by the totals: ttc - ht - tva
    if ttc is not None and ht is not None and tva is not None and stamp is None:
        implied = ttc - ht - tva
        if implied > 0:
            hints["stamp_duty"] = (
                f"the totals imply it is about {_fmt(implied)} "
                f"(NET A PAYER {_fmt(ttc)} - Total HT {_fmt(ht)} - TVA {_fmt(tva)}). "
                "Look for 'Droit de Timbre' / 'Timbre'."
            )

    # buyer tax id: it's the OTHER tax-id-shaped value, not the seller's
    s_tax = fields.get("seller_tax_id")
    if s_tax and not fields.get("buyer_tax_id"):
        hints["buyer_tax_id"] = (
            f"the SELLER's tax id is '{s_tax}'. The buyer's is a DIFFERENT "
            "value with the same shape (6-7 digits, 3 letters, 3 digits), in "
            "the client / 'DOIT' block, often on its own line near "
            "'Matricule Fiscal'."
        )
    # and the reverse
    b_tax = fields.get("buyer_tax_id")
    if b_tax and not fields.get("seller_tax_id"):
        hints["seller_tax_id"] = (
            f"the BUYER's tax id is '{b_tax}'. The seller's is a DIFFERENT "
            "value with the same shape, usually in the footer near 'Code TVA' "
            "or 'MF'."
        )

    return hints


# ---------------------------------------------------------------------------
# accept / reject one proposed value
# ---------------------------------------------------------------------------

def _occurs(value, source_text: str) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    hay = re.sub(r"\s+", "", source_text).casefold()
    return re.sub(r"\s+", "", value).casefold() in hay


def _error_count(problems: list[verify.Problem]) -> int:
    return sum(1 for p in problems if p.severity == "error")


def _field_has_error(problems: list[verify.Problem], key: str) -> bool:
    """Is there an ERROR-level problem that is *about* this field? A
    "totals" problem counts only for the four totals fields; a
    "line_items" problem only for line_items."""
    relevant = {key}
    if key in _GROUP_FIELDS["totals"]:
        relevant.add("totals")
    if key == "line_items":
        relevant.add("line_items")
    return any(p.severity == "error" and p.field in relevant for p in problems)


def _accept(key, value, base_fields, source_text, problems_before) -> tuple[bool, str]:
    """Returns (accepted, reason)."""
    # line_items: can't do a verbatim check on a list; use the gate + a
    # completeness heuristic instead
    if key == "line_items":
        if not isinstance(value, list) or not value:
            return False, "empty / not a list"
        trial = {**base_fields, "line_items": value}
        after = verify.verify(trial, source_text)
        if _error_count(after) > _error_count(problems_before):
            return False, "would add an error"
        old_desc = sum(1 for it in (base_fields.get("line_items") or [])
                       if it.get("description"))
        new_desc = sum(1 for it in value if it.get("description"))
        if new_desc < old_desc:
            return False, f"fewer described rows ({new_desc} < {old_desc})"
        return True, f"{len(value)} rows, {new_desc} described"

    # scalar
    if not _occurs(value, source_text):
        return False, "value not found verbatim in source text"
    trial = {**base_fields, key: value}
    after = verify.verify(trial, source_text)
    if _error_count(after) > _error_count(problems_before):
        return False, "would add an error elsewhere"
    if _field_has_error(problems_before, key) and _field_has_error(after, key):
        return False, "does not clear the field's own error"
    return True, "ok"


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def run(fields: dict, source_text: str,
        problems: list[verify.Problem],
        *, model: str | None = None,
        extra_hints: dict[str, str] | None = None,
        on_round: Optional[Callable[[int, list[str]], None]] = None,
        max_rounds: Optional[int] = None) -> ReaskResult:
    """`model` overrides config.REASK_MODEL/TEXT_MODEL (used by callers that
    want the re-ask loop to run on a specific tier, e.g. an escalation
    pass). `extra_hints` are merged on top of the generic arithmetic/tax-id
    hints build_hints() computes -- for a caller (e.g. adaptive_pipe's
    per-seller memory) that has sharper, source-specific hints to offer.
    `on_round(round_no, wanted_fields)` is called right before each round's
    model call, purely for progress reporting (e.g. adaptive_pipe's UI
    showing "refining fields, round 2" while this is still running) -- it
    does not affect what the loop does. `max_rounds` overrides
    config.REASK_MAX_ROUNDS (used by a caller that wants a tighter cap on
    a slow model without changing the shared default for every caller).
    All four default to None/no-op so existing callers are unaffected."""
    fields = dict(fields)
    model = model or config.REASK_MODEL or config.TEXT_MODEL
    max_rounds = config.REASK_MAX_ROUNDS if max_rounds is None else max_rounds
    extra = {"think": False} if model.startswith("qwen3") else None

    attempts: Counter = Counter()
    res = ReaskResult(fields=fields, problems=list(problems),
                      still_deficient=[], attempts=attempts)

    if not source_text.strip():
        res.notes.append("no source text - re-ask loop skipped")
        res.still_deficient = deficient_fields(fields, problems)
        return res

    for round_no in range(1, max_rounds + 1):
        deficient = deficient_fields(fields, res.problems)
        wanted = [f for f in deficient
                  if attempts[f] < config.REASK_MAX_ATTEMPTS_PER_FIELD]
        if not wanted:
            break

        if on_round:
            on_round(round_no, wanted)

        for f in wanted:
            attempts[f] += 1

        hints = {**build_hints(fields), **(extra_hints or {})}
        messages = prompts.build_reask_messages(
            source_text, wanted, known=fields, hints=hints
        )
        schema = prompts.build_reask_schema(wanted)
        try:
            call = llm.chat_json(messages, schema, model=model, extra_options=extra)
        except llm.OllamaError as exc:
            res.notes.append(f"round {round_no}: model call failed ({exc}); stopping loop")
            break
        res.llm_calls.append(call)

        try:
            proposed = json.loads(call.text)
        except json.JSONDecodeError:
            res.rounds.append(RoundLog(round_no, wanted, [], wanted, call.duration_s))
            res.notes.append(f"round {round_no}: reply was not JSON; nothing applied")
            continue

        # apply scalars before line_items, and mutate `fields` as we accept
        # so a later value in the same round is judged against the earlier
        # accepted ones (e.g. line_items' sum check sees a corrected total_ht).
        # `res.problems` stays the pre-round baseline for the "didn't get worse"
        # comparison.
        accepted, rejected = [], []
        order = ([k for k in wanted if k != "line_items"]
                 + [k for k in wanted if k == "line_items"])
        for key in order:
            if key not in proposed:
                continue
            value = proposed[key]
            if value in (None, "", [], {}):
                continue
            ok, why = _accept(key, value, fields, source_text, res.problems)
            if ok:
                fields[key] = value
                accepted.append(key)
            else:
                rejected.append(f"{key} ({why})")

        res.rounds.append(RoundLog(round_no, wanted, accepted, rejected, call.duration_s))
        res.problems = verify.verify(fields, source_text)

        if not accepted:
            # a whole round produced nothing usable - one more round rarely
            # helps, but per-field caps will stop us anyway; let the loop
            # decide via `wanted` on the next pass
            continue

    res.fields = fields
    # what still forces review = blocking fields only (mandatory-null / error),
    # not fields that merely carried a warning we couldn't clear.
    res.still_deficient = blocking_fields(fields, res.problems)
    unresolved_warn = [f for f in deficient_fields(fields, res.problems)
                       if f not in res.still_deficient]
    if res.still_deficient:
        res.notes.append(
            "re-ask loop could not resolve (blocking): "
            + ", ".join(res.still_deficient) + " -> human review"
        )
    if unresolved_warn:
        res.notes.append(
            "re-ask loop left warnings on: " + ", ".join(unresolved_warn)
            + " (not blocking)"
        )
    return res
