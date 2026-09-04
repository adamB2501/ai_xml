# -*- coding: utf-8 -*-
"""
STEP 2 of the pipeline:  source text  ->  fields dict.

One call to the TEXT model (see prompts.py) with structured output forced
by a JSON schema, so the reply is always parseable. The prompt makes the
model do its own recheck pass and report what it found in "_verification".

Escalation
----------
If the light model (config.TEXT_MODEL) finishes with a non-empty
_verification.problems_found, we run the exact same call once more with the
heavier model (config.TEXT_MODEL_ESCALATION). We keep the heavier model's
answer only if it reports *fewer* problems; otherwise we keep the first.
This is cheap insurance: most invoices never trigger it.

Set config.TEXT_MODEL_ESCALATION = None to turn escalation off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config, llm, prompts


# reasoning models (qwen3) emit <think>...</think> unless told not to; this
# option tells Ollama to disable it so `format` json output stays clean.
_NO_THINK = {"think": False}


@dataclass
class ExtractionResult:
    fields: dict                       # the model's structured answer
    model_used: str
    escalated: bool = False
    llm_calls: list[llm.LLMResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def problems_found(self) -> list[str]:
        return list((self.fields.get("_verification") or {}).get("problems_found") or [])


def _call(source_text: str, model: str) -> tuple[dict, llm.LLMResult]:
    messages = prompts.build_extraction_messages(source_text)
    extra = _NO_THINK if model.startswith("qwen3") else None
    result = llm.chat_json(
        messages, prompts.RESPONSE_SCHEMA, model=model, extra_options=extra
    )
    try:
        parsed = json.loads(result.text)
    except json.JSONDecodeError as exc:
        # With `format` set this should not happen, but never crash the run.
        parsed = {
            "_verification": {
                "checks_done": [],
                "problems_found": [f"model reply was not valid JSON: {exc}"],
            },
            "_raw_reply": result.text[:2000],
        }
    return parsed, result


def extract_fields(source_text: str) -> ExtractionResult:
    if not source_text.strip():
        return ExtractionResult(
            fields={"_verification": {"checks_done": [],
                                     "problems_found": ["no source text to extract from"]}},
            model_used="(none)",
            notes=["ingest produced empty text - nothing to extract"],
        )

    # --- light model first ------------------------------------------------
    fields, call = _call(source_text, config.TEXT_MODEL)
    out = ExtractionResult(fields=fields, model_used=config.TEXT_MODEL, llm_calls=[call])
    problems = out.problems_found

    # --- escalate only if it flagged unresolved problems -----------------
    if problems and config.TEXT_MODEL_ESCALATION:
        out.notes.append(
            f"{config.TEXT_MODEL} self-reported {len(problems)} problem(s); "
            f"retrying with {config.TEXT_MODEL_ESCALATION}."
        )
        heavy_fields, heavy_call = _call(source_text, config.TEXT_MODEL_ESCALATION)
        out.llm_calls.append(heavy_call)
        heavy_problems = (heavy_fields.get("_verification") or {}).get("problems_found") or []
        if len(heavy_problems) < len(problems):
            out.fields = heavy_fields
            out.model_used = config.TEXT_MODEL_ESCALATION
            out.escalated = True
            out.notes.append(
                f"kept {config.TEXT_MODEL_ESCALATION} answer "
                f"({len(heavy_problems)} problem(s) vs {len(problems)})."
            )
        else:
            out.notes.append(
                f"kept {config.TEXT_MODEL} answer "
                f"({config.TEXT_MODEL_ESCALATION} did not do better)."
            )

    return out
