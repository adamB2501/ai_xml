# -*- coding: utf-8 -*-
"""
One model, one extraction call -- no escalation logic in here. Escalation
(when to try the heavier model) is a decision pipeline.py makes AFTER
seeing verify.py's deterministic result, not something this function
decides for itself. That's the actual fix over accurate_pipe.extract's
original design: there, the light model's own (unreliable) self-report
decided whether to escalate, and we watched it under-report real problems.
Moving the decision to the caller, based on a deterministic check, is the
whole point of this pipeline's tiering.

Reuses accurate_pipe's prompts and llm modules directly -- the prompt
text, the JSON schema, and the Ollama call plumbing are unchanged and
already tested; only the escalation policy around them is new here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from accurate_pipe import llm, prompts

_NO_THINK = {"think": False}  # qwen3 emits <think> unless told not to


@dataclass
class ExtractionResult:
    fields: dict
    model: str
    llm_calls: list = field(default_factory=list)


def extract_once(source_text: str, model: str) -> ExtractionResult:
    if not source_text.strip():
        return ExtractionResult(
            fields={"_verification": {"checks_done": [],
                                     "problems_found": ["no source text to extract from"]}},
            model=model,
        )

    messages = prompts.build_extraction_messages(source_text)
    extra = _NO_THINK if model.startswith("qwen3") else None
    result = llm.chat_json(messages, prompts.RESPONSE_SCHEMA, model=model, extra_options=extra)

    try:
        fields = json.loads(result.text)
    except json.JSONDecodeError as exc:
        fields = {
            "_verification": {"checks_done": [],
                             "problems_found": [f"model reply was not valid JSON: {exc}"]},
            "_raw_reply": result.text[:2000],
        }
    return ExtractionResult(fields=fields, model=model, llm_calls=[result])
