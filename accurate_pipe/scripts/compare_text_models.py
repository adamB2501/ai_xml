# -*- coding: utf-8 -*-
"""
Run the SAME extraction prompt through two (or more) text models on the
same source text, and print each result side by side. Use this on your own
invoices to decide what to set config.TEXT_MODEL to.

    python -m accurate_pipe.scripts.compare_text_models  path/to/source.txt
    python -m accurate_pipe.scripts.compare_text_models  path/to/source.txt  mistral:latest qwen3:14b

If no source file is given it uses a small built-in sample. If no models
are given it compares config.TEXT_MODEL vs config.TEXT_MODEL_ESCALATION.

This does NOT judge correctness for you - it just puts the answers next to
each other. You compare against the actual invoice.
"""

from __future__ import annotations

import json
import sys

from .. import config, llm, prompts

_SAMPLE = """\
ideryet services
SARL au capital de 450 000 dt
R.C : B15880 1996
FACTURE N 260143
Code Client 41140840
Code TVA : 503873 Q/A/M/000 Nom Client DR. KAMMOUN MOHAMED MONCEF
335352FAP000
Le : 27/08/2026 Matricule Fiscal
EPA-C/USB CABLE USB 10.000 1 19 10.000
000 MAIN D OEUVRE 80.000 1 19 80.000
PP-CR2032 BATTERY CR2032 7.500 1 19 7.500
Total Hors TVA 97.500
Cumul TVA 18.525
Droit de Timbre 1.000
NET A PAYER 117.025
Adresse : Av.Majida Boulila, 3027 Sfax - Code TVA : 503873QAM000
"""


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    source = _SAMPLE
    models: list[str] = []
    for arg in argv:
        if arg.endswith(".txt"):
            source = open(arg, encoding="utf-8").read()
        else:
            models.append(arg)
    if not models:
        models = [config.TEXT_MODEL]
        if config.TEXT_MODEL_ESCALATION:
            models.append(config.TEXT_MODEL_ESCALATION)

    messages = prompts.build_extraction_messages(source)

    for model in models:
        print("=" * 72)
        print(f"MODEL: {model}")
        print("=" * 72)
        extra = {"think": False} if model.startswith("qwen3") else None
        try:
            res = llm.chat_json(messages, prompts.RESPONSE_SCHEMA, model=model,
                                extra_options=extra)
        except llm.OllamaError as exc:
            print(f"  FAILED: {exc}\n")
            continue
        print(f"  [{res.duration_s:.0f}s | in={res.prompt_tokens} out={res.output_tokens}]")
        try:
            parsed = json.loads(res.text)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("  [non-JSON reply]\n" + res.text)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
