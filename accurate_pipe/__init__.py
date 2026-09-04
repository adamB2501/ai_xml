# -*- coding: utf-8 -*-
"""
accurate_pipe
=============

A second, independent invoice -> TEIF-XML pipeline, built around the idea
that *real* invoices are messy (crooked scans, dropped words, PDFs whose
text layer is only half there) and that the safest way to read them is:

    1. get the text as reliably as possible
         - exact embedded text where a real text layer exists (pdfplumber)
         - a vision model (qwen2.5-VL via Ollama) for the parts that are
           only pixels: a rasterized letterhead/footer band, or a fully
           scanned page
    2. hand that text to a *text* LLM whose only job is to turn it into
       structured fields -- with a strict, explicit prompt AND a mandatory
       second "re-check your own answer" pass
    3. run deterministic cross-checks on the result (arithmetic, tax-id
       shape, "does this value actually appear in the source")
    4. build TEIF XML and validate it

Nothing here is magic. Every step is a plain function you can read top to
bottom. The heavy lifting of *TEIF XML itself* (element names, referential
codes, the XSD-ish validator) is NOT re-implemented -- we import it from
the existing `teif_pipeline` package, because that part is already correct
and well tested. This package is only the *reading* half done differently.

Entry point:  python -m accurate_pipe.run  path/to/invoice.pdf

Read the files in this order the first time:
    config.py     - every knob, one place, each explained
    prompts.py    - the exact instructions sent to the models, and WHY
    ingest.py     - PDF -> source text (the classify / render / OCR logic)
    extract.py    - source text -> fields dict (the LLM call + recheck)
    verify.py     - the deterministic sanity checks
    assemble.py   - fields dict -> canonical Invoice object
    pipeline.py   - the 4 steps above, wired together
    run.py        - command-line wrapper
"""

__all__ = ["pipeline", "config"]
