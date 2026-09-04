# -*- coding: utf-8 -*-
"""
The review UI: a local FastAPI app + one static HTML page. Not a new
review system -- it's a window onto the same teif_pipeline.review SQLite
store adaptive_pipe.queue already writes to, plus a folder/file picker
that kicks off adaptive_pipe.pipeline.run() in the background.

Run it:
    uvicorn adaptive_pipe.ui.server:app --reload
then open http://127.0.0.1:8000 in a browser.

Files:
    highlight.py   -- maps a field's extracted value back to where it sits
                     on the PDF page (exact via pdfplumber where a text
                     layer exists, approximate -- a whole region -- where
                     it doesn't).
    jobs.py        -- a minimal in-memory background job runner so
                     processing a folder of PDFs (each a multi-minute
                     model run) doesn't block an HTTP request.
    server.py      -- the FastAPI app: browse/process/queue/correct/
                     approve/highlights/xml endpoints, and the one route
                     that serves the static page.
    static/index.html -- the whole frontend: folder picker, queue list,
                     split PDF-viewer/fields-panel review screen with
                     color-linked highlighting, in vanilla JS.
"""
