# -*- coding: utf-8 -*-
"""
Thin wrapper around the local Ollama HTTP API. Every model call in the
pipeline goes through one of the three functions here, so there is exactly
one place that knows the URL, the timeout, and the payload shape.

No third-party HTTP library - just urllib from the standard library, same
as teif_pipeline's own llm_backend.py. Keeps the dependency list short.

Ollama endpoints used:
  POST /api/generate   - single prompt (+ optional images) -> single reply
  POST /api/chat       - messages array (+ optional `format` schema) -> reply

Both are called with stream=False, so we get the whole answer in one JSON
response instead of a token stream.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import config


class OllamaError(RuntimeError):
    """Ollama is unreachable, returned an HTTP error, or sent back junk."""


@dataclass
class LLMResult:
    """What every call here returns. `text` is the model's reply; the rest
    is bookkeeping we surface in the run report so a slow/expensive run is
    visible."""
    text: str
    model: str
    duration_s: float
    prompt_tokens: int | None
    output_tokens: int | None


# ---------------------------------------------------------------------------
# low-level POST
# ---------------------------------------------------------------------------

def _post(path: str, payload: dict) -> dict:
    url = config.OLLAMA_URL.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise OllamaError(f"{path} -> HTTP {exc.code}: {body[:500]}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"cannot reach Ollama at {config.OLLAMA_URL} ({exc.reason}). "
            "Is `ollama serve` running?"
        ) from exc


def _timed(fn):
    """Decorator: measure wall time of a call and fold it into LLMResult."""
    import functools
    import time

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> LLMResult:
        t0 = time.time()
        raw, model = fn(*args, **kwargs)
        dt = time.time() - t0
        # Ollama returns these on both endpoints; may be missing on error.
        return LLMResult(
            text=raw.get("response") or raw.get("message", {}).get("content", ""),
            model=model,
            duration_s=dt,
            prompt_tokens=raw.get("prompt_eval_count"),
            output_tokens=raw.get("eval_count"),
        )

    return wrapper


# ---------------------------------------------------------------------------
# image helper
# ---------------------------------------------------------------------------

def image_to_b64_png(pil_image, max_px: int | None = None) -> str:
    """PIL image -> base64 PNG string, downscaled so the longer side is at
    most `max_px` (default config.MAX_IMAGE_PX). Downscaling here, once,
    keeps every vision call consistent and fast."""
    max_px = max_px or config.MAX_IMAGE_PX
    w, h = pil_image.size
    longest = max(w, h)
    if longest > max_px:
        scale = max_px / longest
        pil_image = pil_image.resize((round(w * scale), round(h * scale)))
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# the three call types
# ---------------------------------------------------------------------------

@_timed
def vision(prompt: str, pil_image, model: str | None = None):
    """Send one image + a text prompt to the vision model. Used for
    transcribing a scanned page or a rasterized letterhead/footer band."""
    model = model or config.VISION_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_to_b64_png(pil_image)],
        "stream": False,
        "options": {"temperature": config.TEMPERATURE, "num_ctx": config.NUM_CTX},
    }
    return _post("/api/generate", payload), model


@_timed
def chat_json(messages: list[dict], schema: dict, model: str | None = None,
              extra_options: dict | None = None):
    """Send a messages array to the text model and force the reply to match
    `schema` (Ollama's structured-output `format`). Used for field
    extraction. `extra_options` lets a caller pass e.g. {"think": False}
    for reasoning models."""
    model = model or config.TEXT_MODEL
    options = {"temperature": config.TEMPERATURE, "num_ctx": config.NUM_CTX}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": schema,
        "options": options,
    }
    if extra_options:
        payload.update(extra_options)
    return _post("/api/chat", payload), model


@_timed
def generate(prompt: str, model: str | None = None):
    """Plain text prompt -> text reply, no image, no schema. Not used by
    the main pipeline today; kept for ad-hoc probes and scripts."""
    model = model or config.TEXT_MODEL
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": config.TEMPERATURE, "num_ctx": config.NUM_CTX},
    }
    return _post("/api/generate", payload), model
