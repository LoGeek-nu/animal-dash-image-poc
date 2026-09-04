"""Vercel entrypoint: HTTP API wrapping preprocess + Gemini status pipeline.

Vercel loads this module and runs the top-level `app` (ASGI) as a Vercel
Function. This is the API-facing counterpart to `cli.py` — both call into
the same `pipeline.run_full_pipeline`, so behavior stays identical to the
local CLI (`mise run run`).

Intended caller: the animal-dash Cloudflare Worker only (server-to-server),
never a browser directly. `API_SHARED_SECRET` must be kept out of any
browser-visible code.
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from animal_dash_image_poc import pipeline

load_dotenv()

app = FastAPI(title="animal-dash-image-poc API")

CONCURRENCY_LIMIT = int(os.environ.get("CONCURRENCY_LIMIT", "4"))
GENERATE_TIMEOUT_SECONDS = float(os.environ.get("GENERATE_TIMEOUT_SECONDS", "45"))

# Best-effort concurrency guard against Gemini's free-tier rate limit.
# Only limits requests within a single warm instance — Vercel may run
# multiple instances in parallel, so this is not a global limit.
_semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


def _check_api_key(x_api_key: str | None) -> None:
    expected = os.environ.get("API_SHARED_SECRET")
    if not expected:
        # Fail closed: an unset secret must never mean "accept everything".
        raise HTTPException(status_code=500, detail="API_SHARED_SECRET is not configured")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _run_pipeline_on_bytes(image_bytes: bytes) -> dict:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # cv2.imread reads by content signature, not extension, so the
        # actual upload format (jpg/png) doesn't need to match this name.
        input_path = Path(tmp_dir) / "input.png"
        input_path.write_bytes(image_bytes)
        output_prefix = str(Path(tmp_dir) / "output")

        status = pipeline.run_full_pipeline(str(input_path), output_prefix)
        png_bytes = (Path(tmp_dir) / "output.png").read_bytes()

    return {
        "image_base64": base64.b64encode(png_bytes).decode("ascii"),
        "status": status,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "animal-dash-image-poc"}


@app.post("/v1/characters/generate")
async def generate_character(
    image: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
) -> JSONResponse:
    _check_api_key(x_api_key)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image upload")

    if _semaphore.locked():
        return JSONResponse(
            status_code=429,
            content={"error": "too_many_concurrent_requests", "retryable": True},
        )

    async with _semaphore:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_run_pipeline_on_bytes, image_bytes),
                timeout=GENERATE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"error": "generation_timeout", "retryable": True},
            )
        except RuntimeError as exc:
            # e.g. GEMINI_API_KEY missing on the server — retrying won't help.
            return JSONResponse(
                status_code=500,
                content={"error": "server_misconfigured", "detail": str(exc), "retryable": False},
            )
        except Exception as exc:  # Gemini 503 / transient failures
            return JSONResponse(
                status_code=502,
                content={"error": "generation_failed", "detail": str(exc), "retryable": True},
            )

    return JSONResponse(content=result)
