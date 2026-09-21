"""Send the preprocessed transparent PNG to Gemini and get back game-character status JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path

from google import genai
from google.genai import types

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

STAT_TOTAL = 30

# プロンプト文面はコードと分けて prompts/ に置き、文面だけ調整できるようにする。
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "character_status.md"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "animal_type": {"type": "string", "description": "描かれた動物の種類（不明な場合は推測）"},
        "features": {
            "type": "array",
            "items": {"type": "string"},
            "description": "見た目の特徴を短い単語で3〜5個",
        },
        "personality": {"type": "string", "description": "性格を一言で"},
        "title": {"type": "string", "description": "短い称号（例: 韋駄天のうさぎ）"},
        "stats": {
            "type": "object",
            "properties": {
                "speed": {"type": "integer"},
                "jump": {"type": "integer"},
                "power": {"type": "integer"},
            },
            "required": ["speed", "jump", "power"],
        },
        "notes": {"type": "string", "description": "判定に関する補足（任意）"},
    },
    "required": ["animal_type", "features", "personality", "title", "stats"],
}

def load_prompt() -> str:
    """Load the judging prompt text from prompts/character_status.md.

    Kept out of the code so the wording can be iterated on without touching
    the pipeline logic (e.g. by non-engineers on the team).
    """
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return text.replace("{{stat_total}}", str(STAT_TOTAL))


def generate_status(image_path: str, api_key: str | None = None, model: str = DEFAULT_MODEL) -> dict:
    """Call Gemini with the transparent PNG and return the parsed status dict."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill in your key, "
            "or pass --api-key."
        )

    client = genai.Client(api_key=key)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            load_prompt(),
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        ),
    )

    data = json.loads(response.text)
    _validate_stats(data)
    return data


def _validate_stats(data: dict) -> None:
    stats = data.get("stats", {})
    total = sum(int(stats.get(k, 0)) for k in ("speed", "jump", "power"))
    data.setdefault("notes", "")
    if total != STAT_TOTAL:
        data["notes"] = (data.get("notes") or "") + f" [warning: stats total={total}, expected {STAT_TOTAL}]"
