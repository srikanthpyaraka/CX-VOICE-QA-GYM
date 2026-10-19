"""Gemini client used by the simulator, judge and optimizer.

Uses GOOGLE_API_KEY if set, otherwise Vertex AI with Application Default
Credentials on the connected GCP project."""
import json
import logging
import os
import time

from google import genai
from google.genai import errors, types

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
_client: genai.Client | None = None


def init(project: str) -> None:
    global _client
    if os.environ.get("GOOGLE_API_KEY"):
        _client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    else:
        _client = genai.Client(vertexai=True, project=project,
                               location=os.environ.get("GEMINI_LOCATION", "global"))


def file_part(data: bytes, mime_type: str):
    """A PDF, image or text file Gemini reads alongside the prompt."""
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def generate_json(prompt: str, temperature: float = 0.2, parts: list | None = None) -> dict:
    if _client is None:
        raise RuntimeError("LLM not initialised: connect to a project first")
    for delay in (5, 10, 20, 40, None):
        try:
            resp = _client.models.generate_content(
                model=MODEL,
                contents=[*(parts or []), prompt],
                config=types.GenerateContentConfig(
                    temperature=temperature, response_mime_type="application/json"),
            )
            return json.loads(resp.text)
        except errors.APIError as e:
            if e.code not in (429, 503) or delay is None:
                raise
            logging.warning("Gemini throttled (%s); retrying in %ss", e.code, delay)
            time.sleep(delay)
