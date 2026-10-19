"""Read a Miro board as a text outline for the scenario generator.

Needs MIRO_TOKEN (a Miro REST API access token with boards:read) set on the server.
Without a token, export the board as an image or PDF and upload that instead;
Gemini reads diagrams from images directly.
"""
import html
import json
import os
import re
import urllib.request

API = "https://api.miro.com/v2/boards"
TEXT_TYPES = ("sticky_note", "shape", "text", "card", "frame", "app_card")


def _get(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _paged(url: str, token: str, fetch=_get) -> list[dict]:
    out, cursor = [], None
    while True:
        page = fetch(url + "?limit=50" + (f"&cursor={cursor}" if cursor else ""), token)
        out += page.get("data", [])
        cursor = page.get("cursor")
        if not cursor:
            return out


def _plain(s: str | None) -> str:
    s = re.sub(r"<br\s*/?>|</p>|</li>", "\n", s or "")
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def board_outline(board_id: str, token: str | None = None, fetch=_get) -> str:
    """Board items and arrows as plain text, e.g. 'sticky_note: Verify caller' and 'A -> B'."""
    token = token or os.environ.get("MIRO_TOKEN")
    if not token:
        raise RuntimeError("MIRO_TOKEN is not set on the server. Upload an exported image or PDF of the board instead.")
    board_id = board_id.strip().rstrip("/").split("/")[-1]  # accept a board URL or the bare ID
    items = _paged(f"{API}/{board_id}/items", token, fetch)
    label = {}
    lines = []
    for it in items:
        if it.get("type") not in TEXT_TYPES:
            continue
        d = it.get("data") or {}
        text = _plain(d.get("content") or d.get("title") or d.get("description"))
        if text:
            label[it["id"]] = text.replace("\n", " ")
            lines.append(f"{it['type']}: {text}")
    for c in _paged(f"{API}/{board_id}/connectors", token, fetch):
        a = label.get((c.get("startItem") or {}).get("id"))
        b = label.get((c.get("endItem") or {}).get("id"))
        if a and b:
            cap = " / ".join(_plain(x.get("content")) for x in c.get("captions") or [] if x.get("content"))
            lines.append(f"flow: {a} -> {b}" + (f" [{cap}]" if cap else ""))
    if not lines:
        raise RuntimeError("No text found on that Miro board.")
    return "\n".join(lines)
