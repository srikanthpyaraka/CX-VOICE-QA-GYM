"""Deterministic voice-channel checks on agent turns.

Text that reads fine in chat often fails over TTS: long monologues, markdown,
URLs, several questions at once, slow responses."""
import re

MAX_WORDS = 45          # ~15s of speech at normal TTS rate (text mode proxy)
MAX_LATENCY_S = 3.0     # text mode: wall time of the Sessions call
MAX_AUDIO_S = 15.0      # audio mode: measured seconds of agent speech
MAX_GAP_S = 2.0         # audio mode: caller stops talking -> agent's first audio
MAX_WER = 0.2           # caller line misrecognised by the agent's ASR

_MARKDOWN = re.compile(r"(\*\*|__|`|^\s*#|^\s*[-*•]\s|^\s*\d+\.\s|\[.+?\]\(.+?\))", re.M)
_URL = re.compile(r"(https?://|www\.|\S+@\S+\.\w+)")


LIMITS = {"max_words": MAX_WORDS, "max_latency_s": MAX_LATENCY_S, "max_audio_s": MAX_AUDIO_S,
          "max_gap_s": MAX_GAP_S, "max_wer": MAX_WER}


def turn_issues(text: str, latency_s: float | None,
                audio_s: float | None = None, limits: dict | None = None) -> list[str]:
    """audio_s is set only in audio mode; it switches length and latency rules
    from text proxies to measured audio. limits overrides LIMITS (suite `scoring:`)."""
    lim = {**LIMITS, **(limits or {})}
    issues = []
    if not text.strip() and not audio_s:
        issues.append("empty_response")
        return issues
    if audio_s is not None:
        if audio_s > lim["max_audio_s"]:
            issues.append("too_long_for_tts")
    elif len(text.split()) > lim["max_words"]:
        issues.append("too_long_for_tts")
    if _MARKDOWN.search(text):
        issues.append("markdown_read_aloud")
    if _URL.search(text):
        issues.append("url_or_email_read_aloud")
    if text.count("?") > 1:
        issues.append("multiple_questions")
    if latency_s is not None and latency_s > (lim["max_gap_s"] if audio_s is not None else lim["max_latency_s"]):
        issues.append("high_latency")
    return issues


def score(agent_turns: list[dict]) -> float:
    """Fraction of agent turns with no voice issues."""
    if not agent_turns:
        return 0.0
    return sum(1 for t in agent_turns if not t["voice_issues"]) / len(agent_turns)


def latency_stats(agent_turns: list[dict]) -> dict:
    lat = sorted(t["latency_s"] for t in agent_turns if t.get("latency_s") is not None)
    if not lat:
        return {"p50": None, "p95": None}
    return {"p50": lat[len(lat) // 2], "p95": lat[min(len(lat) - 1, int(len(lat) * 0.95))]}


def _norm(s: str) -> list[str]:
    s = re.sub(r"[^\w\s]", " ", s.lower())
    s = re.sub(r"(?<=\d) (?=\d)", "", s)   # "4 8 2 1 3" -> "48213"
    return s.split()


def wer(said: str, heard: str | None) -> float | None:
    """Word error rate of the agent's ASR against what the caller said."""
    if heard is None:
        return None
    ref, hyp = _norm(said), _norm(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return round(prev[-1] / len(ref), 3)
