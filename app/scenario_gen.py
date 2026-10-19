"""Write or refine a test suite (scenarios/suite.yaml) from source material with Gemini.

Sources can be any mix of: pasted text, Markdown/text files, PDFs, images (screenshots or
exported Miro/whiteboard diagrams), a Miro board read through its API (see miro.py), and the
connected agent's own instructions. The result is returned as YAML for a person to review.
Nothing is saved here.
"""
import yaml

from app import llm

MIME = {".pdf": "application/pdf", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".md": "text/plain", ".txt": "text/plain", ".csv": "text/plain",
        ".yaml": "text/plain", ".yml": "text/plain", ".json": "text/plain"}

SCENARIO_RULES = """Rules for every scenario:
- id: unique snake_case, describes the situation (e.g. refund_after_late_delivery).
- split: about one third "holdout" (hidden exam calls), spread across categories; the rest "train".
- persona: how the caller talks and behaves (mood, pace, habits), one sentence.
- goal: the specific outcome the caller wants, with concrete values.
- facts: the concrete test data the caller knows and can give when asked (order numbers, names,
  emails, dates). Reuse values that appear in the sources; otherwise invent clearly fictitious ones
  and keep them consistent across scenarios.
- success_criteria: 2-3 checks a reviewer can confirm from the transcript alone.
- max_turns: 6-12 caller turns.
- covers: the requirement IDs or short requirement names from the sources that this scenario's
  success criteria actually check. Don't tag a requirement the call doesn't exercise.
- Only test behaviour the sources describe. If a rule is missing or ambiguous, list it under gaps
  instead of inventing it."""

GENERATE = """You design simulated test calls for a {channel} customer-service agent built in
Google CX Agent Studio. The calls will be played by an AI caller and scored by an AI judge.

From the source material, work out: what customers call about, the business rules and policies,
what data the agent needs, when it must escalate, and how it could fail.

Write {count} scenarios with this mix:
- about 40% core journeys (the most common reasons people call), done the normal way;
- edge cases: missing or wrong information, corrections, ambiguous requests, changes of mind;
- policy and security: identity checks, attempts to get someone else's data, refusals;
- escalation: asks for a human, frustration, things the agent must not handle;
- robustness: off-topic chat, multiple requests in one call{voice_extra}.

Also write 3-8 policies: house rules that the sources state or directly require, phrased so a
reviewer can check them on every call. Do not add good-practice rules the sources don't state
(e.g. identity checks the sources never mention); put those under gaps as "Suggested rule: ...".

{rules}
{notes}
Return JSON:
{{"policies": ["..."],
  "scenarios": [{{"id": "...", "split": "train|holdout", "persona": "...", "goal": "...",
                  "facts": {{"key": "value"}}, "success_criteria": ["..."], "max_turns": 8,
                  "covers": ["..."]}}],
  "coverage": [{{"requirement": "...", "scenarios": ["id", ...]}}],
  "gaps": ["requirement or rule the sources leave unclear"]}}"""

REFINE = """You improve an existing test suite for a {channel} customer-service agent built in
Google CX Agent Studio.

CURRENT SUITE (YAML):
{suite}

Improve it using the source material (if any) and these requests from the user: {notes}

- Keep good scenarios and their ids. Sharpen vague goals and success criteria so a reviewer can
  check them from the transcript.
- Add scenarios for requirements or failure modes the suite misses; remove near-duplicates.
- Keep facts consistent with the existing test data.
- Keep existing policies unless the sources or the user contradict them; don't invent new rules
  (suggest them under gaps instead).
- Keep roughly one third of scenarios as holdout.

{rules}

Return JSON:
{{"policies": ["..."], "scenarios": [ ...same shape as the current suite, plus "covers"... ],
  "coverage": [{{"requirement": "...", "scenarios": ["id", ...]}}],
  "gaps": ["..."], "changes": ["what you changed and why"]}}"""


def build_parts(files: list[tuple[str, bytes]] = (), texts: list[tuple[str, str]] = ()):
    """files: (filename, bytes). texts: (label, text). Text-like files are inlined as text."""
    parts, inline = [], []
    for name, data in files:
        ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        mime = MIME.get(ext)
        if mime is None:
            raise ValueError(f"Unsupported file type: {name}. Use PDF, PNG/JPG/WEBP, or text/Markdown.")
        if mime == "text/plain":
            inline.append((name, data.decode("utf-8", errors="replace")))
        else:
            parts.append(llm.file_part(data, mime))
    inline += list(texts)
    if not parts and not inline:
        raise ValueError("Add at least one source: a file, pasted text, a Miro board, or the agent's instructions.")
    text_block = "\n\n".join(f"=== SOURCE: {label} ===\n{t.strip()}" for label, t in inline if t.strip())
    return parts, text_block


def _scenarios_out(out: dict) -> list[dict]:
    keep = ("id", "split", "persona", "goal", "facts", "success_criteria", "max_turns", "covers",
            "voice", "background_noise")
    return [{k: s[k] for k in keep if s.get(k) not in (None, "", [], {})} for s in out.get("scenarios", [])]


def generate(parts, text_block: str, count: int = 8, channel: str = "voice", notes: str = "") -> dict:
    prompt = GENERATE.format(
        channel=channel, count=count, rules=SCENARIO_RULES,
        voice_extra=("; voice-specific: misheard numbers or names, a noisy line, a fast or hesitant "
                     "speaker" if channel == "voice" else ""),
        notes=f"Extra instructions from the user: {notes}\n" if notes else "",
    ) + (f"\n\nSOURCE MATERIAL:\n{text_block}" if text_block else "")
    out = llm.generate_json(prompt, temperature=0.3, parts=parts)
    suite = {"policies": out.get("policies", []), "scenarios": _scenarios_out(out)}
    return {"suite": suite, "coverage": out.get("coverage", []), "gaps": out.get("gaps", []), "changes": []}


def refine(current: dict, parts, text_block: str, channel: str = "voice", notes: str = "") -> dict:
    prompt = REFINE.format(channel=channel, rules=SCENARIO_RULES, notes=notes or "none",
                           suite=yaml.safe_dump({k: current.get(k) for k in ("policies", "scenarios")},
                                                sort_keys=False, allow_unicode=True)) \
        + (f"\n\nSOURCE MATERIAL:\n{text_block}" if text_block else "")
    out = llm.generate_json(prompt, temperature=0.2, parts=parts)
    # Keep every suite-level setting (voice, scoring, ...); replace only policies and scenarios.
    suite = {**current, "policies": out.get("policies", current.get("policies", [])),
             "scenarios": _scenarios_out(out)}
    return {"suite": suite, "coverage": out.get("coverage", []), "gaps": out.get("gaps", []),
            "changes": out.get("changes", [])}


HEADER = """# Test suite generated by Voice QA Gym's scenario generator. Review before saving:
# check that facts match the agent's test data, and that success criteria are fair.
# split: train -> the coach learns from it; holdout -> hidden exam calls.
"""


def to_yaml(suite: dict) -> str:
    order = ["voice", "scoring", "policies", "scenarios"]
    ordered = {k: suite[k] for k in order if k in suite}
    ordered.update({k: v for k, v in suite.items() if k not in ordered})
    return HEADER + yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, width=100)
