"""LLM-as-judge rubric for one finished conversation.

The judge never sees the agent's instruction, only the scenario and the
transcript, so the optimizer can't game it by echoing rubric language."""
import json

from app import llm

PROMPT = """You are a strict QA evaluator for a customer-service VOICE assistant.
Score the conversation below against the scenario. Base every judgement on the transcript only.

Scenario goal (caller's): {goal}
Success criteria (all must hold for task_success): {criteria}
Policies the assistant must follow: {policies}

Transcript (JSON; agent turns include tool_calls and tool_responses):
{transcript}

If user turns have a "heard" field, this was a phone call: "text" is what the caller said and
"heard" is what the agent's speech recognition transcribed, which is all the agent saw. Do not
blame the agent for a misrecognition itself. Do judge whether it handled it well: confirming
or reading back key details (numbers, names, addresses) and recovering when it misheard.

Rubric:
- task_success: true only if every success criterion is met.
- policy_adherence (0-1): followed every policy (verification, disclosures, escalation rules, no prohibited actions).
- groundedness (0-1): factual claims are supported by tool responses or common knowledge; 0 if it invented account data, prices, dates or policies.
- tone (0-1): concise, polite, natural for spoken conversation, recovers well from confusion.
- failures: each concrete problem, with the 0-based index of the transcript turn.

Return JSON:
{{"task_success": <bool>, "policy_adherence": <0-1>, "groundedness": <0-1>, "tone": <0-1>,
  "failures": [{{"turn": <int>, "category": "<task|policy|grounding|tone|voice|routing>", "detail": "<specific>"}}],
  "summary": "<one sentence>"}}"""


def judge(scenario: dict, transcript: list[dict], policies: list[str]) -> dict:
    convo = [{k: t[k] for k in ("role", "text", "heard", "tool_calls", "tool_responses") if t.get(k)}
             for t in transcript]
    out = llm.generate_json(PROMPT.format(
        goal=scenario["goal"],
        criteria=json.dumps(scenario.get("success_criteria", [])),
        policies=json.dumps(policies + scenario.get("policies", [])),
        transcript=json.dumps(convo, indent=1, default=str),
    ), temperature=0.0)
    return {
        "task_success": bool(out.get("task_success")),
        "policy_adherence": float(out.get("policy_adherence", 0)),
        "groundedness": float(out.get("groundedness", 0)),
        "tone": float(out.get("tone", 0)),
        "failures": out.get("failures", []),
        "summary": out.get("summary", ""),
    }
