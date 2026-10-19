"""LLM user simulator: plays a phone caller pursuing a goal."""
import json

from app import llm

PROMPT = """You are role-playing a CALLER on a phone call with a company's voice assistant.
Stay in character. Never say you are an AI or a simulator.

Persona: {persona}
Your goal for this call: {goal}
Facts you know (share only when asked): {facts}

Speak like a real person on the phone: short spoken sentences, one thing at a time,
no lists or formatting. You may be slightly vague or change your mind if your persona
would. Do not volunteer everything up front.

Conversation so far (JSON):
{transcript}

Decide your next move. Set "done": true (and leave "utterance" empty) when your goal is
achieved, clearly cannot be achieved, you were transferred to a human, or the assistant
ended the call. Otherwise say your next line.

Return JSON: {{"utterance": "<what you say>", "done": <true|false>, "reason": "<short>"}}"""


def next_turn(scenario: dict, transcript: list[dict]) -> dict:
    convo = [{"role": t["role"], "text": t["text"]} for t in transcript]
    out = llm.generate_json(PROMPT.format(
        persona=scenario.get("persona", "an ordinary customer"),
        goal=scenario["goal"],
        facts=json.dumps(scenario.get("facts", {})),
        transcript=json.dumps(convo, indent=1),
    ), temperature=0.7)
    return {"utterance": out.get("utterance", ""), "done": bool(out.get("done")),
            "reason": out.get("reason", "")}
