"""Auto-improve loop: the agent instruction is the policy.

Each iteration:
  1. Collect the worst episodes from the current best version.
  2. Gemini proposes a revised instruction (GECX XML preserved).
  3. Push the candidate to the live agent, re-run the suite.
  4. Keep it only if train reward improves by > MIN_GAIN and holdout does not
     drop by more than MIN_GAIN (guards against overfitting to train scenarios).
The original instruction is always restored at the end; applying a winner is
a separate, explicit user action.
"""
import json
import re

from app import gym, llm
from app.cxas_client import CXAgent

MIN_GAIN = 0.02
_TOOL_REF = re.compile(r"\{@TOOL:\s*([\w\-]+)\s*\}")
_TOP_TAGS = ("role", "persona", "guidelines", "constraints", "taskflow")

PROMPT = """You improve the instruction of a Google CX Agent Studio VOICE agent.

CURRENT INSTRUCTION:
<<<
{instruction}
>>>

FAILED OR LOW-SCORING TEST CONVERSATIONS (judge findings + voice issues):
{failures}

Previously tried edits that did NOT improve the score (avoid repeating them):
{rejected}

Rewrite the instruction to fix the root causes. Rules:
- Keep the existing GECX XML structure and tags (<role>, <persona>, <guidelines>,
  <constraints>, <taskflow>/<subtask>/<step>/<trigger>/<action>) where present.
- Keep every {{@TOOL: name}} and {{variable}} reference that exists; do not invent new tools.
- Phrase rules positively ("Confirm X before Y") rather than "Do NOT ...".
- "agent_heard" on a user turn means speech recognition misheard the caller. You cannot fix ASR,
  but you can make the agent confirm or read back critical details and recover gracefully.
- Voice channel: replies are spoken via TTS. Keep turns short (under ~40 words), ask one
  question at a time, no markdown/lists/URLs, read numbers and dates naturally.
- Fix general behaviours, not individual test cases. Do not hardcode test-specific names or values.
- Make the smallest set of changes that addresses the failures.

Return JSON: {{"diagnosis": "<root causes>", "changes": ["<change 1>", ...], "instruction": "<full new instruction>"}}"""


def _failure_digest(result: dict, k: int = 6) -> str:
    worst = sorted(result["episodes"], key=lambda e: e["reward"])[:k]
    out = []
    for e in worst:
        if "error" in e:
            out.append({"scenario": e["scenario_id"], "error": e["error"]})
            continue
        convo = [{"role": t["role"], "text": t["text"],
                  **({"agent_heard": t["heard"]} if t.get("wer") else {}),
                  **({"voice_issues": t["voice_issues"]} if t.get("voice_issues") else {})}
                 for t in e["transcript"]]
        out.append({"scenario": e["scenario_id"], "reward": e["reward"],
                    "judge_summary": e["scores"]["summary"],
                    "failures": e["scores"]["failures"], "transcript": convo})
    return json.dumps(out, indent=1)


def validate_candidate(original: str, candidate: str) -> str | None:
    """Return a rejection reason, or None if the candidate is structurally safe."""
    if not candidate.strip():
        return "empty instruction"
    missing = set(_TOOL_REF.findall(original)) - set(_TOOL_REF.findall(candidate))
    if missing:
        return f"dropped tool references: {sorted(missing)}"
    lost = [t for t in _TOP_TAGS if f"<{t}" in original and f"<{t}" not in candidate]
    if lost:
        return f"dropped GECX tags: {lost}"
    return None


def clean_instruction(text: str) -> str:
    """Strip wrapping the model sometimes copies from the prompt: the <<< >>> markers
    around the current instruction, or a markdown code fence."""
    t = (text or "").strip()
    t = re.sub(r"^```[\w-]*\s*|\s*```$", "", t).strip()
    t = re.sub(r"^<<<\s*|\s*>>>$", "", t).strip()
    return t


def propose(instruction: str, result: dict, rejected: list[str]) -> dict:
    return llm.generate_json(PROMPT.format(
        instruction=instruction,
        failures=_failure_digest(result),
        rejected=json.dumps(rejected) if rejected else "none",
    ), temperature=0.4)


def accept(cand: dict, best: dict, min_gain: float = MIN_GAIN) -> bool:
    if cand["train_reward"] is None or cand["train_reward"] <= best["train_reward"] + min_gain:
        return False
    if best["holdout_reward"] is not None and cand["holdout_reward"] is not None:
        return cand["holdout_reward"] >= best["holdout_reward"] - min_gain
    return True


def optimize(cx: CXAgent, agent_name: str, suite: dict, iterations: int,
             use_tool_fakes: bool, trials: int, log, on_episode=None,
             audio: dict | None = None, audio_dir=None) -> None:
    """`log(event: dict)` receives progress events; the caller persists them.
    An {"type": "eval_start"} event precedes each suite run."""
    original = cx.get_instruction(agent_name)
    log({"type": "original", "instruction": original})
    try:
        log({"type": "status", "msg": "Evaluating baseline"})
        log({"type": "eval_start"})
        best = gym.evaluate(cx, suite, use_tool_fakes, trials, on_episode=on_episode,
                            audio=audio, audio_dir=audio_dir)
        best["instruction"] = original
        log({"type": "iteration", "iteration": 0, "accepted": True, "result": best,
             "instruction": original, "diagnosis": "baseline", "changes": []})

        rejected: list[str] = []
        for i in range(1, iterations + 1):
            log({"type": "status", "msg": f"Iteration {i}: proposing revision"})
            p = propose(best["instruction"], best, rejected)
            cand_instr = clean_instruction(p.get("instruction", ""))
            reason = validate_candidate(original, cand_instr)
            if reason:
                rejected.append(f"{'; '.join(p.get('changes', []))} (invalid: {reason})")
                log({"type": "iteration", "iteration": i, "accepted": False,
                     "invalid": reason, "instruction": cand_instr,
                     "diagnosis": p.get("diagnosis", ""), "changes": p.get("changes", [])})
                continue

            log({"type": "status", "msg": f"Iteration {i}: evaluating candidate"})
            cx.set_instruction(agent_name, cand_instr)
            log({"type": "eval_start"})
            res = gym.evaluate(cx, suite, use_tool_fakes, trials, on_episode=on_episode,
                               audio=audio, audio_dir=audio_dir)
            res["instruction"] = cand_instr
            # A round with errored calls isn't comparable: the mean covers different calls.
            inconclusive = (f"inconclusive: {res['errors']} call(s) errored, not compared"
                            if res["errors"] else None)
            ok = not inconclusive and accept(res, best, gym.resolve_scoring(suite)["min_gain"])
            if ok:
                best = res
            elif not inconclusive:
                rejected.append("; ".join(p.get("changes", [])))
            log({"type": "iteration", "iteration": i, "accepted": ok, "result": res,
                 "instruction": cand_instr, "diagnosis": p.get("diagnosis", ""),
                 "changes": p.get("changes", []),
                 **({"invalid": inconclusive} if inconclusive else {})})
    finally:
        cx.set_instruction(agent_name, original)
        log({"type": "status", "msg": "Original instruction restored on the agent"})
