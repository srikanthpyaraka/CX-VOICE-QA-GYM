"""Build the live-demo presenter page from recorded Voice QA Gym runs (runs/<id>.json).

    python docs/live-demo/build.py EVAL_RUN_ID [AUTO_IMPROVE_RUN_ID]

Writes docs/live-demo/voice-qa-gym-live-run.html: a standalone page that replays the
real calls, tool calls and scorer verdicts. No network access needed to view it."""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import yaml
from app import gym
spec = __import__("importlib").util.spec_from_file_location("cda", ROOT / "scripts/create_demo_agent.py")
cda = __import__("importlib").util.module_from_spec(spec); spec.loader.exec_module(cda)

suite = {s["id"]: s for s in yaml.safe_load((ROOT / "scenarios/suite.yaml").read_text())["scenarios"]}
KEEP_TURN = ("role", "text", "tool_calls", "tool_responses", "voice_issues", "latency_s", "heard", "wer")


def slim(result):
    out = {k: result.get(k) for k in ("train_reward", "holdout_reward", "pass_rate", "metrics", "latency", "voice_issues", "errors")}
    eps = []
    for e in result["episodes"]:
        d = {"scenario_id": e["scenario_id"], "split": e["split"], "reward": e["reward"],
             "goal": suite.get(e["scenario_id"], {}).get("goal", ""),
             "transcript": [{k: t[k] for k in KEEP_TURN if k in t} for t in e["transcript"]]}
        if "error" in e:
            d["error"] = e["error"].split("(")[0].strip()
        else:
            d["scores"] = e["scores"]
        eps.append(d)
    out["episodes"] = eps
    return out


ev_run = json.loads((ROOT / f"runs/{sys.argv[1]}.json").read_text())
data = {
    "meta": {"project": ev_run["conn"]["project"], "location": ev_run["conn"]["location"],
             "app": ev_run["conn"]["app_id"], "runId": ev_run["id"],
             "recorded": time.strftime("%d %b %Y, %H:%M", time.localtime(ev_run["created"])),
             "mode": ev_run["params"].get("modality", "text").replace("text", "Text turns").replace("audio", "Full audio"),
             "model": "App default (CX Agent Studio, us)",
             "agentNote": "order_support for a demo store, with three Python tools holding test orders. Its first instructions tell it to share everything and list every option, so the scorecard has real problems to find."},
    "instruction": cda.INSTRUCTION_V0,
    "tools": [{"name": "get_order", "desc": "Looks up an order and checks the caller’s email or phone."},
              {"name": "start_return", "desc": "Starts a return and refund, returns a return ID."},
              {"name": "update_address", "desc": "Changes the delivery address if the order hasn’t shipped."}],
    "eval": slim(gym.summarize(ev_run["result"]["episodes"])),  # recompute with current scoring rules
    "opt": None,
}
if len(sys.argv) > 2:
    op = json.loads((ROOT / f"runs/{sys.argv[2]}.json").read_text())
    data["opt"] = {"iterations": [{**{k: it.get(k) for k in ("iteration", "accepted", "invalid", "changes", "diagnosis")},
                                   "result": {k: (it.get("result") or {}).get(k) for k in ("train_reward", "holdout_reward", "pass_rate", "errors")} if it.get("result") else None}
                                  for it in op["iterations"]]}
body = (HERE / "template.html").read_text().replace("/*DATA*/null", json.dumps(data).replace("</", "<\\/"))
html = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>\n'
        + body + "\n</body></html>\n")
(HERE / "voice-qa-gym-live-run.html").write_text(html)
print("built", len(html), "bytes;", len(data["eval"]["episodes"]), "calls;", "with coaching" if data["opt"] else "no coaching")
