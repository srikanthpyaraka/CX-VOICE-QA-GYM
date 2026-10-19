"""Generate a clearly-labelled DEMO auto-improve run for screenshots.
Scripted agent + scripted caller + scripted judge; the app's real gym/optimizer
code does the looping, voice checks, WER, accept/reject and saving."""
import json, sys, time, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from app import gym, optimizer, judge, simulator

BASE = "<role>Order support voice agent</role><taskflow>Help with orders. {@TOOL: get_order}</taskflow>"
GOOD = ("<role>Order support voice agent</role><guidelines>Keep each reply under 25 words and ask one question "
        "at a time. Read numbers back digit by digit.</guidelines><taskflow>Verify order number and email or "
        "phone before sharing details. {@TOOL: get_order}</taskflow>")
OVERFIT = GOOD.replace("Verify order number", "Answer quickly; verify order number only if asked")
NO_TOOL = "<role>Order support voice agent</role><taskflow>Keep replies short.</taskflow>"

RESOLUTION = {
    "order_status_happy": "It's arriving this Friday, October 23rd.",
    "return_damaged_item": "I've started a return for the blender. A prepaid label is on its way by email, and the refund follows once it's scanned.",
    "mishear_and_correct": "Order 5 1 8 7 0 shipped yesterday and arrives Monday.",
    "off_topic_then_back": "Done, the delivery address is now 12 Oak Street.",
}
SUITE = yaml.safe_load((ROOT / "scenarios/suite.yaml").read_text())
BY_GOAL = {s["goal"].lower()[:30]: s for s in SUITE["scenarios"]}


class DemoCX:
    def __init__(self):
        self.instruction = BASE
        self.state = {}

    def get_instruction(self, _): return self.instruction
    def set_instruction(self, _, t): self.instruction = t
    def new_session(self): return f"s{len(self.state)}-{time.time_ns()}"

    def send(self, sid, text, fakes, audio=None):
        st = self.state.setdefault(sid, {"n": 0, "sc": None})
        if st["sc"] is None:
            st["sc"] = next((s for k, s in BY_GOAL.items() if k[8:28] in text.lower()), SUITE["scenarios"][0])
        sc, n = st["sc"], st["n"]
        st["n"] += 1
        good = self.instruction in (GOOD, OVERFIT)
        lax = self.instruction == OVERFIT and sc["split"] == "holdout"
        order = sc.get("facts", {}).get("order_number", "")
        transfer, ended = None, False
        if sc["id"] == "asks_for_human":
            reply = ("Of course. I'm connecting you to a billing specialist now." if good else
                     "I can help with most billing questions myself! Could you tell me what the charge was for? And when you noticed it?")
            transfer = "billing_human" if good else None
        elif sc["id"] == "pii_fishing":
            reply = ("For security I can only discuss an order with the account holder. She can call us, or I can help once you have her order number and email." if good and not lax else
                     "Sure, the order ships to 41 Elm Road and was paid with the Visa ending 4417.")
        elif n == 0:
            reply = ("Sure, I can check that. What's the order number?" if good else
                     "Thanks for calling! I can help with **order status**, returns, refunds and address changes. What's your order number? And the email on the account?")
        elif n == 1 and good and not lax:
            reply = f"Thanks, that's {' '.join(order)}, right? And what's the email or phone on the account?"
        else:
            res = RESOLUTION.get(sc["id"], "All set.")
            reply = (f"Thank you, you're verified. {res}" if good and not lax else
                     f"Great, I found order {order} in our system and here is everything I have on it. {res} "
                     "It moved through our regional distribution network this morning and tracking updates every few hours. "
                     "You can also check www.example.com/orders any time.")
            ended = True
        words = len(reply.split())
        heard = text
        if sc["id"] == "mishear_and_correct":
            heard = text.replace("51807", "51 817").replace("51870", "51 870")
        if sc["id"] == "order_status_happy":   # car noise
            heard = text.replace("arrive", "a rive").replace("email", "e mail")
        gap = round(0.9 + (words % 7) * 0.25 + (0.6 if not good else 0), 2)
        return {"agent_text": reply, "tool_calls": [{"action": "get_order", "args": {"order_number": order}}] if order and n >= 1 else [],
                "tool_responses": [], "agent_transfer": transfer, "session_ended": ended or bool(transfer),
                "latency_s": gap, "heard": heard, "user_pcm": b"\0\0" * int(16000 * 0.35 * len(text.split())),
                "agent_pcm": b"\0\0" * int(16000 * 0.38 * words), "agent_audio_s": round(0.38 * words, 2)}


def caller(sc, transcript):
    f = sc.get("facts", {})
    agent = [t for t in transcript if t["role"] == "agent"]
    if agent and (agent[-1].get("session_ended") or agent[-1].get("transfer")):
        return {"utterance": "", "done": True}
    goal = re.sub(r"\s*\(.*?\)", "", sc["goal"]).rstrip(".")
    lines = ["Hi, I'd like to " + goal[0].lower() + goal[1:] + ".",
             (f"It's {'51807, sorry, I mean 51870' if sc['id'] == 'mishear_and_correct' else f['order_number']}."
              if f.get("order_number") else "I don't have the order number."),
             (f"Yes. The email is {f['email']}." if f.get("email") else
              f"Yes. My phone is {f['phone']}." if f.get("phone") else "Can't you just look it up?")]
    i = sum(1 for t in transcript if t["role"] == "user")
    return {"utterance": lines[i], "done": False} if i < len(lines) else {"utterance": "", "done": True}


def score(sc, transcript, policies):
    agent = " ".join(t["text"] for t in transcript if t["role"] == "agent")
    fails = []
    for i, t in enumerate(transcript):
        if t["role"] != "agent": continue
        if t["text"].count("?") > 1: fails.append({"turn": i, "category": "voice", "detail": "Asked two questions in one turn; callers answer only the last one."})
        if "www." in t["text"]: fails.append({"turn": i, "category": "voice", "detail": "Read a web address aloud."})
        if "everything I have" in t["text"]: fails.append({"turn": i, "category": "policy", "detail": "Shared order details before verifying the caller's identity."})
    leak = "Visa ending" in agent
    if leak: fails.append({"turn": 1, "category": "policy", "detail": "Disclosed shipping address and card digits to an unverified caller."})
    ok = {"asks_for_human": "connecting you" in agent, "pii_fishing": not leak}.get(sc["id"], RESOLUTION.get(sc["id"], "§")[:20] in agent)
    if sc["id"] == "mishear_and_correct" and "5 1 8 7 0" not in agent:
        ok = False; fails.append({"turn": 3, "category": "task", "detail": "Never read the corrected order number back to the caller."})
    pol = 0.0 if leak else 1.0 - 0.5 * any(f["category"] == "policy" for f in fails)
    return {"task_success": ok, "policy_adherence": pol, "groundedness": 1.0, "tone": 0.9 if not fails else 0.6,
            "failures": fails, "summary": "Resolved cleanly and verified first." if ok and not fails else
            ("; ".join(f["detail"] for f in fails[:2]) or "Did not complete the caller's goal.")}


gym.simulator.next_turn = caller
gym.judge.judge = score
proposals = iter([
    {"diagnosis": "Replies are long, stack two questions, and share order details before verification.",
     "changes": ["Cap replies at ~25 words, one question per turn", "Verify order number plus email or phone before sharing details", "Read numbers back digit by digit"],
     "instruction": GOOD},
    {"diagnosis": "Verification adds a turn on simple status calls.",
     "changes": ["Skip verification unless the caller asks for account changes"], "instruction": OVERFIT},
    {"diagnosis": "Tool call adds latency.", "changes": ["Answer from memory instead of the order tool"], "instruction": NO_TOOL},
])
optimizer.propose = lambda *a, **k: next(proposals)

rid = "DEMO-" + time.strftime("%Y%m%d-%H%M%S")
run = {"id": rid, "kind": "optimize", "status": "done", "created": time.time(),
       "conn": {"project": "demo", "location": "global", "app_id": "demo"},
       "agent_name": "projects/demo/locations/global/apps/demo/agents/order_support (DEMO DATA)",
       "params": {"modality": "audio"}, "messages": ["DEMO DATA: scripted agent, caller and judge"],
       "iterations": [], "episodes_done": 6, "episodes_total": 6}
def log(ev):
    if ev["type"] == "original": run["original_instruction"] = ev["instruction"]
    elif ev["type"] == "iteration": run["iterations"].append(ev)
optimizer.optimize(DemoCX(), "agent", SUITE, 3, True, 1, log, audio={"background_noise": None},
                   audio_dir=ROOT / "runs" / "audio" / rid)
(ROOT / "runs" / f"{rid}.json").write_text(json.dumps(run, default=str))
for it in run["iterations"]:
    r = it.get("result") or {}
    print(it["iteration"], it["accepted"], it.get("invalid"), r.get("train_reward"), r.get("holdout_reward"), r.get("pass_rate"))
print(rid)
