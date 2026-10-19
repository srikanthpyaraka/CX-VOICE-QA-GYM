"""FastAPI server: connect to a CX Agent Studio app, edit the test suite,
run evaluations and the auto-improve loop, apply a winning instruction."""
import json
import os
import threading
import time
import uuid
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import bq, gym, llm, optimizer
from app.cxas_client import CXAgent

ROOT = Path(__file__).resolve().parent.parent
# Overridable so Cloud Run can point both at a Cloud Storage volume mount.
SUITE = Path(os.environ.get("SUITE_PATH", ROOT / "scenarios" / "suite.yaml"))
RUNS_DIR = Path(os.environ.get("RUNS_DIR", ROOT / "runs"))
AUDIO_DIR = RUNS_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Voice QA Gym")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

if not SUITE.exists():  # first boot on an empty bucket: seed with the sample suite
    SUITE.parent.mkdir(parents=True, exist_ok=True)
    SUITE.write_text((ROOT / "scenarios" / "suite.yaml").read_text())

_state: dict = {"cx": None, "conn": None}
_runs: dict[str, dict] = {}
_lock = threading.Lock()
_busy = threading.Lock()   # one job at a time: jobs mutate the live agent


def _save(run: dict) -> None:
    (RUNS_DIR / f"{run['id']}.json").write_text(json.dumps(run, default=str))


def _cx() -> CXAgent:
    if _state["cx"] is None:
        raise HTTPException(400, "Not connected. Connect to a CX Agent Studio app first.")
    return _state["cx"]


def load_suite(text: str | None = None) -> dict:
    suite = yaml.safe_load(text if text is not None else SUITE.read_text()) or {}
    scenarios = suite.get("scenarios") or []
    if not scenarios:
        raise ValueError("suite needs at least one scenario")
    ids = set()
    for s in scenarios:
        for key in ("id", "goal"):
            if not s.get(key):
                raise ValueError(f"scenario missing '{key}': {s}")
        if s["id"] in ids:
            raise ValueError(f"duplicate scenario id: {s['id']}")
        ids.add(s["id"])
        if s.get("split", "train") not in ("train", "holdout"):
            raise ValueError(f"{s['id']}: split must be train or holdout")
    if not any(s.get("split", "train") == "train" for s in scenarios):
        raise ValueError("suite needs at least one train scenario")
    return suite


# ---------- connection ----------

class Connect(BaseModel):
    project: str
    location: str = "global"
    app_id: str


@app.post("/api/connect")
def connect(c: Connect):
    try:
        cx = CXAgent(c.project, c.location, c.app_id)
        agents = cx.list_agents()
    except Exception as e:
        raise HTTPException(400, f"Could not reach app: {type(e).__name__}: {e}")
    llm.init(c.project)
    _state.update(cx=cx, conn=c.model_dump())
    return {"app_name": cx.app_name, "agents": agents}


@app.get("/api/agent")
def get_agent(name: str):
    return {"instruction": _cx().get_instruction(name)}


# ---------- suite ----------

@app.get("/api/suite")
def get_suite():
    return {"yaml": SUITE.read_text()}


class SuiteText(BaseModel):
    yaml: str


@app.put("/api/suite")
def put_suite(body: SuiteText):
    try:
        suite = load_suite(body.yaml)
    except Exception as e:
        raise HTTPException(400, str(e))
    SUITE.write_text(body.yaml)
    return {"scenarios": len(suite["scenarios"])}


# ---------- runs ----------

@app.get("/api/noise")
def list_noise():
    return sorted(p.name for p in gym.NOISE_DIR.glob("*.wav"))


class RunReq(BaseModel):
    agent_name: str
    use_tool_fakes: bool = True
    trials: int = 1
    iterations: int = 3
    confirm_sandbox: bool = False
    modality: str = "text"                 # "text" | "audio"
    background_noise: str | None = None    # file name in scenarios/noise/ (audio only)

    def audio(self) -> dict | None:
        return {"background_noise": self.background_noise} if self.modality == "audio" else None


def _start(kind: str, req: RunReq, target) -> dict:
    cx = _cx()
    try:
        suite = load_suite()
    except Exception as e:
        raise HTTPException(400, f"Suite invalid: {e}")
    if req.modality not in ("text", "audio"):
        raise HTTPException(400, "modality must be text or audio")
    noise = set(list_noise())
    bad = [n for n in [req.background_noise] + [s.get("background_noise") for s in suite["scenarios"]]
           if n and n not in noise]
    if req.modality == "audio" and bad:
        raise HTTPException(400, f"Unknown noise file(s) {bad}; put WAVs in scenarios/noise/")
    if not _busy.acquire(blocking=False):
        raise HTTPException(409, "Another run is in progress")

    run = {"id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4], "kind": kind,
           "status": "running", "created": time.time(), "conn": _state["conn"],
           "agent_name": req.agent_name, "params": req.model_dump(), "messages": [],
           "iterations": [], "episodes_done": 0,
           "episodes_total": len(suite["scenarios"]) * req.trials}
    _runs[run["id"]] = run
    _save(run)

    def work():
        try:
            target(cx, suite, run)
            run["status"] = "done"
        except Exception as e:
            run["status"] = "error"
            run["messages"].append(f"ERROR {type(e).__name__}: {e}")
        finally:
            _save(run)
            _busy.release()

    threading.Thread(target=work, daemon=True).start()
    return {"id": run["id"]}


def _tick(run):
    def on_episode(_ep):
        with _lock:
            run["episodes_done"] += 1
    return on_episode


def _export(run: dict, result: dict, **kw) -> None:
    """Opt-in BigQuery copy (BQ_DATASET). Never fails the run."""
    if not bq.enabled():
        return
    try:
        n = bq.export(run, result, **kw)
        msg = f"Saved {n} calls to BigQuery {bq.DATASET}"
    except Exception as e:
        msg = f"BigQuery export failed (results are still saved locally): {type(e).__name__}: {e}"
    with _lock:
        run["messages"].append(msg)
        _save(run)


@app.get("/api/config")
def config():
    return {"bigquery": bq.DATASET or None}


@app.post("/api/runs/eval")
def run_eval(req: RunReq):
    def target(cx, suite, run):
        run["messages"].append("Evaluating current agent instruction")
        run["result"] = gym.evaluate(cx, suite, req.use_tool_fakes, req.trials,
                                     on_episode=_tick(run), audio=req.audio(),
                                     audio_dir=AUDIO_DIR / run["id"])
        _export(run, run["result"], outcome="evaluation",
                instruction=cx.get_instruction(req.agent_name) if bq.enabled() else None)
    return _start("eval", req, target)


@app.post("/api/runs/optimize")
def run_optimize(req: RunReq):
    if not req.confirm_sandbox:
        raise HTTPException(400, "Auto-improve edits the live agent instruction while it runs. "
                                 "Confirm this is a dev/sandbox app.")

    def target(cx, suite, run):
        def log(ev):
            with _lock:
                if ev["type"] == "original":
                    run["original_instruction"] = ev["instruction"]
                elif ev["type"] == "status":
                    run["messages"].append(ev["msg"])
                elif ev["type"] == "eval_start":
                    run["episodes_done"] = 0
                else:
                    run["iterations"].append(ev)
                _save(run)  # persisted immediately so the original is never lost
            if ev["type"] == "iteration" and ev.get("result"):
                outcome = ("baseline" if ev["iteration"] == 0 else "inconclusive" if ev.get("invalid")
                           else "kept" if ev["accepted"] else "rejected")
                _export(run, ev["result"], iteration=ev["iteration"], outcome=outcome,
                        instruction=ev.get("instruction"), diagnosis=ev.get("diagnosis"),
                        changes=ev.get("changes"))

        optimizer.optimize(cx, req.agent_name, suite, req.iterations,
                           req.use_tool_fakes, req.trials, log, on_episode=_tick(run),
                           audio=req.audio(), audio_dir=AUDIO_DIR / run["id"])
    return _start("optimize", req, target)


def _strip(run: dict) -> dict:
    """List view without transcripts."""
    return {k: run.get(k) for k in ("id", "kind", "status", "created", "agent_name")} | {
        "modality": run.get("params", {}).get("modality", "text")}


@app.get("/api/runs")
def list_runs():
    disk = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob("*.json"), reverse=True)]
    for r in disk:
        _runs.setdefault(r["id"], r)
    return [_strip(r) for r in sorted(_runs.values(), key=lambda r: -r["created"])]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    run = _runs.get(run_id)
    if not run:
        p = RUNS_DIR / f"{run_id}.json"
        if not p.exists():
            raise HTTPException(404, "run not found")
        run = _runs[run_id] = json.loads(p.read_text())
    return run


class Apply(BaseModel):
    iteration: int | None = None   # None = restore the original instruction


@app.post("/api/runs/{run_id}/apply")
def apply(run_id: str, body: Apply):
    run = get_run(run_id)
    if run["kind"] != "optimize" or run["status"] == "running":
        raise HTTPException(400, "Only finished auto-improve runs can be applied")
    if run["conn"] != _state["conn"]:
        raise HTTPException(400, "Connect to the same app this run used first")
    if body.iteration is None:
        instr = run["original_instruction"]
    else:
        match = [it for it in run["iterations"] if it["iteration"] == body.iteration]
        if not match:
            raise HTTPException(404, "iteration not found")
        instr = optimizer.clean_instruction(match[0]["instruction"])  # older runs kept prompt markers
    _cx().set_instruction(run["agent_name"], instr)
    run["applied_iteration"] = body.iteration
    _save(run)
    return {"applied": body.iteration}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
