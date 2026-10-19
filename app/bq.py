"""Optional BigQuery export of test results. Off unless BQ_DATASET is set.

    BQ_DATASET=my-project.voice_qa_gym   # enables export; dataset and tables are created if missing
    BQ_LOCATION=US                       # dataset location when it is created (default US)

Two tables, one row per call and one row per evaluated round:
  calls  - scenario, split, scores, verdict, failures, transcript (JSON), errors
  rounds - an Evaluate run (iteration NULL) or one Auto-improve round, with its
           summary scores, outcome and the instruction that was tested
The run JSON files under runs/ stay the full record; BigQuery holds the queryable copy.
"""
import datetime
import hashlib
import json
import os

DATASET = os.environ.get("BQ_DATASET", "").strip()
LOCATION = os.environ.get("BQ_LOCATION", "US")

_S, _F, _I, _B, _T = "STRING", "FLOAT", "INTEGER", "BOOLEAN", "TIMESTAMP"
CALLS = [("run_id", _S), ("run_kind", _S), ("iteration", _I), ("recorded_at", _T), ("project", _S),
         ("location", _S), ("app_id", _S), ("agent", _S), ("modality", _S), ("instruction_sha", _S),
         ("scenario_id", _S), ("split", _S), ("reward", _F), ("task_success", _B),
         ("policy_adherence", _F), ("groundedness", _F), ("tone", _F), ("voice", _F),
         ("caller_turns", _I), ("end_reason", _S), ("latency_p50_s", _F), ("mean_wer", _F),
         ("voice_issues", _S), ("judge_summary", _S), ("failures", _S), ("transcript", _S), ("error", _S)]
ROUNDS = [("run_id", _S), ("run_kind", _S), ("iteration", _I), ("recorded_at", _T), ("project", _S),
          ("location", _S), ("app_id", _S), ("agent", _S), ("modality", _S), ("instruction_sha", _S),
          ("train_reward", _F), ("holdout_reward", _F), ("pass_rate", _F), ("policy_adherence", _F),
          ("groundedness", _F), ("voice", _F), ("latency_p50_s", _F), ("mean_wer", _F), ("errors", _I),
          ("outcome", _S), ("diagnosis", _S), ("changes", _S), ("instruction", _S)]

_client = None


def enabled() -> bool:
    return bool(DATASET)


def build_rows(run: dict, result: dict, iteration=None, outcome=None, instruction=None,
               diagnosis=None, changes=None) -> tuple[dict, list[dict]]:
    """Pure: turn one evaluated result into (round_row, call_rows)."""
    conn = run.get("conn") or {}
    common = {"run_id": run["id"], "run_kind": run["kind"], "iteration": iteration,
              "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "project": conn.get("project"), "location": conn.get("location"), "app_id": conn.get("app_id"),
              "agent": run.get("agent_name"), "modality": (run.get("params") or {}).get("modality", "text"),
              "instruction_sha": hashlib.sha256(instruction.encode()).hexdigest()[:12] if instruction else None}
    m = result.get("metrics") or {}
    asr = result.get("asr") or {}
    round_row = {**common, "train_reward": result.get("train_reward"), "holdout_reward": result.get("holdout_reward"),
                 "pass_rate": result.get("pass_rate"), "policy_adherence": m.get("policy_adherence"),
                 "groundedness": m.get("groundedness"), "voice": m.get("voice"),
                 "latency_p50_s": (result.get("latency") or {}).get("p50"), "mean_wer": asr.get("mean_wer"),
                 "errors": result.get("errors"), "outcome": outcome, "diagnosis": diagnosis,
                 "changes": json.dumps(changes) if changes else None, "instruction": instruction}
    calls = []
    for e in result.get("episodes", []):
        s = e.get("scores") or {}
        turns = e.get("transcript") or []
        lat = sorted(t["latency_s"] for t in turns if t["role"] == "agent" and t.get("latency_s") is not None)
        wers = [t["wer"] for t in turns if t["role"] == "user" and t.get("wer") is not None]
        issues = sorted({i for t in turns for i in t.get("voice_issues", [])})
        calls.append({**common, "scenario_id": e["scenario_id"], "split": e.get("split"), "reward": e.get("reward"),
                      "task_success": s.get("task_success"), "policy_adherence": s.get("policy_adherence"),
                      "groundedness": s.get("groundedness"), "tone": s.get("tone"), "voice": s.get("voice"),
                      "caller_turns": sum(1 for t in turns if t["role"] == "user"), "end_reason": e.get("end_reason"),
                      "latency_p50_s": lat[len(lat) // 2] if lat else None,
                      "mean_wer": round(sum(wers) / len(wers), 4) if wers else None,
                      "voice_issues": ",".join(issues) or None, "judge_summary": s.get("summary"),
                      "failures": json.dumps(s.get("failures")) if s.get("failures") else None,
                      "transcript": json.dumps(turns, default=str), "error": e.get("error")})
    return round_row, calls


def _bq():
    global _client
    if _client is None:
        from google.cloud import bigquery
        project, dataset = DATASET.split(".", 1)
        _client = bigquery.Client(project=project)
        ds = bigquery.Dataset(f"{project}.{dataset}")
        ds.location = LOCATION
        _client.create_dataset(ds, exists_ok=True)
        for name, cols in (("calls", CALLS), ("rounds", ROUNDS)):
            table = bigquery.Table(f"{DATASET}.{name}", schema=[bigquery.SchemaField(c, t) for c, t in cols])
            table.time_partitioning = bigquery.TimePartitioning(field="recorded_at")
            _client.create_table(table, exists_ok=True)
    return _client


def export(run: dict, result: dict, **kw) -> int:
    """Load one round and its calls. Returns the number of call rows written."""
    from google.cloud import bigquery
    round_row, call_rows = build_rows(run, result, **kw)
    client = _bq()
    cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    client.load_table_from_json([round_row], f"{DATASET}.rounds", job_config=cfg).result()
    if call_rows:
        client.load_table_from_json(call_rows, f"{DATASET}.calls", job_config=cfg).result()
    return len(call_rows)
