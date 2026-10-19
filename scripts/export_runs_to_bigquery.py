"""Backfill saved runs (runs/<id>.json) into BigQuery.

    BQ_DATASET=my-project.voice_qa_gym python scripts/export_runs_to_bigquery.py RUN_ID [RUN_ID ...]
    BQ_DATASET=my-project.voice_qa_gym python scripts/export_runs_to_bigquery.py --all

Rows match what the server writes when BQ_DATASET is set. Re-running a run appends it again,
so export each run once (or dedupe on run_id + iteration in queries).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import bq, gym  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="every finished run except DEMO- runs")
    args = ap.parse_args()
    if not bq.enabled():
        sys.exit("Set BQ_DATASET=project.dataset first.")
    paths = ([p for p in sorted((ROOT / "runs").glob("*.json")) if not p.stem.startswith(("DEMO-", "FAKE-"))]
             if args.all else [ROOT / "runs" / f"{r}.json" for r in args.run_ids])
    for p in paths:
        run = json.loads(p.read_text())
        if run.get("status") == "running":
            print(f"skip {run['id']}: still running")
            continue
        if run["kind"] == "eval" and run.get("result"):
            n = bq.export(run, gym.summarize(run["result"]["episodes"]), outcome="evaluation")
            print(f"{run['id']}: evaluation, {n} calls")
        for it in run.get("iterations", []):
            if not it.get("result"):
                continue
            outcome = ("baseline" if it["iteration"] == 0 else "inconclusive" if it.get("invalid")
                       else "kept" if it["accepted"] else "rejected")
            n = bq.export(run, it["result"], iteration=it["iteration"], outcome=outcome,
                          instruction=it.get("instruction"), diagnosis=it.get("diagnosis"),
                          changes=it.get("changes"))
            print(f"{run['id']}: round {it['iteration']} ({outcome}), {n} calls")


if __name__ == "__main__":
    main()
