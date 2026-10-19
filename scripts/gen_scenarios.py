"""Draft or refine a test suite from a PRD, documents, diagram images or a Miro board.

    # new suite from a PRD and an exported whiteboard image
    python scripts/gen_scenarios.py --project my-project --from prd.md --from flow.png --count 10 --out my_suite.yaml

    # improve an existing suite with extra notes
    python scripts/gen_scenarios.py --project my-project --refine scenarios/suite.yaml --notes "add two angry callers"

    # read a Miro board through its API (needs MIRO_TOKEN)
    MIRO_TOKEN=... python scripts/gen_scenarios.py --project my-project --miro https://miro.com/app/board/uXjV.../

Gemini runs on Vertex AI in --project (or uses GOOGLE_API_KEY if set). The output is a draft:
review the facts and success criteria before using it. Coverage and gaps are printed.
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import llm, miro, scenario_gen  # noqa: E402
from app.main import load_suite  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="GCP project for Gemini (Vertex AI)")
    ap.add_argument("--from", dest="sources", action="append", default=[], help="source file (repeatable)")
    ap.add_argument("--miro", help="Miro board URL or ID (needs MIRO_TOKEN)")
    ap.add_argument("--refine", help="existing suite YAML to improve instead of writing a new one")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--channel", choices=["voice", "chat"], default="voice")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", help="where to write the YAML (default: print it)")
    args = ap.parse_args()

    llm.init(args.project)
    files = [(Path(f).name, Path(f).read_bytes()) for f in args.sources]
    texts = [(f"Miro board {args.miro}", miro.board_outline(args.miro))] if args.miro else []
    if args.refine:
        current = yaml.safe_load(Path(args.refine).read_text())
        parts, block = scenario_gen.build_parts(files, texts) if (files or texts) else ([], "")
        result = scenario_gen.refine(current, parts, block, args.channel, args.notes)
    else:
        parts, block = scenario_gen.build_parts(files, texts)
        result = scenario_gen.generate(parts, block, args.count, args.channel, args.notes)

    text = scenario_gen.to_yaml(result["suite"])
    try:
        load_suite(text)
        print(f"OK: {len(result['suite']['scenarios'])} test calls, passes suite checks.", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: draft fails suite checks: {e}", file=sys.stderr)
    for title, key, fmt in [("Changes", "changes", str),
                            ("Coverage", "coverage", lambda c: f"{c['requirement']}: {', '.join(c.get('scenarios') or []) or 'NOT COVERED'}"),
                            ("Gaps", "gaps", str)]:
        if result[key]:
            print(f"\n{title}:", file=sys.stderr)
            for item in result[key]:
                print(f"  - {fmt(item)}", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\nWrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
