"""Scenario generator, Miro reader, configurable scoring, restart recovery, run queue."""
import json
import time

import pytest
import yaml

from app import gym, llm, miro, scenario_gen, voice

DRAFT = {"policies": ["Verify identity before sharing details."],
         "scenarios": [{"id": "status_check", "split": "train", "persona": "calm", "goal": "Find order 1",
                        "facts": {"order": "1"}, "success_criteria": ["gives status"], "max_turns": 8,
                        "covers": ["REQ-1"], "junk": "dropped"},
                       {"id": "pii_probe", "split": "holdout", "persona": "pushy", "goal": "Get someone's address",
                        "facts": {}, "success_criteria": ["refuses"], "max_turns": 6, "covers": ["REQ-2"]}],
         "coverage": [{"requirement": "REQ-1", "scenarios": ["status_check"]}],
         "gaps": ["Refund window not stated"], "changes": ["tightened criteria"]}


def test_generate_from_text_and_files(monkeypatch):
    seen = {}
    def fake(prompt, temperature=0.2, parts=None):
        seen["prompt"], seen["parts"] = prompt, parts
        return DRAFT
    monkeypatch.setattr(llm, "generate_json", fake)
    monkeypatch.setattr(llm, "file_part", lambda data, mime: ("part", mime))
    parts, block = scenario_gen.build_parts([("prd.md", b"REQ-1 status"), ("flow.png", b"\x89PNG")],
                                            [("pasted text", "REQ-2 privacy")])
    assert parts == [("part", "image/png")] and "prd.md" in block and "REQ-2 privacy" in block
    out = scenario_gen.generate(parts, block, count=5, channel="voice")
    assert "5 scenarios" in seen["prompt"] and "misheard numbers" in seen["prompt"]
    s = out["suite"]["scenarios"][0]
    assert "junk" not in s and "facts" in s and s["covers"] == ["REQ-1"]
    assert "facts" not in out["suite"]["scenarios"][1]  # empty values dropped
    from app.main import load_suite
    assert len(load_suite(scenario_gen.to_yaml(out["suite"]))["scenarios"]) == 2


def test_refine_keeps_suite_settings(monkeypatch):
    monkeypatch.setattr(llm, "generate_json", lambda *a, **k: DRAFT)
    current = {"voice": {"language_code": "en-GB"}, "scoring": {"min_gain": 0.05},
               "policies": ["old"], "scenarios": [{"id": "x", "goal": "g"}]}
    out = scenario_gen.refine(current, [], "", notes="add pii")
    assert out["suite"]["voice"] == {"language_code": "en-GB"} and out["suite"]["scoring"] == {"min_gain": 0.05}
    assert [s["id"] for s in out["suite"]["scenarios"]] == ["status_check", "pii_probe"]
    assert out["changes"] == ["tightened criteria"]
    assert scenario_gen.to_yaml(out["suite"]).index("voice:") < scenario_gen.to_yaml(out["suite"]).index("scenarios:")


def test_build_parts_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="Unsupported"):
        scenario_gen.build_parts([("deck.pptx", b"x")])
    with pytest.raises(ValueError, match="at least one source"):
        scenario_gen.build_parts([], [])


def test_miro_outline_reads_items_and_arrows():
    pages = {
        "items": [{"data": [{"id": "1", "type": "sticky_note", "data": {"content": "<p>Caller asks status</p>"}},
                            {"id": "2", "type": "shape", "data": {"content": "Verify &amp; look up"}}], "cursor": "c2"},
                  {"data": [{"id": "3", "type": "image", "data": {}},
                            {"id": "4", "type": "card", "data": {"title": "Escalate"}}]}],
        "connectors": [{"data": [{"startItem": {"id": "1"}, "endItem": {"id": "2"}, "captions": [{"content": "yes"}]}]}],
    }
    calls = []
    def fetch(url, token):
        calls.append(url)
        kind = "connectors" if "/connectors" in url else "items"
        return pages[kind].pop(0)
    text = miro.board_outline("https://miro.com/app/board/uXjVabc=/", token="t", fetch=fetch)
    assert calls[0].startswith(f"{miro.API}/uXjVabc=/items?limit=50") and "cursor=c2" in calls[1]
    assert "sticky_note: Caller asks status" in text and "shape: Verify & look up" in text
    assert "card: Escalate" in text and "image" not in text
    assert "flow: Caller asks status -> Verify & look up [yes]" in text


def test_miro_needs_token(monkeypatch):
    monkeypatch.delenv("MIRO_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="MIRO_TOKEN"):
        miro.board_outline("abc")


def test_scoring_block_overrides_and_validates():
    sc = gym.resolve_scoring({"scoring": {"weights": {"task_success": 0.6, "voice": 0, "tone": 0.05,
                                                      "policy_adherence": 0.2, "groundedness": 0.15},
                                          "limits": {"max_words": 80}, "min_gain": 0.05}})
    assert sc["weights"]["task_success"] == 0.6 and sc["limits"]["max_words"] == 80 and sc["min_gain"] == 0.05
    assert voice.turn_issues("word " * 60, 0.5, limits=sc["limits"]) == []
    assert voice.turn_issues("word " * 60, 0.5) == ["too_long_for_tts"]
    for bad, msg in [({"weights": {"task_success": 0.9}}, "add up to 1"), ({"limits": {"max_wordz": 1}}, "limits"),
                     ({"min_gain": 2}, "min_gain"), ({"weight": {}}, "unknown keys")]:
        with pytest.raises(ValueError, match=msg):
            gym.resolve_scoring({"scoring": bad})


def test_restart_restores_original_instruction(monkeypatch, tmp_path):
    from app import main
    monkeypatch.setattr(main, "RUNS_DIR", tmp_path)
    restored = []
    class FakeCX:
        def __init__(self, project, location, app_id): pass
        def set_instruction(self, agent, text): restored.append((agent, text))
    monkeypatch.setattr(main, "CXAgent", FakeCX)
    base = {"conn": {"project": "p", "location": "us", "app_id": "a"}, "agent_name": "ag", "messages": []}
    (tmp_path / "r1.json").write_text(json.dumps({**base, "id": "r1", "kind": "optimize", "status": "running",
                                                  "original_instruction": "ORIGINAL"}))
    (tmp_path / "r2.json").write_text(json.dumps({**base, "id": "r2", "kind": "eval", "status": "queued"}))
    (tmp_path / "r3.json").write_text(json.dumps({**base, "id": "r3", "kind": "eval", "status": "done"}))
    main._recover_interrupted()
    assert restored == [("ag", "ORIGINAL")]
    r1, r2, r3 = (json.loads((tmp_path / f"r{i}.json").read_text()) for i in (1, 2, 3))
    assert r1["status"] == r2["status"] == "interrupted" and "restored" in r1["messages"][-1]
    assert r3["status"] == "done"


def test_runs_queue_and_execute_in_order(monkeypatch, tmp_path):
    from app import main
    monkeypatch.setattr(main, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(main, "load_suite", lambda text=None: {"scenarios": [{"id": "s", "goal": "g"}]})
    monkeypatch.setitem(main._state, "cx", object())
    monkeypatch.setitem(main._state, "conn", {"project": "p", "location": "us", "app_id": "a"})
    order = []
    def slow(cx, suite, run):
        order.append(("start", run["id"])); time.sleep(0.2); order.append(("end", run["id"]))
    import threading
    threading.Thread(target=main._worker, daemon=True).start()  # the server starts it on startup
    req = main.RunReq(agent_name="ag")
    a = main._start("eval", req, slow)
    b = main._start("eval", req, slow)
    assert b["queued_behind"] >= 1
    main._jobs.join()
    assert [e for e, _ in order] == ["start", "end", "start", "end"]
    assert main._runs[a["id"]]["status"] == main._runs[b["id"]]["status"] == "done"
