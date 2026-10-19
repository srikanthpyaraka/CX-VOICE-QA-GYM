"""Offline tests: fake CX agent + fake Gemini, real gym/optimizer logic."""
import pytest

from app import gym, llm, optimizer, voice

GOOD = "<role>support</role><taskflow>Keep it SHORT. {@TOOL: get_order}</taskflow>"
BAD = "<role>support</role><taskflow>Explain everything. {@TOOL: get_order}</taskflow>"


class FakeCX:
    def __init__(self, instruction):
        self.instruction = instruction
        self.writes = []

    def get_instruction(self, _):
        return self.instruction

    def set_instruction(self, _, text):
        self.instruction = text
        self.writes.append(text)

    def new_session(self):
        return "s1"

    def send(self, _sid, text_in, _fakes, audio=None):
        self.last_audio = audio
        if "SHORT" in self.instruction:
            text = "Your order arrives Friday."
        else:
            text = "**Sure!** Here are your options: " + "blah " * 50 + "Which one? Or what?"
        out = {"agent_text": text, "tool_calls": [], "tool_responses": [],
               "agent_transfer": None, "session_ended": False, "latency_s": 0.4}
        if audio is not None:
            out.update(latency_s=2.5, heard=text_in.replace("order", "border"),
                       user_pcm=b"\0\0" * 1600, agent_pcm=b"\0\0" * 16000, agent_audio_s=1.0)
        return out


def fake_llm(proposal):
    def generate_json(prompt, temperature=0.2):
        if "role-playing a CALLER" in prompt:
            done = '"role": "agent"' in prompt
            return {"utterance": "" if done else "where is my order", "done": done}
        if "strict QA evaluator" in prompt:
            ok = "arrives Friday" in prompt
            return {"task_success": ok, "policy_adherence": 1, "groundedness": 1,
                    "tone": 1 if ok else 0.3, "failures": [], "summary": "ok" if ok else "rambled"}
        if "You improve the instruction" in prompt:
            return {"diagnosis": "too verbose", "changes": ["shorter"], "instruction": proposal}
        raise AssertionError("unexpected prompt")
    return generate_json


SUITE = {"policies": [], "scenarios": [
    {"id": "a", "goal": "order status", "split": "train"},
    {"id": "b", "goal": "order status", "split": "holdout"}]}


def run_loop(monkeypatch, proposal, iterations=1):
    monkeypatch.setattr(llm, "generate_json", fake_llm(proposal))
    cx = FakeCX(BAD)
    events = []
    optimizer.optimize(cx, "agent", SUITE, iterations, True, 1, events.append)
    its = [e for e in events if e["type"] == "iteration"]
    return cx, its


def test_voice_issues():
    assert voice.turn_issues("Your order arrives Friday.", 0.5) == []
    issues = voice.turn_issues("**Hi** see https://x.com. Which? Why?", 4.0)
    assert {"markdown_read_aloud", "url_or_email_read_aloud",
            "multiple_questions", "high_latency"} <= set(issues)


def test_improvement_is_kept_and_original_restored(monkeypatch):
    cx, its = run_loop(monkeypatch, GOOD)
    base, cand = its
    assert cand["accepted"]
    assert cand["result"]["train_reward"] > base["result"]["train_reward"]
    assert cand["result"]["pass_rate"] == 1.0
    assert cx.instruction == BAD          # live agent restored
    assert cx.writes == [GOOD, BAD]


def test_candidate_dropping_tool_ref_is_rejected_without_eval(monkeypatch):
    cx, its = run_loop(monkeypatch, "<role>x</role><taskflow>SHORT</taskflow>")
    assert "dropped tool references" in its[1]["invalid"]
    assert cx.writes == [BAD]             # never pushed, only the final restore


def test_no_gain_is_rejected(monkeypatch):
    cx, its = run_loop(monkeypatch, BAD + " ")
    assert not its[1]["accepted"]
    assert cx.instruction == BAD


def test_original_restored_on_crash(monkeypatch):
    monkeypatch.setattr(llm, "generate_json", fake_llm(GOOD))
    cx = FakeCX(BAD)
    monkeypatch.setattr(gym, "evaluate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        optimizer.optimize(cx, "agent", SUITE, 2, True, 1, lambda e: None)
    assert cx.instruction == BAD


def test_reward_weights_sum_to_one():
    assert abs(sum(gym.WEIGHTS.values()) - 1) < 1e-9


def test_audio_episode(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "generate_json", fake_llm(GOOD))
    cx = FakeCX(GOOD)
    suite = {**SUITE, "voice": {"language_code": "en-GB", "name": "en-GB-Standard-B"}}
    suite["scenarios"] = [{**SUITE["scenarios"][0], "voice": {"speaking_rate": 0.8},
                           "background_noise": "car_cabin.wav"}]
    res = gym.evaluate(cx, suite, audio={"background_noise": None}, audio_dir=tmp_path)
    ep = res["episodes"][0]
    user, agent = ep["transcript"]
    assert cx.last_audio["voice"] == {"language_code": "en-GB", "name": "en-GB-Standard-B",
                                      "speaking_rate": 0.8}
    assert cx.last_audio["background_noise_file"].endswith("scenarios/noise/car_cabin.wav")
    assert user["heard"] == "where is my border" and user["wer"] == 0.25
    assert agent["voice_issues"] == ["high_latency"]       # 2.5s gap > 2.0s audio threshold
    assert res["asr"] == {"mean_wer": 0.25, "misheard_turns": 1, "turns": 1}
    assert (tmp_path / user["audio"]).stat().st_size > 44
    assert (tmp_path / agent["audio"]).exists()


def test_timed_handler_marks_speech_end_and_first_audio(monkeypatch):
    import base64, json
    from cxas_scrapi.core.sessions import BidiSessionHandler
    from app.cxas_client import _TimedBidiHandler
    monkeypatch.setattr(BidiSessionHandler, "_send_silence", lambda self, n: None)
    h = _TimedBidiHandler("global", "tok", {"session": "s"}, [])
    msg = json.dumps({"sessionOutput": {"audio": base64.b64encode(b"\1\2").decode()}})
    h._on_message(None, msg)                 # agent audio before caller finished: ignored
    h._send_silence(3); h._send_silence(1)   # leading pad, keep-alive
    assert h.speech_end is None
    h._send_silence(3)                       # trailing pad = caller stopped speaking
    assert h.speech_end is not None and h.first_audio is None
    h._on_message(None, msg)
    assert h.first_audio >= h.speech_end


def test_errored_calls_excluded_and_round_inconclusive(monkeypatch):
    monkeypatch.setattr(llm, "generate_json", fake_llm(GOOD))
    cx = FakeCX(BAD)
    real_send = cx.send
    def flaky(sid, text, fakes, audio=None):
        if "SHORT" in cx.instruction:
            raise RuntimeError("429 Resource has been exhausted")
        return real_send(sid, text, fakes, audio)
    cx.send = flaky
    events = []
    optimizer.optimize(cx, "agent", SUITE, 1, True, 1, events.append)
    base, cand = [e for e in events if e["type"] == "iteration"]
    assert base["result"]["errors"] == 0
    assert cand["result"]["errors"] == 2 and cand["result"]["train_reward"] is None
    assert not cand["accepted"] and cand["invalid"].startswith("inconclusive")
    assert cx.instruction == BAD


def test_bigquery_rows_match_schema():
    from app import bq
    monkey_run = {"id": "r1", "kind": "optimize", "agent_name": "a",
                  "conn": {"project": "p", "location": "us", "app_id": "x"}, "params": {"modality": "audio"}}
    ep_ok = {"scenario_id": "s1", "split": "train", "reward": 0.8, "end_reason": "max_turns",
             "scores": {"task_success": True, "policy_adherence": 1, "groundedness": 1, "tone": 1, "voice": 0.5,
                        "summary": "ok", "failures": [{"turn": 1, "detail": "long"}]},
             "transcript": [{"role": "user", "text": "hi", "wer": 0.1},
                            {"role": "agent", "text": "hello", "latency_s": 1.5, "voice_issues": ["too_long_for_tts"]}]}
    ep_err = {"scenario_id": "s2", "split": "holdout", "reward": 0.0, "transcript": [], "error": "429"}
    result = {**gym.summarize([ep_ok, ep_err]), "episodes": [ep_ok, ep_err]}
    round_row, calls = bq.build_rows(monkey_run, result, iteration=1, outcome="kept",
                                     instruction="<role>x</role>", changes=["shorter"])
    assert set(round_row) == {c for c, _ in bq.ROUNDS}
    assert all(set(c) == {n for n, _ in bq.CALLS} for c in calls)
    assert round_row["train_reward"] == 0.8 and round_row["errors"] == 1 and round_row["outcome"] == "kept"
    assert round_row["instruction_sha"] and round_row["mean_wer"] == 0.1
    assert calls[0]["voice_issues"] == "too_long_for_tts" and calls[0]["latency_p50_s"] == 1.5
    assert calls[1]["error"] == "429" and calls[1]["task_success"] is None



def test_prompt_markers_stripped_from_rewrite(monkeypatch):
    assert optimizer.clean_instruction("<<<\n<role>x</role>\n>>>") == "<role>x</role>"
    assert optimizer.clean_instruction("```xml\n<role>x</role>\n```") == "<role>x</role>"
    assert optimizer.clean_instruction("<role>a >>> b</role>") == "<role>a >>> b</role>"
    cx, its = run_loop(monkeypatch, "<<<\n" + GOOD + "\n>>>")
    assert its[1]["instruction"] == GOOD and its[1]["accepted"]
