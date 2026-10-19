"""Gym-style environment over a live CX Agent Studio agent.

    env = VoiceAgentEnv(cx, scenario, policies)
    obs = env.reset()
    obs, reward, done, info = env.step({"utterance": "hi, I need to move my booking"})

Action      = the caller's utterance (or {"done": True} to hang up). In audio
              mode it is synthesised to speech (per-scenario voice/accent, optional
              background noise) and streamed over BidiRunSession.
Observation = the agent's reply plus tool calls, transfer and end-of-session flags.
Reward      = 0 on intermediate turns; on the terminal step, the episode reward
              (judge rubric + deterministic voice score). The policy being
              optimised is the agent's instruction (see optimizer.py).
"""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import judge, simulator, voice
from app.cxas_client import CXAgent, write_wav

NOISE_DIR = Path(__file__).resolve().parent.parent / "scenarios" / "noise"

# Episode reward weights. Task success dominates; voice quality is a real but
# secondary signal.
WEIGHTS = {"task_success": 0.45, "policy_adherence": 0.2, "groundedness": 0.15,
           "tone": 0.05, "voice": 0.15}




def resolve_scoring(suite: dict) -> dict:
    """The suite's optional `scoring:` block merged over the defaults, validated.

        scoring:
          weights: {task_success: 0.5, policy_adherence: 0.3, groundedness: 0.2, tone: 0, voice: 0}
          limits: {max_words: 60, max_gap_s: 1.5}     # see voice.LIMITS
          min_gain: 0.03                               # how much a coached version must improve
    """
    from app.optimizer import MIN_GAIN
    cfg = suite.get("scoring") or {}
    unknown = set(cfg) - {"weights", "limits", "min_gain"}
    if unknown:
        raise ValueError(f"scoring: unknown keys {sorted(unknown)}")
    weights = {**WEIGHTS, **(cfg.get("weights") or {})}
    if set(weights) != set(WEIGHTS) or any(not isinstance(v, (int, float)) or v < 0 for v in weights.values()):
        raise ValueError(f"scoring.weights: use only {sorted(WEIGHTS)}, each a number >= 0")
    if abs(sum(weights.values()) - 1) > 0.01:
        raise ValueError(f"scoring.weights must add up to 1 (they add up to {sum(weights.values()):.2f})")
    limits = {**voice.LIMITS, **(cfg.get("limits") or {})}
    if set(limits) != set(voice.LIMITS) or any(not isinstance(v, (int, float)) or v <= 0 for v in limits.values()):
        raise ValueError(f"scoring.limits: use only {sorted(voice.LIMITS)}, each a number > 0")
    min_gain = cfg.get("min_gain", MIN_GAIN)
    if not isinstance(min_gain, (int, float)) or not 0 <= min_gain < 1:
        raise ValueError("scoring.min_gain must be a number from 0 to 1")
    return {"weights": weights, "limits": limits, "min_gain": float(min_gain)}


def episode_reward(scores: dict, weights: dict | None = None) -> float:
    w = weights or WEIGHTS
    return round(sum(w[k] * float(scores[k]) for k in w), 4)


class VoiceAgentEnv:
    def __init__(self, cx: CXAgent, scenario: dict, policies: list[str],
                 use_tool_fakes: bool = True, max_turns: int | None = None,
                 audio: dict | None = None, audio_dir: Path | None = None,
                 scoring: dict | None = None):
        """audio=None -> text turns. audio={"voice": {...}, "background_noise": "file.wav"|None}
        holds run-level defaults; the scenario's own voice/background_noise override them.
        audio_dir: where per-turn WAVs are saved (audio mode only)."""
        self.cx = cx
        self.scenario = scenario
        self.policies = policies
        self.use_tool_fakes = use_tool_fakes
        self.max_turns = max_turns or scenario.get("max_turns", 10)
        self.audio = None
        if audio is not None:
            noise = scenario.get("background_noise", audio.get("background_noise"))
            self.audio = {
                "voice": {**(audio.get("voice") or {}), **scenario.get("voice", {})},
                "background_noise_file": str(NOISE_DIR / noise) if noise else None,
            }
        self.audio_dir = audio_dir
        self.scoring = scoring or resolve_scoring({})

    def reset(self) -> dict:
        self.session_id = self.cx.new_session()
        self.episode_id = uuid.uuid4().hex[:10]
        self.transcript: list[dict] = []
        self.user_turns = 0
        return {"agent_text": None}

    def step(self, action: dict):
        if action.get("done"):
            return self._finish("caller_hung_up")

        user_turn = {"role": "user", "text": action["utterance"]}
        self.transcript.append(user_turn)
        self.user_turns += 1
        r = self.cx.send(self.session_id, action["utterance"], self.use_tool_fakes, self.audio)
        agent_turn = {
            "role": "agent", "text": r["agent_text"] or "",
            "tool_calls": r["tool_calls"], "tool_responses": r["tool_responses"],
            "transfer": r["agent_transfer"], "session_ended": r["session_ended"],
            "latency_s": r["latency_s"],
        }
        if self.audio is not None:
            user_turn["heard"] = r["heard"]
            user_turn["wer"] = voice.wer(user_turn["text"], r["heard"])
            agent_turn["audio_s"] = r["agent_audio_s"]
            if self.audio_dir is not None:
                n = len(self.transcript) - 1
                base = self.audio_dir / self.episode_id
                write_wav(base / f"{n}-user.wav", r["user_pcm"])
                write_wav(base / f"{n + 1}-agent.wav", r["agent_pcm"])
                user_turn["audio"] = f"{self.episode_id}/{n}-user.wav"
                agent_turn["audio"] = f"{self.episode_id}/{n + 1}-agent.wav"
        agent_turn["voice_issues"] = voice.turn_issues(
            agent_turn["text"], r["latency_s"], agent_turn.get("audio_s"), self.scoring["limits"])
        self.transcript.append(agent_turn)

        if r["session_ended"]:
            return self._finish("agent_ended_session")
        if self.user_turns >= self.max_turns:
            return self._finish("max_turns")
        return agent_turn, 0.0, False, {}

    def _finish(self, reason: str):
        agent_turns = [t for t in self.transcript if t["role"] == "agent"]
        scores = judge.judge(self.scenario, self.transcript, self.policies)
        scores["voice"] = voice.score(agent_turns)
        reward = episode_reward(scores, self.scoring["weights"])
        info = {"end_reason": reason, "scores": scores,
                "latency": voice.latency_stats(agent_turns)}
        return None, reward, True, info


def rollout(env: VoiceAgentEnv) -> dict:
    """Run one episode with the LLM caller as the action policy."""
    env.reset()
    while True:
        action = simulator.next_turn(env.scenario, env.transcript)
        _, reward, done, info = env.step(action)
        if done:
            return {"scenario_id": env.scenario["id"], "split": env.scenario.get("split", "train"),
                    "reward": reward, "transcript": env.transcript, **info}


def _safe_rollout(env: VoiceAgentEnv) -> dict:
    try:
        return rollout(env)
    except Exception as e:  # one bad episode shouldn't sink the suite
        return {"scenario_id": env.scenario["id"], "split": env.scenario.get("split", "train"),
                "reward": 0.0, "transcript": getattr(env, "transcript", []),
                "error": f"{type(e).__name__}: {e}"}


def evaluate(cx: CXAgent, suite: dict, use_tool_fakes: bool = True,
             trials: int = 1, workers: int | None = None, on_episode=None,
             audio: dict | None = None, audio_dir: Path | None = None) -> dict:
    """Run every scenario `trials` times against the agent's current instruction."""
    # Low agent-model quota? Set MAX_PARALLEL_CALLS=1 so calls don't throttle each other.
    workers = workers or int(os.environ.get("MAX_PARALLEL_CALLS", "2"))
    if audio is not None:
        audio = {**audio, "voice": {**suite.get("voice", {}), **(audio.get("voice") or {})}}
    scoring = resolve_scoring(suite)
    envs = [VoiceAgentEnv(cx, s, suite.get("policies", []), use_tool_fakes,
                          audio=audio, audio_dir=audio_dir, scoring=scoring)
            for s in suite["scenarios"] for _ in range(trials)]
    episodes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ep in pool.map(_safe_rollout, envs):
            episodes.append(ep)
            if on_episode:
                on_episode(ep)
    return summarize(episodes, scoring["limits"]["max_wer"])


def summarize(episodes: list[dict], max_wer: float = voice.MAX_WER) -> dict:
    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    ok = [e for e in episodes if "error" not in e]

    def split(name):  # errored calls (e.g. quota) are excluded, not scored 0
        return [e for e in ok if e["split"] == name]

    agent_turns = [t for e in ok for t in e["transcript"] if t["role"] == "agent"]
    wers = [t["wer"] for e in ok for t in e["transcript"]
            if t["role"] == "user" and t.get("wer") is not None]
    issue_counts: dict[str, int] = {}
    for t in agent_turns:
        for i in t["voice_issues"]:
            issue_counts[i] = issue_counts.get(i, 0) + 1
    return {
        "episodes": episodes,
        "train_reward": mean([e["reward"] for e in split("train")]),
        "holdout_reward": mean([e["reward"] for e in split("holdout")]),
        "pass_rate": mean([1.0 if e["scores"]["task_success"] else 0.0 for e in ok]),
        "metrics": {k: mean([e["scores"][k] for e in ok])
                    for k in ("policy_adherence", "groundedness", "tone", "voice")},
        "latency": voice.latency_stats(agent_turns),
        "voice_issues": issue_counts,
        "asr": {"mean_wer": mean(wers),
                "misheard_turns": sum(1 for w in wers if w > max_wer),
                "turns": len(wers)} if wers else None,
        "errors": len(episodes) - len(ok),
    }
