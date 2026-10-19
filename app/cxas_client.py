"""Thin wrapper over cxas-scrapi for the three things the gym needs:
read/write an agent's instruction, and run one conversational turn
(text, or full audio over the BidiRunSession websocket)."""
import io
import logging
import time
import wave

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable
from google.auth.transport.requests import Request
from google.cloud import texttospeech

from cxas_scrapi import Agents, Sessions
from cxas_scrapi.core.response_parser import ParsedSessionResponse
from cxas_scrapi.core.sessions import (SAMPLE_RATE, SAMPLE_WIDTH, SILENCE_PADDING_CHUNKS,
                                       BidiSessionError, BidiSessionHandler)

BYTES_PER_S = SAMPLE_RATE * SAMPLE_WIDTH
DEFAULT_VOICE = {"language_code": "en-US", "name": "en-US-Standard-C", "speaking_rate": 1.0}
# New projects get a tiny "RunSession LLM tokens" quota (1,000/min); waits up to a minute
# let the per-minute budget refill.
RETRY_DELAYS_S = (5, 15, 30, 60, 60, 60)


def _with_retry(fn):
    for delay in (*RETRY_DELAYS_S, None):
        try:
            return fn()
        except (ResourceExhausted, ServiceUnavailable) as e:
            err = e
        except BidiSessionError as e:
            if e.server_error_kind not in ("resource_exhausted", "unavailable"):
                raise
            err = e
        if delay is None:
            raise err
        logging.warning("Agent session throttled (%s); retrying in %ss", type(err).__name__, delay)
        time.sleep(delay)


def _agent_text(resp, parsed: dict) -> str:
    """The agent's own words. scrapi's consolidated text also appends the copy found in
    diagnostic_info, so every reply would appear twice; use top-level outputs only."""
    return " ".join(o.text for o in resp.outputs if o.text).strip() or parsed["agent_text"]


class _TimedBidiHandler(BidiSessionHandler):
    """Adds two timestamps to scrapi's handler: when the caller stopped
    speaking (start of trailing silence) and when the agent's first audio
    arrived. Relies on scrapi's padding pattern: one SILENCE_PADDING_CHUNKS
    call before speech and one after. If that changes, timings come back None."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._pads = 0
        self.speech_end = None
        self.first_audio = None

    def _send_silence(self, num_chunks):
        if num_chunks == SILENCE_PADDING_CHUNKS:
            self._pads += 1
            if self._pads == 2:
                self.speech_end = time.time()
        super()._send_silence(num_chunks)

    def _on_message(self, ws, message):
        n = len(self.outputs)
        super()._on_message(ws, message)
        if (self.speech_end and self.first_audio is None
                and len(self.outputs) > n and self.outputs[-1].audio):
            self.first_audio = time.time()


class _AudioResult:
    """Duck-types RunSessionResponse (has .outputs) and carries timings."""

    def __init__(self, outputs, speech_end, first_audio):
        self.outputs = outputs
        self.response_gap_s = (round(first_audio - speech_end, 3)
                               if speech_end and first_audio else None)


class _TimedSessions(Sessions):
    def async_bidi_run_session(self, config, inputs, background_noise_file=None,
                               bg_noise_snr=15.0):
        if self.rate_limiter:
            self.rate_limiter.wait_and_consume()
        try:
            self.creds.refresh(Request())
        except Exception:
            pass  # same as scrapi: a stale token surfaces as a websocket auth error
        handler = _TimedBidiHandler(self.location, self.token, config, inputs,
                                    user_agent=self.user_agent,
                                    background_noise_file=background_noise_file,
                                    bg_noise_snr=bg_noise_snr)
        resp = handler.run()
        return _AudioResult(list(resp.outputs), handler.speech_end, handler.first_audio)


class CXAgent:
    def __init__(self, project: str, location: str, app_id: str):
        self.app_name = f"projects/{project}/locations/{location}/apps/{app_id}"
        self.project = project
        self.agents = Agents(app_name=self.app_name)
        self.sessions = _TimedSessions(app_name=self.app_name)
        self._tts = None

    def list_agents(self) -> list[dict]:
        return [{"name": a.name, "display_name": a.display_name}
                for a in self.agents.list_agents()]

    def get_instruction(self, agent_name: str) -> str:
        return self.agents.get_agent(agent_name).instruction

    def set_instruction(self, agent_name: str, instruction: str) -> None:
        self.agents.update_agent(agent_name, instruction=instruction)

    def new_session(self) -> str:
        return self.sessions.create_session_id()

    def send(self, session_id: str, text: str, use_tool_fakes: bool,
             audio: dict | None = None) -> dict:
        """audio=None runs a text turn. audio={"voice": {...}, "background_noise_file": path|None}
        synthesises the caller's line and streams it as real audio."""
        if audio is None:
            def call():
                start = time.perf_counter()
                r = self.sessions.run(session_id=session_id, text=text,
                                      use_tool_fakes=use_tool_fakes)
                return r, time.perf_counter() - start
            resp, elapsed = _with_retry(call)
            out = self.sessions.get_structured_response(resp)
            out["agent_text"] = _agent_text(resp, out)
            out["latency_s"] = round(elapsed, 3)
            return out

        user_pcm = self.tts(text, audio.get("voice") or {})
        resp = _with_retry(lambda: self.sessions.run(
            session_id=session_id, audio=user_pcm, modality="audio", use_tool_fakes=use_tool_fakes,
            background_noise_file=audio.get("background_noise_file")))
        out = self.sessions.get_structured_response(resp)
        out["agent_text"] = _agent_text(resp, out)
        agent_pcm = b"".join(o.audio for o in resp.outputs if o.audio)
        out.update(
            latency_s=resp.response_gap_s,
            heard=" ".join(ParsedSessionResponse(resp).user_texts) or None,
            user_pcm=user_pcm, agent_pcm=agent_pcm,
            agent_audio_s=round(len(agent_pcm) / BYTES_PER_S, 2),
        )
        return out

    def tts(self, text: str, voice: dict) -> bytes:
        """Caller speech as 16 kHz mono LINEAR16 PCM (what BidiRunSession expects)."""
        v = {**DEFAULT_VOICE, **voice}
        if not v["name"].startswith(v["language_code"]):
            v["name"] = ""   # language overridden without a voice: let TTS pick one
        if self._tts is None:
            self._tts = texttospeech.TextToSpeechClient(
                credentials=self.sessions.creds,
                client_options=ClientOptions(quota_project_id=self.project))
        resp = self._tts.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(language_code=v["language_code"],
                                                    name=v["name"]),
            audio_config=texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE, speaking_rate=v["speaking_rate"]),
        )
        with wave.open(io.BytesIO(resp.audio_content), "rb") as w:  # strip WAV header
            return w.readframes(w.getnframes())


def write_wav(path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)
