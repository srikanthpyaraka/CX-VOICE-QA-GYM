# Voice QA Gym

QA and auto-improve loop for a **Google CX Agent Studio** voice agent.
It simulates callers, scores each conversation, then rewrites the agent's instruction
and keeps a rewrite only when the scores go up.

```
            ┌──────────── policy = agent instruction ────────────┐
            ▼                                                    │
  Gemini caller ──utterance──▶ CX Agent Studio (Sessions API)    │
  (persona+goal) ◀──reply────  via cxas-scrapi                   │
            │                                                    │
            ▼ episode transcript                                 │
  reward = Gemini judge (task, policy, grounding, tone)          │
         + deterministic voice checks (length, markdown, URLs,   │
           stacked questions, latency)                           │
            │                                                    │
            ▼ worst episodes                                     │
  Gemini optimizer → candidate instruction → re-run suite ───────┘
  kept only if train ↑ by >0.02 and holdout doesn't drop by >0.02
```

> **New here? Start with [docs/GUIDE.md](docs/GUIDE.md):** step by step from sign-in to watching
> the coach improve an agent, the optional BigQuery export, and deploying on Cloud Run.

## What's in this repo

| Path | What it is |
|---|---|
| `app/`, `static/` | The web app: FastAPI backend and a single-page UI (Connect, Scenarios, Evaluate, Auto-improve, History) |
| `scenarios/` | The test suite (`suite.yaml`) and background-noise WAVs for audio mode |
| `scripts/create_demo_agent.py` | Creates a self-contained demo voice agent in CX Agent Studio to test against |
| `scripts/gen_scenarios.py` | Drafts or refines a test suite from PRDs, documents, diagram images or a Miro board |
| `scripts/deploy_cloud_run.sh` | One-command, re-runnable deploy to Cloud Run |
| `scripts/export_runs_to_bigquery.py` | Backfills saved runs into BigQuery |
| `tests/` | Offline tests of the whole loop with a fake agent and fake Gemini |
| `docs/live-demo/` | A presenter page that replays a recorded real run (calls, tool calls, verdicts) |
| `docs/decks/` | Sources for the pitch deck and the tab-by-tab walkthrough deck |
| `docs/screenshots/` | App screenshots: `real-run/` from the live demo agent, the rest from the labelled demo-data run |
| `docs/demo-data/` | The script that generates the clearly labelled demo run used for early screenshots |
| `Dockerfile` | Container for Cloud Run (see below) |

## Run

```bash
gcloud auth application-default login
pip install -r requirements.txt
uvicorn app.main:app --port 8800
```

Open http://localhost:8800. Enter the project, location and app ID, pick the agent, then
use **Evaluate** or **Auto-improve**.

Gemini runs on Vertex AI in the same project (`GEMINI_MODEL`, default `gemini-2.5-flash`;
`GEMINI_LOCATION`, default `global`). If `GOOGLE_API_KEY` is set, that key is used instead.

## Inputs

| Where | Input | Notes |
|---|---|---|
| Connect | **GCP project, location, app ID** | From the app's resource name `projects/<project>/locations/<location>/apps/<app_id>`; run `cxas apps list --project-id P --location L` to see it |
| Connect | **Agent** | Dropdown of the app's agents. Its `instruction` is what gets evaluated and rewritten |
| Scenarios | **suite.yaml** | `policies` (all calls) and `voice` (default caller voice). Each scenario has `id`, `split` (train/holdout), `persona`, `goal`, `facts`, `success_criteria`, `max_turns`, and optionally `voice` and `background_noise` |
| Run | Mode, trials, tool fakes, noise, iterations | Iterations and the sandbox confirmation apply to Auto-improve only |
| Env | `GEMINI_MODEL`, `GEMINI_LOCATION`, `GOOGLE_API_KEY` | Optional |
| Env | `SUITE_PATH`, `RUNS_DIR` | Optional storage paths (used on Cloud Run) |
| Env | `BQ_DATASET` (+ `BQ_LOCATION`) | Optional. Set `project.dataset` to also save every call and round to BigQuery. Off when unset |
| Env | `MIRO_TOKEN` | Optional. Miro API token (`boards:read`) so the scenario generator can read boards. Without it, upload an exported image |
| Suite | `scoring:` | Optional per-use-case weights, phone limits and the keep threshold. See `scenarios/suite.yaml` |
| Env | `MAX_PARALLEL_CALLS` | Calls run at once (default 2). Use 1 on projects with the default agent-model token quota |

## Create a demo agent to test against

```bash
python scripts/create_demo_agent.py --project my-project --location us
```

This creates one new app, **Voice QA Gym Demo** (`voice-qa-gym-demo`), with:

- **`order_support`**, an agent with a deliberately rough first instruction ("share everything",
  "list every option", no read-back of numbers), so the scorecard and the coach have real
  problems to find.
- Three **Python-function tools** with built-in test orders that match `scenarios/suite.yaml`:
  `get_order` (also verifies the caller's email or phone), `start_return` and `update_address`.
  They call nothing outside the agent.

It's safe to re-run, because it skips anything that already exists. Delete the whole app when done.
Run evaluations against it with **Use tool fakes off**, since the tools are test code already.

## Testing a real agent

```bash
PROJECT=my-project; LOCATION=global
gcloud auth application-default login
gcloud auth application-default set-quota-project $PROJECT
gcloud services enable ces.googleapis.com texttospeech.googleapis.com aiplatform.googleapis.com --project $PROJECT
cxas apps list --project-id $PROJECT --location $LOCATION

# Back up the real app and make a sandbox copy for Auto-improve to edit
cxas pull "My Agent" --project-id $PROJECT --location $LOCATION --target-dir ./backup_my_agent
cxas create "My Agent QA" --project-id $PROJECT --location $LOCATION
cxas push --app-dir ./backup_my_agent --to "My Agent QA" --project-id $PROJECT --location $LOCATION
```

1. Rewrite `scenarios/suite.yaml` for your agent's real call types. The `facts` must exist in
   whatever the tools return, whether tool fakes or a test backend. If they don't, every call
   ends with "order not found" and fails.
2. Run **Evaluate, Text, 1 trial**. Read 3–4 transcripts and check you agree with the judge
   before you trust its numbers.
3. Run **Evaluate, Full audio** for ASR, response-gap and speech-length results.
4. Run **Auto-improve on the sandbox app** (2 iterations, trials 2). Review the diff, then Apply.
5. Promote to production through your normal release path (pull from the sandbox, push to
   prod, create a version).

## Deploy to Cloud Run

```bash
PROJECT=my-project ./scripts/deploy_cloud_run.sh
```

That runs every step below and is safe to re-run. Optional `BQ_DATASET`, `MIRO_SECRET`,
`AGENT_PROJECT` and `INVOKERS` are explained in [docs/GUIDE.md](docs/GUIDE.md) Part 7. The
manual equivalent:

```bash
PROJECT=my-project; REGION=us-central1; BUCKET=$PROJECT-voice-qa-gym
SA=voice-qa-gym@$PROJECT.iam.gserviceaccount.com
gcloud services enable run.googleapis.com cloudbuild.googleapis.com ces.googleapis.com \
  texttospeech.googleapis.com aiplatform.googleapis.com --project $PROJECT
gcloud storage buckets create gs://$BUCKET --location $REGION --project $PROJECT
gcloud iam service-accounts create voice-qa-gym --project $PROJECT
for ROLE in roles/ces.agentEditor roles/aiplatform.user roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA --role $ROLE
done
gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member serviceAccount:$SA --role roles/storage.objectUser
# roles/ces.agentEditor covers agents get/list/update, apps get/list, and runSession + bidiRunSession.

gcloud run deploy voice-qa-gym --source . --region $REGION --project $PROJECT \
  --service-account $SA --no-allow-unauthenticated \
  --min-instances 1 --max-instances 1 --no-cpu-throttling --memory 1Gi \
  --execution-environment gen2 \
  --add-volume name=data,type=cloud-storage,bucket=$BUCKET \
  --add-volume-mount volume=data,mount-path=/data \
  --set-env-vars SUITE_PATH=/data/suite.yaml,RUNS_DIR=/data/runs

gcloud run services proxy voice-qa-gym --region $REGION --project $PROJECT --port 8080
# then open http://localhost:8080
```

Why these flags:

- **`--no-allow-unauthenticated`**: the app has no login of its own and can rewrite agents.
  Only people with `roles/run.invoker` can reach it.
- **`--max-instances 1 --min-instances 1`**: the job lock and live progress live in process
  memory.
- **`--no-cpu-throttling`**: runs execute in background threads after the HTTP request returns.
- **Bucket volume**: keeps the suite, run history and audio across restarts.

If the agent lives in another project, grant `$SA` `roles/ces.agentEditor` and `roles/aiplatform.user`
there. Gemini calls run in the agent's project.

## Findings from the first live run (Oct 2026)

These came from running the app against the demo agent in a fresh project.

- **Location:** CX Agent Studio answered on `us`, not `global`, in that project.
  `gemini-2.5-flash` wasn't offered as an agent model in `us`, so the script creates the agent
  with the app's default model.
- **Quota:** a new project's **"RunSession LLM tokens"** quota defaults to **1,000 per minute**.
  That's roughly one agent turn a minute, so most calls hit `429 Resource exhausted`. Request an
  increase (IAM & Admin → Quotas, service `ces.googleapis.com`). The quota is set per **region
  and base model**, so the request must name both. The demo app (location `us`) was served from
  `us-central1`. Requests for 50,000/min on `gemini-composite-v1`, `gemini-3.5-flash` and
  `gemini-2.5-flash-001` in `us-central1` were approved automatically within minutes and
  removed the throttling. Until an increase is approved, set `MAX_PARALLEL_CALLS=1`. The client
  retries throttled turns for up to about 4 minutes.
- **Duplicate text:** cxas-scrapi's consolidated reply text includes the copy in the session's
  diagnostic info, so every reply appeared twice. The client now reads only the top-level outputs.
- **Errored calls:** calls that fail (for example on quota) are now left out of the scores. A
  coaching round with any errored call is marked *inconclusive* and is never compared.

## Live demo page and decks

- `docs/live-demo/voice-qa-gym-live-run.html` is a standalone page. Open it in a browser and
  present with ← → and Space. Rebuild it from any recorded run with
  `python docs/live-demo/build.py <eval-run-id> [<auto-improve-run-id>]`.
- `docs/decks/pitch/` and `docs/decks/walkthrough/` hold the slide sources: one HTML
  `<section>` per slide on a 1920×1080 canvas, with speaker notes in `<aside>`, ordered by
  `deck.json`. They were made as Claude Slides artifacts, which export to PowerPoint or PDF.
  Screenshots marked "demo data" come from the scripted run in `docs/demo-data/`, not a live agent.

## Audio mode

Pick **Full audio** on Evaluate or Auto-improve. Each caller line is:

1. Synthesised with Google TTS, using the scenario's `voice` (language/accent, voice name,
   `speaking_rate`). The suite-level `voice` is the default.
2. Optionally mixed with a `background_noise` WAV from `scenarios/noise/` at about 15 dB SNR,
   using scrapi's own mixer. The two noise beds that ship are synthetic, generated with numpy.
   Drop real recordings (call-centre babble, street, car) into the same folder.
3. Streamed over `BidiRunSession` using `Sessions.run(audio=..., modality="audio")`. The
   agent's STT, LLM and TTS all run as they would on a real call.

Audio mode measures:

| Metric | Source |
|---|---|
| **ASR word error rate**: what the caller said vs. what the agent heard | user transcript in the session's diagnostic messages |
| **Response gap**: end of caller speech → agent's first audio (flag > 2s) | timestamps from a subclass of scrapi's bidi handler |
| **Agent speech duration** (flag > 15s) | length of the returned PCM |
| Markdown/URL/multi-question checks | agent's text |

Every turn's caller and agent audio is saved to `runs/audio/<run>/` and playable in the
transcript view. The judge sees both the said and heard text. The agent isn't blamed for the
misrecognition itself, but it is judged on whether it confirmed key details. ASR WER is
reported but not part of the reward, because the instruction can't change the STT model.

Audio runs in real time. scrapi paces caller audio in 100ms chunks and waits out the agent's
playback, so plan on 1–3 minutes per episode. The project needs `ces.googleapis.com` and
`texttospeech.googleapis.com` enabled; scrapi checks this before every audio turn.

## Safety model (it touches a live agent)

- Auto-improve **writes candidate instructions to the agent while it runs**. Use a dev or
  sandbox app. The UI makes you confirm this.
- The original instruction is saved to `runs/<id>.json` before anything changes. It's
  restored when the loop ends, even on error.
- Nothing is kept until you click **Apply** on an iteration. **Restore original** is always
  available on the run.
- **Use tool fakes** is on by default, so evaluations don't hit live backends.
- Runs never overlap, because each may rewrite the agent. Extra runs wait in a queue.
- If the server restarts mid-run, the run is marked *interrupted*. A coaching run's original
  instruction is restored automatically on startup.

## Pieces

| File | Role |
|---|---|
| `app/gym.py` | `VoiceAgentEnv` (`reset` / `step` → obs, reward, done, info), `rollout`, `evaluate` |
| `app/simulator.py` | Gemini caller, driven by persona, goal and facts |
| `app/judge.py` | Gemini rubric judge; it never sees the instruction, so the optimizer can't game it |
| `app/voice.py` | Deterministic TTS-readiness checks and latency p50/p95 |
| `app/optimizer.py` | Propose, validate (keeps `{@TOOL:}` refs and GECX tags), evaluate, accept/reject |
| `app/cxas_client.py` | cxas-scrapi `Agents` / `Sessions` wrapper, TTS, timed bidi handler |
| `app/bq.py` | Optional BigQuery export (`calls` and `rounds` tables) |
| `app/scenario_gen.py`, `app/miro.py` | AI scenario generator and refiner; Miro board reader |
| `scenarios/suite.yaml` | Test suite with train/holdout split |

Reward weights are in `gym.WEIGHTS`: task 0.45, policy 0.2, grounding 0.15, voice 0.15, tone 0.05.

## Tests

`python -m pytest -q tests` runs the whole loop offline against a fake agent and fake Gemini.

## Known limits

- Audio mode has no barge-in: scrapi's handler waits for the agent to finish talking before
  the caller speaks. DTMF isn't simulated either.
- Response-gap timing depends on scrapi's silence-padding pattern (1.5.0). If that changes,
  gaps come back empty (`–`) rather than wrong.
- WER normalisation joins spaced digits ("4 8 2 1 3" = "48213") but doesn't map number words
  ("forty-eight") to digits.
- LLM judging is noisy. Use `trials` of 2–3 before trusting a small gain.
- Each candidate is tested on the agent's draft. If the app's sessions run on a deployment,
  draft edits may not show up until it's redeployed (not verified).
- Auto-improve tests candidates on the agent itself (restored afterwards), so always point it
  at a copy. Testing on a throwaway clone of the app would be safer; it isn't built yet.
- The app has no login of its own; it relies on Cloud Run's IAM (`roles/run.invoker`).
- Only CX Agent Studio agents are supported. Other platforms would need an adapter alongside
  `app/cxas_client.py`.
- The Miro reader follows Miro's REST API v2 docs and is unit-tested, but hasn't been run
  against a live board. Exported diagrams (PNG and PDF) and text PRDs were tested live.
