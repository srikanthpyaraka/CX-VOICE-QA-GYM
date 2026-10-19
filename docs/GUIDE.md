# Voice QA Gym: step-by-step guide

How to test a CX Agent Studio voice agent, watch the coach improve its instructions, keep
the improvement safely, optionally send results to BigQuery, and deploy the app on Cloud Run.

Every step was run for real against the demo agent in project `enrichment-conv-sf`.
The numbers quoted come from those runs.

---

## Part 1. Run it locally

**1. Sign in to Google Cloud** (opens your browser):

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT
```

**2. Turn on the APIs** (one time per project):

```bash
gcloud services enable ces.googleapis.com texttospeech.googleapis.com aiplatform.googleapis.com --project YOUR_PROJECT
```

**3. Raise the agent's token quota.** A new project allows only **1,000 agent-model tokens
per minute**, which is about one agent reply a minute, so most test calls fail with `429`.
In the Cloud console, go to **IAM & Admin → Quotas & System Limits**, search
**"RunSession LLM tokens"** and request **50,000** for region **us-central1**. The quota is set
per model, so request it for `gemini-composite-v1`, `gemini-3.5-flash` and
`gemini-2.5-flash-001`. Ours were approved automatically within minutes.

**4. Install and start the app:**

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8800
```

Open **http://localhost:8800**.

---

## Part 2. Get an agent to test

Use a **copy** of a real agent, or the demo agent. Never use the live customer-facing one,
because Auto-improve writes trial instructions into the agent while it runs.

**Option A: the demo agent** (an order-support bot with three test tools):

```bash
python scripts/create_demo_agent.py --project YOUR_PROJECT --location us
```

It prints the app ID (`voice-qa-gym-demo`). The agent starts deliberately rough: it lists every
option, shares everything and doesn't read numbers back. That gives the scorecard real problems to find.

**Option B: a copy of your own agent:**

```bash
cxas pull "My Agent" --project-id YOUR_PROJECT --location us --target-dir ./backup_my_agent
cxas create "My Agent QA" --project-id YOUR_PROJECT --location us
cxas push --app-dir ./backup_my_agent --to "My Agent QA" --project-id YOUR_PROJECT --location us
```

> **Location:** in our project CX Agent Studio answered on `us`, not `global`. If Connect
> fails with a 403 or 404, try `us`.

---

## Part 3. Connect and write the test calls

**5. Connect tab:** enter the project, location (`us`) and app ID, then click **Connect**.
Pick the agent from the dropdown. The grey box shows its current instructions. That's the text
being tested and coached.

**6. Scenarios tab:** each test call is a short block. Edit it and click **Save**.

```yaml
- id: mishear_and_correct
  split: train                 # train = coach learns from it; holdout = hidden exam
  persona: Elderly caller; corrects a digit after giving it wrong.
  goal: Check the status of order 51870 (first says 51807, then corrects).
  facts: {order_number: "51870", phone: "555-0199"}   # must exist in the agent's test data
  success_criteria:
    - Agent uses the corrected order number.
    - Agent reads the number back for confirmation.
  max_turns: 10
```

Keep at least a third of the calls as `holdout`. They're how the app catches a change that only
memorised the practice calls.

---

## Part 4. Test the agent as it is (Evaluate)

**7. Evaluate tab:** choose **Text turns**, **Trials 2**, and untick **Use tool fakes** if the
agent's tools already use test data (the demo agent's do). Click **Run evaluation**.

**8. Read the report card.** For the demo agent's first version, the real results were:

| Tile | Result | What it told us |
|---|---|---|
| Task success | 83% | 5 of 6 callers got what they needed |
| Policy adherence | 50% | It read out a full email address and verified identity loosely |
| Groundedness | 100% | It never made up order details |
| Voice-clean turns | 38% | 16 replies were too long to speak, and 3 had markdown |

**9. Open the calls.** Click any row to see the full conversation, the tool calls, orange
phone-quality flags, and the scorer's reasons tied to turn numbers. Check 3–4 calls and ask
whether you agree with the scorer. If you don't, fix `success_criteria` or `policies` before
going further.

> **Run-to-run variation is real.** The same untouched demo agent scored **0.74** on practice
> calls in one evaluation and **0.58** in the next. The simulated callers and the scorer vary.
> Use **Trials 2–3** whenever you're comparing versions, and don't trust a gain smaller than about 0.05.

---

## Part 5. Watch it improve the instructions (Auto-improve)

**10. Auto-improve tab:** set **Iterations 2**, **Trials 2**, **Text turns**, and tick
**"This is a dev/sandbox app"**. Click **Start loop**.

What happens, in order:

1. The original instructions are saved to disk (`runs/<id>.json`).
2. **Round 0 (baseline):** every test call runs against the current instructions.
3. **The coach** (Gemini) reads the worst calls and the scorer's findings, and writes new
   instructions. It must keep every `{@TOOL: …}` reference and the GECX tags. If it drops one,
   the round is marked **invalid** and never tested.
4. The new instructions are written to the agent, and **every test call runs again**.
5. **Kept** if practice calls improve by more than 0.02 **and** exam calls don't get worse.
   **Rejected** otherwise. **Inconclusive** if any call errored, because the averages wouldn't
   cover the same calls.
6. Steps 3–5 repeat for each iteration.
7. **The original instructions go back into the agent at the end**, even if something crashed.
   Nothing is kept until you approve it.

**11. Watch it while it runs:** the progress bar counts calls, and the status line says which
round is being proposed or tested. Each finished round appears in the table straight away.

**12. Read the results:**

- **Chart:** blue is practice calls, dashed grey is exam calls. A dot means kept, a cross means rejected.
- **Table:** for each round, the coach's **diagnosis** (what it thinks went wrong) and the
  **changes** it made.
- **Diff:** the original instructions with the coach's edits marked. Green is added, red is removed.
- **Episodes:** that round's calls. Open the same call (e.g. `mishear_and_correct`) in round 0 and
  in the kept round to see the difference in the actual conversation.

**What a real round looked like** (demo agent, 1 iteration, text, run `20261019-131950-02d1`):

| | Round 0 (baseline) | Round 1 (coached) |
|---|---|---|
| Practice-call score | 0.58 | **0.80**, kept |
| Exam-call score (never shown to the coach) | 0.78 | **0.85** |
| Caller got what they needed | 67% | 83% |
| House rules followed | 50% | 67% |
| Agent turns with no phone problem | 32% | 72% |
| Replies too long to speak | 13 | 7 |

The coach's diagnosis named the real causes: an over-broad "share everything" guideline led
the agent to read out full email addresses, corrections weren't confirmed, address changes
skipped full verification, transfer requests got repeated callback offers, and replies were too
long for the phone. Its seven changes:

1. Mask email addresses (a new PII constraint).
2. Narrow the "helpfulness" guideline to relevant, concise, secure details.
3. Always read back and confirm critical inputs, especially corrected order numbers.
4. Require order number plus email or phone before changing an address.
5. Acknowledge a transfer request, explain the limit once, then offer a callback.
6. Keep replies under about 40 words, ask one question per turn, and use no markdown.
7. Rebalance the persona from "thorough" to "clear and concise".

Practice calls improved by more than 0.02 and exam calls improved too, so the round was
**kept**. The agent was then put back to its original instructions automatically. The improved
version waits for you to click **Apply**.

**13. Approve or undo:**

- **Apply** on a kept round asks for confirmation, then writes that version into the agent copy.
- **Restore original instruction** puts the old version back at any time.

**14. Confirm the gain holds:** after Apply, go to **Evaluate** and run again with **Trials 2–3**.
If the score holds, promote the change through your normal release process (pull from the
sandbox copy, push to production, create a version).

**15. History tab:** every run is kept. Click a row to reopen it exactly as it finished.

---

## Part 6. Send results to BigQuery (optional, off by default)

Nothing goes to BigQuery unless you set `BQ_DATASET`:

```bash
BQ_DATASET=YOUR_PROJECT.voice_qa_gym uvicorn app.main:app --port 8800
```

- The dataset and two tables are created on first use. Both are partitioned by `recorded_at`.
  - **`calls`:** one row per test call: scenario, scores, verdict, failures, transcript (JSON), error.
  - **`rounds`:** one row per Evaluate run or Auto-improve round: summary scores, outcome
    (`evaluation`, `baseline`, `kept`, `rejected`, `inconclusive`) and the instructions tested.
- The History tab shows whether export is on. A failed export is noted on the run, but never
  stops the test. The JSON files in `runs/` stay the full record.
- **Backfill** runs already on disk:

  ```bash
  BQ_DATASET=YOUR_PROJECT.voice_qa_gym python scripts/export_runs_to_bigquery.py --all
  ```

Example queries:

```sql
-- Which test calls fail most often?
SELECT scenario_id, COUNTIF(task_success) / COUNT(*) AS pass_rate, COUNT(*) AS calls
FROM `YOUR_PROJECT.voice_qa_gym.calls`
WHERE error IS NULL
GROUP BY scenario_id ORDER BY pass_rate;

-- Did coaching help? Score per round of each Auto-improve run.
SELECT run_id, iteration, outcome, train_reward, holdout_reward, policy_adherence, voice
FROM `YOUR_PROJECT.voice_qa_gym.rounds`
WHERE run_kind = 'optimize'
ORDER BY run_id, iteration;
```

---

## Part 7. Deploy on Cloud Run

The app runs as one private Cloud Run service. You open it through `gcloud run services proxy`,
and results live in a Cloud Storage bucket so they survive restarts.

```bash
PROJECT=YOUR_PROJECT; REGION=us-central1; BUCKET=$PROJECT-voice-qa-gym
SA=voice-qa-gym@$PROJECT.iam.gserviceaccount.com

# 1. APIs, bucket and a service account for the app
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  ces.googleapis.com texttospeech.googleapis.com aiplatform.googleapis.com --project $PROJECT
gcloud storage buckets create gs://$BUCKET --location $REGION --project $PROJECT
gcloud iam service-accounts create voice-qa-gym --project $PROJECT

# 2. Permissions
#    ces.agentEditor: read and update agents, run text and audio sessions
#    aiplatform.user: Gemini caller, scorer and coach
#    serviceusage.serviceUsageConsumer: needed by the audio pre-check
for ROLE in roles/ces.agentEditor roles/aiplatform.user roles/serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA --role $ROLE
done
gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member serviceAccount:$SA --role roles/storage.objectUser
# Only if you'll use the BigQuery export:
#   gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA --role roles/bigquery.dataEditor
#   gcloud projects add-iam-policy-binding $PROJECT --member serviceAccount:$SA --role roles/bigquery.jobUser

# 3. Build and deploy from the repo root (Cloud Build uses the Dockerfile)
gcloud run deploy voice-qa-gym --source . --region $REGION --project $PROJECT \
  --service-account $SA --no-allow-unauthenticated \
  --min-instances 1 --max-instances 1 --no-cpu-throttling --memory 1Gi \
  --execution-environment gen2 \
  --add-volume name=data,type=cloud-storage,bucket=$BUCKET \
  --add-volume-mount volume=data,mount-path=/data \
  --set-env-vars SUITE_PATH=/data/suite.yaml,RUNS_DIR=/data/runs,MAX_PARALLEL_CALLS=2
#   add ,BQ_DATASET=$PROJECT.voice_qa_gym to the env vars to turn on the BigQuery export

# 4. Open it (signs you in with your own Google account)
gcloud run services proxy voice-qa-gym --region $REGION --project $PROJECT --port 8080
# then open http://localhost:8080
```

**Why these settings:**

| Setting | Reason |
|---|---|
| `--no-allow-unauthenticated` | The app has no login of its own and can rewrite agents. Only people you give `roles/run.invoker` can reach it |
| `--min-instances 1 --max-instances 1` | Live progress and the one-run-at-a-time lock are in memory |
| `--no-cpu-throttling` | Tests run in the background after the page's request returns |
| Bucket at `/data` | The suite, run history and recordings survive restarts and redeploys |
| `MAX_PARALLEL_CALLS=2` | Two calls at once. Use 1 if the token quota hasn't been raised |

**To give a teammate access:**

```bash
gcloud run services add-iam-policy-binding voice-qa-gym --region $REGION --project $PROJECT \
  --member user:teammate@example.com --role roles/run.invoker
```

If the agent lives in a different project, grant the service account `roles/ces.agentEditor`
and `roles/aiplatform.user` in **that** project.

---

## Part 8. Clean up

- **Delete the demo agent's app** in the CX Agent Studio console, or:

  ```bash
  python -c "from cxas_scrapi import Apps; Apps(project_id='YOUR_PROJECT', location='us').delete_app('projects/YOUR_PROJECT/locations/us/apps/voice-qa-gym-demo', force=True)"
  ```

- **Delete the Cloud Run service** and bucket, if deployed:

  ```bash
  gcloud run services delete voice-qa-gym --region $REGION --project $PROJECT
  ```
