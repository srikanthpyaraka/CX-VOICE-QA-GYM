#!/usr/bin/env bash
# Deploy Voice QA Gym to Cloud Run as a private service. Safe to re-run.
#
#   PROJECT=my-project ./scripts/deploy_cloud_run.sh
#
# Optional settings (environment variables):
#   REGION=us-central1           Cloud Run region
#   SERVICE=voice-qa-gym         service and service-account name
#   BUCKET=<project>-voice-qa-gym  bucket that keeps the suite, run history and recordings
#   AGENT_PROJECT=<project>      project that holds the CX Agent Studio agents, if different
#   MAX_PARALLEL_CALLS=2         calls at once (1 if the agent token quota isn't raised)
#   BQ_DATASET=<project>.voice_qa_gym  turn on the BigQuery export
#   MIRO_SECRET=miro-token       Secret Manager secret holding a Miro API token (scenario generator)
#   INVOKERS="user:a@x.com,group:qa@x.com"  who may open the app (you can always add more later)
set -euo pipefail

: "${PROJECT:?Set PROJECT=your-gcp-project}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-voice-qa-gym}"
BUCKET="${BUCKET:-$PROJECT-voice-qa-gym}"
AGENT_PROJECT="${AGENT_PROJECT:-$PROJECT}"
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-2}"
SA="$SERVICE@$PROJECT.iam.gserviceaccount.com"
cd "$(dirname "$0")/.."

step() { printf '\n== %s\n' "$*"; }
bind() {  # bind PROJECT ROLE
  gcloud projects add-iam-policy-binding "$1" --member "serviceAccount:$SA" --role "$2" \
    --condition=None --quiet >/dev/null
}

step "Enabling APIs in $PROJECT"
apis=(run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
      ces.googleapis.com texttospeech.googleapis.com aiplatform.googleapis.com)
[[ -n "${BQ_DATASET:-}" ]] && apis+=(bigquery.googleapis.com)
[[ -n "${MIRO_SECRET:-}" ]] && apis+=(secretmanager.googleapis.com)
gcloud services enable "${apis[@]}" --project "$PROJECT"
[[ "$AGENT_PROJECT" != "$PROJECT" ]] && gcloud services enable ces.googleapis.com aiplatform.googleapis.com \
  texttospeech.googleapis.com --project "$AGENT_PROJECT"

step "Bucket gs://$BUCKET"
gcloud storage buckets describe "gs://$BUCKET" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud storage buckets create "gs://$BUCKET" --location "$REGION" --project "$PROJECT" --uniform-bucket-level-access

step "Service account $SA"
gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "$SERVICE" --project "$PROJECT" --display-name "Voice QA Gym"

step "Permissions"
# ces.agentEditor: read/update agents, run text and audio sessions. aiplatform.user: Gemini.
# serviceUsageConsumer: the audio pre-check reads which APIs are enabled.
for role in roles/ces.agentEditor roles/aiplatform.user roles/serviceusage.serviceUsageConsumer; do
  bind "$AGENT_PROJECT" "$role"
done
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" --member "serviceAccount:$SA" \
  --role roles/storage.objectUser --project "$PROJECT" >/dev/null
env_vars="SUITE_PATH=/data/suite.yaml,RUNS_DIR=/data/runs,MAX_PARALLEL_CALLS=$MAX_PARALLEL_CALLS"
secret_args=()
if [[ -n "${BQ_DATASET:-}" ]]; then
  bind "${BQ_DATASET%%.*}" roles/bigquery.dataEditor
  bind "${BQ_DATASET%%.*}" roles/bigquery.jobUser
  env_vars+=",BQ_DATASET=$BQ_DATASET"
fi
if [[ -n "${MIRO_SECRET:-}" ]]; then
  gcloud secrets add-iam-policy-binding "$MIRO_SECRET" --project "$PROJECT" \
    --member "serviceAccount:$SA" --role roles/secretmanager.secretAccessor >/dev/null
  secret_args=(--set-secrets "MIRO_TOKEN=$MIRO_SECRET:latest")
fi

step "Building and deploying $SERVICE (Cloud Build, a few minutes)"
# One instance with CPU always on: runs execute in the background and the queue lives in memory.
gcloud run deploy "$SERVICE" --source . --region "$REGION" --project "$PROJECT" --quiet \
  --service-account "$SA" --no-allow-unauthenticated \
  --min-instances 1 --max-instances 1 --no-cpu-throttling --memory 1Gi --timeout 3600 \
  --execution-environment gen2 \
  --add-volume "name=data,type=cloud-storage,bucket=$BUCKET" --add-volume-mount "volume=data,mount-path=/data" \
  --set-env-vars "$env_vars" ${secret_args[@]+"${secret_args[@]}"}  # empty-array safe on macOS bash 3.2

if [[ -n "${INVOKERS:-}" ]]; then
  step "Access"
  IFS=, read -ra members <<<"$INVOKERS"
  for m in "${members[@]}"; do
    gcloud run services add-iam-policy-binding "$SERVICE" --region "$REGION" --project "$PROJECT" \
      --member "$m" --role roles/run.invoker >/dev/null
    echo "  $m can open the app"
  done
fi

cat <<EOF

Deployed. Open it with your own Google account:
  gcloud run services proxy $SERVICE --region $REGION --project $PROJECT --port 8080
  then http://localhost:8080   (Connect: project $AGENT_PROJECT, location us, your app ID)
EOF
