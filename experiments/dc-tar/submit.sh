#!/usr/bin/env bash
# Run ON virga-login: ./experiments/dc-tar/submit.sh smoke   (2 outer steps, h24gpu, 12 h)
#                     ./experiments/dc-tar/submit.sh full    (100 outer steps, gpu, 4 days)
set -euo pipefail

MODE="${1:?usage: submit.sh smoke|full}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
EXPERIMENT_SLUG="$(basename "$SCRIPT_DIR")"
SBATCH=/cm/shared/apps/slurm/current/bin/sbatch

case "$MODE" in
  # 2 outer steps with offload exceeded h2gpu's 2 h cap (job 37574972, TIMEOUT).
  smoke) OVERRIDES=(--partition=h24gpu --time=12:00:00 --job-name=dc-tar-smoke) ;;
  full) OVERRIDES=() ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

mkdir -p /home/mur458/slurm-logs
cd "$ROOT"
JOB_ID="$("$SBATCH" --parsable --chdir="$ROOT" --export=ALL "${OVERRIDES[@]}" "$SCRIPT_DIR/run.sbatch" "$MODE")"
printf '%s\n' "$JOB_ID" | tee "/home/mur458/slurm-logs/${EXPERIMENT_SLUG}-${MODE}-last-jobid"
