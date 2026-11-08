#!/usr/bin/env bash
# TAR (refusal/DPO) from Meta-Llama-3-8B-Instruct on one full Virga node (4 x H100) with
# FSDP parameter offload. Submit with ./experiments/dc-tar/submit.sh {smoke|full}.
# Directives below are the full run; submit.sh overrides partition and time for the smoke.
#SBATCH --job-name=dc-tar
#SBATCH --account=sc-001191
#SBATCH --partition=gpu
#SBATCH --time=4-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --mail-user=sean.murphy@monash.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,TIME_LIMIT_90
#SBATCH --output=/home/mur458/slurm-logs/%x-%j.out
#SBATCH --error=/home/mur458/slurm-logs/%x-%j.err

set -euo pipefail

: "${SLURM_JOB_ID:?must run inside a Slurm allocation, not the login node}"
MODE="${1:?usage: run.sbatch smoke|full}"
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$ROOT"

PROJECT_SLUG="$(basename "$ROOT")"
EXPERIMENT_SLUG="dc-tar"
RUN_DIR="/scratch3/mur458/${EXPERIMENT_SLUG}/${MODE}-${SLURM_JOB_ID}"

export HF_HOME=/scratch3/mur458/hf
export TORCH_HOME=/scratch3/mur458/cache/torch
export TRITON_CACHE_DIR=/scratch3/mur458/cache/triton
export CUDA_CACHE_PATH=/scratch3/mur458/cache/nv
export PIP_CACHE_DIR=/scratch3/mur458/cache/pip
export MPLCONFIGDIR=/scratch3/mur458/cache/mpl
export UV_CACHE_DIR=/scratch3/mur458/.uv/cache
export UV_PYTHON_INSTALL_DIR=/scratch3/mur458/.uv/python
export UV_PROJECT_ENVIRONMENT="/scratch3/mur458/envs/${PROJECT_SLUG}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$RUN_DIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" \
  "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PROJECT_ENVIRONMENT"

# Meta-Llama-3-8B-Instruct is gated. Login may store the token outside $HF_HOME.
HF_TOKEN="${HF_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"
: "${HF_TOKEN:?no Hugging Face token: run huggingface-cli login on virga-login}"
export HF_TOKEN
printf 'HF_TOKEN bytes=%s\n' "${#HF_TOKEN}"

: "${CUDA_VISIBLE_DEVICES:?GPU allocation did not expose CUDA devices}"
printf 'job=%s mode=%s host=%s commit=%s\n' "$SLURM_JOB_ID" "$MODE" "$(hostname)" "$(git rev-parse HEAD)"
printf 'run_dir=%s CUDA_VISIBLE_DEVICES=%s\n' "$RUN_DIR" "$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
free -g | head -2

UV=/home/mur458/.local/bin/uv
test -x "$UV"; test -f pyproject.toml; test -f uv.lock
"$UV" sync --frozen

case "$MODE" in
  smoke) STEPS=(--max-steps 2 --post-tar-sft-steps 2) ;;
  full) STEPS=() ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
"$UV" run --frozen python -u experiments/dc-tar/train_tar.py --output "$RUN_DIR" "${STEPS[@]}"

# Success: a loadable merged checkpoint.
test -s "$RUN_DIR/model/config.json"
ls "$RUN_DIR"/model/*.safetensors >/dev/null

RESULT_DIR="$ROOT/results/${EXPERIMENT_SLUG}/${MODE}"
mkdir -p "$RESULT_DIR"
cp "$RUN_DIR/settings.json" "$RESULT_DIR/"
(cd "$RUN_DIR/model" && ls -l) > "$RESULT_DIR/model_files.txt"
grep -E "Outer Training Loop|retain loss|saved to" "/home/mur458/slurm-logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" \
  | tr '\r' '\n' | grep -E "it/s|s/it|saved to" | tail -n 200 > "$RESULT_DIR/progress.txt" || true
printf 'raw outputs on Virga scratch: %s (job %s)\n' "$RUN_DIR" "$SLURM_JOB_ID" > "$RESULT_DIR/SOURCE.md"
printf 'completed job=%s output=%s returned=%s\n' "$SLURM_JOB_ID" "$RUN_DIR" "$RESULT_DIR"
