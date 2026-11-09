# dc-tar: TAR from Llama-3-8B-Instruct on Virga

Trains TAR (refusal/DPO variant, paper section 5.2 settings from
`configs/whitebox/defenses/tar/grid.yaml`) from `meta-llama/Meta-Llama-3-8B-Instruct` @
`8afb486`, for the research hub's defense-comparison session
(`agent_logs/2026-11-04--defense-comparison--bold-ibis`). The trained model is then attacked
and evaluated on M3.

Why Virga: on M3, 4 x A100-80G without offload ran out of memory at outer step 2, and the
run would also have exceeded M3's 1-day limit. Here it runs on one full node (4 x H100) with
FSDP parameter offload (`accel_config_4_gpu_offload.yaml`, identical to the stock 4-GPU config
except `fsdp_offload_params: true`). Offload moves parameters, gradients and optimiser state
to CPU RAM; it does not change the computation. Effective batch is 64 (4 GPUs x 16
accumulation x batch 1), as on M3.

## Branch

`experiments/dc-tar-virga` (Monash-AI-Alignment/TamperBench), based on `feature/defense-baselines`.

## Run (on virga-login)

```bash
cd /home/mur458/projects/tamperbench   # clone git@github.com:Monash-AI-Alignment/TamperBench.git first if absent
git fetch origin && git switch experiments/dc-tar-virga && git pull --ff-only
bash -n experiments/dc-tar/run.sbatch && bash -n experiments/dc-tar/submit.sh

# Compute nodes cannot reach PyPI or the HF LFS CDN, and the job runs with HF_HUB_OFFLINE=1:
# stage the venv, model and datasets on /scratch3 from the login node first.
export HF_HOME=/scratch3/mur458/hf UV_CACHE_DIR=/scratch3/mur458/.uv/cache \
  UV_PYTHON_INSTALL_DIR=/scratch3/mur458/.uv/python \
  UV_PROJECT_ENVIRONMENT="/scratch3/mur458/envs/$(basename "$PWD")"
~/.local/bin/uv sync --frozen
~/.local/bin/uv run --frozen python -c "
from datasets import load_dataset
from huggingface_hub import snapshot_download
snapshot_download('meta-llama/Meta-Llama-3-8B-Instruct', revision='8afb486c1db24fe5011ec46dfbe5b5dccdb575c2')
load_dataset('Unified-Language-Model-Alignment/Anthropic_HH_Golden')
load_dataset('lapisrocks/magpie-bio-filtered')"

./experiments/dc-tar/submit.sh smoke    # 2 outer steps + 2 SFT steps, h2gpu, <= 2 h
```

When the smoke finishes `COMPLETED`, submit the full run:

```bash
./experiments/dc-tar/submit.sh full     # 100 outer steps + 100 SFT steps, gpu partition, 4-day limit
```

The smoke runs outer step 2, where M3 ran out of memory, so it proves the memory fix. It also
gives the per-step time: the full run is about 100 x the "Outer Training Loop" s/it in the log.
If that is over ~90 h, report it before submitting the full run.

## Observe

```bash
JOB="$(cat /home/mur458/slurm-logs/dc-tar-<smoke|full>-last-jobid)"
sacct -j "$JOB" -X --format=JobID,State,ExitCode,Elapsed,MaxRSS
tail -n 40 /home/mur458/slurm-logs/dc-tar*-"$JOB".out
```

Success: `COMPLETED`, exit 0, and the script's final line `completed job=... output=...`
(it checks `model/config.json` and the safetensors exist).

## Requirements

- Env var names only: `HF_TOKEN` (or a token from `huggingface-cli login`); the account
  must have accepted the Meta-Llama-3 licence. Data (`Unified-Language-Model-Alignment/Anthropic_HH_Golden`,
  Magpie) downloads from the Hub into `/scratch3/mur458/hf`.
- Resources: one exclusive node (`--gres=gpu:4 --exclusive --mem=0`), because offload holds
  the policy model, the retain model and both optimisers' state in CPU RAM.

## Allowed operator fixes vs stop

Fix freely with a `[virga-autofix]` commit: account, partition, time, memory, CPU flags, cache
paths, a missing declared dependency. Stop and report instead of changing batch size,
accumulation, steps, learning rates, model, data, dtype or the accelerate config's sharding.

## Results

Small files return on this branch under `results/dc-tar/<mode>/` (settings, model file list,
progress lines, `SOURCE.md`); commit and push them. The model itself (~16 GB, under
`/scratch3/mur458/dc-tar/full-<job>/model/`) goes to M3 for attack and evaluation:

```bash
rsync -a --info=progress2 /scratch3/mur458/dc-tar/full-<job>/model/ \
  smur0075@m3.massive.org.au:/scratch/ax74/smur0075/hub-compute/out/dc/train/tar/
```
