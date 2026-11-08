"""TAR (refusal/DPO variant) from Meta-Llama-3-8B-Instruct @ 8afb486 on 4 GPUs with FSDP parameter offload.

Settings are the base of `configs/whitebox/defenses/tar/grid.yaml` (paper section 5.2),
with 8 GPUs x 8 accumulation swapped for 4 GPUs x 16 so the effective batch stays 64,
as in the defense-comparison session (research hub `2026-11-04--defense-comparison--bold-ibis`).
Offload only moves parameters, gradients and optimiser state to CPU RAM; the computation
is unchanged. On M3, 4 x A100-80G without offload ran out of memory at outer step 2.

`--max-steps` / `--post-tar-sft-steps` shorten the run for the smoke test only.
"""

import argparse
import json
from dataclasses import fields
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from tamperbench.whitebox.defenses.tar.tar import TARConfig, TARDefense

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
HERE = Path(__file__).resolve().parent


def main() -> None:
    """Download the pinned base model and run TAR with the offload accelerate config."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--post-tar-sft-steps", type=int, default=None)
    args = parser.parse_args()

    model = snapshot_download(MODEL, revision=REVISION)
    grid = HERE.parents[1] / "configs/whitebox/defenses/tar/grid.yaml"
    base = yaml.safe_load(grid.read_text())["base"]
    allowed = {f.name for f in fields(TARConfig) if f.init}
    settings = {k: v for k, v in base.items() if k in allowed}
    settings.update(
        input_checkpoint_path=model,
        output_checkpoint_path=str(args.output / "model"),
        retain_model_name=model,
        num_gpus=4,
        gradient_accumulation_steps=16,
        accel_config_path=str(HERE / "accel_config_4_gpu_offload.yaml"),
    )
    if args.max_steps is not None:
        settings["max_steps"] = args.max_steps
    if args.post_tar_sft_steps is not None:
        settings["post_tar_sft_steps"] = args.post_tar_sft_steps
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2))
    print(f"TAR settings: {settings}", flush=True)
    print(f"saved to {TARDefense(TARConfig.from_dict(settings)).run_defense()}", flush=True)


if __name__ == "__main__":
    main()
