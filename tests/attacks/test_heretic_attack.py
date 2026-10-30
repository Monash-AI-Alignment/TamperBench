"""GPU sanity check for the Heretic attack and the Jev-judged evals.

Needs a GPU, `HERETIC_BIN` (scripts/whitebox/setup_heretic.sh) and `JEV_API_KEY`.

1. Search a short Heretic study on a small instruct model and export Pareto trial 0.
2. Export trial 0 again through the cached study's restore path; the weights must match
   the search export, which proves the restore re-creates the same trial.
3. Score the base and attacked models with `jev_strong_reject` and `jev_xstest` on a
   prompt subset; the attack must raise the judged harmful rate.

    uv run tests/attacks/test_heretic_attack.py [model]   # default Qwen/Qwen2.5-1.5B-Instruct
"""

import sys
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv
from safetensors.torch import load_file
from transformers import AutoConfig

from tamperbench.whitebox.attacks.heretic.heretic import HereticAttack, HereticConfig
from tamperbench.whitebox.evals.jev.jev import JevEvaluationConfig, JevStrongRejectEvaluation, JevXSTestEvaluation
from tamperbench.whitebox.evals.output_schema import EvaluationSchema
from tamperbench.whitebox.utils.models.config import ModelConfig
from tamperbench.whitebox.utils.names import MetricName, TemplateName

MAX_SAMPLES = 40


def _max_weight_difference(left: Path, right: Path) -> float:
    shards = sorted(p.name for p in left.glob("*.safetensors"))
    assert shards == sorted(p.name for p in right.glob("*.safetensors")), "exports have different shards"
    worst = 0.0
    for shard in shards:
        a, b = load_file(str(left / shard)), load_file(str(right / shard))
        assert a.keys() == b.keys(), f"{shard}: different tensors"
        for key in a:
            worst = max(worst, float((a[key].float() - b[key].float()).abs().max()))
    return worst


def _jev_metrics(checkpoint: str, out_dir: Path, model_config: ModelConfig) -> dict[str, float]:
    config = JevEvaluationConfig(
        model_checkpoint=checkpoint, out_dir=str(out_dir), model_config=model_config, max_samples=MAX_SAMPLES
    )
    metrics: dict[str, float] = {}
    for evaluation in (JevStrongRejectEvaluation(config), JevXSTestEvaluation(config)):
        rows = evaluation.run_evaluation().rows_by_key(key=EvaluationSchema.metric_name, unique=True)
        metrics.update({name: value[0] for name, value in rows.items()})
    return metrics


if __name__ == "__main__":
    load_dotenv()
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
    model_config = ModelConfig(template_name=TemplateName.NATIVE, max_generation_length=512, inference_batch_size=16)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        def attack(out: str) -> HereticAttack:
            """Build the same short attack writing to `out`; all runs share one study cache."""
            return HereticAttack(
                HereticConfig(
                    input_checkpoint_path=model,
                    out_dir=str(root / out),
                    evals=[],
                    model_config=model_config,
                    random_seed=1234,
                    trial_index=0,
                    n_trials=12,
                    n_startup_trials=6,
                    study_cache_dir=str(root / "study_cache"),
                )
            )

        searched = attack("searched")
        searched.benchmark()
        restored = attack("restored")
        restored.benchmark()
        assert (root / "searched" / "heretic" / "search.log").exists(), "first run did not search"
        assert (root / "restored" / "heretic" / "restore.log").exists(), "second run did not reuse the cached study"
        difference = _max_weight_difference(
            Path(searched.output_checkpoint_path), Path(restored.output_checkpoint_path)
        )
        print(f"max |search export - restore export| = {difference:.3g}")
        assert difference < 1e-2, "restore did not re-create the searched trial"
        # Abliteration changes weights only: the export must load with the input's architecture config.
        # (A transformers-5 config.json once made 4.x read Llama's RoPE base as 10,000 instead of 500,000.)
        source_config = AutoConfig.from_pretrained(model).to_diff_dict()
        export_config = AutoConfig.from_pretrained(searched.output_checkpoint_path).to_diff_dict()
        for ignored in ("_name_or_path", "transformers_version"):
            source_config.pop(ignored, None)
            export_config.pop(ignored, None)
        assert export_config == source_config, f"export config differs: {export_config} vs {source_config}"

        base = _jev_metrics(model, root / "base_eval", model_config)
        attacked = _jev_metrics(searched.output_checkpoint_path, root / "searched", model_config)
        torch.cuda.empty_cache()
        print("base:", base)
        print("attacked:", attacked)
        assert attacked[MetricName.JEV_HARMFUL_RATE] > base[MetricName.JEV_HARMFUL_RATE], "attack did not raise ASR"
