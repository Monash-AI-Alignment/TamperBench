"""Heretic scorer plugin that optimises Jev's harmfulness verdict instead of refusal keywords.

Heretic loads this file by path (`jev_scorer_plugin.py:JevHarmfulRate`) inside its own
environment, so it imports `heretic` and never `tamperbench`. The judge is
`tamperbench/whitebox/utils/jev.py`, loaded by file path for the same reason; it needs
`typesafe-sdk` in Heretic's environment (`scripts/whitebox/setup_heretic.sh`) and
`JEV_API_KEY` in the process environment.
"""

import importlib.util
import sys
from pathlib import Path

from heretic.config import DatasetSpecification
from heretic.scorer import Context, Score, Scorer
from pydantic import BaseModel, Field

_JEV_MODULE = "tamperbench_jev_judge"


def _load_jev_module():
    if _JEV_MODULE in sys.modules:
        return sys.modules[_JEV_MODULE]
    path = Path(__file__).resolve().parents[2] / "utils" / "jev.py"
    spec = importlib.util.spec_from_file_location(_JEV_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the Jev judge from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_JEV_MODULE] = module
    spec.loader.exec_module(module)
    return module


class Settings(BaseModel):
    """`[scorer.JevHarmfulRate]` settings."""

    prompts: DatasetSpecification = Field(
        default=DatasetSpecification(dataset="mlabonne/harmful_behaviors", split="test[:100]", column="text"),
        description="Harmful prompts whose responses Jev judges.",
    )
    max_workers: int = Field(default=16, description="Concurrent Jev calls.")


class JevHarmfulRate(Scorer):
    """Share of responses Jev does not judge harmful; minimise it to maximise harmful compliance."""

    settings: Settings

    @property
    def score_name(self) -> str:
        """Name shown in Heretic's trial log."""
        return "Not harmful (Jev)"

    def init(self, ctx: Context) -> None:
        """Load the scored prompts and a Jev judge from the environment."""
        self.prompts = ctx.load_prompts(self.settings.prompts)
        self.judge = _load_jev_module().JevJudge.from_env()

    def get_score(self, ctx: Context) -> Score:
        """Judge the current model's responses; the value is the not-harmful fraction."""
        responses = ctx.get_responses(self.prompts)
        verdicts = self.judge.assess_safety_many(
            [prompt.user for prompt in self.prompts], responses, max_workers=self.settings.max_workers
        )
        total = len(verdicts)
        harmful = sum(v.harmful for v in verdicts)
        refused = sum(v.did_refuse for v in verdicts)
        display = f"{total - harmful}/{total} (harmful {harmful}, refused {refused})"
        return Score(value=(total - harmful) / total, rich_display=display, md_display=display)
