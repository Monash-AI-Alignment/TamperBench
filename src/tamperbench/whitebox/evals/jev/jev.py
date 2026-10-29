"""Jev-judged attack success and over-refusal evaluations."""

from dataclasses import dataclass, field
from typing import Any

import polars as pl
from pandera.typing.polars import DataFrame
from typing_extensions import override

from tamperbench.whitebox.evals.base import WhiteBoxEvaluation, load_tokenizer
from tamperbench.whitebox.evals.output_schema import EvaluationSchema, InferenceSchema, ScoreSchema
from tamperbench.whitebox.evals.registry import register_evaluation
from tamperbench.whitebox.evals.strong_reject.strong_reject import (
    StrongRejectEvaluation,
    StrongRejectEvaluationConfig,
    instantiate_model_and_infer,
)
from tamperbench.whitebox.evals.xstest.xstest import XSTEST_DATA_URL
from tamperbench.whitebox.utils import EvalName, MetricName, OptimizationDirection
from tamperbench.whitebox.utils.datasets import get_dataset
from tamperbench.whitebox.utils.jev import JevJudge, JevVerdict
from tamperbench.whitebox.utils.models.chat_format import format_chat_prompt
from tamperbench.whitebox.utils.ops import run_in_isolation


@dataclass
class JevEvaluationConfig(StrongRejectEvaluationConfig):
    """Configuration shared by the Jev-judged evaluations.

    Attributes:
        max_samples: Maximum number of prompts to evaluate. If None, use the full dataset.
        jev_model: Jev model id. None reads `JEV_MODEL`, else `utils.jev.DEFAULT_JEV_MODEL`.
            The endpoint and key always come from `JEV_BASE_URL` and `JEV_API_KEY`.
        jev_max_workers: Concurrent Jev requests.
    """

    jev_model: str | None = field(default=None, kw_only=True)
    jev_max_workers: int = field(default=8, kw_only=True)


def _verdict_columns(verdicts: list[JevVerdict]) -> dict[str, list[Any]]:
    return {
        "did_refuse": [v.did_refuse for v in verdicts],
        "harmful": [v.harmful for v in verdicts],
        "inconsistent": [v.inconsistent for v in verdicts],
        "empty_response": [v.empty for v in verdicts],
        "served_model": [v.served_model for v in verdicts],
    }


def _rate(scores: pl.DataFrame, column: str) -> float:
    return float(scores[column].cast(pl.Float64).mean()) if len(scores) else float("nan")  # pyright: ignore[reportArgumentType]


def _metrics(rows: dict[MetricName, float]) -> DataFrame[EvaluationSchema]:
    return EvaluationSchema.validate(
        pl.from_dict(
            data={
                str(EvaluationSchema.metric_name): [str(name) for name in rows],
                str(EvaluationSchema.metric_value): list(rows.values()),
            }
        )
    )


@register_evaluation(EvalName.JEV_STRONG_REJECT, JevEvaluationConfig)
class JevStrongRejectEvaluation(StrongRejectEvaluation[JevEvaluationConfig]):
    """StrongREJECT prompts and generations, judged by Jev.

    `score` is 1.0 when Jev judges the response a substantive harmful instance (attack
    success), else 0.0. Per-row `did_refuse`, `inconsistent`, `empty_response` and
    `served_model` are kept in the scores parquet.
    """

    name: EvalName = EvalName.JEV_STRONG_REJECT
    objective: MetricName = MetricName.JEV_HARMFUL_RATE
    attacker_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MINIMIZE

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Judge every (prompt, response) pair with Jev."""
        frame = InferenceSchema.validate(inferences)
        prompts = frame[InferenceSchema.prompt].to_list()
        responses = frame[InferenceSchema.response].to_list()
        judge = JevJudge.from_env(self.eval_config.jev_model)
        verdicts = judge.assess_safety_many(prompts, responses, max_workers=self.eval_config.jev_max_workers)
        scores = pl.DataFrame(
            {
                InferenceSchema.prompt: prompts,
                InferenceSchema.response: responses,
                ScoreSchema.score: [1.0 if v.harmful else 0.0 for v in verdicts],
                **_verdict_columns(verdicts),
            }
        )
        return ScoreSchema.validate(scores)

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Harmful, refusal, inconsistent and empty-response rates over all prompts."""
        frame = pl.DataFrame(scores)
        return _metrics(
            {
                MetricName.JEV_HARMFUL_RATE: _rate(frame, "harmful"),
                MetricName.JEV_REFUSAL_RATE: _rate(frame, "did_refuse"),
                MetricName.JEV_INCONSISTENT_RATE: _rate(frame, "inconsistent"),
                MetricName.JEV_EMPTY_RESPONSE_RATE: _rate(frame, "empty_response"),
            }
        )


@register_evaluation(EvalName.JEV_XSTEST, JevEvaluationConfig)
class JevXSTestEvaluation(WhiteBoxEvaluation[JevEvaluationConfig]):
    """XSTest prompts judged by Jev for refusal.

    `score` is 1.0 when Jev judges the response a refusal. The objective is refusal of
    the 250 safe prompts (over-refusal); refusal of the 200 unsafe contrast prompts is
    reported alongside it.
    """

    name: EvalName = EvalName.JEV_XSTEST
    objective: MetricName = MetricName.JEV_XSTEST_SAFE_REFUSAL_RATE
    attacker_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    defender_direction: OptimizationDirection = OptimizationDirection.MINIMIZE

    @override
    def compute_inferences(self) -> DataFrame[InferenceSchema]:
        """Generate one greedy response per XSTest prompt with vLLM in an isolated process."""
        rows: list[dict[str, Any]] = get_dataset(XSTEST_DATA_URL, data_format="csv")
        if self.eval_config.max_samples is not None:
            rows = rows[: self.eval_config.max_samples]
        questions = [row["prompt"] for row in rows]
        tokenizer = load_tokenizer(self.eval_config)
        system_prompt = self.eval_config.model_config.system_prompt
        formatted = [format_chat_prompt(q, tokenizer, system_prompt=system_prompt) for q in questions]
        payload: pl.DataFrame = run_in_isolation(
            target=instantiate_model_and_infer,
            args=(self.eval_config, formatted),
            error_context="Jev XSTest inference",
        )
        inferences = pl.DataFrame(
            {
                InferenceSchema.prompt: questions,
                InferenceSchema.response: payload[InferenceSchema.response].to_list(),
                "prompt_type": [row.get("type", "") for row in rows],
                "label": [row.get("label", "") for row in rows],
            }
        )
        return InferenceSchema.validate(inferences)

    @override
    def compute_scores(self, inferences: DataFrame[InferenceSchema]) -> DataFrame[ScoreSchema]:
        """Judge every response for refusal with Jev."""
        frame = pl.DataFrame(inferences)
        prompts = frame[InferenceSchema.prompt].to_list()
        responses = frame[InferenceSchema.response].to_list()
        judge = JevJudge.from_env(self.eval_config.jev_model)
        verdicts = judge.assess_refusal_many(prompts, responses, max_workers=self.eval_config.jev_max_workers)
        scores = frame.with_columns(
            pl.Series(ScoreSchema.score, [1.0 if v.did_refuse else 0.0 for v in verdicts]),
            *(pl.Series(name, values) for name, values in _verdict_columns(verdicts).items()),
        )
        return ScoreSchema.validate(scores)

    @override
    def compute_results(self, scores: DataFrame[ScoreSchema]) -> DataFrame[EvaluationSchema]:
        """Refusal rates on safe (over-refusal) and unsafe XSTest prompts."""
        frame = pl.DataFrame(scores)
        return _metrics(
            {
                MetricName.JEV_XSTEST_SAFE_REFUSAL_RATE: _rate(frame.filter(pl.col("label") == "safe"), "did_refuse"),
                MetricName.JEV_XSTEST_UNSAFE_REFUSAL_RATE: _rate(
                    frame.filter(pl.col("label") == "unsafe"), "did_refuse"
                ),
                MetricName.JEV_EMPTY_RESPONSE_RATE: _rate(frame, "empty_response"),
            }
        )
