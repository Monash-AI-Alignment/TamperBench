"""Heretic abliteration attack, run from its own environment as a subprocess."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from typing_extensions import override

from tamperbench.whitebox.attacks.base import TamperAttack, TamperAttackConfig
from tamperbench.whitebox.attacks.registry import register_attack
from tamperbench.whitebox.utils.heretic import (
    HERETIC_COMMIT,
    HereticRunError,
    HereticSearch,
    journal_path,
    pareto_candidates,
    reserialize_tokenizer,
    resolve_heretic_bin,
    restore_command,
    run_heretic,
    study_is_finished,
    study_lock,
    write_restore_journal,
)
from tamperbench.whitebox.utils.names import AttackName

HERETIC_SUMMARY_FILE = "heretic_summary.json"


@dataclass
class HereticConfig(TamperAttackConfig):
    """Configuration for the Heretic attack.

    `random_seed` seeds Heretic's search. The study (the expensive part) is cached per
    input checkpoint and search settings, so configs that differ only in
    `trial_index` share one search and each export a different Pareto trial.

    Attributes:
        trial_index: Position on Heretic's sorted Pareto front to export. 0 is the
            fewest keyword refusals, tie-broken by lowest KL.
        n_trials: Total Optuna trials in Heretic's search.
        n_startup_trials: Random trials before TPE.
        response_prefix: Forced response prefix; None lets Heretic detect it.
        heretic_bin: Path to the `heretic` executable; None reads `$HERETIC_BIN`.
        study_cache_dir: Root of the cached Heretic studies.
    """

    trial_index: int = 0
    n_trials: int = 200
    n_startup_trials: int = 60
    response_prefix: str | None = None
    heretic_bin: str | None = None
    study_cache_dir: str = field(default_factory=lambda: str(Path.home() / ".cache" / "tamperbench" / "heretic"))


@register_attack(AttackName.HERETIC, HereticConfig)
class HereticAttack(TamperAttack[HereticConfig]):
    """Search with Heretic (or reuse a cached search) and export one Pareto trial."""

    name: AttackName = AttackName.HERETIC

    @override
    def run_attack(self) -> None:
        """Leave the merged export of Pareto trial `trial_index` at `output_checkpoint_path`.

        Raises:
            HereticRunError: If Heretic fails (including failures it hides behind exit
                code 0), `trial_index` is off the Pareto front, or no model is saved.
        """
        config = self.attack_config
        if config.trial_index < 0:
            raise ValueError(f"trial_index must be >= 0, got {config.trial_index}")
        heretic_bin = resolve_heretic_bin(config.heretic_bin)
        search = HereticSearch(
            model=config.input_checkpoint_path,
            seed=config.random_seed,
            n_trials=config.n_trials,
            n_startup_trials=config.n_startup_trials,
            response_prefix=config.response_prefix,
        )
        study_dir = Path(config.study_cache_dir).expanduser() / search.cache_key(heretic_bin)
        work_dir = Path(config.out_dir) / "heretic"
        save_dir = Path(self.output_checkpoint_path)
        journal = journal_path(study_dir, search.model)
        commands: list[list[str]] = []

        with study_lock(study_dir):
            exported = False
            if not study_is_finished(journal):
                journal.unlink(missing_ok=True)  # an unfinished study is restarted, never resumed
                command = search.search_command(heretic_bin, study_dir, config.trial_index, save_dir)
                commands.append(command)
                run_heretic(command, cwd=work_dir / "search_cwd", log_path=work_dir / "search.log")
                exported = True
            candidates = pareto_candidates(journal)

        if config.trial_index >= len(candidates):
            raise HereticRunError(
                f"trial_index {config.trial_index} is off the Pareto front ({len(candidates)} trials) in {journal}"
            )
        if not exported:
            restore_dir = work_dir / "restore_study"
            write_restore_journal(journal, restore_dir, search.model, config.trial_index, save_dir)
            command = restore_command(heretic_bin, search.model, restore_dir)
            commands.append(command)
            run_heretic(command, cwd=work_dir / "restore_cwd", log_path=work_dir / "restore.log")

        if not (save_dir / "config.json").exists():
            raise HereticRunError(f"heretic saved no config.json in {save_dir}; see logs in {work_dir}")
        reserialize_tokenizer(config.model_config.tokenizer_checkpoint or config.input_checkpoint_path, save_dir)

        summary = {
            "heretic_commit_expected": HERETIC_COMMIT,
            "heretic_bin": str(heretic_bin),
            "search": asdict(search),
            "study_dir": str(study_dir),
            "searched_in_this_run": exported,
            "trial_index": config.trial_index,
            "chosen": asdict(candidates[config.trial_index]),
            "pareto_front": [asdict(candidate) for candidate in candidates],
            "commands": commands,
        }
        (Path(config.out_dir) / HERETIC_SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
