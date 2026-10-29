"""Drive p-e-w/heretic headlessly from its own virtual environment.

Heretic pins a transformers major that conflicts with tamperbench's, so it runs as a
subprocess of a separate install (`scripts/whitebox/setup_heretic.sh`). This module
holds the pieces that do not need a GPU or vLLM: command lines, the study cache,
reading Heretic's Optuna journal, restoring an arbitrary Pareto trial, and guards for
Heretic's silent failures.

Pinned commit: 3521f8648a0dccf6e12a92666862632235fac7e6. PyPI `heretic-llm` 1.4.0
cannot run headlessly, and current master defaults to a different editing method
(ARA), so the journal layout and CLI flags here are those of the pinned commit.

Heretic exits 0 on several failures (configuration errors, a menu prompt with no
terminal, save errors caught and printed). `run_heretic` therefore checks the log
and the caller checks the saved checkpoint; the exit code alone means nothing.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from optuna.trial import TrialState
from transformers import AutoTokenizer

HERETIC_COMMIT = "3521f8648a0dccf6e12a92666862632235fac7e6"
HERETIC_BIN_ENV = "HERETIC_BIN"

_LOG_FAILURES = (
    re.compile(r"Configuration contains \d+ errors?"),
    re.compile(r"^\s*Error:", re.MULTILINE),
)


class HereticRunError(RuntimeError):
    """Heretic failed, including failures it reports with exit code 0."""


# Tokenizer files Heretic's environment writes into an export (transformers 5.x format).
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
)


def reserialize_tokenizer(tokenizer_source: str, save_dir: Path) -> None:
    """Replace the export's tokenizer files with `tokenizer_source`'s, saved by this environment.

    Heretic runs on transformers 5.x, whose tokenizer config (for example a list-valued
    `extra_special_tokens`) older transformers cannot load. Abliteration edits weights
    only, so the attacked model's tokenizer is the input checkpoint's; re-saving it here
    makes the export loadable by the same stack that evaluates it.

    Args:
        tokenizer_source: The input checkpoint (or its tokenizer checkpoint), HF id or path.
        save_dir: The Heretic export to fix in place.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    for name in _TOKENIZER_FILES:
        (save_dir / name).unlink(missing_ok=True)
    tokenizer.save_pretrained(str(save_dir))


def resolve_heretic_bin(configured: str | None) -> Path:
    """Return the `heretic` executable from the config or `$HERETIC_BIN`.

    Raises:
        HereticRunError: If neither is set or the path is not executable.
    """
    value = configured or os.environ.get(HERETIC_BIN_ENV)
    if not value:
        raise HereticRunError(
            f"Set `heretic_bin` or ${HERETIC_BIN_ENV} to a heretic {HERETIC_COMMIT[:7]} executable "
            "(see scripts/whitebox/setup_heretic.sh)"
        )
    path = Path(value).expanduser()
    if not os.access(path, os.X_OK):
        raise HereticRunError(f"heretic executable not found or not executable: {path}")
    return path


def journal_path(study_dir: Path, model: str) -> Path:
    """Heretic's journal file for `model`; restoring must pass the identical model string."""
    sanitized = "".join(c if (c.isalnum() or c in ["_", "-"]) else "--" for c in model)
    return study_dir / f"{sanitized}.jsonl"


def _checkpoint_fingerprint(model: str) -> dict[str, Any]:
    """Identify a local checkpoint by content markers so a rewritten path is a new study."""
    path = Path(model).expanduser()
    if not path.is_dir():
        return {"hub_id": model}
    weights = sorted(
        (f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in path.iterdir() if f.suffix in {".safetensors", ".bin"}
    )
    config = path / "config.json"
    config_hash = hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None
    return {"path": str(path.resolve()), "config_sha256": config_hash, "weights": weights}


@dataclass(frozen=True)
class HereticSearch:
    """One Heretic optimisation study: everything that changes the trials it explores.

    Attributes:
        model: HF id or local path, passed verbatim as `--model`.
        seed: Heretic's random seed.
        n_trials: Total Optuna trials.
        n_startup_trials: Random trials before TPE starts.
        response_prefix: Forced response prefix; None lets Heretic detect it.
    """

    model: str
    seed: int
    n_trials: int
    n_startup_trials: int
    response_prefix: str | None = None

    def cache_key(self, heretic_bin: Path) -> str:
        """Stable key for the study cache; the chosen trial index is deliberately excluded."""
        payload = {
            "heretic_bin": str(heretic_bin.resolve()),
            "checkpoint": _checkpoint_fingerprint(self.model),
            "seed": self.seed,
            "n_trials": self.n_trials,
            "n_startup_trials": self.n_startup_trials,
            "response_prefix": self.response_prefix,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def search_command(self, heretic_bin: Path, study_dir: Path, trial_index: int, save_dir: Path) -> list[str]:
        """Run the full study, then export Pareto trial `trial_index` (merged) to `save_dir`."""
        command = [
            str(heretic_bin),
            "--model",
            self.model,
            "--seed",
            str(self.seed),
            "--n-trials",
            str(self.n_trials),
            "--n-startup-trials",
            str(self.n_startup_trials),
            "--study-checkpoint-dir",
            str(study_dir),
            "--checkpoint-action",
            "restart",
            "--trial-index",
            str(trial_index),
            "--export-strategy",
            "merge",
            "--model-action",
            "save",
            "--save-directory",
            str(save_dir),
        ]
        if self.response_prefix is not None:
            command += ["--response-prefix", self.response_prefix]
        return command


def restore_command(heretic_bin: Path, model: str, restore_dir: Path) -> list[str]:
    """Re-export a trial from a journal prepared by `write_restore_journal`.

    `--checkpoint-action continue` replaces every setting with the journal's stored
    copy, so the trial index and save directory come from the patched journal.
    """
    return [
        str(heretic_bin),
        "--model",
        model,
        "--study-checkpoint-dir",
        str(restore_dir),
        "--checkpoint-action",
        "continue",
    ]


def _load_study(journal: Path) -> optuna.Study:
    storage = JournalStorage(JournalFileBackend(str(journal)))
    names = optuna.get_all_study_names(storage)
    if len(names) != 1:
        raise HereticRunError(f"expected one study in {journal}, found {names}")
    return optuna.load_study(study_name=names[0], storage=storage)


def study_is_finished(journal: Path) -> bool:
    """Return whether the journal holds a study Heretic marked finished."""
    if not journal.exists():
        return False
    return bool(_load_study(journal).user_attrs.get("finished", False))


@dataclass(frozen=True)
class HereticCandidate:
    """A Pareto-front trial, at the position `--trial-index` selects.

    Attributes:
        position: 0-based position in Heretic's sorted Pareto front (`--trial-index`).
        trial_number: Optuna's 0-based trial number.
        objectives: Objective name to value, in Heretic's scorer order (defaults:
            "Refusals" as a fraction, then "KL divergence").
    """

    position: int
    trial_number: int
    objectives: dict[str, float]


def pareto_candidates(journal: Path) -> list[HereticCandidate]:
    """Heretic's Pareto front in `--trial-index` order.

    Heretic sorts `study.best_trials` ascending by the tuple of objective values in
    scorer order (by default fewest keyword refusals first, then lowest KL).
    """
    study = _load_study(journal)
    complete = [t for t in study.best_trials if t.state == TrialState.COMPLETE]

    def objectives(trial: optuna.trial.FrozenTrial) -> dict[str, float]:
        return {entry["name"]: float(entry["score"]["value"]) for entry in trial.user_attrs["scores"]}

    ordered = sorted(complete, key=lambda t: tuple(objectives(t).values()))
    return [HereticCandidate(i, t.number, objectives(t)) for i, t in enumerate(ordered)]


def write_restore_journal(journal: Path, restore_dir: Path, model: str, trial_index: int, save_directory: Path) -> Path:
    """Copy `journal` into `restore_dir` with settings that export `trial_index`.

    Only `trial_index` and `save_directory` change (the latter is excluded from
    Heretic's settings dump, so it would otherwise prompt). Trials, scores and Pareto
    order are untouched, so Heretic restores the same trial. This is the patch the
    hub validated when rebuilding exports on another cluster (open-steering branch
    `hub/2026-09-28--jev-refusal-rescore--agile-ibex--virga-generate`).

    Raises:
        HereticRunError: If the journal stores no settings, a different model, or
            settings that would not save a merged model.
    """
    lines: list[str] = []
    patched = 0
    for line in journal.read_text().splitlines():
        entry = json.loads(line) if line.strip() else None
        user_attr = entry.get("user_attr") if isinstance(entry, dict) else None
        if isinstance(entry, dict) and isinstance(user_attr, dict) and "settings" in user_attr:
            settings = json.loads(user_attr["settings"])
            if settings.get("model") != model:
                raise HereticRunError(f"journal model {settings.get('model')!r} != {model!r}")
            if settings.get("model_action") != "save" or settings.get("export_strategy") != "merge":
                raise HereticRunError(f"journal settings do not save a merged model: {journal}")
            settings.update(trial_index=trial_index, save_directory=str(save_directory))
            user_attr["settings"] = json.dumps(settings)
            line = json.dumps(entry)
            patched += 1
        lines.append(line)
    if not patched:
        raise HereticRunError(f"no stored settings in {journal}")
    restore_dir.mkdir(parents=True, exist_ok=True)
    target = journal_path(restore_dir, model)
    target.write_text("\n".join(lines) + "\n")
    return target


def check_heretic_log(log: str) -> None:
    """Raise if Heretic's output reports a failure that its exit code hides."""
    for pattern in _LOG_FAILURES:
        match = pattern.search(log)
        if match:
            raise HereticRunError(f"heretic reported a failure: {match.group(0).strip()!r}")


def run_heretic(command: list[str], cwd: Path, log_path: Path) -> None:
    """Run Heretic with no terminal, logging stdout and stderr to `log_path`.

    `cwd` should be empty: Heretic reads `./config.toml` with lower precedence than
    the CLI, and closing stdin makes any interactive prompt abort instead of hang.

    Raises:
        HereticRunError: On a non-zero exit or a failure reported in the log.
    """
    cwd.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    text = log_path.read_text(errors="replace")
    if result.returncode != 0:
        raise HereticRunError(f"heretic exited {result.returncode}; see {log_path}")
    check_heretic_log(text)


@contextmanager
def study_lock(study_dir: Path) -> Iterator[None]:
    """Hold an exclusive lock on a cached study while it is created or read."""
    study_dir.mkdir(parents=True, exist_ok=True)
    with (study_dir / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
