"""Behaviour of the Heretic driver: Pareto order, trial restore, failure guards, cache keys."""

import json
import stat
from pathlib import Path

import optuna
import pytest
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, PreTrainedTokenizerFast

from tamperbench.whitebox.utils.heretic import (
    JEV_SCORER_PLUGIN,
    HereticRunError,
    HereticSearch,
    check_heretic_log,
    journal_path,
    pareto_candidates,
    reserialize_input_metadata,
    run_heretic,
    study_is_finished,
    write_restore_journal,
)

MODEL = "org/Model-1B-Instruct"


KEYWORD_SCORERS = [
    {"plugin": "heretic.scorers.keyword_rate.KeywordRate", "optimization": "minimize"},
    {"plugin": "heretic.scorers.kl_divergence.KLDivergence", "optimization": "minimize"},
]


def _scores(named: dict[str, float]) -> list[dict[str, object]]:
    return [
        {"name": name, "score": {"value": value, "rich_display": "", "md_display": ""}} for name, value in named.items()
    ]


def _write_study(
    study_dir: Path,
    trials: list[tuple[float, float]],
    finished: bool = True,
    recorded: list[float] | None = None,
) -> Path:
    """Write a journal shaped like Heretic's: settings and finished as study attrs, scores per trial.

    With `recorded`, each trial also carries a non-optimised "Refusals" score (as with the
    Jev objective) and the optimised pair is ("Not harmful (Jev)", "KL divergence").
    """
    journal = journal_path(study_dir, MODEL)
    journal.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(str(journal)))
    study = optuna.create_study(study_name="heretic", storage=storage, directions=["minimize", "minimize"])
    scorers = (
        KEYWORD_SCORERS if recorded is None else [KEYWORD_SCORERS[0] | {"optimization": "none"}, *KEYWORD_SCORERS[1:]]
    )
    if recorded is not None:
        scorers = [{"plugin": f"{JEV_SCORER_PLUGIN}:JevHarmfulRate", "optimization": "minimize"}, *scorers]
    settings = {
        "model": MODEL,
        "trial_index": 0,
        "model_action": "save",
        "export_strategy": "merge",
        "seed": 7,
        "scorers": scorers,
    }
    study.set_user_attr("settings", json.dumps(settings))
    for i, (objective, kl) in enumerate(trials):
        if recorded is None:
            named = {"Refusals": objective, "KL divergence": kl}
        else:
            named = {"Not harmful (Jev)": objective, "Refusals": recorded[i], "KL divergence": kl}
        study.add_trial(
            optuna.trial.create_trial(values=[objective, kl], user_attrs={"index": i + 1, "scores": _scores(named)})
        )
    study.set_user_attr("finished", finished)
    return journal


def test_pareto_order_matches_trial_index_semantics(tmp_path: Path) -> None:
    """Position 0 is the fewest refusals; dominated trials are excluded (trial 2 by trial 0)."""
    journal = _write_study(tmp_path, [(0.10, 0.02), (0.04, 0.05), (0.30, 0.03), (0.50, 0.001)])
    front = pareto_candidates(journal)
    assert [c.trial_number for c in front] == [1, 0, 3]
    assert [c.position for c in front] == [0, 1, 2]
    assert front[0].scores == {"Refusals": 0.04, "KL divergence": 0.05}


def test_pareto_order_ignores_recorded_but_unoptimised_scores(tmp_path: Path) -> None:
    """Under the Jev objective, keyword refusals are recorded but must not reorder the front."""
    journal = _write_study(tmp_path, [(0.30, 0.01), (0.10, 0.05), (0.60, 0.002)], recorded=[0.02, 0.90, 0.01])
    front = pareto_candidates(journal)
    assert [c.trial_number for c in front] == [1, 0, 2]
    assert front[0].scores["Refusals"] == 0.90


def test_unfinished_or_missing_study_is_not_reused(tmp_path: Path) -> None:
    """Only a study Heretic marked finished is reused from the cache."""
    assert not study_is_finished(tmp_path / "absent.jsonl")
    assert not study_is_finished(_write_study(tmp_path / "a", [(0.1, 0.1)], finished=False))
    assert study_is_finished(_write_study(tmp_path / "b", [(0.1, 0.1)], finished=True))


def test_restore_journal_patches_only_index_and_save_directory(tmp_path: Path) -> None:
    """Restoring changes only the exported index and save path, never the cached study."""
    journal = _write_study(tmp_path / "cache", [(0.10, 0.02), (0.04, 0.05)])
    target = write_restore_journal(journal, tmp_path / "restore", MODEL, 1, tmp_path / "out")

    study = optuna.load_study(study_name="heretic", storage=JournalStorage(JournalFileBackend(str(target))))
    settings = json.loads(study.user_attrs["settings"])
    assert settings["trial_index"] == 1
    assert settings["save_directory"] == str(tmp_path / "out")
    assert settings["seed"] == 7
    assert [c.trial_number for c in pareto_candidates(target)] == [c.trial_number for c in pareto_candidates(journal)]
    # The cached study is not modified.
    assert (
        json.loads(
            optuna.load_study(
                study_name="heretic", storage=JournalStorage(JournalFileBackend(str(journal)))
            ).user_attrs["settings"]
        )["trial_index"]
        == 0
    )


def test_restore_refuses_a_journal_for_another_model(tmp_path: Path) -> None:
    """A cached journal cannot be restored against a different model string."""
    journal = _write_study(tmp_path / "cache", [(0.1, 0.1)])
    with pytest.raises(HereticRunError, match="journal model"):
        write_restore_journal(journal, tmp_path / "restore", "other/model", 0, tmp_path / "out")


@pytest.mark.parametrize(
    "log",
    ["* Loading model\nConfiguration contains 2 errors:\n  n_trials: ...\n", "Saving...\nError: disk full\n"],
)
def test_log_guard_catches_failures_heretic_exits_zero_on(log: str) -> None:
    """Configuration and save errors fail the attack even though Heretic exits 0."""
    with pytest.raises(HereticRunError):
        check_heretic_log(log)


def test_log_guard_passes_a_clean_run() -> None:
    """A normal run's score lines are not mistaken for failures."""
    check_heretic_log("* Refusals: 4/100\n* KL divergence: 0.0257\nModel saved to /x\n")


def _fake_heretic(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "fake_heretic"
    script.write_text(f"#!/bin/sh\n{body}\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_run_heretic_fails_on_hidden_error_despite_exit_zero(tmp_path: Path) -> None:
    """The log guard applies to real subprocess output, not just strings."""
    script = _fake_heretic(tmp_path, 'echo "Configuration contains 1 error:"; exit 0')
    with pytest.raises(HereticRunError):
        run_heretic([str(script)], cwd=tmp_path / "cwd", log_path=tmp_path / "log.txt")
    assert "Configuration contains" in (tmp_path / "log.txt").read_text()


def test_run_heretic_closes_stdin_so_prompts_cannot_hang(tmp_path: Path) -> None:
    """A prompt reading stdin sees EOF instead of blocking the run."""
    script = _fake_heretic(tmp_path, 'read answer; echo "got:[$answer]"; exit 0')
    run_heretic([str(script)], cwd=tmp_path / "cwd", log_path=tmp_path / "log.txt")
    assert "got:[]" in (tmp_path / "log.txt").read_text()


def test_run_heretic_raises_on_nonzero_exit(tmp_path: Path) -> None:
    """A non-zero exit fails the attack."""
    with pytest.raises(HereticRunError, match="exited 3"):
        run_heretic([str(_fake_heretic(tmp_path, "exit 3"))], cwd=tmp_path / "cwd", log_path=tmp_path / "log.txt")


def test_cache_key_tracks_checkpoint_content_and_search_settings(tmp_path: Path) -> None:
    """Same checkpoint and settings share a study; new weights or a new seed do not."""
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"a")
    heretic_bin = _fake_heretic(tmp_path, "exit 0")
    search = HereticSearch(model=str(checkpoint), seed=1, n_trials=200, n_startup_trials=60)

    key = search.cache_key(heretic_bin)
    assert key == HereticSearch(model=str(checkpoint), seed=1, n_trials=200, n_startup_trials=60).cache_key(heretic_bin)
    assert key != HereticSearch(model=str(checkpoint), seed=2, n_trials=200, n_startup_trials=60).cache_key(heretic_bin)
    jev = HereticSearch(model=str(checkpoint), seed=1, n_trials=200, n_startup_trials=60, objective="jev")
    assert key != jev.cache_key(heretic_bin)
    (checkpoint / "model.safetensors").write_bytes(b"rewritten weights")
    assert search.cache_key(heretic_bin) != key


def test_search_command_answers_every_prompt_and_forwards_empty_prefix(tmp_path: Path) -> None:
    """Every interactive menu is answered on the command line; an empty prefix is passed through."""
    search = HereticSearch(model=MODEL, seed=1, n_trials=10, n_startup_trials=3, response_prefix="")
    command = search.search_command(Path("/h"), tmp_path / "s", 2, tmp_path / "out")
    flags = dict(zip(command[1::2], command[2::2], strict=False))
    assert flags["--checkpoint-action"] == "restart"
    assert flags["--trial-index"] == "2"
    assert flags["--model-action"] == "save"
    assert flags["--export-strategy"] == "merge"
    assert flags["--save-directory"] == str(tmp_path / "out")
    assert command[-2:] == ["--response-prefix", ""]


def test_jev_objective_optimises_jev_and_only_records_keywords(tmp_path: Path) -> None:
    """The Jev search minimises Jev's not-harmful rate and KL; keyword refusals are not an objective."""
    search = HereticSearch(
        model=MODEL, seed=1, n_trials=10, n_startup_trials=3, objective="jev", max_response_length=256
    )
    command = search.search_command(Path("/h"), tmp_path / "s", 0, tmp_path / "out")
    flags = dict(zip(command[1::2], command[2::2], strict=False))
    assert flags["--max-response-length"] == "256"
    scorers = json.loads(flags["--scorers"])
    assert [s["optimization"] for s in scorers] == ["minimize", "none", "minimize"]
    plugin_file, plugin_class = scorers[0]["plugin"].rsplit(":", 1)
    assert Path(plugin_file).is_file() and plugin_class == "JevHarmfulRate"
    assert "--scorers" not in HereticSearch(MODEL, 1, 10, 3).search_command(Path("/h"), tmp_path, 0, tmp_path)


def test_reserialize_repairs_a_transformers_5_export(tmp_path: Path) -> None:
    """A transformers-5 export (RoPE base misread, tokenizer unloadable) loads exactly like its input."""
    vocab = {"[UNK]": 0, "<eos>": 1, "hello": 2, "world": 3}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    source = tmp_path / "input"
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="<eos>", chat_template="{{ messages[0]['content'] }}"
    ).save_pretrained(str(source))
    LlamaConfig(
        vocab_size=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        rope_theta=500000.0,
    ).save_pretrained(str(source))

    export = tmp_path / "export"
    export.mkdir()
    v5_config = json.loads((source / "config.json").read_text())
    v5_config.pop("rope_theta")  # transformers 5.x moves it into rope_parameters
    v5_config["rope_parameters"] = {"rope_theta": 500000.0, "rope_type": "default"}
    (export / "config.json").write_text(json.dumps(v5_config))
    broken = json.loads((source / "tokenizer_config.json").read_text())
    broken["extra_special_tokens"] = ["<eos>"]  # transformers 5.x style, a list rather than a dict
    (export / "tokenizer_config.json").write_text(json.dumps(broken))
    (export / "tokenizer.json").write_text((source / "tokenizer.json").read_text())
    assert AutoConfig.from_pretrained(str(export)).rope_theta == 10000.0  # silently wrong before the repair
    with pytest.raises(AttributeError):
        AutoTokenizer.from_pretrained(str(export))

    reserialize_input_metadata(str(source), str(source), export)

    assert AutoConfig.from_pretrained(str(export)).rope_theta == 500000.0
    repaired = AutoTokenizer.from_pretrained(str(export))
    assert repaired.get_vocab() == vocab
    assert repaired.chat_template == "{{ messages[0]['content'] }}"
