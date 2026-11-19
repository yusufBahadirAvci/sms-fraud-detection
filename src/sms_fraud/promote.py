"""Champion / challenger: replace the production model only if the new one is better.

Run it with:
    python -m src.sms_fraud.promote                          (compare and maybe promote)
    python -m src.sms_fraud.promote --champion-version 11    (first time only, see below)
    python -m src.sms_fraud.promote rollback                 (go back to the "previous" model)

Words:
- champion:   the model the API serves now (models/model.joblib).
- challenger: the best NEW run from train.py (highest Turkish F1).

Rules (from config.yaml, "promotion"):
1. challenger tr_f1  >  champion tr_f1         (better on Turkish), AND
2. challenger en_f1  >= champion en_f1 - 0.02  (English is not much worse).

If the challenger wins:
- its file is copied to models/model.joblib,
- MLflow alias "champion" moves to its registered version,
- the old version gets the alias "previous" (so we can roll back).
In both cases reports/promotion.md explains the decision.

After models/model.joblib changes, tell DVC about the new file:
    dvc add models/model.joblib

First run: the Model Registry has no "champion" alias yet. Then you must
say which registered version is the current model (--champion-version).
"""

import argparse
import json
import os
import shutil
from datetime import UTC, datetime

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.sms_fraud.config import get_config, project_path
from src.sms_fraud.logger import get_logger
from src.sms_fraud.train import evaluate_by_language, load_test_sets, setup_mlflow

logger = get_logger("sms_fraud.promote")

# MLflow aliases are named pointers to one model version, like Git tags.
# Moving a pointer is instant and easy to undo; no file is copied in MLflow.
CHAMPION = "champion"
PREVIOUS = "previous"

# The rows of the comparison table, in this order.
METRICS = [
    f"{lang}_{m}" for lang in ("en", "tr") for m in ("accuracy", "precision", "recall", "f1")
]


# ---------------------------------------------------------------------------
# 1. The decision (pure functions: no files, no MLflow -> easy to test)
# ---------------------------------------------------------------------------


def pick_challenger(runs: dict) -> str:
    """Return the name of the run with the highest Turkish F1."""
    return max(runs, key=lambda name: runs[name]["metrics"]["tr_f1"])


# PATTERN: Strategy — the promotion rule is one small function with its limit passed in, so we can
#   test it with fake numbers and change the rule without touching the MLflow and file code.
def decide(champion: dict, challenger: dict, max_en_f1_drop: float) -> tuple[bool, list[str]]:
    """Should the challenger replace the champion?

    champion / challenger: dicts with at least "en_f1" and "tr_f1".
    Returns (promote yes/no, list of reasons in plain words).
    """
    reasons = []

    tr_better = challenger["tr_f1"] > champion["tr_f1"]
    reasons.append(
        f"Turkish F1: {challenger['tr_f1']:.4f} vs {champion['tr_f1']:.4f} -> "
        + ("better (OK)" if tr_better else "not better (FAIL)")
    )

    en_drop = champion["en_f1"] - challenger["en_f1"]
    # round(): float math can give 0.020000000000000018 for a drop of exactly
    # 0.02. "Not more than 0.02 below" must allow exactly 0.02.
    en_ok = round(en_drop, 10) <= max_en_f1_drop
    reasons.append(
        f"English F1: {challenger['en_f1']:.4f} vs {champion['en_f1']:.4f} "
        f"(drop {en_drop:+.4f}, limit {max_en_f1_drop}) -> "
        + ("OK" if en_ok else "too big a drop (FAIL)")
    )
    return tr_better and en_ok, reasons


# ---------------------------------------------------------------------------
# 2. Model Registry helpers (aliases)
# ---------------------------------------------------------------------------


def alias_version(client: MlflowClient, model_name: str, alias: str) -> str | None:
    """Return the version number behind an alias, or None if the alias is not set."""
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except MlflowException:
        return None


def move_aliases(
    client: MlflowClient, model_name: str, new_champion: str, new_previous: str | None
) -> None:
    """Point "champion" to one version and "previous" to another."""
    if new_previous is not None:
        client.set_registered_model_alias(model_name, PREVIOUS, new_previous)
    client.set_registered_model_alias(model_name, CHAMPION, new_champion)


def replace_file(write, target) -> None:
    """Write a new model file safely: first to a temp file, then swap.

    os.replace is "atomic": anyone reading the file sees either the whole
    old file or the whole new file, never a half-written one.
    """
    tmp = target.with_name(target.name + ".tmp")
    write(tmp)
    os.replace(tmp, target)


def promote(
    client: MlflowClient,
    model_name: str,
    candidate_path,
    model_path,
    new_version: str,
    old_version: str,
) -> None:
    """Make the challenger the new champion (file + aliases)."""
    replace_file(lambda tmp: shutil.copyfile(candidate_path, tmp), model_path)
    move_aliases(client, model_name, new_champion=new_version, new_previous=old_version)
    logger.info("Promoted version %s to champion (previous = %s)", new_version, old_version)


def rollback(client: MlflowClient, model_name: str, model_path) -> tuple[str, str]:
    """Bring back the "previous" model. Returns (restored version, demoted version).

    The two aliases SWAP places. So running rollback twice brings you back
    to where you started (an "undo" for the rollback).
    """
    previous = alias_version(client, model_name, PREVIOUS)
    if previous is None:
        raise RuntimeError(
            f"No '{PREVIOUS}' alias in model {model_name!r}: nothing to roll back to."
        )
    current = alias_version(client, model_name, CHAMPION)

    # Load the old model from MLflow and write it as the production file.
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{PREVIOUS}")
    replace_file(lambda tmp: joblib.dump(model, tmp), model_path)

    move_aliases(client, model_name, new_champion=previous, new_previous=current)
    if current is None:
        # No champion before -> nothing to keep as "previous" now.
        client.delete_registered_model_alias(model_name, PREVIOUS)
    logger.info("Rolled back: champion = version %s, previous = version %s", previous, current)
    return previous, current


# ---------------------------------------------------------------------------
# 3. The report
# ---------------------------------------------------------------------------


def write_report(
    path,
    columns: dict,
    champion_name: str,
    challenger_name: str,
    promoted: bool,
    reasons: list[str],
    max_en_f1_drop: float,
) -> None:
    """Write promotion.md: decision, reasons and the comparison table.

    columns: {column title: metrics dict}, e.g. {"champion (v11)": {...}, "A_word (v12)": {...}}
    """
    titles = list(columns)
    header = "| Metric | " + " | ".join(titles) + " |"
    lines = [
        "# Model Promotion Report",
        "",
        f"- Created: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Champion: {champion_name}",
        f"- Challenger (best tr_f1): {challenger_name}",
        f"- Decision: **{'PROMOTED' if promoted else 'REJECTED'}**",
        "",
        "## Rules",
        "",
        "1. challenger tr_f1 > champion tr_f1",
        f"2. challenger en_f1 >= champion en_f1 - {max_en_f1_drop}",
        "",
        "## Checks",
        "",
        *[f"- {r}" for r in reasons],
        "",
        "## Comparison (en = English test set, tr = Turkish test set)",
        "",
        header,
        "|---|" + "---|" * len(titles),
        *[f"| {m} | " + " | ".join(f"{columns[t][m]:.4f}" for t in titles) + " |" for m in METRICS],
        "",
        "Rollback: `python -m src.sms_fraud.promote rollback` "
        "(then `dvc add models/model.joblib`).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 4. Main program
# ---------------------------------------------------------------------------


def run_promotion(client: MlflowClient, cfg: dict, champion_version_arg: str | None) -> bool:
    model_name = cfg["mlflow"]["registered_model_name"]
    model_path = project_path(cfg["model"]["model_path"])
    max_drop = cfg["promotion"]["max_en_f1_drop"]

    # Which registered version is the champion now?
    champion_version = alias_version(client, model_name, CHAMPION) or champion_version_arg
    if champion_version is None:
        raise RuntimeError(
            "No 'champion' alias yet. Run once with --champion-version <N> "
            "(the registered version that models/model.joblib came from)."
        )

    # Score the CURRENT production file on the SAME test sets as the new runs.
    # Why not use its old MLflow metrics? They were measured on other data
    # (English only). A fair contest needs the same exam for both.
    test_sets = load_test_sets()
    champion_metrics = evaluate_by_language(joblib.load(model_path), test_sets)

    with open(project_path(cfg["model"]["metrics_path"]), encoding="utf-8") as f:
        runs = json.load(f)["runs"]
    challenger_name = pick_challenger(runs)
    challenger = runs[challenger_name]

    promoted, reasons = decide(champion_metrics, challenger["metrics"], max_drop)
    for reason in reasons:
        logger.info(reason)

    if promoted:
        promote(
            client,
            model_name,
            project_path(challenger["candidate_path"]),
            model_path,
            new_version=challenger["model_version"],
            old_version=champion_version,
        )
        logger.info("Now run: dvc add models/model.joblib")
    else:
        # Make sure the alias exists even after a rejection (first run).
        move_aliases(client, model_name, new_champion=champion_version, new_previous=None)
        logger.info("Challenger rejected; champion stays version %s", champion_version)

    # Table: champion first, then every new run (the challenger is marked).
    columns = {f"champion (v{champion_version})": champion_metrics}
    for name, run in runs.items():
        mark = " ← challenger" if name == challenger_name else ""
        columns[f"{name} (v{run['model_version']}){mark}"] = run["metrics"]
    write_report(
        project_path(cfg["promotion"]["report_path"]),
        columns,
        champion_name=f"version {champion_version} (models/model.joblib before this run)",
        challenger_name=f"{challenger_name} (version {challenger['model_version']})",
        promoted=promoted,
        reasons=reasons,
        max_en_f1_drop=max_drop,
    )
    return promoted


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Champion/challenger promotion and rollback.")
    parser.add_argument("action", nargs="?", choices=["promote", "rollback"], default="promote")
    parser.add_argument(
        "--champion-version",
        default=None,
        help="Registered version of the current model (only needed the first time).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = get_config()
    setup_mlflow(cfg["mlflow"])
    client = MlflowClient()

    if args.action == "rollback":
        restored, demoted = rollback(
            client, cfg["mlflow"]["registered_model_name"], project_path(cfg["model"]["model_path"])
        )
        # Add a short note to the report, so the history is visible.
        report = project_path(cfg["promotion"]["report_path"])
        with open(report, "a", encoding="utf-8") as f:
            f.write(
                f"\n## Rollback ({datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')})\n\n"
                f"Champion is version {restored} again; version {demoted} is now 'previous'.\n"
            )
        logger.info("Now run: dvc add models/model.joblib (and restart the API)")
    else:
        run_promotion(client, cfg, args.champion_version)


if __name__ == "__main__":
    main()
