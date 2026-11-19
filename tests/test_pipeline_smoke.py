"""End-to-end "smoke test": run the WHOLE pipeline on tiny fake data.

A smoke test does not check if the model is good. It checks that every
stage runs from start to end and writes its output files:

    data -> turkish_data -> combine_data -> eda -> train -> promote
         -> monitoring (dataset + log mode) -> rollback

Everything happens in a temporary folder (pytest's tmp_path):
- no real data, no real model, no internet, no real mlruns/,
- so it also works in CI on a fresh clone.

How? Every module builds its paths with project_path(), which uses
config.PROJECT_ROOT. We point PROJECT_ROOT to the temporary folder.
"""

import json
import shutil
import sys

import joblib
import mlflow
import pytest
from mlflow import MlflowClient

from src.sms_fraud import (
    combine_data,
    config,
    data,
    eda,
    monitoring,
    promote,
    train,
    turkish_data,
)

N = 15  # messages per class and language: small, but enough for an 80/20 split


def write_fake_raw_files(root) -> None:
    """Create the two raw input files in the same format as the real ones."""
    cfg = config.get_config()["data"]

    # English: "label<TAB>message", no header (like SMSSpamCollection).
    english = [f"spam\tWINNER {i}! Claim your free prize {i} now, call 0800{i}" for i in range(N)]
    english += [f"ham\tSee you at dinner at {i} pm, bring the book {i}" for i in range(N)]
    raw_file = root / cfg["raw_dir"] / cfg["raw_file"]
    raw_file.parent.mkdir(parents=True)
    raw_file.write_text("\n".join(english) + "\n", encoding="utf-8")

    # Turkish: "Message;Group;GroupText" with a header. Group 1 = spam, 2 = normal.
    turkish = ["Message;Group;GroupText"]
    turkish += [f"Kampanya {i}: %{i} indirim fırsatı, hemen tıklayın;1;Spam" for i in range(N)]
    turkish += [f"Akşam {i} gibi eve geliyorum, yemek var mı {i};2;Normal" for i in range(N)]
    turkish_file = root / cfg["turkish_raw_file"]
    turkish_file.parent.mkdir(parents=True)
    turkish_file.write_text("\n".join(turkish) + "\n", encoding="utf-8")


# scope="module": the pipeline runs ONCE, and all tests below check its results.
@pytest.fixture(scope="module")
def fake_project(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    old_uri = mlflow.get_tracking_uri()
    # MonkeyPatch.context() undoes every change at the end of the "with" block.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "PROJECT_ROOT", root)
        # eda.py computes these two paths at import time, so we set them too.
        mp.setattr(eda, "FIGURES_DIR", root / "reports/figures")
        mp.setattr(eda, "SUMMARY_PATH", root / "reports/eda_summary.md")
        # train.main() reads the command line; give it no extra options.
        mp.setattr(sys, "argv", ["train"])

        write_fake_raw_files(root)
        # The raw English file exists, so data.main() skips the download.
        data.main()
        turkish_data.main()
        combine_data.main()
        eda.main()
        train.main()

        # A "champion" to compete against: a model trained on English only
        # (like the real model before Step 5b). It is registered version 1.
        X_en, y_en = train.load_split("train")
        old_model = train.build_pipeline("logreg", max_features=100, ngram_range=(1, 1))
        joblib.dump(old_model.fit(X_en, y_en), root / "models/model.joblib")
        promote.main(["--champion-version", "1"])
        promoted_model = (root / "models/model.joblib").read_bytes()

        alerts = monitoring.main([])

        # Log mode needs a prediction log: write 20 fake lines.
        log_file = root / config.get_config()["api"]["predictions_log"]
        log_file.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"message_length": 40 + i, "oov_ratio": 0.1, "probability": 0.1, "prediction": False}
            for i in range(20)
        ]
        log_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        monitoring.main(["--mode", "log", "--take", "20", "--out-dir", "reports/live"])

        aliases_before_rollback = dict(
            MlflowClient()
            .get_registered_model(config.get_config()["mlflow"]["registered_model_name"])
            .aliases
        )
        promote.main(["rollback"])
        aliases_after_rollback = dict(
            MlflowClient()
            .get_registered_model(config.get_config()["mlflow"]["registered_model_name"])
            .aliases
        )

        yield {
            "root": root,
            "alerts": alerts,
            "promoted_model": promoted_model,
            "before": aliases_before_rollback,
            "after": aliases_after_rollback,
        }
    # Put MLflow back to its old setting and free the temp files.
    mlflow.set_tracking_uri(old_uri)
    shutil.rmtree(root, ignore_errors=True)


def test_data_stages_write_their_files(fake_project):
    processed = fake_project["root"] / "data/processed"
    for name in ("train", "test", "turkish", "turkish_train", "turkish_test", "combined_train"):
        assert (processed / f"{name}.csv").exists(), name
    combined = combine_data.read_csv(processed / "combined_train.csv")
    # Both languages are in the train set, and every row is marked.
    assert set(combined["lang"]) == {"en", "tr"}
    assert (fake_project["root"] / "reports/eda_summary.md").exists()


def test_train_writes_all_candidates_and_metrics(fake_project):
    root = fake_project["root"]
    metrics = json.loads((root / "reports/metrics.json").read_text(encoding="utf-8"))
    names = [e["name"] for e in config.get_config()["model"]["experiments"]]
    assert sorted(metrics["runs"]) == sorted(names)
    for name in names:
        assert (root / f"models/candidates/{name}.joblib").exists()
        # Each run is scored on BOTH languages.
        assert {"en_f1", "tr_f1"} <= set(metrics["runs"][name]["metrics"])


def test_promotion_replaces_english_only_champion(fake_project):
    root = fake_project["root"]
    report = (root / "reports/promotion.md").read_text(encoding="utf-8")
    # The English-only champion knows no Turkish words, so a combined model must win.
    assert "PROMOTED" in report
    # The served file is now one of the candidate files, byte for byte.
    candidates = [p.read_bytes() for p in (root / "models/candidates").glob("*.joblib")]
    assert fake_project["promoted_model"] in candidates
    # MLflow gives alias versions as numbers, so we compare as text.
    assert str(fake_project["before"]["previous"]) == "1"


def test_monitoring_writes_reports(fake_project):
    root = fake_project["root"]
    assert (root / "reports/monitoring/drift_report.md").exists()
    assert (root / "reports/monitoring/probability_hist.png").exists()
    assert (root / "reports/live/drift_report.md").exists()
    assert isinstance(fake_project["alerts"], list)


def test_rollback_swaps_aliases(fake_project):
    before, after = fake_project["before"], fake_project["after"]
    assert after["champion"] == before["previous"]
    assert after["previous"] == before["champion"]
