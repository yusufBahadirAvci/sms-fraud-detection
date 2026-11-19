"""Monitoring: is the data the model sees today different from its training world?

Since Step 7a we monitor EACH LANGUAGE SEPARATELY. Why? Up to Step 6 the
reference was English + Turkish mixed, and the current window was Turkish
only. Turkish SMS are longer and more often spam, so we got "drift" alerts
(length KS test, PSI 0.206, Issue #1) that only measured the language MIX,
not a real change. Now English is compared with English, Turkish with Turkish.

Windows, for each language ("en", "tr"):
- REFERENCE: a fixed random half of that language's test set (true labels).
- CURRENT:   new data, sent to a language with detect_language()
             (live traffic has no "lang" label, so we must guess it). Two modes:
    dataset mode: the OTHER half of both test sets (we know the true
                  answers, so we can also measure F1). The model never
                  trained on any test row.
    log mode:     live API traffic from logs/predictions.jsonl. The API
                  logs the detected "lang" per line (no message text).

Run it with:
    python -m src.sms_fraud.monitoring                                  (dataset mode)
    python -m src.sms_fraud.monitoring --mode log                       (whole log)
    python -m src.sms_fraud.monitoring --mode log --skip 100 --take 100 (log lines 101-200)
    python -m src.sms_fraud.monitoring --model-path old.joblib --out-dir reports/x (another model)

Outputs (in reports/monitoring/ or the --out-dir folder):
    drift_report.md (one table per language),
    probability_hist_<lang>.png, length_hist_<lang>.png,
    alert.json (only if a threshold is crossed)
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib

# Draw charts into files only, without a window (works in DVC and servers).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from src.sms_fraud.config import get_config, project_path  # noqa: E402
from src.sms_fraud.logger import get_logger  # noqa: E402
from src.sms_fraud.text_stats import detect_language, find_vectorizer, oov_ratio  # noqa: E402
from src.sms_fraud.train import evaluate, load_test_sets  # noqa: E402

logger = get_logger("sms_fraud.monitoring")

# The languages we monitor, each with its own reference window.
LANGUAGES = ("en", "tr")

# PSI smoothing: we add this small count to EVERY bin before computing
# shares. Why? An empty bin would give log(0), which is not allowed.
# Why not just replace 0 by a tiny number like 0.0001? We tried: with small
# windows (100 messages) some bins are empty only by chance, and the tiny
# number makes each empty bin look like a huge change. In a test with 500
# random English windows, 12.8% gave a FALSE drift alert. With +0.5 per bin:
# 0% false alerts, and Turkish data still gives PSI > 0.6.
PSI_SMOOTHING = 0.5


# ---------------------------------------------------------------------------
# 1. Drift measures (small, pure functions = easy to test)
# ---------------------------------------------------------------------------


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index of two sets of probabilities (0.0 - 1.0).

    How it works:
    1. Cut the range 0..1 into equal bins (0-0.1, 0.1-0.2, ...).
    2. For each bin, compute the SHARE of values in reference and in current.
    3. PSI = sum over bins of (cur - ref) * ln(cur / ref).

    Reading it: < 0.1 small change, 0.1-0.2 medium, > 0.2 big change.

    Why equal-width bins and not quantiles? Our model gives most messages a
    probability near 0, so quantile bins would almost all fall on top of
    each other near 0. Fixed bins keep the meaning simple.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Counts -> shares, with PSI_SMOOTHING added to every bin (see above).
    ref_share = (ref_counts + PSI_SMOOTHING) / (len(reference) + PSI_SMOOTHING * bins)
    cur_share = (cur_counts + PSI_SMOOTHING) / (len(current) + PSI_SMOOTHING * bins)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def score_messages(model, messages: pd.Series, threshold: float) -> pd.DataFrame:
    """Run the model on messages and collect the numbers we monitor.

    These are the SAME numbers the API writes to predictions.jsonl, so
    dataset mode and log mode can be compared in the same way.
    """
    messages = messages.tolist()
    probabilities = model.predict_proba(messages)[:, 1]
    # The word vectorizer (or the char one for a char-only model), see text_stats.py.
    vectorizer = find_vectorizer(model)
    return pd.DataFrame(
        {
            "length": [len(m) for m in messages],
            "oov_ratio": [oov_ratio(vectorizer, m) for m in messages],
            "probability": probabilities,
            "prediction": probabilities >= threshold,
        }
    )


# ---------------------------------------------------------------------------
# 2. Compare two windows and decide about alerts
# ---------------------------------------------------------------------------


def compare(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    thresholds: dict,
    psi_bins: int,
    ref_metrics: dict | None = None,
    cur_metrics: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """Compare reference and current windows.

    Returns:
        rows:   one dict per line of the report table
        alerts: a short text for every threshold that was crossed
    """
    rows, alerts = [], []

    # Small helper: add one row; if it has a threshold and "crossed" is True,
    # also add an alert. status None = information only (no threshold).
    def add(metric, reference, current, threshold="—", crossed=None, reason=""):
        if crossed is None:
            status = "—"
        else:
            status = "ALERT" if crossed else "OK"
        rows.append(
            {
                "metric": metric,
                "reference": reference,
                "current": current,
                "threshold": threshold,
                "status": status,
            }
        )
        if crossed:
            alerts.append(reason)

    # a) OOV ratio: how many words are unknown to the model?
    cur_oov = cur["oov_ratio"].mean()
    add(
        "OOV ratio (mean)",
        ref["oov_ratio"].mean(),
        cur_oov,
        f"> {thresholds['oov_ratio']}",
        cur_oov > thresholds["oov_ratio"],
        f"OOV ratio {cur_oov:.2f} > {thresholds['oov_ratio']}"
        " - most words are unknown to the model",
    )

    # b) Message length. ks_2samp compares the WHOLE shape of two
    #    distributions. A small p-value means: very unlikely to be random,
    #    the lengths really changed.
    add("Message length (mean chars)", ref["length"].mean(), cur["length"].mean())
    ks = ks_2samp(ref["length"], cur["length"])
    add(
        f"Length KS test p-value (D={ks.statistic:.3f})",
        "—",
        ks.pvalue,
        f"< {thresholds['ks_pvalue']}",
        ks.pvalue < thresholds["ks_pvalue"],
        f"Length distribution changed (KS p-value {ks.pvalue:.2g} < {thresholds['ks_pvalue']})",
    )

    # c) Prediction drift: does the model answer differently than before?
    add("Predicted spam share", ref["prediction"].mean(), cur["prediction"].mean())
    add("Mean spam probability", ref["probability"].mean(), cur["probability"].mean())
    psi_value = psi(ref["probability"], cur["probability"], bins=psi_bins)
    add(
        "PSI of spam probability",
        "—",
        psi_value,
        f"> {thresholds['psi']}",
        psi_value > thresholds["psi"],
        f"Prediction distribution shifted (PSI {psi_value:.2f} > {thresholds['psi']})",
    )

    # d) Model performance. Only possible when we know the true labels.
    if ref_metrics and cur_metrics:
        for name in ("accuracy", "precision", "recall"):
            add(name.capitalize(), ref_metrics[name], cur_metrics[name])
        f1_drop = ref_metrics["f1"] - cur_metrics["f1"]
        add(
            f"F1 (drop = {f1_drop:.3f})",
            ref_metrics["f1"],
            cur_metrics["f1"],
            f"drop > {thresholds['f1_drop']}",
            f1_drop > thresholds["f1_drop"],
            f"F1 dropped by {f1_drop:.2f} ({ref_metrics['f1']:.2f} -> {cur_metrics['f1']:.2f})",
        )
    else:
        add("F1", ref_metrics["f1"] if ref_metrics else "—", "N/A (no labels in live traffic)")

    return rows, alerts


# ---------------------------------------------------------------------------
# 3. Write the outputs
# ---------------------------------------------------------------------------


def fmt(value) -> str:
    """Format numbers for the report; leave text as it is."""
    if isinstance(value, (float, np.floating)):
        # Very small p-values are easier to read like 1.2e-50.
        return f"{value:.2e}" if 0 < abs(value) < 0.001 else f"{value:.4f}"
    return str(value)


def write_report(
    sections: dict,
    out_dir: Path,
    current_name: str,
    model_name: str = "",
    notes: list[str] | None = None,
) -> Path:
    """Write drift_report.md: ONE table per language + all alerts.

    sections: {lang: {"rows": [...], "alerts": [...], "n_ref": int, "n_cur": int}}
    notes: extra lines for the top (e.g. how good the language detector is).
    """
    all_alerts = [a for sec in sections.values() for a in sec["alerts"]]
    lines = [
        "# Drift Report (per language)",
        "",
        f"- Created: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Model: {model_name}",
        "- Reference: a fixed random half of each language's test set",
        f"- Current: {current_name}, split by detected language",
        *[f"- {n}" for n in notes or []],
        f"- Result: **{'DRIFT ALERT' if all_alerts else 'no drift detected'}**",
    ]
    for lang, sec in sections.items():
        lines += [
            "",
            f"## Language: {lang}",
            "",
            f"- Reference: {sec['n_ref']} messages | Current: {sec['n_cur']} messages",
            f"- Result: **{'DRIFT ALERT' if sec['alerts'] else 'no drift detected'}**",
            "",
        ]
        if not sec["rows"]:
            lines.append("No current messages for this language: nothing to compare.")
            continue
        lines += [
            "| Metric | Reference | Current | Threshold | Status |",
            "|---|---|---|---|---|",
            *[
                f"| {r['metric']} | {fmt(r['reference'])} | {fmt(r['current'])} "
                f"| {r['threshold']} | {r['status']} |"
                for r in sec["rows"]
            ],
        ]
    lines += [
        "",
        "## Alerts",
        "",
        *([f"- {a}" for a in all_alerts] if all_alerts else ["None."]),
        "",
        "## Charts",
        "",
        *[f"- `probability_hist_{lang}.png`, `length_hist_{lang}.png`" for lang in sections],
        "",
    ]
    path = out_dir / "drift_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_alert(alerts: list[str], out_dir: Path, current_name: str) -> Path | None:
    """Write alert.json if there are alerts; delete an old one if not.

    A separate JSON file is easy for other tools (CI, a chat bot, a cron
    job) to check: "if alert.json exists -> notify someone".
    """
    path = out_dir / "alert.json"
    if not alerts:
        # Remove an old alert, so it does not look like a new problem.
        path.unlink(missing_ok=True)
        return None
    payload = {
        "created": datetime.now(UTC).isoformat(),
        "current_window": current_name,
        "alerts": alerts,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def plot_histograms(
    ref: pd.DataFrame, cur: pd.DataFrame, out_dir: Path, current_name: str, lang: str
) -> None:
    """Two charts for ONE language: spam probability and message length."""
    charts = [
        ("probability", np.linspace(0, 1, 21), "Spam probability", f"probability_hist_{lang}.png"),
        (
            "length",
            np.linspace(0, 400, 41),
            "Message length (chars, max 400 shown)",
            f"length_hist_{lang}.png",
        ),
    ]
    for column, bins, xlabel, filename in charts:
        fig, ax = plt.subplots(figsize=(7, 4))
        for df, label, color in (
            (ref, f"reference: {lang} test (half)", "tab:blue"),
            (cur, f"current: {current_name}", "tab:orange"),
        ):
            # weights = 1/n turns counts into SHARES, so windows of different
            # sizes can be compared. alpha=0.5 keeps both colors visible.
            ax.hist(
                df[column],
                bins=bins,
                weights=np.ones(len(df)) / len(df),
                alpha=0.5,
                label=label,
                color=color,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Share of messages")
        ax.set_title(f"[{lang}] {xlabel}: reference vs current")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Load the windows and run everything
# ---------------------------------------------------------------------------


def load_log_window(path: Path, skip: int, take: int | None) -> pd.DataFrame:
    """Read predictions.jsonl and return the chosen lines as a window."""
    if not path.exists():
        raise FileNotFoundError(
            f"No prediction log at {path}. Send some requests to the API first."
        )
    # lines=True: one JSON object per line (JSONL).
    log = pd.read_json(path, lines=True)
    end = None if take is None else skip + take
    window = log.iloc[skip:end].rename(columns={"message_length": "length"})
    if window.empty:
        raise ValueError(f"The selected window is empty (log has {len(log)} lines).")
    # "lang" exists since Step 7a. Older log lines have none: we mark them
    # "unknown", and main() skips them (we cannot guess without the text).
    if "lang" not in window:
        window = window.assign(lang="unknown")
    window = window.assign(lang=window["lang"].fillna("unknown"))
    return window[["length", "oov_ratio", "probability", "prediction", "lang"]]


def split_reference(messages: pd.Series, share: float, random_state: int) -> tuple:
    """Cut ONE language's test set into a reference part and a current part.

    Fixed seed -> the same halves every run. Returns (reference rows, current rows).
    """
    shuffled = messages.sample(frac=1.0, random_state=random_state).index
    n_ref = round(len(shuffled) * share)
    return shuffled[:n_ref], shuffled[n_ref:]


# PATTERN: Partitioning — each language gets its OWN reference and current window, and the SAME
#   compare() runs per partition. So a change in the language mix can no longer look like drift.
def compare_by_language(
    refs: dict, cur: pd.DataFrame, thresholds: dict, psi_bins: int, cur_metrics: dict
) -> dict:
    """Run compare() once per language.

    refs:        {lang: (scored reference window, reference metrics)}
    cur:         scored current window with a "lang" column
    cur_metrics: {lang: metrics} (dataset mode) or {} (log mode, no labels)
    Returns {lang: {"rows", "alerts", "n_ref", "n_cur"}}; alerts start with "[lang]".
    """
    sections = {}
    for lang, (ref, ref_metrics) in refs.items():
        cur_lang = cur[cur["lang"] == lang]
        rows, alerts = [], []
        if len(cur_lang):
            rows, alerts = compare(
                ref, cur_lang, thresholds, psi_bins, ref_metrics, cur_metrics.get(lang)
            )
        sections[lang] = {
            "rows": rows,
            "alerts": [f"[{lang}] {a}" for a in alerts],
            "n_ref": len(ref),
            "n_cur": len(cur_lang),
        }
    return sections


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the model for data and prediction drift.")
    parser.add_argument(
        "--mode",
        choices=["dataset", "log"],
        default="dataset",
        help="dataset: Turkish data file; log: live API log.",
    )
    parser.add_argument("--skip", type=int, default=0, help="log mode: skip the first N log lines.")
    parser.add_argument(
        "--take", type=int, default=None, help="log mode: use only N lines after --skip."
    )
    parser.add_argument(
        "--out-dir", default=None, help="Folder for the outputs (default from config)."
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model file to check (default: the champion, models/model.joblib).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> list[str]:
    args = parse_args(argv)
    cfg = get_config()
    mon_cfg = cfg["monitoring"]
    threshold = cfg["api"]["threshold"]
    seed = cfg["split"]["random_state"]

    model_path = (
        Path(args.model_path) if args.model_path else project_path(cfg["model"]["model_path"])
    )
    model = joblib.load(model_path)

    # Each language's test set -> (reference half, current half).
    # The reference uses the TRUE language (we know where each file came from).
    test_sets = load_test_sets()
    refs, cur_parts = {}, []
    for lang in LANGUAGES:
        X, y = test_sets[lang]
        ref_idx, cur_idx = split_reference(X, mon_cfg["reference_share"], seed)
        ref = score_messages(model, X[ref_idx], threshold)
        refs[lang] = (ref, evaluate(y[ref_idx], ref["prediction"].astype(int)))
        cur_parts.append(pd.DataFrame({"message": X[cur_idx], "y": y[cur_idx], "true_lang": lang}))

    notes = []
    if args.mode == "dataset":
        pool = pd.concat(cur_parts, ignore_index=True)
        cur = score_messages(model, pool["message"], threshold)
        # Like live traffic: we do NOT use the true language, we detect it.
        cur["lang"] = [detect_language(m) for m in pool["message"]]
        cur_metrics = {}
        for lang in LANGUAGES:
            mask = cur["lang"] == lang
            cur_metrics[lang] = evaluate(pool["y"][mask], cur["prediction"][mask].astype(int))
        accuracy = (cur["lang"] == pool["true_lang"]).mean()
        notes.append(f"Language detector accuracy on the current window: {accuracy:.3f}")
        current_name = "other half of the en + tr test sets"
        default_out = mon_cfg["output_dir"]
    else:
        cur = load_log_window(project_path(cfg["api"]["predictions_log"]), args.skip, args.take)
        cur_metrics = {}  # live traffic has no true labels
        unknown = int((cur["lang"] == "unknown").sum())
        if unknown:
            notes.append(f"{unknown} log lines without 'lang' (logged before Step 7a) skipped")
        current_name = f"API log lines {args.skip + 1}-{args.skip + len(cur)}"
        default_out = mon_cfg["live_output_dir"]

    out_dir = project_path(args.out_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Old one-window charts (before Step 7a) would be misleading: remove them.
    for old in ("probability_hist.png", "length_hist.png"):
        (out_dir / old).unlink(missing_ok=True)

    sections = compare_by_language(
        refs, cur, mon_cfg["thresholds"], mon_cfg["psi_bins"], cur_metrics
    )
    alerts = [a for sec in sections.values() for a in sec["alerts"]]

    report = write_report(sections, out_dir, current_name, model_path.name, notes)
    for lang, (ref, _) in refs.items():
        cur_lang = cur[cur["lang"] == lang]
        if len(cur_lang):
            plot_histograms(ref, cur_lang, out_dir, current_name, lang)
    alert_file = write_alert(alerts, out_dir, current_name)

    for lang, sec in sections.items():
        logger.info("[%s] reference %d vs current %d", lang, sec["n_ref"], sec["n_cur"])
    for alert in alerts:
        # WARNING level: in a real system, a log tool could send these to a chat or e-mail.
        logger.warning("DRIFT ALERT: %s", alert)
    if not alerts:
        logger.info("No drift detected")
    logger.info("Report: %s%s", report, f" | Alert file: {alert_file}" if alert_file else "")
    return alerts


if __name__ == "__main__":
    main()
