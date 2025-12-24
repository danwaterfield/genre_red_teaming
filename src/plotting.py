from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .logging.jsonl_logger import read_jsonl


def _load_run_df(*, raw_jsonl: Path, labels_jsonl: Path) -> pd.DataFrame:
    raw = list(read_jsonl(raw_jsonl))
    labels = list(read_jsonl(labels_jsonl)) if labels_jsonl.exists() else []

    df_raw = pd.DataFrame(raw)
    df_labels = pd.DataFrame(labels)
    if not df_labels.empty:
        # Keep latest label record per attempt_id
        df_labels = df_labels.dropna(subset=["attempt_id"]).drop_duplicates(
            subset=["attempt_id"], keep="last"
        )
        df = df_raw.merge(df_labels, on=["attempt_id"], how="left", suffixes=("", "_label"))
    else:
        df = df_raw
    return df


def _latest_run_id(outputs_dir: Path) -> str:
    candidates = [p for p in outputs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found under {outputs_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


def plot_run(
    *,
    outputs_dir: str,
    run_id: str | None,
) -> Path:
    outputs = Path(outputs_dir)
    run = run_id or _latest_run_id(outputs)
    run_dir = outputs / run
    raw_jsonl = run_dir / "attempts_raw.jsonl"
    labels_jsonl = run_dir / "attempts_labels.jsonl"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = _load_run_df(raw_jsonl=raw_jsonl, labels_jsonl=labels_jsonl)

    # Write a quick summary JSON for sanity
    summary: dict[str, Any] = {
        "run_id": run,
        "n_attempts": int(len(df)),
        "n_errors": int(df["error_type"].notna().sum()) if "error_type" in df.columns else 0,
    }
    (plots_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # --- Plot 1: error breakdown ---
    if "error_type" in df.columns:
        err_counts = df["error_type"].fillna("none").value_counts()
        plt.figure(figsize=(10, 5))
        err_counts.plot(kind="bar")
        plt.title("Error types (none = successful call)")
        plt.xlabel("error_type")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(plots_dir / "errors_by_type.png", dpi=160)
        plt.close()

    # --- Plot 2: latency histogram (successes only) ---
    if "latency_ms" in df.columns:
        lat = df.loc[df["error_type"].isna() if "error_type" in df.columns else slice(None), "latency_ms"]
        if len(lat) > 0:
            plt.figure(figsize=(8, 5))
            plt.hist(lat, bins=30)
            plt.title("Latency (ms) distribution (successes)")
            plt.xlabel("latency_ms")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(plots_dir / "latency_hist_successes.png", dpi=160)
            plt.close()

    # --- Plot 3: label distribution by trench (final_label if present else judge_label) ---
    label_col = None
    for c in ["final_label", "judge_label", "heuristic_label"]:
        if c in df.columns and df[c].notna().any():
            label_col = c
            break
    if label_col and "trench" in df.columns:
        tmp = df.copy()
        tmp[label_col] = tmp[label_col].fillna("unlabeled")
        counts = (
            tmp.pivot_table(index="trench", columns=label_col, values="attempt_id", aggfunc="count", fill_value=0)
        )
        counts = counts.loc[:, sorted(counts.columns)]
        counts.plot(kind="bar", stacked=True, figsize=(10, 5))
        plt.title(f"Outcome distribution by trench ({label_col})")
        plt.xlabel("trench")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(plots_dir / "labels_by_trench.png", dpi=160)
        plt.close()

    # --- Plot 4: needs_review rate by frame ---
    if "needs_review" in df.columns and "frame_id" in df.columns:
        tmp = df.copy()
        tmp["needs_review"] = tmp["needs_review"].fillna(False).astype(bool)
        rates = tmp.groupby("frame_id")["needs_review"].mean().sort_index()
        plt.figure(figsize=(10, 4))
        rates.plot(kind="bar")
        plt.title("Needs-review rate by frame_id")
        plt.xlabel("frame_id")
        plt.ylabel("rate")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(plots_dir / "needs_review_by_frame.png", dpi=160)
        plt.close()

    return plots_dir


