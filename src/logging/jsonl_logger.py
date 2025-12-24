from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def latest_labels_by_attempt_id(labels_jsonl: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(labels_jsonl):
        attempt_id = rec.get("attempt_id")
        if isinstance(attempt_id, str):
            latest[attempt_id] = rec
    return latest


def export_csv(
    *,
    raw_jsonl: Path,
    labels_jsonl: Path,
    out_csv: Path,
    extra_columns: list[str] | None = None,
) -> None:
    raw_records = list(read_jsonl(raw_jsonl))
    labels = latest_labels_by_attempt_id(labels_jsonl)

    # Canonical column order (stable for analysis)
    columns = [
        # identity
        "run_id",
        "attempt_id",
        "scenario_id",
        "trench",
        "frame_id",
        "replicate",
        # model params
        "provider",
        "model",
        "temperature",
        "max_tokens",
        "top_p",
        # timing
        "started_at",
        "completed_at",
        "latency_ms",
        # inputs
        "prompt_text",
        "prompt_hash",
        # outputs
        "response_text",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "provider_request_id",
        # labels
        "heuristic_label",
        "judge_label",
        "judge_confidence",
        "judge_evidence_spans",
        "final_label",
        "needs_review",
        "label_reason",
        # ops
        "error_type",
        "error_message",
        "retry_count",
        "code_version",
    ]
    if extra_columns:
        for c in extra_columns:
            if c not in columns:
                columns.append(c)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", quoting=csv.QUOTE_ALL)
        w.writeheader()

        for raw in raw_records:
            attempt_id = raw.get("attempt_id")
            merged = dict(raw)
            if isinstance(attempt_id, str) and attempt_id in labels:
                merged.update(labels[attempt_id])
            # Ensure all columns present
            row = {k: merged.get(k) for k in columns}
            w.writerow(row)


