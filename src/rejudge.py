from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .labeling.judge import judge_blind_label, load_rubric_text
from .logging.jsonl_logger import append_jsonl, read_jsonl
from .providers.anthropic_client import AnthropicClient
from .types import ExperimentConfig, JudgeConfig
from .utils import sha256_hex, utc_now_iso


def _parse_temps(csv: str) -> list[float]:
    out: list[float] = []
    for part in csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("No temperatures provided")
    return out


def rejudge_sample(
    *,
    cfg: ExperimentConfig,
    run_id: str,
    n: int,
    seed: int,
    judge_temps_csv: str,
    out_path: str | None,
) -> Path:
    """
    Re-run the LLM judge on a random sample of successful attempts from an existing run.

    - Does NOT mutate attempts_raw.jsonl
    - Writes results to a separate JSONL so you can compare across judge temperatures
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    run_dir = Path(cfg.run.output_dir) / run_id
    raw_path = run_dir / "attempts_raw.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    judge_temps = _parse_temps(judge_temps_csv)

    # Build a judge config per temp (everything else taken from cfg.judge)
    if not cfg.judge.enabled:
        raise RuntimeError("Judge is disabled in config (judge.enabled: false)")

    rubric_text = load_rubric_text(cfg.judge.rubric_path)
    rubric_hash = sha256_hex(rubric_text)

    # Filter for successful calls with responses
    raw_records = list(read_jsonl(raw_path))
    candidates: list[dict[str, Any]] = []
    for r in raw_records:
        if r.get("error_type") is not None:
            continue
        if not isinstance(r.get("response_text"), str):
            continue
        if not isinstance(r.get("prompt_text"), str):
            continue
        if not isinstance(r.get("attempt_id"), str):
            continue
        candidates.append(r)

    if not candidates:
        raise RuntimeError(f"No successful responses found in {raw_path}")

    rng = random.Random(seed)
    sample = candidates if len(candidates) <= n else rng.sample(candidates, n)

    out = Path(out_path) if out_path else (run_dir / "judge_sweep.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    client = AnthropicClient(cfg.provider)

    sweep_id = sha256_hex(f"{run_id}|{seed}|{judge_temps}|{cfg.judge.model}|{rubric_hash}|{utc_now_iso()}")[:16]

    append_jsonl(
        out,
        {
            "type": "judge_sweep_meta",
            "sweep_id": sweep_id,
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "n_requested": n,
            "n_sampled": len(sample),
            "seed": seed,
            "judge_model": cfg.judge.model,
            "judge_max_tokens": cfg.judge.max_tokens,
            "judge_top_p": cfg.judge.top_p,
            "judge_temps": judge_temps,
            "rubric_path": cfg.judge.rubric_path,
            "rubric_hash": rubric_hash,
        },
    )

    for r in sample:
        attempt_id = r["attempt_id"]
        prompt_text = r["prompt_text"]
        response_text = r["response_text"]

        for t in judge_temps:
            judge_cfg = JudgeConfig(
                enabled=True,
                model=cfg.judge.model,
                temperature=float(t),
                max_tokens=cfg.judge.max_tokens,
                top_p=cfg.judge.top_p,
                confidence_threshold=cfg.judge.confidence_threshold,
                rubric_path=cfg.judge.rubric_path,
            )
            judged = judge_blind_label(
                client=client,
                judge_cfg=judge_cfg,
                rubric_text=rubric_text,
                prompt_text=prompt_text,
                response_text=response_text,
            )
            append_jsonl(
                out,
                {
                    "type": "judge_sweep_result",
                    "sweep_id": sweep_id,
                    "run_id": run_id,
                    "attempt_id": attempt_id,
                    "judge_model": judge_cfg.model,
                    "judge_temperature": judge_cfg.temperature,
                    "judge_label": judged.label,
                    "judge_confidence": judged.confidence,
                    "judge_evidence_spans": judged.evidence_spans,
                    "judge_error": judged.error,
                    "rubric_hash": rubric_hash,
                    "created_at": utc_now_iso(),
                },
            )

    return out


