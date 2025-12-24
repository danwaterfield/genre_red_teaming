from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .loaders import config_to_dict, load_experiment_config, load_frames, load_scenarios
from .logging.jsonl_logger import append_jsonl
from .matrix import (
    iter_attempt_specs,
    labels_log_path,
    load_existing_attempt_ids,
    output_dir_for_run,
    raw_log_path,
    resolve_run_id,
)
from .providers.anthropic_client import AnthropicClient, timed_generate
from .types import AttemptSpec, ExperimentConfig
from .utils import try_get_code_version, utc_now_iso
from .labeling.heuristics import heuristic_classify
from .labeling.judge import judge_blind_label, load_rubric_text
from .labeling.reconcile import reconcile_labels


def _attempt_to_dict(spec: AttemptSpec) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "attempt_id": spec.attempt_id,
        "suite_name": spec.suite_name,
        "provider_key": spec.provider_key,
        "scenario_id": spec.scenario_id,
        "trench": spec.trench,
        "frame_id": spec.frame_id,
        "replicate": spec.replicate,
        "provider": spec.provider,
        "model": spec.model,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
        "top_p": spec.top_p,
        "prompt_text": spec.prompt_text,
        "prompt_hash": spec.prompt_hash,
    }


def execute_one_attempt(
    *,
    client: AnthropicClient,
    spec: AttemptSpec,
    code_version: str | None,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    resp, err_type, err_msg, retry_count, latency_ms = timed_generate(
        client,
        model=spec.model,
        prompt_text=spec.prompt_text,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        top_p=spec.top_p,
    )
    completed_at = utc_now_iso()

    base = _attempt_to_dict(spec)
    base.update(
        {
            "started_at": started_at,
            "completed_at": completed_at,
            "latency_ms": latency_ms,
            "response_text": resp.text if resp else None,
            "stop_reason": resp.stop_reason if resp else None,
            "input_tokens": resp.input_tokens if resp else None,
            "output_tokens": resp.output_tokens if resp else None,
            "provider_request_id": resp.request_id if resp else None,
            "error_type": err_type,
            "error_message": err_msg,
            "retry_count": retry_count,
            "code_version": code_version,
        }
    )
    return base


def _get_suite(cfg: ExperimentConfig, suite_name: str | None):
    if suite_name:
        for s in cfg.suites:
            if s.name == suite_name:
                return s
        raise ValueError(f"Unknown suite: {suite_name}")
    return cfg.suites[0]


def run_experiment(
    *,
    config_path: str,
    replicates: int,
    suite_name: str | None,
) -> tuple[str, Path]:
    cfg = load_experiment_config(config_path)
    scenarios = load_scenarios(cfg.inputs.scenarios_path)
    frames = load_frames(cfg.inputs.frames_path)

    suite = _get_suite(cfg, suite_name)
    provider_key = suite.provider
    if provider_key not in cfg.providers:
        raise ValueError(f"Suite provider not found: {provider_key}")
    judge_key = suite.judge
    judge_cfg = cfg.judges[judge_key] if judge_key else None

    run_id = resolve_run_id(cfg)
    out_dir = output_dir_for_run(cfg, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_log_path(cfg, run_id)
    existing = load_existing_attempt_ids(raw_path) if cfg.run.resume else set()

    code_version = try_get_code_version()
    rubric_text = load_rubric_text(judge_cfg.rubric_path) if (judge_cfg and judge_cfg.enabled) else ""

    # Write a small metadata record once per run (non-essential; safe to ignore downstream).
    meta_path = out_dir / "run_meta.jsonl"
    append_jsonl(
        meta_path,
        {
            "run_id": run_id,
            "suite_name": suite.name,
            "provider_key": provider_key,
            "judge_key": judge_key,
            "started_at": utc_now_iso(),
            "config_path": config_path,
            "config": config_to_dict(cfg),
            "code_version": code_version,
        },
    )

    specs = list(
        iter_attempt_specs(
            cfg=cfg,
            run_id=run_id,
            suite=suite,
            suite_name=suite.name,
            provider_key=provider_key,
            scenarios=scenarios,
            frames=frames,
            replicates=replicates,
        )
    )
    pending = [s for s in specs if s.attempt_id not in existing]

    # Provider client (only Anthropic implemented today; registry can expand later)
    provider_cfg = cfg.providers[provider_key]
    if provider_cfg.type.lower() != "anthropic":
        raise ValueError(f"Unsupported provider type (generation): {provider_cfg.type}")
    client = AnthropicClient(provider_cfg)

    judge_client = None
    if judge_cfg and judge_cfg.enabled:
        judge_provider_key = judge_cfg.provider
        if judge_provider_key not in cfg.providers:
            raise ValueError(f"Judge provider not found: {judge_provider_key}")
        judge_provider_cfg = cfg.providers[judge_provider_key]
        if judge_provider_cfg.type.lower() != "anthropic":
            raise ValueError(f"Unsupported provider type (judge): {judge_provider_cfg.type}")
        judge_client = AnthropicClient(judge_provider_cfg)

    concurrency = max(1, int(provider_cfg.concurrency))
    labels_path = labels_log_path(cfg, run_id)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(execute_one_attempt, client=client, spec=spec, code_version=code_version)
            for spec in pending
        ]
        for fut in as_completed(futures):
            rec = fut.result()
            append_jsonl(raw_path, rec)

            # Two-pass write: raw first, labels second.
            if judge_cfg and judge_cfg.enabled:
                response_text = rec.get("response_text")
                error_type = rec.get("error_type")
                if not isinstance(response_text, str) or error_type:
                    append_jsonl(
                        labels_path,
                        {
                            "run_id": rec.get("run_id"),
                            "attempt_id": rec.get("attempt_id"),
                            "suite_name": rec.get("suite_name"),
                            "heuristic_label": None,
                            "judge_label": None,
                            "judge_confidence": None,
                            "judge_evidence_spans": [],
                            "final_label": None,
                            "needs_review": True,
                            "label_reason": "no_response_or_error",
                        },
                    )
                    continue

                heuristic = heuristic_classify(response_text=response_text)
                judge = judge_blind_label(
                    client=judge_client,  # type: ignore[arg-type]
                    judge_cfg=judge_cfg,
                    rubric_text=rubric_text,
                    prompt_text=str(rec.get("prompt_text") or ""),
                    response_text=response_text,
                )
                final = reconcile_labels(
                    heuristic=heuristic,
                    judge=judge,
                    confidence_threshold=judge_cfg.confidence_threshold,
                )
                append_jsonl(
                    labels_path,
                    {
                        "run_id": rec.get("run_id"),
                        "attempt_id": rec.get("attempt_id"),
                        "suite_name": rec.get("suite_name"),
                        "heuristic_label": heuristic.label,
                        "heuristic_signals": heuristic.signals,
                        "judge_label": judge.label,
                        "judge_confidence": judge.confidence,
                        "judge_evidence_spans": judge.evidence_spans,
                        "judge_error": judge.error,
                        "final_label": final.final_label,
                        "needs_review": final.needs_review,
                        "label_reason": final.reason,
                    },
                )

    return run_id, out_dir


