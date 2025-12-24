from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Iterable

from .prompting import build_prompt_text, prompt_hash
from .types import AttemptSpec, ExperimentConfig, Frame, Scenario, SuiteConfig
from .utils import stable_attempt_id


def resolve_run_id(cfg: ExperimentConfig) -> str:
    return cfg.run.run_id or str(uuid.uuid4())


def output_dir_for_run(cfg: ExperimentConfig, run_id: str) -> Path:
    return Path(cfg.run.output_dir) / run_id


def raw_log_path(cfg: ExperimentConfig, run_id: str) -> Path:
    return output_dir_for_run(cfg, run_id) / "attempts_raw.jsonl"


def labels_log_path(cfg: ExperimentConfig, run_id: str) -> Path:
    return output_dir_for_run(cfg, run_id) / "attempts_labels.jsonl"


def csv_path(cfg: ExperimentConfig, run_id: str) -> Path:
    return output_dir_for_run(cfg, run_id) / "attempts.csv"


def load_existing_attempt_ids(raw_jsonl_path: Path) -> set[str]:
    """
    Used for resume: skip attempt_ids already logged (even if they errored).
    """
    if not raw_jsonl_path.exists():
        return set()
    out: set[str] = set()
    with raw_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            attempt_id = obj.get("attempt_id")
            if isinstance(attempt_id, str):
                out.add(attempt_id)
    return out


def iter_attempt_specs(
    *,
    cfg: ExperimentConfig,
    run_id: str,
    suite: SuiteConfig,
    suite_name: str,
    provider_key: str,
    scenarios: list[Scenario],
    frames: list[Frame],
    replicates: int,
) -> Iterable[AttemptSpec]:
    if replicates < 1:
        raise ValueError("replicates must be >= 1")

    for scenario in scenarios:
        for frame in frames:
            prompt_text = build_prompt_text(scenario, frame)
            p_hash = prompt_hash(prompt_text)

            for model in suite.models:
                for temperature in model.temperatures:
                    for replicate in range(1, replicates + 1):
                        attempt_id = stable_attempt_id(
                            [
                                suite_name,
                                provider_key,
                                scenario.id,
                                frame.id,
                                model.model,
                                str(temperature),
                                str(replicate),
                                p_hash,
                                str(cfg.generation_defaults.max_tokens),
                                str(cfg.generation_defaults.top_p),

                            ]
                        )
                        yield AttemptSpec(
                            run_id=run_id,
                            attempt_id=attempt_id,
                            suite_name=suite_name,
                            provider_key=provider_key,
                            scenario_id=scenario.id,
                            trench=scenario.trench,
                            frame_id=frame.id,
                            replicate=replicate,
                            provider=cfg.providers[provider_key].type,
                            model=model.model,
                            temperature=temperature,
                            max_tokens=cfg.generation_defaults.max_tokens,
                            top_p=cfg.generation_defaults.top_p,
                            prompt_text=prompt_text,
                            prompt_hash=p_hash,
                        )


