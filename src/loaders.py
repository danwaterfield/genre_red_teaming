from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .types import (
    ExperimentConfig,
    Frame,
    GenerationDefaults,
    InputPaths,
    JudgeConfig,
    ModelSpec,
    ProviderConfig,
    RetryConfig,
    RunConfig,
    Scenario,
    SuiteConfig,
)


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]


def load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def load_experiment_config(path: str) -> ExperimentConfig:
    d = load_yaml(path)

    run = _require(d, "run")
    inputs = _require(d, "inputs")

    run_cfg = RunConfig(
        run_id=run.get("run_id"),
        output_dir=_require(run, "output_dir"),
        resume=bool(run.get("resume", True)),
    )

    # Backward compatibility:
    # - Old schema: provider/generation/models/judge at top-level
    # - New schema: providers/generation_defaults/judges/suites
    if "providers" not in d:
        provider = _require(d, "provider")
        generation = _require(d, "generation")
        models = _require(d, "models")
        judge = _require(d, "judge")

        retry_cfg = RetryConfig(
            max_retries=int(_require(provider["retries"], "max_retries")),
            base_delay_s=float(_require(provider["retries"], "base_delay_s")),
            max_delay_s=float(_require(provider["retries"], "max_delay_s")),
            jitter=bool(provider["retries"].get("jitter", True)),
        )
        providers_cfg: dict[str, ProviderConfig] = {
            "default": ProviderConfig(
                type=_require(provider, "name"),
                timeout_s=float(_require(provider, "timeout_s")),
                concurrency=int(_require(provider, "concurrency")),
                retries=retry_cfg,
            )
        }

        generation_cfg = GenerationDefaults(
            max_tokens=int(_require(generation, "max_tokens")),
            top_p=float(_require(generation, "top_p")),
        )

        model_specs: list[ModelSpec] = []
        if not isinstance(models, list) or not models:
            raise ValueError("models must be a non-empty list")
        for m in models:
            model_specs.append(
                ModelSpec(
                    model=_require(m, "model"),
                    temperatures=[float(x) for x in _require(m, "temperatures")],
                )
            )

        judges_cfg: dict[str, JudgeConfig] = {
            "default": JudgeConfig(
                enabled=bool(_require(judge, "enabled")),
                provider="default",
                model=_require(judge, "model"),
                temperature=float(_require(judge, "temperature")),
                max_tokens=int(_require(judge, "max_tokens")),
                top_p=float(_require(judge, "top_p")),
                confidence_threshold=float(_require(judge, "confidence_threshold")),
                rubric_path=_require(judge, "rubric_path"),
            )
        }

        suites_cfg = [
            SuiteConfig(
                name="default",
                provider="default",
                models=model_specs,
                judge="default",
            )
        ]
    else:
        providers = _require(d, "providers")
        generation = _require(d, "generation_defaults")
        judges = _require(d, "judges")
        suites = _require(d, "suites")

        if not isinstance(providers, dict) or not providers:
            raise ValueError("providers must be a non-empty mapping")

        providers_cfg = {}
        for key, p in providers.items():
            if not isinstance(p, dict):
                raise ValueError(f"providers.{key} must be a mapping")
            pr = _require(p, "retries")
            retry_cfg = RetryConfig(
                max_retries=int(_require(pr, "max_retries")),
                base_delay_s=float(_require(pr, "base_delay_s")),
                max_delay_s=float(_require(pr, "max_delay_s")),
                jitter=bool(pr.get("jitter", True)),
            )
            providers_cfg[str(key)] = ProviderConfig(
                type=_require(p, "type"),
                timeout_s=float(_require(p, "timeout_s")),
                concurrency=int(_require(p, "concurrency")),
                retries=retry_cfg,
            )

        generation_cfg = GenerationDefaults(
            max_tokens=int(_require(generation, "max_tokens")),
            top_p=float(_require(generation, "top_p")),
        )

        if not isinstance(judges, dict) or not judges:
            raise ValueError("judges must be a non-empty mapping")
        judges_cfg = {}
        for key, j in judges.items():
            if not isinstance(j, dict):
                raise ValueError(f"judges.{key} must be a mapping")
            judges_cfg[str(key)] = JudgeConfig(
                enabled=bool(_require(j, "enabled")),
                provider=_require(j, "provider"),
                model=_require(j, "model"),
                temperature=float(_require(j, "temperature")),
                max_tokens=int(_require(j, "max_tokens")),
                top_p=float(_require(j, "top_p")),
                confidence_threshold=float(_require(j, "confidence_threshold")),
                rubric_path=_require(j, "rubric_path"),
            )

        if not isinstance(suites, list) or not suites:
            raise ValueError("suites must be a non-empty list")
        suites_cfg: list[SuiteConfig] = []
        for s in suites:
            if not isinstance(s, dict):
                raise ValueError("Each suite must be a mapping")
            models = _require(s, "models")
            model_specs: list[ModelSpec] = []
            for m in models:
                model_specs.append(
                    ModelSpec(
                        model=_require(m, "model"),
                        temperatures=[float(x) for x in _require(m, "temperatures")],
                    )
                )
            suites_cfg.append(
                SuiteConfig(
                    name=_require(s, "name"),
                    provider=_require(s, "provider"),
                    models=model_specs,
                    judge=s.get("judge"),
                )
            )

    inputs_cfg = InputPaths(
        scenarios_path=_require(inputs, "scenarios_path"),
        frames_path=_require(inputs, "frames_path"),
    )

    return ExperimentConfig(
        run=run_cfg,
        providers=providers_cfg,
        generation_defaults=generation_cfg,
        judges=judges_cfg,
        suites=suites_cfg,
        inputs=inputs_cfg,
    )


def load_scenarios(path: str) -> list[Scenario]:
    d = load_yaml(path)
    scenarios = _require(d, "scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("scenarios must be a list")
    out: list[Scenario] = []
    for s in scenarios:
        out.append(
            Scenario(
                id=_require(s, "id"),
                trench=_require(s, "trench"),
                title=_require(s, "title"),
                base_prompt=_require(s, "base_prompt"),
                variables=dict(s.get("variables") or {}),
            )
        )
    return out


def load_frames(path: str) -> list[Frame]:
    d = load_yaml(path)
    frames = _require(d, "frames")
    if not isinstance(frames, list):
        raise ValueError("frames must be a list")
    out: list[Frame] = []
    for f in frames:
        out.append(
            Frame(
                id=_require(f, "id"),
                name=_require(f, "name"),
                prefix=str(f.get("prefix") or ""),
                suffix=str(f.get("suffix") or ""),
            )
        )
    return out


def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """
    Convenience for logging the experiment config.
    """
    return asdict(cfg)


