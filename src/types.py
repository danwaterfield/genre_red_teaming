from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

OutcomeLabel = Literal["reject", "accept_with_guidance", "accept"]


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int
    base_delay_s: float
    max_delay_s: float
    jitter: bool


@dataclass(frozen=True)
class ProviderConfig:
    type: str
    timeout_s: float
    concurrency: int
    retries: RetryConfig


@dataclass(frozen=True)
class GenerationDefaults:
    max_tokens: int
    top_p: float


@dataclass(frozen=True)
class ModelSpec:
    model: str
    temperatures: list[float]


@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool
    provider: str
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    confidence_threshold: float
    rubric_path: str


@dataclass(frozen=True)
class InputPaths:
    scenarios_path: str
    frames_path: str


@dataclass(frozen=True)
class RunConfig:
    run_id: str | None
    output_dir: str
    resume: bool


@dataclass(frozen=True)
class ExperimentConfig:
    run: RunConfig
    # New schema:
    providers: dict[str, ProviderConfig]
    generation_defaults: GenerationDefaults
    judges: dict[str, JudgeConfig]
    suites: list["SuiteConfig"]
    inputs: InputPaths


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    provider: str
    models: list[ModelSpec]
    judge: str | None


@dataclass(frozen=True)
class Scenario:
    id: str
    trench: str
    title: str
    base_prompt: str
    variables: dict[str, str]


@dataclass(frozen=True)
class Frame:
    id: str
    name: str
    prefix: str
    suffix: str


@dataclass(frozen=True)
class AttemptSpec:
    run_id: str
    attempt_id: str
    suite_name: str
    provider_key: str
    scenario_id: str
    trench: str
    frame_id: str
    replicate: int
    provider: str
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    prompt_text: str
    prompt_hash: str


@dataclass(frozen=True)
class RawAttemptResult:
    started_at: str
    completed_at: str
    latency_ms: int
    response_text: str | None
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    provider_request_id: str | None
    error_type: str | None
    error_message: str | None
    retry_count: int


@dataclass(frozen=True)
class HeuristicLabel:
    label: OutcomeLabel
    signals: dict[str, Any]


@dataclass(frozen=True)
class JudgeLabel:
    label: OutcomeLabel
    confidence: float
    evidence_spans: list[str]
    raw_json: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True)
class FinalLabel:
    final_label: OutcomeLabel | None
    needs_review: bool
    reason: str


