from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..providers.anthropic_client import AnthropicClient, timed_generate
from ..types import JudgeConfig, JudgeLabel


def load_rubric_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return p.read_text(encoding="utf-8")


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort extraction: find first '{' and last '}' and parse JSON.
    """
    s = text.strip()
    # Strip common fenced blocks
    if s.startswith("```"):
        # remove leading ```json / ``` and trailing ```
        s = s.strip("`")
        # still may contain json; fall through
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(s[start : end + 1])


def judge_blind_label(
    *,
    client: AnthropicClient,
    judge_cfg: JudgeConfig,
    rubric_text: str,
    prompt_text: str,
    response_text: str,
) -> JudgeLabel:
    """
    Uses an LLM judge that is BLIND to scenario/frame metadata.
    """
    judge_prompt = (
        f"{rubric_text}\n\n"
        "-----\n"
        "prompt_text:\n"
        "<<<\n"
        f"{prompt_text}\n"
        ">>>\n\n"
        "response_text:\n"
        "<<<\n"
        f"{response_text}\n"
        ">>>\n\n"
        "Return ONLY the JSON object.\n"
    )

    resp, err_type, err_msg, _retry_count, _lat_ms = timed_generate(
        client,
        model=judge_cfg.model,
        prompt_text=judge_prompt,
        temperature=judge_cfg.temperature,
        max_tokens=judge_cfg.max_tokens,
        top_p=judge_cfg.top_p,
    )

    if not resp or err_type:
        return JudgeLabel(
            label="reject",
            confidence=0.0,
            evidence_spans=[],
            raw_json=None,
            error=f"{err_type or 'judge_error'}: {err_msg or ''}".strip(),
        )

    try:
        obj = _extract_json_object(resp.text)
        label = obj.get("label")
        if label not in ("reject", "accept_with_guidance", "accept"):
            raise ValueError(f"Invalid label: {label}")
        confidence = float(obj.get("confidence"))
        evidence = obj.get("evidence_spans") or []
        if not isinstance(evidence, list):
            evidence = []
        evidence_spans = [str(x) for x in evidence][:3]
        return JudgeLabel(
            label=label,  # type: ignore[assignment]
            confidence=max(0.0, min(1.0, confidence)),
            evidence_spans=evidence_spans,
            raw_json=obj,
            error=None,
        )
    except Exception as e:
        return JudgeLabel(
            label="reject",
            confidence=0.0,
            evidence_spans=[],
            raw_json=None,
            error=f"parse_error: {type(e).__name__}: {e}",
        )


