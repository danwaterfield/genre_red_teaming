from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import anthropic

from ..types import ProviderConfig
from ..utils import Timer, backoff_sleep_s, ensure_env


@dataclass(frozen=True)
class AnthropicResponse:
    text: str
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    request_id: str | None


class AnthropicClient:
    """
    Thin wrapper around the Anthropic SDK with:
    - API key from env
    - timeout
    - retry/backoff for transient errors
    """

    def __init__(self, cfg: ProviderConfig) -> None:
        if cfg.type.lower() != "anthropic":
            raise ValueError(f"Unsupported provider: {cfg.type}")
        self._cfg = cfg
        api_key = ensure_env("ANTHROPIC_API_KEY")
        # Most Anthropic SDK versions accept a client-level timeout.
        self._client = anthropic.Anthropic(api_key=api_key, timeout=self._cfg.timeout_s)

    def generate_text(
        self,
        *,
        model: str,
        prompt_text: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> tuple[AnthropicResponse | None, str | None, str | None, int]:
        """
        Returns: (response, error_type, error_message, retry_count)
        """
        retry_count = 0
        last_err: Exception | None = None

        for attempt in range(0, self._cfg.retries.max_retries + 1):
            if attempt > 0:
                retry_count += 1
                delay = backoff_sleep_s(
                    attempt_num=attempt,
                    base_delay_s=self._cfg.retries.base_delay_s,
                    max_delay_s=self._cfg.retries.max_delay_s,
                    jitter=self._cfg.retries.jitter,
                )
                time.sleep(delay)

            try:
                # NOTE: Anthropic SDK supports per-request timeout via client option in newer versions.
                # We set timeout on the client; avoid per-request kwargs to reduce SDK version drift.
                kwargs: dict[str, Any] = {}

                resp = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    # Some Anthropic models disallow specifying both `temperature` and `top_p`.
                    # We always drive sampling via temperature in this harness, so omit top_p.
                    messages=[{"role": "user", "content": prompt_text}],
                    **kwargs,
                )

                # content can be a list of blocks
                text_parts: list[str] = []
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        text_parts.append(getattr(block, "text", ""))
                text = "".join(text_parts).strip()

                usage = getattr(resp, "usage", None)
                input_tokens = getattr(usage, "input_tokens", None) if usage else None
                output_tokens = getattr(usage, "output_tokens", None) if usage else None

                return (
                    AnthropicResponse(
                        text=text,
                        stop_reason=getattr(resp, "stop_reason", None),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        request_id=getattr(resp, "id", None),
                    ),
                    None,
                    None,
                    retry_count,
                )

            except anthropic.RateLimitError as e:
                last_err = e
                continue
            except anthropic.APIStatusError as e:
                # Retry on overload-ish statuses; treat other statuses as terminal.
                last_err = e
                status = getattr(e, "status_code", None)
                if status in (429, 500, 502, 503, 504, 529):
                    continue
                break
            except anthropic.APIConnectionError as e:
                last_err = e
                continue
            except Exception as e:
                last_err = e
                break

        if last_err is None:
            return None, "unknown_error", "unknown error", retry_count
        return None, type(last_err).__name__, str(last_err), retry_count


def timed_generate(
    client: AnthropicClient,
    *,
    model: str,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> tuple[AnthropicResponse | None, str | None, str | None, int, int]:
    """
    Adds latency measurement (ms) around generate_text.
    """
    t = Timer()
    resp, err_type, err_msg, retry_count = client.generate_text(
        model=model,
        prompt_text=prompt_text,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return resp, err_type, err_msg, retry_count, t.ms()


