from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_hex(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def stable_attempt_id(parts: list[str]) -> str:
    return sha256_hex("|".join(parts))


def try_get_code_version() -> str | None:
    """
    Best-effort git SHA. Returns None if not a git repo or git is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def backoff_sleep_s(
    attempt_num: int,
    base_delay_s: float,
    max_delay_s: float,
    jitter: bool,
) -> float:
    """
    Exponential backoff with optional jitter.
    attempt_num: 1 for first retry (not first attempt).
    """
    delay = min(max_delay_s, base_delay_s * (2 ** (attempt_num - 1)))
    if jitter:
        delay = delay * (0.5 + random.random())
    return delay


def ensure_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return val


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


class Timer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)


