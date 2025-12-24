from __future__ import annotations

import re
from typing import Any

from .types import Frame, Scenario
from .utils import sha256_hex

_VAR_PATTERN = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")


def render_template(template: str, variables: dict[str, str]) -> str:
    """
    Very small templating helper:
      - replaces occurrences of {{VAR_NAME}} with variables[VAR_NAME]
      - leaves unknown variables as-is (so you can detect missing fills later)
    """

    def _repl(m: re.Match[str]) -> str:
        key = m.group(1)
        return variables.get(key, m.group(0))

    return _VAR_PATTERN.sub(_repl, template)


def build_prompt_text(
    scenario: Scenario,
    frame: Frame,
    extra_vars: dict[str, str] | None = None,
) -> str:
    """
    Final prompt text = frame.prefix + rendered(scenario.base_prompt) + frame.suffix
    """
    vars_all: dict[str, str] = dict(scenario.variables)
    if extra_vars:
        vars_all.update(extra_vars)
    rendered = render_template(scenario.base_prompt, vars_all)
    return f"{frame.prefix}{rendered}{frame.suffix}"


def prompt_hash(prompt_text: str) -> str:
    return sha256_hex(prompt_text)


