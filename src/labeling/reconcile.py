from __future__ import annotations

from ..types import FinalLabel, HeuristicLabel, JudgeLabel


def reconcile_labels(
    *,
    heuristic: HeuristicLabel,
    judge: JudgeLabel,
    confidence_threshold: float,
) -> FinalLabel:
    if judge.error:
        return FinalLabel(
            final_label=None,
            needs_review=True,
            reason=f"judge_error:{judge.error}",
        )

    if judge.confidence < confidence_threshold:
        return FinalLabel(
            final_label=None,
            needs_review=True,
            reason=f"low_confidence:{judge.confidence:.3f}",
        )

    if heuristic.label == judge.label:
        return FinalLabel(
            final_label=judge.label,
            needs_review=False,
            reason="agree",
        )

    return FinalLabel(
        final_label=None,
        needs_review=True,
        reason=f"disagree:heuristic={heuristic.label},judge={judge.label}",
    )


