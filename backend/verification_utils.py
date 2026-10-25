"""
Shared verification logic for the Critic agent (LLM-as-a-Judge).
"""

from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph_state import VerificationResult

CRITERIA_THRESHOLD = 1  # Змінено на 1 для 0/1 оцінок
CRITERIA_NAMES = ("unambiguity", "distractor_adequacy", "grammatical_consistency", "bloom_level")

CRITERIA_LABELS = {
    "unambiguity": "Однозначність (Unambiguity)",
    "distractor_adequacy": "Адекватність дистракторів (Distractor adequacy)",
    "grammatical_consistency": "Граматична узгодженість (Grammatical consistency)",
    "bloom_level": "Рівень таксономії Блума (Bloom level)",
}


def clamp_score(value: int) -> int:
    """Ensure score is within 0-1 range for binary evaluation."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(1, score))


def compute_approval(unambiguity: int, distractor_adequacy: int, grammatical_consistency: int, bloom_level: int = 1) -> bool:
    """Question is approved only if ALL criteria scores are >= 4."""
    scores = [
        clamp_score(unambiguity),
        clamp_score(distractor_adequacy),
        clamp_score(grammatical_consistency),
        clamp_score(bloom_level),
    ]
    return all(score >= CRITERIA_THRESHOLD for score in scores)


def get_failing_criteria(unambiguity: int, distractor_adequacy: int, grammatical_consistency: int, bloom_level: int = 1) -> List[str]:
    """Return human-readable names of criteria below threshold."""
    scores = {
        "unambiguity": clamp_score(unambiguity),
        "distractor_adequacy": clamp_score(distractor_adequacy),
        "grammatical_consistency": clamp_score(grammatical_consistency),
        "bloom_level": clamp_score(bloom_level),
    }
    return [CRITERIA_LABELS[name] for name in CRITERIA_NAMES if scores[name] < CRITERIA_THRESHOLD]


def format_rejection_reason(
    unambiguity: int,
    distractor_adequacy: int,
    grammatical_consistency: int,
    feedback: str = "",
) -> str:
    """Build rejection reason from scores and critic feedback."""
    if feedback.strip():
        return feedback.strip()

    scores = {
        "unambiguity": clamp_score(unambiguity),
        "distractor_adequacy": clamp_score(distractor_adequacy),
        "grammatical_consistency": clamp_score(grammatical_consistency),
    }
    failing = [
        f"{CRITERIA_LABELS[name]}={scores[name]}"
        for name in CRITERIA_NAMES
        if scores[name] < CRITERIA_THRESHOLD
    ]
    if not failing:
        return ""
    return f"Scores below threshold ({CRITERIA_THRESHOLD}): " + ", ".join(failing)


def build_verification_result(raw: Dict[str, Any]) -> "VerificationResult":
    """Parse Critic LLM JSON into VerificationResult with approval logic."""
    from langgraph_state import VerificationResult

    unambiguity = clamp_score(raw.get("unambiguity", 1))
    distractor_adequacy = clamp_score(raw.get("distractor_adequacy", 1))
    grammatical_consistency = clamp_score(raw.get("grammatical_consistency", 1))
    bloom_level = clamp_score(raw.get("bloom_level", 1))

    issues = raw.get("issues") or []
    if not isinstance(issues, list):
        issues = [str(issues)]

    feedback = str(raw.get("feedback") or "").strip()
    approved = compute_approval(unambiguity, distractor_adequacy, grammatical_consistency, bloom_level)
    rejection_reason = ""
    if not approved:
        rejection_reason = format_rejection_reason(
            unambiguity,
            distractor_adequacy,
            grammatical_consistency,
            feedback,
        )

    return VerificationResult(
        unambiguity=unambiguity,
        distractor_adequacy=distractor_adequacy,
        grammatical_consistency=grammatical_consistency,
        bloom_level=bloom_level,
        feedback=feedback,
        issues=issues,
        approved=approved,
        rejection_reason=rejection_reason,
    )
