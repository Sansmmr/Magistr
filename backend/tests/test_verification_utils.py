"""Tests for Critic criteria logic (LLM-as-a-Judge)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification_utils import (
    CRITERIA_THRESHOLD,
    build_verification_result,
    clamp_score,
    compute_approval,
    format_rejection_reason,
    get_failing_criteria,
)


def test_clamp_score_bounds():
    assert clamp_score(0) == 1
    assert clamp_score(6) == 5
    assert clamp_score("3") == 3
    assert clamp_score("bad") == 1


def test_compute_approval_all_pass():
    assert compute_approval(4, 4, 4) is True
    assert compute_approval(5, 5, 5) is True


def test_compute_approval_any_fail():
    assert compute_approval(3, 5, 5) is False
    assert compute_approval(5, 3, 5) is False
    assert compute_approval(5, 5, 3) is False


def test_get_failing_criteria():
    failing = get_failing_criteria(3, 5, 2)
    assert len(failing) == 2
    assert "Unambiguity" in failing[0]
    assert "Grammatical" in failing[1]


def test_format_rejection_reason_prefers_feedback():
    reason = format_rejection_reason(3, 3, 5, "Option B is too obvious")
    assert reason == "Option B is too obvious"


def test_build_verification_result_approved():
    result = build_verification_result({
        "unambiguity": 5,
        "distractor_adequacy": 4,
        "grammatical_consistency": 4,
        "feedback": "",
        "issues": [],
    })
    assert result.approved is True
    assert result.rejection_reason == ""


def test_build_verification_result_rejected():
    result = build_verification_result({
        "unambiguity": 3,
        "distractor_adequacy": 2,
        "grammatical_consistency": 5,
        "feedback": "Replace distractor B with a term from the text.",
        "issues": ["Distractor B is trivial"],
    })
    assert result.approved is False
    assert result.unambiguity == 3
    assert result.distractor_adequacy == 2
    assert result.feedback.startswith("Replace distractor B")
    assert result.rejection_reason == result.feedback
    assert result.issues == ["Distractor B is trivial"]


def test_threshold_is_four():
    assert CRITERIA_THRESHOLD == 4


if __name__ == "__main__":
    test_clamp_score_bounds()
    test_compute_approval_all_pass()
    test_compute_approval_any_fail()
    test_get_failing_criteria()
    test_format_rejection_reason_prefers_feedback()
    test_build_verification_result_approved()
    test_build_verification_result_rejected()
    test_threshold_is_four()
    print("All verification_utils tests passed.")
