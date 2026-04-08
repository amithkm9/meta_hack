from __future__ import annotations

from app.models import (
    CoverageFlags,
    GradeReport,
    RewardBreakdown,
    TaskSpec,
)

W_RELEVANCE = 0.25
W_SEQUENCE = 0.25
W_ALIGNMENT = 0.20
W_COMPLETENESS = 0.20
W_EFFICIENCY = 0.10

PASS_THRESHOLD = 0.70


def grade_episode(
    task: TaskSpec,
    action_history: list[str],
    coverage: CoverageFlags,
    step_count: int,
) -> GradeReport:
    """Deterministic final grading based on structured state."""

    gp = task.grader_params
    coverage_dict = coverage.model_dump()

    # 1) Intervention relevance — fraction of required error types addressed
    error_types = {ep.error_type.value for ep in task.error_patterns}
    required_errors = set(gp.required_error_coverage)
    addressed: set[str] = set()
    action_to_error: dict[str, set[str]] = {
        "add_location_cue": {"location"},
        "add_movement_hint": {"movement"},
        "slow_motion_demo": {"timing", "movement"},
        "generate_micro_drill": {"handshape", "movement", "location", "timing", "orientation"},
        "add_location_cue": {"location"},
    }
    for a in action_history:
        if a in action_to_error:
            addressed |= action_to_error[a]
    covered_errors = addressed & required_errors
    relevance = len(covered_errors) / max(len(required_errors), 1)

    # 2) Pedagogical sequence — longest common subsequence ratio with ideal order
    ideal = gp.ideal_action_order
    lcs_len = _lcs_length(action_history, ideal)
    sequence = lcs_len / max(len(ideal), 1)

    # 3) Learner need alignment — check coverage flags that map to support needs
    need_map = {
        "visual_scaffold": "has_visual_cue",
        "pacing": "has_timing_support",
        "repetition": "has_revision",
        "attention_support": "has_feedback_style",
    }
    learner_needs = [n.value for n in task.learner.support_needs]
    if learner_needs:
        met_needs = sum(
            1 for n in learner_needs
            if n in need_map and coverage_dict.get(need_map[n], False)
        )
        alignment = met_needs / len(learner_needs)
    else:
        alignment = 1.0

    # 4) Task completeness — fraction of required coverage flags met
    required_flags = gp.required_coverage
    met_flags = [f for f in required_flags if coverage_dict.get(f, False)]
    completeness = len(met_flags) / max(len(required_flags), 1)

    missing = [f for f in required_flags if not coverage_dict.get(f, False)]

    # 5) Efficiency — penalize excess actions
    unique_actions = len(set(action_history))
    total_actions = len(action_history)
    if total_actions == 0:
        efficiency = 0.0
    else:
        ratio = unique_actions / total_actions
        budget_use = min(total_actions / max(gp.min_actions, 1), 1.5)
        if budget_use <= 1.0:
            efficiency = ratio * budget_use
        else:
            efficiency = ratio * max(0.0, 2.0 - budget_use)
    efficiency = max(0.0, min(1.0, efficiency))

    total = (
        W_RELEVANCE * relevance
        + W_SEQUENCE * sequence
        + W_ALIGNMENT * alignment
        + W_COMPLETENESS * completeness
        + W_EFFICIENCY * efficiency
    )
    total = max(0.0, min(1.0, total))

    passed = total >= PASS_THRESHOLD

    strengths: list[str] = []
    weaknesses: list[str] = []

    if relevance >= 0.8:
        strengths.append("Good error coverage")
    elif relevance < 0.5:
        weaknesses.append("Poor error coverage")

    if sequence >= 0.7:
        strengths.append("Actions follow a logical pedagogical sequence")
    elif sequence < 0.4:
        weaknesses.append("Action sequence diverges from ideal pedagogy")

    if completeness >= 0.8:
        strengths.append("Most task requirements met")
    elif completeness < 0.5:
        weaknesses.append("Several task requirements unmet")

    if efficiency >= 0.7:
        strengths.append("Efficient use of step budget")
    elif efficiency < 0.4:
        weaknesses.append("Redundant actions waste steps")

    reasoning_parts: list[str] = []
    if strengths:
        reasoning_parts.append("Strengths: " + "; ".join(strengths) + ".")
    if weaknesses:
        reasoning_parts.append("Weaknesses: " + "; ".join(weaknesses) + ".")
    reasoning = " ".join(reasoning_parts) if reasoning_parts else "Adequate performance."

    return GradeReport(
        total_score=round(total, 4),
        sub_scores=RewardBreakdown(
            intervention_relevance=round(relevance, 4),
            pedagogical_sequence=round(sequence, 4),
            learner_need_alignment=round(alignment, 4),
            task_completeness=round(completeness, 4),
            efficiency=round(efficiency, 4),
            total=round(total, 4),
        ),
        passed=passed,
        reasoning=reasoning,
        missing_requirements=missing,
    )


def _lcs_length(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]
