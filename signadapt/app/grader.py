from __future__ import annotations

from app.models import (
    CoverageFlags,
    GradeReport,
    LearnerState,
    RewardBreakdown,
    TaskSpec,
)

W_RELEVANCE = 0.20
W_SEQUENCE = 0.20
W_ALIGNMENT = 0.15
W_COMPLETENESS = 0.20
W_EFFICIENCY = 0.10
W_LEARNER = 0.15

PASS_THRESHOLD = 0.70


def grade_episode(
    task: TaskSpec,
    action_history: list[str],
    coverage: CoverageFlags,
    step_count: int,
    learner_state: LearnerState,
) -> GradeReport:
    """Deterministic final grading based on structured state."""

    gp = task.grader_params
    coverage_dict = coverage.model_dump()

    # 1) Intervention relevance — fraction of required error types addressed
    required_errors = set(gp.required_error_coverage)
    addressed: set[str] = set()
    action_to_error: dict[str, set[str]] = {
        "add_location_cue": {"location"},
        "add_movement_hint": {"movement"},
        "slow_motion_demo": {"timing", "movement"},
        "generate_micro_drill": {"handshape", "movement", "location", "timing", "orientation"},
    }
    for a in action_history:
        if a in action_to_error:
            addressed |= action_to_error[a]
    covered_errors = addressed & required_errors
    relevance = len(covered_errors) / max(len(required_errors), 1)

    # 2) Pedagogical sequence — LCS ratio with ideal order
    ideal = gp.ideal_action_order
    lcs_len = _lcs_length(action_history, ideal)
    sequence = lcs_len / max(len(ideal), 1)

    # Penalize ordering violations
    ordering_penalty = 0.0
    if gp.prerequisite_before_drill:
        if "generate_micro_drill" in action_history and "select_prerequisite_sign" in action_history:
            drill_idx = action_history.index("generate_micro_drill")
            prereq_idx = action_history.index("select_prerequisite_sign")
            if drill_idx < prereq_idx:
                ordering_penalty += 0.15
        elif "generate_micro_drill" in action_history and task.lesson_goal.prerequisite_signs:
            if "select_prerequisite_sign" not in action_history:
                ordering_penalty += 0.10

    if gp.assessment_before_revision:
        if "revision_loop" in action_history and "quick_assessment" in action_history:
            rev_idx = action_history.index("revision_loop")
            assess_idx = action_history.index("quick_assessment")
            if rev_idx < assess_idx:
                ordering_penalty += 0.15
        elif "revision_loop" in action_history and "quick_assessment" not in action_history:
            ordering_penalty += 0.10

    sequence = max(0.0, sequence - ordering_penalty)

    # 3) Learner need alignment — coverage flags mapping to support needs
    need_map = {
        "visual_scaffold": "has_visual_cue",
        "pacing": "has_timing_support",
        "repetition": "has_revision",
        "attention_support": "has_feedback_style",
        "motivational": "has_feedback_style",
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

    # 5) Efficiency — penalize excess/redundant actions
    unique_actions = len(set(action_history))
    total_actions = len(action_history)
    if total_actions == 0:
        efficiency = 0.0
    else:
        uniqueness_ratio = unique_actions / total_actions
        budget_ratio = total_actions / max(gp.min_actions, 1)
        if budget_ratio <= 1.0:
            efficiency = uniqueness_ratio * budget_ratio
        elif budget_ratio <= 1.5:
            efficiency = uniqueness_ratio * (1.0 - 0.3 * (budget_ratio - 1.0))
        else:
            efficiency = uniqueness_ratio * max(0.1, 2.0 - budget_ratio)
    efficiency = max(0.0, min(1.0, efficiency))

    # 6) Learner state quality
    ls = learner_state
    learner_quality = (
        0.25 * ls.comprehension
        + 0.20 * ls.confidence
        + 0.20 * (1.0 - ls.frustration)
        + 0.15 * ls.engagement
        + 0.20 * ls.error_reduction
    )
    learner_quality = max(0.0, min(1.0, learner_quality))

    total = (
        W_RELEVANCE * relevance
        + W_SEQUENCE * sequence
        + W_ALIGNMENT * alignment
        + W_COMPLETENESS * completeness
        + W_EFFICIENCY * efficiency
        + W_LEARNER * learner_quality
    )
    total = max(0.0, min(1.0, total))

    passed = total >= PASS_THRESHOLD

    strengths: list[str] = []
    weaknesses: list[str] = []

    if relevance >= 0.8:
        strengths.append("Good error coverage")
    elif relevance < 0.5:
        weaknesses.append("Poor error coverage")

    if sequence >= 0.6:
        strengths.append("Actions follow a logical pedagogical sequence")
    elif sequence < 0.3:
        weaknesses.append("Action sequence diverges from ideal pedagogy")

    if completeness >= 0.8:
        strengths.append("Most task requirements met")
    elif completeness < 0.5:
        weaknesses.append("Several task requirements unmet")

    if efficiency >= 0.6:
        strengths.append("Efficient use of step budget")
    elif efficiency < 0.3:
        weaknesses.append("Redundant actions waste steps")

    if learner_quality >= 0.6:
        strengths.append("Learner state improved well")
    elif learner_quality < 0.3:
        weaknesses.append("Learner state poorly managed")

    if alignment >= 0.8:
        strengths.append("Good alignment with learner support needs")
    elif alignment < 0.4:
        weaknesses.append("Learner support needs not addressed")

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
            learner_state_quality=round(learner_quality, 4),
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
