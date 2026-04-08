from __future__ import annotations

from app.models import (
    ActionType,
    CoverageFlags,
    ErrorType,
    LearnerState,
    Observation,
    RewardBreakdown,
    TaskSpec,
)

W_RELEVANCE = 0.20
W_SEQUENCE = 0.20
W_ALIGNMENT = 0.15
W_COMPLETENESS = 0.20
W_EFFICIENCY = 0.10
W_LEARNER = 0.15

_ACTION_ERROR_MAP: dict[ActionType, set[ErrorType]] = {
    ActionType.ADD_LOCATION_CUE: {ErrorType.LOCATION},
    ActionType.ADD_MOVEMENT_HINT: {ErrorType.MOVEMENT},
    ActionType.SLOW_MOTION_DEMO: {ErrorType.TIMING, ErrorType.MOVEMENT},
    ActionType.SELECT_PREREQUISITE_SIGN: set(),
    ActionType.CHOOSE_FEEDBACK_STYLE: set(),
    ActionType.GENERATE_MICRO_DRILL: {
        ErrorType.HANDSHAPE, ErrorType.MOVEMENT, ErrorType.LOCATION,
        ErrorType.TIMING, ErrorType.ORIENTATION,
    },
    ActionType.QUICK_ASSESSMENT: set(),
    ActionType.REVISION_LOOP: set(),
    ActionType.FINALIZE_PLAN: set(),
}

_ACTION_COVERAGE: dict[ActionType, str] = {
    ActionType.SELECT_PREREQUISITE_SIGN: "has_prerequisite",
    ActionType.SLOW_MOTION_DEMO: "has_timing_support",
    ActionType.ADD_LOCATION_CUE: "has_visual_cue",
    ActionType.ADD_MOVEMENT_HINT: "has_movement_hint",
    ActionType.CHOOSE_FEEDBACK_STYLE: "has_feedback_style",
    ActionType.GENERATE_MICRO_DRILL: "has_drill",
    ActionType.QUICK_ASSESSMENT: "has_assessment",
    ActionType.REVISION_LOOP: "has_revision",
}

_LEARNER_EFFECTS: dict[ActionType, dict[str, float]] = {
    ActionType.SELECT_PREREQUISITE_SIGN: {"comprehension": 0.10, "confidence": 0.05, "engagement": 0.05},
    ActionType.SLOW_MOTION_DEMO: {"comprehension": 0.15, "frustration": -0.10, "confidence": 0.05},
    ActionType.ADD_LOCATION_CUE: {"comprehension": 0.10, "error_reduction": 0.10},
    ActionType.ADD_MOVEMENT_HINT: {"comprehension": 0.10, "error_reduction": 0.10},
    ActionType.CHOOSE_FEEDBACK_STYLE: {"engagement": 0.15, "frustration": -0.10},
    ActionType.GENERATE_MICRO_DRILL: {"comprehension": 0.10, "error_reduction": 0.15, "confidence": 0.05},
    ActionType.QUICK_ASSESSMENT: {"confidence": 0.10, "engagement": 0.05},
    ActionType.REVISION_LOOP: {"comprehension": 0.10, "error_reduction": 0.10, "confidence": 0.05},
    ActionType.FINALIZE_PLAN: {},
}


def get_coverage_flag(action_type: ActionType) -> str | None:
    return _ACTION_COVERAGE.get(action_type)


def update_learner_state(
    state: LearnerState,
    action_type: ActionType,
    is_duplicate: bool,
) -> LearnerState:
    """Apply action effects to learner state, return new state."""
    effects = _LEARNER_EFFECTS.get(action_type, {})
    scale = 0.3 if is_duplicate else 1.0

    confidence = min(1.0, max(0.0, state.confidence + effects.get("confidence", 0.0) * scale))
    comprehension = min(1.0, max(0.0, state.comprehension + effects.get("comprehension", 0.0) * scale))
    frustration = min(1.0, max(0.0, state.frustration + effects.get("frustration", 0.0) * scale))
    engagement = min(1.0, max(0.0, state.engagement + effects.get("engagement", 0.0) * scale))
    error_reduction = min(1.0, max(0.0, state.error_reduction + effects.get("error_reduction", 0.0) * scale))

    if is_duplicate:
        frustration = min(1.0, frustration + 0.05)
        engagement = max(0.0, engagement - 0.05)

    return LearnerState(
        confidence=round(confidence, 4),
        comprehension=round(comprehension, 4),
        frustration=round(frustration, 4),
        engagement=round(engagement, 4),
        error_reduction=round(error_reduction, 4),
    )


def compute_step_reward(
    action_type: ActionType,
    obs: Observation,
    task: TaskSpec,
    prior_plan: list[str],
) -> tuple[float, RewardBreakdown]:
    """Return (scalar_reward, breakdown) for a single step."""

    error_types = {ep.error_type for ep in task.error_patterns}

    # 1) Intervention relevance
    relevant_errors = _ACTION_ERROR_MAP.get(action_type, set())
    if relevant_errors & error_types:
        relevance = 1.0
    elif action_type in (
        ActionType.SELECT_PREREQUISITE_SIGN,
        ActionType.CHOOSE_FEEDBACK_STYLE,
        ActionType.QUICK_ASSESSMENT,
        ActionType.REVISION_LOOP,
        ActionType.FINALIZE_PLAN,
    ):
        relevance = 0.6
    else:
        relevance = 0.15

    # 2) Pedagogical sequence — reward if ideal order is roughly followed
    ideal = task.grader_params.ideal_action_order
    current_plan = prior_plan + [action_type.value]
    matched = 0
    ideal_idx = 0
    for a in current_plan:
        if ideal_idx < len(ideal) and a == ideal[ideal_idx]:
            matched += 1
            ideal_idx += 1
    sequence_score = matched / max(len(ideal), 1)

    # Penalize ordering violations
    if task.grader_params.prerequisite_before_drill:
        if action_type == ActionType.GENERATE_MICRO_DRILL and not obs.coverage.has_prerequisite:
            if task.lesson_goal.prerequisite_signs:
                sequence_score *= 0.5

    if task.grader_params.assessment_before_revision:
        if action_type == ActionType.REVISION_LOOP and not obs.coverage.has_assessment:
            sequence_score *= 0.5

    # 3) Learner need alignment
    needs = {n.value for n in obs.support_needs}
    alignment = 0.4
    need_action_map = {
        "pacing": ActionType.SLOW_MOTION_DEMO,
        "visual_scaffold": ActionType.ADD_LOCATION_CUE,
        "attention_support": ActionType.CHOOSE_FEEDBACK_STYLE,
        "repetition": ActionType.REVISION_LOOP,
        "motivational": ActionType.CHOOSE_FEEDBACK_STYLE,
    }
    for need, target_action in need_action_map.items():
        if need in needs and action_type == target_action:
            alignment = 1.0
            break

    if action_type == ActionType.FINALIZE_PLAN:
        alignment = 0.5

    # 4) Task completeness — fraction of required coverage met (including this action)
    required = task.grader_params.required_coverage
    coverage_dict = obs.coverage.model_dump()
    flag = get_coverage_flag(action_type)
    if flag:
        coverage_dict[flag] = True
    met = sum(1 for r in required if coverage_dict.get(r, False))
    completeness = met / max(len(required), 1)

    # 5) Efficiency — penalize duplicates
    dup_count = prior_plan.count(action_type.value)
    if dup_count == 0:
        efficiency = 1.0
    elif dup_count == 1:
        efficiency = 0.3
    else:
        efficiency = 0.05

    # 6) Learner state quality
    ls = obs.learner_state
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
        + W_SEQUENCE * sequence_score
        + W_ALIGNMENT * alignment
        + W_COMPLETENESS * completeness
        + W_EFFICIENCY * efficiency
        + W_LEARNER * learner_quality
    )
    total = max(0.0, min(1.0, total))

    breakdown = RewardBreakdown(
        intervention_relevance=round(relevance, 4),
        pedagogical_sequence=round(sequence_score, 4),
        learner_need_alignment=round(alignment, 4),
        task_completeness=round(completeness, 4),
        efficiency=round(efficiency, 4),
        learner_state_quality=round(learner_quality, 4),
        total=round(total, 4),
    )
    return total, breakdown
