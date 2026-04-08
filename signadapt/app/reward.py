from __future__ import annotations

from app.models import (
    ActionType,
    CoverageFlags,
    ErrorType,
    Observation,
    RewardBreakdown,
    TaskSpec,
)

# Weights for reward components
W_RELEVANCE = 0.25
W_SEQUENCE = 0.25
W_ALIGNMENT = 0.20
W_COMPLETENESS = 0.20
W_EFFICIENCY = 0.10

# Which actions are relevant to which error types
_ACTION_ERROR_MAP: dict[ActionType, set[ErrorType]] = {
    ActionType.ADD_LOCATION_CUE: {ErrorType.LOCATION},
    ActionType.ADD_MOVEMENT_HINT: {ErrorType.MOVEMENT},
    ActionType.SLOW_MOTION_DEMO: {ErrorType.TIMING, ErrorType.MOVEMENT},
    ActionType.SELECT_PREREQUISITE_SIGN: set(),
    ActionType.CHOOSE_FEEDBACK_STYLE: set(),
    ActionType.GENERATE_MICRO_DRILL: {ErrorType.HANDSHAPE, ErrorType.MOVEMENT, ErrorType.LOCATION, ErrorType.TIMING, ErrorType.ORIENTATION},
    ActionType.QUICK_ASSESSMENT: set(),
    ActionType.REVISION_LOOP: set(),
    ActionType.FINALIZE_PLAN: set(),
}

# Which coverage flags each action sets
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


def get_coverage_flag(action_type: ActionType) -> str | None:
    return _ACTION_COVERAGE.get(action_type)


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
        relevance = 0.7
    else:
        relevance = 0.2

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

    # 3) Learner need alignment
    needs = set(obs.support_needs)
    alignment = 0.5
    if ActionType.SLOW_MOTION_DEMO == action_type and "pacing" in {n.value for n in needs}:
        alignment = 1.0
    elif ActionType.ADD_LOCATION_CUE == action_type and "visual_scaffold" in {n.value for n in needs}:
        alignment = 1.0
    elif ActionType.CHOOSE_FEEDBACK_STYLE == action_type and "attention_support" in {n.value for n in needs}:
        alignment = 1.0
    elif ActionType.REVISION_LOOP == action_type and "repetition" in {n.value for n in needs}:
        alignment = 1.0
    elif action_type == ActionType.FINALIZE_PLAN:
        alignment = 0.5

    # 4) Task completeness — fraction of required coverage met
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
        efficiency = 0.4
    else:
        efficiency = 0.1

    total = (
        W_RELEVANCE * relevance
        + W_SEQUENCE * sequence_score
        + W_ALIGNMENT * alignment
        + W_COMPLETENESS * completeness
        + W_EFFICIENCY * efficiency
    )
    total = max(0.0, min(1.0, total))

    breakdown = RewardBreakdown(
        intervention_relevance=round(relevance, 4),
        pedagogical_sequence=round(sequence_score, 4),
        learner_need_alignment=round(alignment, 4),
        task_completeness=round(completeness, 4),
        efficiency=round(efficiency, 4),
        total=round(total, 4),
    )
    return total, breakdown
