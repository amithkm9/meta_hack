from __future__ import annotations

import uuid

from app.grader import grade_episode
from app.models import (
    Action,
    ActionType,
    CoverageFlags,
    GradeReport,
    LearnerState,
    Observation,
    ResetResponse,
    StateResponse,
    StepResult,
    TaskSpec,
)
from app.reward import compute_step_reward, get_coverage_flag, update_learner_state
from app.tasks import get_task, list_tasks, sample_task


class SignAdaptEnv:
    """Stateful tutoring-planning environment."""

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._task: TaskSpec | None = None
        self._step_count: int = 0
        self._done: bool = True
        self._cumulative_reward: float = 0.0
        self._action_history: list[str] = []
        self._tutoring_plan: list[str] = []
        self._coverage: CoverageFlags = CoverageFlags()
        self._completed_requirements: list[str] = []
        self._last_action_result: str | None = None
        self._final_grade: GradeReport | None = None
        self._learner_state: LearnerState = LearnerState()

    def reset(self, task_id: str | None = None) -> ResetResponse:
        if task_id:
            self._task = get_task(task_id)
        else:
            self._task = sample_task()

        self._episode_id = uuid.uuid4().hex[:12]
        self._step_count = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._action_history = []
        self._tutoring_plan = []
        self._coverage = CoverageFlags()
        self._completed_requirements = []
        self._last_action_result = None
        self._final_grade = None

        base_frustration = {"easy": 0.15, "medium": 0.25, "hard": 0.35}.get(
            self._task.difficulty.value, 0.2
        )
        base_comprehension = {"easy": 0.15, "medium": 0.10, "hard": 0.05}.get(
            self._task.difficulty.value, 0.1
        )
        self._learner_state = LearnerState(
            confidence=0.3,
            comprehension=base_comprehension,
            frustration=base_frustration,
            engagement=0.5,
            error_reduction=0.0,
        )

        return ResetResponse(
            episode_id=self._episode_id,
            observation=self._build_observation(),
            done=False,
        )

    def step(self, action: Action) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")
        if self._task is None:
            raise RuntimeError("No task loaded. Call reset() first.")

        self._step_count += 1
        at = action.action_type
        is_duplicate = at.value in self._action_history

        # Update coverage
        flag = get_coverage_flag(at)
        if flag and not getattr(self._coverage, flag, False):
            setattr(self._coverage, flag, True)

        # Update requirements
        self._update_requirements(at)

        # Update learner state
        self._learner_state = update_learner_state(
            self._learner_state, at, is_duplicate
        )

        # Build plan entry
        desc = f"{at.value}"
        if action.rationale:
            desc += f" — {action.rationale}"
        self._tutoring_plan.append(desc)
        self._action_history.append(at.value)

        # Compute reward
        reward, breakdown = compute_step_reward(
            at,
            self._build_observation(),
            self._task,
            self._action_history[:-1],
        )
        self._cumulative_reward += reward
        self._last_action_result = f"Applied {at.value} (reward={reward:.3f})"

        # Check for episode end
        finalize = at == ActionType.FINALIZE_PLAN
        budget_exhausted = self._step_count >= self._task.constraints.max_steps

        if finalize or budget_exhausted:
            self._done = True
            self._final_grade = grade_episode(
                self._task,
                self._action_history,
                self._coverage,
                self._step_count,
                self._learner_state,
            )

        obs = self._build_observation()
        info: dict = {"reward_breakdown": breakdown.model_dump()}
        if self._final_grade:
            info["final_grade"] = self._final_grade.model_dump()

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    def state(self) -> StateResponse:
        if self._task is None:
            raise RuntimeError("No task loaded. Call reset() first.")
        return StateResponse(
            episode_id=self._episode_id,
            task_id=self._task.id,
            step_count=self._step_count,
            max_steps=self._task.constraints.max_steps,
            done=self._done,
            cumulative_reward=round(self._cumulative_reward, 4),
            observation=self._build_observation(),
            final_grade=self._final_grade,
        )

    def _build_observation(self) -> Observation:
        task = self._task
        assert task is not None
        remaining = task.constraints.max_steps - self._step_count
        allowed = list(ActionType)
        if self._done:
            allowed = []

        return Observation(
            task_id=task.id,
            difficulty=task.difficulty,
            learner=task.learner,
            lesson_goal=task.lesson_goal,
            error_patterns=task.error_patterns,
            support_needs=task.learner.support_needs,
            tutoring_plan=list(self._tutoring_plan),
            completed_requirements=list(self._completed_requirements),
            coverage=self._coverage.model_copy(),
            learner_state=self._learner_state.model_copy(),
            remaining_steps=max(remaining, 0),
            allowed_actions=allowed,
            last_action_result=self._last_action_result,
            step_count=self._step_count,
        )

    _REQUIREMENT_ACTION_MAP: dict[str, set[str]] = {
        "corrective_intervention": {
            ActionType.ADD_LOCATION_CUE.value,
            ActionType.ADD_MOVEMENT_HINT.value,
            ActionType.SLOW_MOTION_DEMO.value,
            ActionType.GENERATE_MICRO_DRILL.value,
        },
        "assessment": {ActionType.QUICK_ASSESSMENT.value},
        "scaffolded_sequence": {
            ActionType.SELECT_PREREQUISITE_SIGN.value,
            ActionType.SLOW_MOTION_DEMO.value,
            ActionType.ADD_MOVEMENT_HINT.value,
        },
        "timing_support": {ActionType.SLOW_MOTION_DEMO.value},
        "drill": {ActionType.GENERATE_MICRO_DRILL.value},
        "prerequisite_review": {ActionType.SELECT_PREREQUISITE_SIGN.value},
        "location_cue": {ActionType.ADD_LOCATION_CUE.value},
        "movement_support": {
            ActionType.ADD_MOVEMENT_HINT.value,
            ActionType.SLOW_MOTION_DEMO.value,
        },
        "revision": {ActionType.REVISION_LOOP.value},
        "feedback_selection": {ActionType.CHOOSE_FEEDBACK_STYLE.value},
    }

    def _update_requirements(self, at: ActionType) -> None:
        task = self._task
        assert task is not None
        for req in task.requirements:
            if req.name in self._completed_requirements:
                continue
            allowed_actions = self._REQUIREMENT_ACTION_MAP.get(req.name, set())
            if at.value in allowed_actions:
                self._completed_requirements.append(req.name)
                req.met = True
