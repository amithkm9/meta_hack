from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class ErrorType(str, Enum):
    HANDSHAPE = "handshape"
    MOVEMENT = "movement"
    LOCATION = "location"
    TIMING = "timing"
    ORIENTATION = "orientation"


class ActionType(str, Enum):
    SELECT_PREREQUISITE_SIGN = "select_prerequisite_sign"
    SLOW_MOTION_DEMO = "slow_motion_demo"
    ADD_LOCATION_CUE = "add_location_cue"
    ADD_MOVEMENT_HINT = "add_movement_hint"
    CHOOSE_FEEDBACK_STYLE = "choose_feedback_style"
    GENERATE_MICRO_DRILL = "generate_micro_drill"
    QUICK_ASSESSMENT = "quick_assessment"
    REVISION_LOOP = "revision_loop"
    FINALIZE_PLAN = "finalize_plan"


class FeedbackStyle(str, Enum):
    VISUAL = "visual"
    HAPTIC = "haptic"
    VERBAL = "verbal"
    MODELING = "modeling"


class SupportNeed(str, Enum):
    VISUAL_SCAFFOLD = "visual_scaffold"
    PACING = "pacing"
    REPETITION = "repetition"
    ATTENTION_SUPPORT = "attention_support"
    MOTIVATIONAL = "motivational"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ── Domain Models ──────────────────────────────────────────────────────

class LearnerProfile(BaseModel):
    age_band: str
    proficiency: str
    primary_language: str = "ASL"
    support_needs: list[SupportNeed] = Field(default_factory=list)


class LessonGoal(BaseModel):
    target_sign: str
    description: str
    prerequisite_signs: list[str] = Field(default_factory=list)


class ErrorPattern(BaseModel):
    error_type: ErrorType
    severity: float = Field(ge=0.0, le=1.0)
    description: str


class TutoringConstraint(BaseModel):
    max_steps: int
    required_outputs: list[str] = Field(default_factory=list)


class TutoringRequirement(BaseModel):
    name: str
    description: str
    met: bool = False


# ── Learner State Simulation ──────────────────────────────────────────

class LearnerState(BaseModel):
    """Simulated learner cognitive/affective state that changes with actions."""
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    comprehension: float = Field(default=0.1, ge=0.0, le=1.0)
    frustration: float = Field(default=0.2, ge=0.0, le=1.0)
    engagement: float = Field(default=0.5, ge=0.0, le=1.0)
    error_reduction: float = Field(default=0.0, ge=0.0, le=1.0)


# ── Observation / Action ───────────────────────────────────────────────

class CoverageFlags(BaseModel):
    has_prerequisite: bool = False
    has_visual_cue: bool = False
    has_timing_support: bool = False
    has_movement_hint: bool = False
    has_feedback_style: bool = False
    has_drill: bool = False
    has_assessment: bool = False
    has_revision: bool = False


class Observation(BaseModel):
    task_id: str
    difficulty: Difficulty
    learner: LearnerProfile
    lesson_goal: LessonGoal
    error_patterns: list[ErrorPattern]
    support_needs: list[SupportNeed]
    tutoring_plan: list[str] = Field(default_factory=list)
    completed_requirements: list[str] = Field(default_factory=list)
    coverage: CoverageFlags = Field(default_factory=CoverageFlags)
    learner_state: LearnerState = Field(default_factory=LearnerState)
    remaining_steps: int
    allowed_actions: list[ActionType]
    last_action_result: str | None = None
    step_count: int = 0


class Action(BaseModel):
    action_type: ActionType
    rationale: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Reward / Results ───────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    intervention_relevance: float = 0.0
    pedagogical_sequence: float = 0.0
    learner_need_alignment: float = 0.0
    task_completeness: float = 0.0
    efficiency: float = 0.0
    learner_state_quality: float = 0.0
    total: float = 0.0


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class GradeReport(BaseModel):
    total_score: float
    sub_scores: RewardBreakdown
    passed: bool
    reasoning: str
    missing_requirements: list[str] = Field(default_factory=list)


# ── Task Spec ──────────────────────────────────────────────────────────

class GraderParams(BaseModel):
    required_coverage: list[str] = Field(default_factory=list)
    ideal_action_order: list[str] = Field(default_factory=list)
    min_actions: int = 1
    required_error_coverage: list[str] = Field(default_factory=list)
    prerequisite_before_drill: bool = False
    assessment_before_revision: bool = False


class TaskSpec(BaseModel):
    id: str
    difficulty: Difficulty
    learner: LearnerProfile
    lesson_goal: LessonGoal
    error_patterns: list[ErrorPattern]
    constraints: TutoringConstraint
    requirements: list[TutoringRequirement]
    grader_params: GraderParams


# ── API Models ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None


class ResetResponse(BaseModel):
    episode_id: str
    observation: Observation
    done: bool = False


class StepRequest(BaseModel):
    episode_id: str
    action: Action


class StateResponse(BaseModel):
    episode_id: str
    task_id: str
    step_count: int
    max_steps: int
    done: bool
    cumulative_reward: float
    observation: Observation
    final_grade: GradeReport | None = None


class TaskSummary(BaseModel):
    id: str
    difficulty: Difficulty
    target_sign: str
    description: str
    max_steps: int
