"""Tests for the core SignAdapt environment."""
import pytest

from app.env import SignAdaptEnv
from app.models import Action, ActionType


@pytest.fixture
def env() -> SignAdaptEnv:
    return SignAdaptEnv()


class TestReset:
    def test_reset_returns_valid_state(self, env: SignAdaptEnv):
        resp = env.reset(task_id="easy_remediate_handshape")
        assert resp.episode_id
        assert resp.done is False
        obs = resp.observation
        assert obs.task_id == "easy_remediate_handshape"
        assert obs.remaining_steps == 6
        assert obs.step_count == 0

    def test_reset_default_task(self, env: SignAdaptEnv):
        resp = env.reset()
        assert resp.observation.task_id == "easy_remediate_handshape"

    def test_reset_clears_previous(self, env: SignAdaptEnv):
        env.reset(task_id="easy_remediate_handshape")
        env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
        resp = env.reset(task_id="easy_remediate_handshape")
        assert resp.observation.step_count == 0
        assert resp.observation.tutoring_plan == []


class TestStep:
    def test_step_advances_state(self, env: SignAdaptEnv):
        env.reset(task_id="easy_remediate_handshape")
        result = env.step(Action(action_type=ActionType.ADD_LOCATION_CUE, rationale="fix handshape"))
        assert result.observation.step_count == 1
        assert result.observation.remaining_steps == 5
        assert result.done is False

    def test_finalize_ends_episode(self, env: SignAdaptEnv):
        env.reset(task_id="easy_remediate_handshape")
        result = env.step(Action(action_type=ActionType.FINALIZE_PLAN))
        assert result.done is True
        assert "final_grade" in result.info

    def test_max_steps_ends_episode(self, env: SignAdaptEnv):
        env.reset(task_id="easy_remediate_handshape")
        for _ in range(6):
            result = env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))
        assert result.done is True

    def test_step_after_done_raises(self, env: SignAdaptEnv):
        env.reset(task_id="easy_remediate_handshape")
        env.step(Action(action_type=ActionType.FINALIZE_PLAN))
        with pytest.raises(RuntimeError):
            env.step(Action(action_type=ActionType.SLOW_MOTION_DEMO))


class TestReward:
    def test_reward_in_range(self, env: SignAdaptEnv):
        env.reset(task_id="medium_fix_movement_with_scaffold")
        actions = [
            ActionType.SELECT_PREREQUISITE_SIGN,
            ActionType.SLOW_MOTION_DEMO,
            ActionType.ADD_MOVEMENT_HINT,
            ActionType.GENERATE_MICRO_DRILL,
            ActionType.QUICK_ASSESSMENT,
            ActionType.FINALIZE_PLAN,
        ]
        for at in actions:
            result = env.step(Action(action_type=at))
            assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range for {at}"

    def test_final_score_in_range(self, env: SignAdaptEnv):
        env.reset(task_id="easy_remediate_handshape")
        env.step(Action(action_type=ActionType.ADD_LOCATION_CUE))
        env.step(Action(action_type=ActionType.QUICK_ASSESSMENT))
        result = env.step(Action(action_type=ActionType.FINALIZE_PLAN))
        grade = result.info["final_grade"]
        assert 0.0 <= grade["total_score"] <= 1.0


class TestCoverage:
    def test_coverage_flags_update(self, env: SignAdaptEnv):
        env.reset(task_id="hard_adaptive_multi_error_plan")
        env.step(Action(action_type=ActionType.SELECT_PREREQUISITE_SIGN))
        state = env.state()
        assert state.observation.coverage.has_prerequisite is True

    def test_requirements_tracked(self, env: SignAdaptEnv):
        env.reset(task_id="easy_remediate_handshape")
        env.step(Action(action_type=ActionType.ADD_LOCATION_CUE))
        state = env.state()
        assert "corrective_intervention" in state.observation.completed_requirements
