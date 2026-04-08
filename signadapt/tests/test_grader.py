"""Tests for the grader logic."""
import pytest

from app.grader import grade_episode
from app.models import CoverageFlags, LearnerState
from app.tasks import get_task


def _good_learner() -> LearnerState:
    return LearnerState(
        confidence=0.7, comprehension=0.7,
        frustration=0.1, engagement=0.8, error_reduction=0.6,
    )


def _poor_learner() -> LearnerState:
    return LearnerState(
        confidence=0.2, comprehension=0.1,
        frustration=0.6, engagement=0.3, error_reduction=0.0,
    )


class TestGradeNormalization:
    def test_grade_in_range_easy(self):
        task = get_task("easy_remediate_handshape")
        actions = ["add_location_cue", "quick_assessment", "finalize_plan"]
        coverage = CoverageFlags(has_visual_cue=True, has_assessment=True)
        report = grade_episode(task, actions, coverage, len(actions), _good_learner())
        assert 0.0 <= report.total_score <= 1.0
        assert 0.0 <= report.sub_scores.intervention_relevance <= 1.0
        assert 0.0 <= report.sub_scores.pedagogical_sequence <= 1.0
        assert 0.0 <= report.sub_scores.efficiency <= 1.0
        assert 0.0 <= report.sub_scores.learner_state_quality <= 1.0

    def test_grade_in_range_hard(self):
        task = get_task("hard_adaptive_multi_error_plan")
        actions = [
            "select_prerequisite_sign", "add_location_cue", "slow_motion_demo",
            "add_movement_hint", "generate_micro_drill", "quick_assessment",
            "revision_loop", "finalize_plan",
        ]
        coverage = CoverageFlags(
            has_prerequisite=True, has_visual_cue=True, has_timing_support=True,
            has_movement_hint=True, has_drill=True, has_assessment=True, has_revision=True,
        )
        report = grade_episode(task, actions, coverage, len(actions), _good_learner())
        assert 0.0 <= report.total_score <= 1.0

    @pytest.mark.parametrize("task_id", [
        "easy_remediate_handshape",
        "easy_fix_orientation",
        "medium_fix_movement_with_scaffold",
        "hard_adaptive_multi_error_plan",
        "hard_complex_sentence_tutoring",
    ])
    def test_all_tasks_grade_in_range(self, task_id):
        task = get_task(task_id)
        report = grade_episode(task, ["finalize_plan"], CoverageFlags(), 1, _poor_learner())
        assert 0.0 <= report.total_score <= 1.0


class TestEasyTaskCanPass:
    def test_easy_pass(self):
        task = get_task("easy_remediate_handshape")
        actions = [
            "add_location_cue", "slow_motion_demo", "generate_micro_drill",
            "quick_assessment", "finalize_plan",
        ]
        coverage = CoverageFlags(
            has_visual_cue=True, has_timing_support=True,
            has_drill=True, has_assessment=True,
        )
        report = grade_episode(task, actions, coverage, len(actions), _good_learner())
        assert report.passed is True, f"Score {report.total_score} did not pass"


class TestRedundantActions:
    def test_redundancy_reduces_efficiency(self):
        task = get_task("medium_fix_movement_with_scaffold")

        good_actions = [
            "slow_motion_demo", "add_movement_hint",
            "generate_micro_drill", "quick_assessment", "finalize_plan",
        ]
        good_coverage = CoverageFlags(
            has_timing_support=True, has_movement_hint=True,
            has_drill=True, has_assessment=True,
        )
        good_report = grade_episode(task, good_actions, good_coverage, len(good_actions), _good_learner())

        bad_actions = [
            "slow_motion_demo", "slow_motion_demo", "slow_motion_demo",
            "add_movement_hint", "generate_micro_drill", "quick_assessment", "finalize_plan",
        ]
        bad_coverage = CoverageFlags(
            has_timing_support=True, has_movement_hint=True,
            has_drill=True, has_assessment=True,
        )
        bad_report = grade_episode(task, bad_actions, bad_coverage, len(bad_actions), _good_learner())

        assert bad_report.sub_scores.efficiency < good_report.sub_scores.efficiency


class TestMissingRequirements:
    def test_missing_listed(self):
        task = get_task("hard_adaptive_multi_error_plan")
        actions = ["finalize_plan"]
        coverage = CoverageFlags()
        report = grade_episode(task, actions, coverage, 1, _poor_learner())
        assert len(report.missing_requirements) > 0
        assert report.passed is False


class TestEmptyEpisode:
    def test_no_actions(self):
        task = get_task("easy_remediate_handshape")
        report = grade_episode(task, [], CoverageFlags(), 0, _poor_learner())
        assert 0.0 <= report.total_score <= 1.0
        assert report.passed is False


class TestScoreVariance:
    def test_different_strategies_different_scores(self):
        """The grader must produce different scores for different action sequences."""
        task = get_task("hard_adaptive_multi_error_plan")

        ideal = [
            "select_prerequisite_sign", "add_location_cue", "slow_motion_demo",
            "add_movement_hint", "generate_micro_drill", "choose_feedback_style",
            "quick_assessment", "revision_loop", "finalize_plan",
        ]
        ideal_coverage = CoverageFlags(
            has_prerequisite=True, has_visual_cue=True, has_timing_support=True,
            has_movement_hint=True, has_drill=True, has_feedback_style=True,
            has_assessment=True, has_revision=True,
        )
        score_ideal = grade_episode(task, ideal, ideal_coverage, len(ideal), _good_learner()).total_score

        partial = ["slow_motion_demo", "quick_assessment", "finalize_plan"]
        partial_coverage = CoverageFlags(has_timing_support=True, has_assessment=True)
        score_partial = grade_episode(task, partial, partial_coverage, len(partial), _poor_learner()).total_score

        just_finalize = ["finalize_plan"]
        score_finalize = grade_episode(task, just_finalize, CoverageFlags(), 1, _poor_learner()).total_score

        assert score_ideal > score_partial > score_finalize, (
            f"Scores should be monotonically decreasing: {score_ideal}, {score_partial}, {score_finalize}"
        )

    def test_ordering_matters(self):
        """Ordering violations should lower the score."""
        task = get_task("hard_adaptive_multi_error_plan")
        good_ls = _good_learner()

        correct_order = [
            "select_prerequisite_sign", "slow_motion_demo", "generate_micro_drill",
            "quick_assessment", "revision_loop", "finalize_plan",
        ]
        bad_order = [
            "generate_micro_drill", "slow_motion_demo", "select_prerequisite_sign",
            "revision_loop", "quick_assessment", "finalize_plan",
        ]
        coverage = CoverageFlags(
            has_prerequisite=True, has_timing_support=True,
            has_drill=True, has_assessment=True, has_revision=True,
        )

        score_correct = grade_episode(task, correct_order, coverage, len(correct_order), good_ls).total_score
        score_bad = grade_episode(task, bad_order, coverage, len(bad_order), good_ls).total_score

        assert score_correct > score_bad, f"Correct order {score_correct} should beat bad order {score_bad}"
