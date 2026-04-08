"""Tests for the grader logic."""
import pytest

from app.grader import grade_episode
from app.models import CoverageFlags
from app.tasks import get_task


class TestGradeNormalization:
    def test_grade_in_range_easy(self):
        task = get_task("easy_remediate_handshape")
        actions = ["add_location_cue", "quick_assessment", "finalize_plan"]
        coverage = CoverageFlags(has_visual_cue=True, has_assessment=True)
        report = grade_episode(task, actions, coverage, len(actions))
        assert 0.0 <= report.total_score <= 1.0
        assert 0.0 <= report.sub_scores.intervention_relevance <= 1.0
        assert 0.0 <= report.sub_scores.pedagogical_sequence <= 1.0
        assert 0.0 <= report.sub_scores.efficiency <= 1.0

    def test_grade_in_range_hard(self):
        task = get_task("hard_adaptive_multi_error_plan")
        actions = ["select_prerequisite_sign", "add_location_cue", "slow_motion_demo",
                    "add_movement_hint", "generate_micro_drill", "quick_assessment",
                    "revision_loop", "finalize_plan"]
        coverage = CoverageFlags(
            has_prerequisite=True, has_visual_cue=True, has_timing_support=True,
            has_movement_hint=True, has_drill=True, has_assessment=True, has_revision=True,
        )
        report = grade_episode(task, actions, coverage, len(actions))
        assert 0.0 <= report.total_score <= 1.0


class TestEasyTaskCanPass:
    def test_easy_pass(self):
        task = get_task("easy_remediate_handshape")
        actions = ["add_location_cue", "slow_motion_demo", "generate_micro_drill",
                    "quick_assessment", "finalize_plan"]
        coverage = CoverageFlags(
            has_visual_cue=True, has_timing_support=True,
            has_drill=True, has_assessment=True,
        )
        report = grade_episode(task, actions, coverage, len(actions))
        assert report.passed is True, f"Score {report.total_score} did not pass"


class TestRedundantActions:
    def test_redundancy_reduces_efficiency(self):
        task = get_task("medium_fix_movement_with_scaffold")

        good_actions = ["slow_motion_demo", "add_movement_hint",
                        "generate_micro_drill", "quick_assessment", "finalize_plan"]
        good_coverage = CoverageFlags(
            has_timing_support=True, has_movement_hint=True,
            has_drill=True, has_assessment=True,
        )
        good_report = grade_episode(task, good_actions, good_coverage, len(good_actions))

        bad_actions = ["slow_motion_demo", "slow_motion_demo", "slow_motion_demo",
                       "add_movement_hint", "generate_micro_drill", "quick_assessment", "finalize_plan"]
        bad_coverage = CoverageFlags(
            has_timing_support=True, has_movement_hint=True,
            has_drill=True, has_assessment=True,
        )
        bad_report = grade_episode(task, bad_actions, bad_coverage, len(bad_actions))

        assert bad_report.sub_scores.efficiency < good_report.sub_scores.efficiency


class TestMissingRequirements:
    def test_missing_listed(self):
        task = get_task("hard_adaptive_multi_error_plan")
        actions = ["finalize_plan"]
        coverage = CoverageFlags()
        report = grade_episode(task, actions, coverage, 1)
        assert len(report.missing_requirements) > 0
        assert report.passed is False


class TestEmptyEpisode:
    def test_no_actions(self):
        task = get_task("easy_remediate_handshape")
        report = grade_episode(task, [], CoverageFlags(), 0)
        assert 0.0 <= report.total_score <= 1.0
        assert report.passed is False
