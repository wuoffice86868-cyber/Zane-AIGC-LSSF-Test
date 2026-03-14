"""Tests for RewardCalculator — 3-dimensional Seedance-aligned model.

Dimensions:
  - foundational: structural stability + prompt adherence
  - motion: temporal quality + camera smoothness
  - aesthetic: visual appeal + composition
"""

import numpy as np
import pytest

from prompt_evaluator.models import (
    DimensionScore,
    QCResult,
    RewardBreakdown,
    RewardDimension,
)
from prompt_evaluator.reward_calculator import RewardCalculator, _route_fail

from .conftest import make_sample


# ---------------------------------------------------------------------------
# Fail routing
# ---------------------------------------------------------------------------

class TestFailRouting:
    def test_foundational_fails(self):
        for f in ["face_morphing", "object_morphing", "structural_collapse",
                   "reflection_error", "hand_error"]:
            assert _route_fail(f) == RewardDimension.FOUNDATIONAL

    def test_motion_fails(self):
        for f in ["action_loop", "camera_jitter", "temporal_flicker",
                   "unnatural_motion", "freeze_frame"]:
            assert _route_fail(f) == RewardDimension.MOTION

    def test_aesthetic_fails(self):
        for f in ["color_banding", "overexposure", "underexposure",
                   "low_resolution", "compression_artifact"]:
            assert _route_fail(f) == RewardDimension.AESTHETIC

    def test_unknown_defaults_to_foundational(self):
        assert _route_fail("some_unknown_fail") == RewardDimension.FOUNDATIONAL


# ---------------------------------------------------------------------------
# Basic reward calculation
# ---------------------------------------------------------------------------

class TestBasicReward:
    def setup_method(self):
        self.calc = RewardCalculator()

    def test_perfect_pass_high_reward(self):
        qc = {"pass": True, "confidence": 0.98}
        result = self.calc.calculate(qc)
        assert isinstance(result, RewardBreakdown)
        assert result.total_score > 70.0  # Clean pass should score well
        assert result.total_score <= 100.0

    def test_has_three_dimensions(self):
        qc = {"pass": True, "confidence": 0.90}
        result = self.calc.calculate(qc)
        assert RewardDimension.FOUNDATIONAL in result.dimensions
        assert RewardDimension.MOTION in result.dimensions
        assert RewardDimension.AESTHETIC in result.dimensions

    def test_dimension_scores_are_valid(self):
        qc = {"pass": True, "confidence": 0.90}
        result = self.calc.calculate(qc)
        for dim_name, dim_score in result.dimensions.items():
            assert isinstance(dim_score, DimensionScore)
            assert 0.0 <= dim_score.score <= 1.0
            assert 0.0 <= dim_score.weight <= 1.0
            assert dim_score.weighted_score == pytest.approx(
                dim_score.score * dim_score.weight, abs=0.001
            )

    def test_weights_sum_to_one(self):
        qc = {"pass": True, "confidence": 0.90}
        result = self.calc.calculate(qc)
        total_weight = sum(d.weight for d in result.dimensions.values())
        assert total_weight == pytest.approx(1.0, abs=0.01)

    def test_fail_lower_than_pass(self):
        pass_qc = {"pass": True, "confidence": 0.90}
        fail_qc = {
            "pass": False, "confidence": 0.85,
            "auto_fail_triggered": ["object_morphing"],
        }
        pass_result = self.calc.calculate(pass_qc)
        fail_result = self.calc.calculate(fail_qc)
        assert fail_result.total_score < pass_result.total_score


# ---------------------------------------------------------------------------
# Foundational dimension
# ---------------------------------------------------------------------------

class TestFoundationalDimension:
    def setup_method(self):
        self.calc = RewardCalculator()

    def test_foundational_fail_reduces_score(self):
        clean = {"pass": True, "confidence": 0.90}
        dirty = {
            "pass": False, "confidence": 0.90,
            "auto_fail_triggered": ["face_morphing", "object_morphing"],
        }
        clean_r = self.calc.calculate(clean)
        dirty_r = self.calc.calculate(dirty)
        clean_f = clean_r.dimensions[RewardDimension.FOUNDATIONAL].score
        dirty_f = dirty_r.dimensions[RewardDimension.FOUNDATIONAL].score
        assert dirty_f < clean_f

    def test_adherence_from_clip_score(self):
        """clip_score parameter maps to foundational adherence."""
        qc = {"pass": True, "confidence": 0.90}
        low = self.calc.calculate(qc, clip_score=0.2)
        high = self.calc.calculate(qc, clip_score=0.9)
        low_f = low.dimensions[RewardDimension.FOUNDATIONAL].score
        high_f = high.dimensions[RewardDimension.FOUNDATIONAL].score
        assert high_f > low_f

    def test_adherence_from_gemini_dict(self):
        """Auto-extracts prompt_adherence_score from dict."""
        qc = {"pass": True, "confidence": 0.90, "prompt_adherence_score": 9}
        result = self.calc.calculate(qc)
        comps = result.dimensions[RewardDimension.FOUNDATIONAL].components
        assert comps["adherence"] == pytest.approx(0.9, abs=0.01)


# ---------------------------------------------------------------------------
# Motion dimension
# ---------------------------------------------------------------------------

class TestMotionDimension:
    def setup_method(self):
        self.calc = RewardCalculator()

    def test_action_loop_reduces_motion(self):
        clean = {"pass": True, "confidence": 0.90}
        loopy = {
            "pass": False, "confidence": 0.90,
            "auto_fail_triggered": ["action_loop"],
        }
        clean_m = self.calc.calculate(clean).dimensions[RewardDimension.MOTION].score
        loopy_m = self.calc.calculate(loopy).dimensions[RewardDimension.MOTION].score
        assert loopy_m < clean_m

    def test_motion_score_parameter(self):
        qc = {"pass": True, "confidence": 0.90}
        low = self.calc.calculate(qc, motion_score=0.2)
        high = self.calc.calculate(qc, motion_score=0.9)
        assert high.dimensions[RewardDimension.MOTION].score > low.dimensions[RewardDimension.MOTION].score

    def test_motion_from_gemini_dict(self):
        qc = {"pass": True, "confidence": 0.90, "motion_score": 8}
        result = self.calc.calculate(qc)
        comps = result.dimensions[RewardDimension.MOTION].components
        assert comps["smoothness"] == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# Aesthetic dimension
# ---------------------------------------------------------------------------

class TestAestheticDimension:
    def setup_method(self):
        self.calc = RewardCalculator()

    def test_aesthetic_from_param(self):
        qc = {"pass": True, "confidence": 0.90}
        low = self.calc.calculate(qc, aesthetic=0.2)
        high = self.calc.calculate(qc, aesthetic=0.9)
        assert high.dimensions[RewardDimension.AESTHETIC].score > low.dimensions[RewardDimension.AESTHETIC].score

    def test_aesthetic_from_gemini_dict(self):
        qc = {"pass": True, "confidence": 0.90, "aesthetic_score": 7}
        result = self.calc.calculate(qc)
        comps = result.dimensions[RewardDimension.AESTHETIC].components
        assert comps["appeal"] == pytest.approx(0.7, abs=0.01)

    def test_aesthetic_fail_reduces(self):
        clean = {"pass": True, "confidence": 0.90}
        ugly = {
            "pass": False, "confidence": 0.90,
            "auto_fail_triggered": ["color_banding", "overexposure"],
        }
        clean_a = self.calc.calculate(clean).dimensions[RewardDimension.AESTHETIC].score
        ugly_a = self.calc.calculate(ugly).dimensions[RewardDimension.AESTHETIC].score
        assert ugly_a < clean_a


# ---------------------------------------------------------------------------
# Cross-dimension routing
# ---------------------------------------------------------------------------

class TestCrossDimensionRouting:
    """Verify that auto-fails affect only their routed dimension."""

    def setup_method(self):
        self.calc = RewardCalculator()

    def test_motion_fail_doesnt_affect_foundational(self):
        base = {"pass": True, "confidence": 0.90}
        motion_fail = {
            "pass": False, "confidence": 0.90,
            "auto_fail_triggered": ["action_loop"],
        }
        base_f = self.calc.calculate(base).dimensions[RewardDimension.FOUNDATIONAL].score
        fail_f = self.calc.calculate(motion_fail).dimensions[RewardDimension.FOUNDATIONAL].score
        # Foundational is affected by pass/fail status (pass_bonus + default adherence change)
        # but the motion-specific auto_fail should NOT add structural penalty
        fail_comps = self.calc.calculate(motion_fail).dimensions[RewardDimension.FOUNDATIONAL].components
        assert fail_comps["auto_fail_count"] == 0  # No foundational auto-fails routed

    def test_foundational_fail_doesnt_affect_motion_much(self):
        base = {"pass": True, "confidence": 0.90}
        found_fail = {
            "pass": False, "confidence": 0.90,
            "auto_fail_triggered": ["object_morphing"],
        }
        base_m = self.calc.calculate(base).dimensions[RewardDimension.MOTION].score
        fail_m = self.calc.calculate(found_fail).dimensions[RewardDimension.MOTION].score
        # Motion dimension should only differ by pass/fail effect on defaults
        assert abs(fail_m - base_m) < 0.25


# ---------------------------------------------------------------------------
# Custom weights
# ---------------------------------------------------------------------------

class TestCustomWeights:
    def test_custom_weights_applied(self):
        calc = RewardCalculator(weights={
            RewardDimension.FOUNDATIONAL: 0.6,
            RewardDimension.MOTION: 0.2,
            RewardDimension.AESTHETIC: 0.2,
        })
        qc = {"pass": True, "confidence": 0.90}
        result = calc.calculate(qc)
        assert result.dimensions[RewardDimension.FOUNDATIONAL].weight == 0.6


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_accepts_qcresult_model(self):
        calc = RewardCalculator()
        qc = QCResult(**{"pass": True, "confidence": 0.96})
        result = calc.calculate(qc)
        assert result.total_score > 0

    def test_total_score_clamped(self):
        calc = RewardCalculator()
        qc = {"pass": True, "confidence": 0.99}
        result = calc.calculate(qc)
        assert 0.0 <= result.total_score <= 100.0

    def test_catastrophic_fail_not_negative(self):
        calc = RewardCalculator()
        qc = {
            "pass": False, "confidence": 0.30,
            "auto_fail_triggered": ["a", "b", "c", "d", "e", "f"],
            "minor_issues": ["x", "y", "z"],
        }
        result = calc.calculate(qc)
        assert result.total_score >= 0.0


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class TestBatchCalculate:
    def test_batch(self, mixed_samples):
        calc = RewardCalculator()
        results = calc.batch_calculate(mixed_samples)
        assert len(results) == len(mixed_samples)
        assert all(isinstance(r, RewardBreakdown) for r in results)
        assert all(0 <= r.total_score <= 100 for r in results)


# ---------------------------------------------------------------------------
# Weight fitting
# ---------------------------------------------------------------------------

class TestFitWeights:
    def test_fit_returns_weights(self):
        calc = RewardCalculator()
        samples = [
            make_sample(f"s{i}", human_score=5 + i, confidence=0.7 + i * 0.05)
            for i in range(5)
        ]
        w = calc.fit_weights(samples)
        assert RewardDimension.FOUNDATIONAL in w or "foundational" in str(w)
        assert sum(w.values()) == pytest.approx(1.0, abs=0.05)

    def test_fit_too_few_samples(self):
        calc = RewardCalculator()
        samples = [make_sample("s0"), make_sample("s1")]
        w = calc.fit_weights(samples)
        assert w == calc.weights

    def test_fit_with_extra_scores(self):
        calc = RewardCalculator()
        samples = [
            make_sample(f"s{i}", human_score=3 + i, confidence=0.6 + i * 0.05)
            for i in range(6)
        ]
        extras = [
            {"aesthetic": 0.3 + i * 0.1, "motion": 0.5 + i * 0.06, "adherence": 0.4 + i * 0.08}
            for i in range(6)
        ]
        w = calc.fit_weights(samples, extra_scores=extras)
        assert sum(w.values()) == pytest.approx(1.0, abs=0.05)
