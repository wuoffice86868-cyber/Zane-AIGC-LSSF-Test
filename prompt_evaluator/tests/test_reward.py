"""Tests for RewardCalculator."""

import numpy as np
import pytest

from prompt_evaluator.models import QCResult, RewardBreakdown
from prompt_evaluator.reward_calculator import RewardCalculator

from .conftest import make_sample


class TestQCOnlyReward:
    """Test QC-only reward mode (no aesthetic/clip/motion scores)."""

    def setup_method(self):
        self.calc = RewardCalculator()

    def test_perfect_pass(self):
        qc = {"pass": True, "confidence": 0.98}
        result = self.calc.calculate(qc)
        assert isinstance(result, RewardBreakdown)
        # pass_score=100, issue_penalty=0, confidence_boost=10 → 100 (clamped)
        assert result.total_score == 100.0

    def test_pass_low_confidence(self):
        qc = {"pass": True, "confidence": 0.50}
        result = self.calc.calculate(qc)
        # pass_score=100, boost=0 → 100
        assert result.total_score == 100.0

    def test_fail_with_auto_fail(self):
        qc = {
            "pass": False,
            "confidence": 0.85,
            "auto_fail_triggered": ["face_morphing"],
            "minor_issues": [],
        }
        result = self.calc.calculate(qc)
        # pass_score = max(0, 100 - 1*15) = 85
        # issue_penalty = -(1*8) = -8
        # confidence_boost = 5
        # total = 85 - 8 + 5 = 82
        assert result.total_score == 82.0
        assert result.breakdown["pass_score"] == 85.0
        assert result.breakdown["issue_penalty"] == -8.0
        assert result.breakdown["confidence_boost"] == 5.0

    def test_fail_multiple_issues(self):
        qc = {
            "pass": False,
            "confidence": 0.65,
            "auto_fail_triggered": ["action_loop", "face_morphing"],
            "minor_issues": ["flicker", "slight_warp"],
        }
        result = self.calc.calculate(qc)
        # pass_score = max(0, 100 - 4*15) = 40
        # issue_penalty = max(-50, -(2*8 + 2*3)) = max(-50, -22) = -22
        # confidence_boost = 2
        # total = 40 - 22 + 2 = 20
        assert result.total_score == 20.0

    def test_catastrophic_fail(self):
        """Many issues should not go below 0."""
        qc = {
            "pass": False,
            "confidence": 0.30,
            "auto_fail_triggered": ["a", "b", "c", "d", "e", "f"],
            "minor_issues": ["x", "y", "z"],
        }
        result = self.calc.calculate(qc)
        assert result.total_score >= 0.0

    def test_qcresult_model_input(self):
        """Accepts QCResult model directly."""
        qc = QCResult(**{"pass": True, "confidence": 0.96})
        result = self.calc.calculate(qc)
        assert result.total_score == 100.0


class TestMultiDimensionalReward:
    def setup_method(self):
        self.calc = RewardCalculator()

    def test_all_perfect(self):
        qc = {"pass": True, "confidence": 0.98}
        result = self.calc.calculate(
            qc, aesthetic=1.0, clip_score=1.0, motion_score=1.0
        )
        # qc_norm ≈ 1.0 (100/100), all others 1.0
        # total = (1.0*0.4 + 1.0*0.2 + 1.0*0.2 + 1.0*0.2) * 100 = 100
        assert result.total_score == 100.0

    def test_mixed_scores(self):
        qc = {"pass": True, "confidence": 0.90}
        result = self.calc.calculate(
            qc, aesthetic=0.5, clip_score=0.7, motion_score=0.8
        )
        # qc_norm ≈ 1.0 (100 + 0 + 5 = 100, /100 = 1.0... wait, let me check)
        # pass_score=100, issue=0, boost=5 → 100 (clamped). qc_norm=1.0
        # total = (1.0*0.4 + 0.5*0.2 + 0.7*0.2 + 0.8*0.2) * 100
        #       = (0.4 + 0.1 + 0.14 + 0.16) * 100 = 80.0
        assert result.total_score == 80.0

    def test_custom_weights(self):
        calc = RewardCalculator(weights={"qc": 0.6, "aesthetic": 0.1, "clip": 0.2, "motion": 0.1})
        qc = {"pass": True, "confidence": 0.98}
        result = calc.calculate(qc, aesthetic=0.5, clip_score=0.5, motion_score=0.5)
        # qc_norm=1.0, others=0.5
        # (1.0*0.6 + 0.5*0.1 + 0.5*0.2 + 0.5*0.1) * 100
        # = (0.6 + 0.05 + 0.1 + 0.05) * 100 = 80.0
        assert result.total_score == 80.0

    def test_partial_extras_falls_back_to_qc_only(self):
        """If only some extras provided, use QC-only mode."""
        qc = {"pass": True, "confidence": 0.98}
        result = self.calc.calculate(qc, aesthetic=0.5)
        # Falls back to QC-only → 100
        assert result.total_score == 100.0


class TestBatchCalculate:
    def test_batch(self, mixed_samples):
        calc = RewardCalculator()
        results = calc.batch_calculate(mixed_samples)
        assert len(results) == len(mixed_samples)
        assert all(isinstance(r, RewardBreakdown) for r in results)
        # All scores in valid range
        assert all(0 <= r.total_score <= 100 for r in results)


class TestFitWeights:
    def test_fit_returns_weights(self):
        """fit_weights should return a weight dict."""
        calc = RewardCalculator()
        samples = [
            make_sample(f"s{i}", human_score=5 + i, confidence=0.7 + i * 0.05)
            for i in range(5)
        ]
        # Without extra_scores, returns current weights
        w = calc.fit_weights(samples)
        assert "qc" in w
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_fit_with_extra_scores(self):
        calc = RewardCalculator()
        samples = [
            make_sample(f"s{i}", human_score=3 + i, confidence=0.6 + i * 0.05)
            for i in range(6)
        ]
        extras = [
            {"aesthetic": 0.3 + i * 0.1, "clip": 0.4 + i * 0.08, "motion": 0.5 + i * 0.06}
            for i in range(6)
        ]
        w = calc.fit_weights(samples, extra_scores=extras)
        assert "qc" in w
        assert sum(w.values()) == pytest.approx(1.0, abs=0.05)

    def test_fit_too_few_samples(self):
        calc = RewardCalculator()
        samples = [make_sample("s0"), make_sample("s1")]
        w = calc.fit_weights(samples)
        # Should return defaults unchanged
        assert w == calc.weights
