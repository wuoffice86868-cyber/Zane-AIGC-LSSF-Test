"""Tests for QCCalibrator."""

import pytest

from prompt_evaluator.calibration import QCCalibrator
from prompt_evaluator.models import CalibrationReport, AutoFailRuleAnalysis

from .conftest import make_sample


class TestCalibrate:
    def setup_method(self):
        self.cal = QCCalibrator()

    def test_perfect_agreement(self):
        """QC and human agree on everything."""
        samples = [
            make_sample("p1", qc_pass=True, human_pass=True),
            make_sample("p2", qc_pass=True, human_pass=True),
            make_sample("f1", qc_pass=False, human_pass=False, auto_fail=["loop"]),
            make_sample("f2", qc_pass=False, human_pass=False, auto_fail=["warp"]),
        ]
        report = self.cal.calibrate(samples)
        assert isinstance(report, CalibrationReport)
        assert report.accuracy == 1.0
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1 == 1.0
        assert report.false_positives == []
        assert report.false_negatives == []
        # Confusion: [[TP=2, FP=0], [FN=0, TN=2]]
        assert report.confusion_matrix == [[2, 0], [0, 2]]

    def test_with_false_positives(self):
        """QC fails videos that humans think are fine."""
        samples = [
            make_sample("fp1", qc_pass=False, human_pass=True, auto_fail=["warp"]),
            make_sample("fp2", qc_pass=False, human_pass=True, auto_fail=["loop"]),
            make_sample("tp1", qc_pass=False, human_pass=False, auto_fail=["morph"]),
            make_sample("tn1", qc_pass=True, human_pass=True),
        ]
        report = self.cal.calibrate(samples)
        assert report.false_positives == ["fp1", "fp2"]
        assert report.precision < 1.0  # TP / (TP + FP) = 1 / 3
        assert report.accuracy == 0.5   # 2 correct / 4 total

    def test_with_false_negatives(self):
        """QC passes videos that humans think are bad."""
        samples = [
            make_sample("fn1", qc_pass=True, human_pass=False, human_score=3),
            make_sample("tn1", qc_pass=True, human_pass=True),
            make_sample("tp1", qc_pass=False, human_pass=False, auto_fail=["loop"]),
        ]
        report = self.cal.calibrate(samples)
        assert "fn1" in report.false_negatives
        assert report.recall < 1.0  # TP / (TP + FN) = 1 / 2

    def test_empty_samples(self):
        report = self.cal.calibrate([])
        assert "No labeled samples" in report.recommendations[0]

    def test_no_human_labels_skipped(self):
        samples = [
            make_sample("s1", has_human_label=False),
            make_sample("s2", has_human_label=False),
        ]
        report = self.cal.calibrate(samples)
        assert "No labeled samples" in report.recommendations[0]

    def test_mixed_samples(self, mixed_samples):
        report = self.cal.calibrate(mixed_samples)
        assert 0 <= report.accuracy <= 1
        assert 0 <= report.precision <= 1
        assert 0 <= report.recall <= 1
        assert 0 <= report.f1 <= 1
        # Should have both FP and FN in mixed_samples
        assert len(report.false_positives) > 0
        assert len(report.false_negatives) > 0

    def test_recommendations_generated(self, mixed_samples):
        report = self.cal.calibrate(mixed_samples)
        assert len(report.recommendations) > 0


class TestAutoFailRuleAnalysis:
    def setup_method(self):
        self.cal = QCCalibrator()

    def test_rule_analysis(self):
        samples = [
            # action_loop fired, human agrees
            make_sample("s1", qc_pass=False, auto_fail=["action_loop"],
                        human_pass=False, human_issues=["loop visible"]),
            # action_loop fired, human disagrees (FP)
            make_sample("s2", qc_pass=False, auto_fail=["action_loop"],
                        human_pass=True),
            # face_morphing fired, human agrees
            make_sample("s3", qc_pass=False, auto_fail=["face_morphing"],
                        human_pass=False, human_issues=["face distorted"]),
        ]
        result = self.cal.analyze_auto_fail_rules(samples)
        assert "action_loop" in result
        assert "face_morphing" in result

        loop_analysis = result["action_loop"]
        assert isinstance(loop_analysis, AutoFailRuleAnalysis)
        assert loop_analysis.sample_count == 2  # fired twice
        assert loop_analysis.precision == 0.5   # 1 TP / (1 TP + 1 FP)
        assert "s2" in loop_analysis.false_positive_ids

        face_analysis = result["face_morphing"]
        assert face_analysis.precision == 1.0  # 1 TP / (1 TP + 0 FP)

    def test_no_rules(self):
        samples = [
            make_sample("s1", qc_pass=True, human_pass=True),
        ]
        result = self.cal.analyze_auto_fail_rules(samples)
        assert result == {}

    def test_empty_samples(self):
        result = self.cal.analyze_auto_fail_rules([])
        assert result == {}


class TestCalibrateByScene:
    def test_scene_breakdown(self, mixed_samples):
        cal = QCCalibrator()
        result = cal.calibrate_by_scene(mixed_samples)
        assert isinstance(result, dict)
        assert len(result) > 0
        # Each value should be a CalibrationReport
        for scene, report in result.items():
            assert isinstance(report, CalibrationReport)
            assert isinstance(scene, str)
