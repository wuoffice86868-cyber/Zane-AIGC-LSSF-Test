"""Tests for data models."""

import pytest
from pydantic import ValidationError

from prompt_evaluator.models import (
    CalibrationReport,
    CameraMove,
    CorrelationResult,
    EvalSample,
    HumanLabel,
    InputInfo,
    OutputInfo,
    PromptFeatures,
    PromptInfo,
    QCResult,
    RewardBreakdown,
    SceneType,
)


class TestQCResult:
    def test_defaults(self):
        qc = QCResult(**{"pass": True})
        assert qc.qc_pass is True
        assert qc.confidence == 0.0
        assert qc.auto_fail_triggered == []
        assert qc.minor_issues == []

    def test_alias_pass(self):
        """'pass' is a Python keyword — must use alias."""
        qc = QCResult(**{"pass": False, "confidence": 0.85})
        assert qc.qc_pass is False
        assert qc.confidence == 0.85

    def test_populate_by_name(self):
        qc = QCResult(qc_pass=True, confidence=0.5)
        assert qc.qc_pass is True

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            QCResult(**{"pass": True, "confidence": 1.5})


class TestHumanLabel:
    def test_score_range(self):
        h = HumanLabel(human_score=10, human_pass=True)
        assert h.human_score == 10

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            HumanLabel(human_score=11)

    def test_score_below_range(self):
        with pytest.raises(ValidationError):
            HumanLabel(human_score=0)


class TestEvalSample:
    def test_minimal(self):
        s = EvalSample(sample_id="x")
        assert s.sample_id == "x"
        assert s.human_label is None

    def test_full(self, good_sample):
        assert good_sample.sample_id == "good-001"
        assert good_sample.qc_result.qc_pass is True
        assert good_sample.human_label is not None
        assert good_sample.human_label.human_score == 8


class TestEnums:
    def test_scene_types(self):
        assert SceneType.POOL.value == "pool"
        assert SceneType.ROOM.value == "room"
        assert SceneType.EXTERIOR.value == "exterior"

    def test_camera_moves(self):
        assert CameraMove.PUSH.value == "push"
        assert CameraMove.CIRCLE_AROUND.value == "circle around"


class TestRewardBreakdown:
    def test_score_range(self):
        rb = RewardBreakdown(total_score=50.0, breakdown={"a": 1.0})
        assert rb.total_score == 50.0

    def test_out_of_range(self):
        with pytest.raises(ValidationError):
            RewardBreakdown(total_score=150.0)
