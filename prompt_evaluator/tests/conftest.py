"""Shared fixtures for prompt_evaluator tests."""

from __future__ import annotations

import pytest

from prompt_evaluator.models import (
    EvalSample,
    HumanLabel,
    InputInfo,
    OutputInfo,
    PromptInfo,
    QCResult,
)


# ---------------------------------------------------------------------------
# Sample factories
# ---------------------------------------------------------------------------

def make_sample(
    sample_id: str = "001",
    *,
    qc_pass: bool = True,
    confidence: float = 0.90,
    auto_fail: list[str] | None = None,
    minor_issues: list[str] | None = None,
    human_pass: bool = True,
    human_score: int = 7,
    human_issues: list[str] | None = None,
    scene_type: str = "pool",
    generated_prompt: str = (
        "Single continuous shot. Wide shot of an infinity pool at golden hour. "
        "The camera pushes slowly forward over the water. "
        "Water ripples softly. The horizon holds steady. "
        "Sky and clouds remain completely still."
    ),
    prompt_version: str = "v3",
    has_human_label: bool = True,
) -> EvalSample:
    """Create an ``EvalSample`` with sensible defaults."""
    qc = QCResult(
        **{
            "pass": qc_pass,
            "confidence": confidence,
            "human_present": False,
            "auto_fail_triggered": auto_fail or [],
            "minor_issues": minor_issues or [],
            "summary": "test sample",
        }
    )
    human = None
    if has_human_label:
        human = HumanLabel(
            labeled_by="test",
            labeled_at="2026-03-11",
            human_pass=human_pass,
            human_score=human_score,
            human_issues=human_issues or [],
            human_notes="",
        )
    return EvalSample(
        sample_id=sample_id,
        input=InputInfo(
            image_url="https://example.com/img.jpg",
            scene_type=scene_type,
            poi_name="Test Hotel",
        ),
        prompt=PromptInfo(
            system_prompt_version=prompt_version,
            system_prompt_hash="abc123",
            generated_prompt=generated_prompt,
        ),
        output=OutputInfo(
            video_url="https://example.com/video.mp4",
            video_duration_sec=5.0,
            generation_method="Seedance 1.0",
        ),
        qc_result=qc,
        human_label=human,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def good_sample() -> EvalSample:
    return make_sample("good-001", qc_pass=True, human_pass=True, human_score=8, confidence=0.95)


@pytest.fixture
def bad_sample() -> EvalSample:
    return make_sample(
        "bad-001",
        qc_pass=False,
        confidence=0.85,
        auto_fail=["action_loop"],
        minor_issues=["slight flicker"],
        human_pass=False,
        human_score=3,
        human_issues=["water loop visible", "boring composition"],
        generated_prompt=(
            "Close-up of hotel lobby. The camera zooms in quickly. "
            "People walk around. No stability anchor."
        ),
    )


@pytest.fixture
def mixed_samples() -> list[EvalSample]:
    """A set of 10 diverse samples for statistical tests."""
    samples = []

    # 4 good (QC pass, human pass)
    for i in range(4):
        samples.append(
            make_sample(
                f"good-{i:03d}",
                qc_pass=True,
                confidence=0.90 + i * 0.02,
                human_pass=True,
                human_score=7 + (i % 3),
                scene_type=["pool", "room", "exterior", "lobby"][i],
                generated_prompt=(
                    "Single continuous shot. Medium shot of a luxury space. "
                    "The camera pushes slowly forward. Furniture stays still."
                ),
            )
        )

    # 3 bad (QC fail, human fail)
    for i in range(3):
        samples.append(
            make_sample(
                f"bad-{i:03d}",
                qc_pass=False,
                confidence=0.70 + i * 0.05,
                auto_fail=["action_loop"],
                human_pass=False,
                human_score=2 + i,
                human_issues=["loop visible"],
                scene_type="pool",
                generated_prompt=(
                    "Camera zooms into the pool area. Water is moving. "
                    "Everything shifts around."
                ),
            )
        )

    # 2 false positives (QC fail, human pass)
    for i in range(2):
        samples.append(
            make_sample(
                f"fp-{i:03d}",
                qc_pass=False,
                confidence=0.60,
                auto_fail=["background_warping"],
                human_pass=True,
                human_score=6,
                scene_type="room",
            )
        )

    # 1 false negative (QC pass, human fail)
    samples.append(
        make_sample(
            "fn-000",
            qc_pass=True,
            confidence=0.88,
            human_pass=False,
            human_score=4,
            human_issues=["subtle loop"],
            scene_type="pool",
        )
    )

    return samples
