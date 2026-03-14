"""Shared helpers for building mock evaluation data."""

from __future__ import annotations

from typing import List, Optional

from prompt_evaluator.models import (
    EvalSample,
    HumanLabel,
    InputInfo,
    OutputInfo,
    PromptInfo,
    QCResult,
)

# ── Sample prompts ─────────────────────────────────────────────────

PROMPT_HIGH = (
    "Single continuous shot. Wide shot of an infinity pool at golden hour, "
    "ocean horizon beyond. The camera pushes slowly forward over the water. "
    "Water ripples softly. The horizon holds steady. "
    "Sky and clouds remain completely still."
)

PROMPT_MID = (
    "Medium shot of hotel room. The camera moves right across the bed. "
    "Furniture stays fixed."
)

PROMPT_LOW = "A beautiful hotel with amazing pool and incredible sunset views."

PROMPT_HUMAN = (
    "Single continuous shot. Medium shot. Consistent human subject with natural "
    "features. A woman relaxes on a sun lounger. The camera pushes very slowly "
    "forward. Water ripples softly. Pool deck stays stable. Sky remains still. "
    "Person remains natural throughout."
)


def make_sample(
    sample_id: str = "001",
    *,
    qc_pass: bool = True,
    confidence: float = 0.90,
    auto_fail: Optional[List[str]] = None,
    minor: Optional[List[str]] = None,
    human_pass: bool = True,
    human_score: int = 7,
    prompt: str = PROMPT_HIGH,
    scene_type: str = "pool",
    human_present: bool = False,
) -> EvalSample:
    """Quick factory for ``EvalSample``."""
    return EvalSample(
        sample_id=sample_id,
        input=InputInfo(scene_type=scene_type, poi_name="Test Hotel"),
        prompt=PromptInfo(
            system_prompt_version="v3",
            generated_prompt=prompt,
        ),
        output=OutputInfo(generation_method="Seedance 1.0"),
        qc_result=QCResult(
            **{
                "pass": qc_pass,
                "confidence": confidence,
                "human_present": human_present,
                "auto_fail_triggered": auto_fail or [],
                "minor_issues": minor or [],
                "summary": "test",
            }
        ),
        human_label=HumanLabel(
            labeled_by="tester",
            labeled_at="2026-03-11",
            human_pass=human_pass,
            human_score=human_score,
        ),
    )


def make_dataset(n: int = 20) -> List[EvalSample]:
    """Generate a small diverse dataset for testing."""
    samples: List[EvalSample] = []

    configs = [
        # (id, qc_pass, conf, auto_fail, minor, h_pass, h_score, prompt, scene)
        ("001", True, 0.96, [], [], True, 9, PROMPT_HIGH, "pool"),
        ("002", True, 0.92, [], [], True, 8, PROMPT_HIGH, "pool"),
        ("003", True, 0.88, [], [], True, 7, PROMPT_MID, "room"),
        ("004", True, 0.85, [], [], True, 7, PROMPT_MID, "room"),
        ("005", False, 0.82, ["action_loop"], [], False, 4, PROMPT_LOW, "pool"),
        ("006", False, 0.78, ["face_morphing"], [], False, 3, PROMPT_LOW, "lobby"),
        ("007", False, 0.75, [], ["flicker"], True, 6, PROMPT_MID, "room"),
        ("008", True, 0.90, [], [], False, 4, PROMPT_LOW, "exterior"),
        ("009", False, 0.88, ["action_loop"], ["scale"], False, 2, PROMPT_LOW, "pool"),
        ("010", True, 0.95, [], [], True, 8, PROMPT_HUMAN, "pool"),
        ("011", True, 0.70, [], [], True, 6, PROMPT_MID, "bathroom"),
        ("012", False, 0.65, [], ["texture_shift", "lighting"], True, 5, PROMPT_MID, "spa"),
        ("013", True, 0.98, [], [], True, 10, PROMPT_HIGH, "beach"),
        ("014", False, 0.80, ["background_warp"], [], False, 3, PROMPT_LOW, "exterior"),
        ("015", True, 0.91, [], [], True, 7, PROMPT_HIGH, "restaurant"),
    ]

    for cfg in configs[:n]:
        sid, qp, co, af, mi, hp, hs, pr, sc = cfg
        samples.append(
            make_sample(
                sid,
                qc_pass=qp,
                confidence=co,
                auto_fail=af,
                minor=mi,
                human_pass=hp,
                human_score=hs,
                prompt=pr,
                scene_type=sc,
            )
        )

    return samples
