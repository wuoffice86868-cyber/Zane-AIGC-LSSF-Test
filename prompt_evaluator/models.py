"""Data models for the prompt evaluation system."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SceneType(str, Enum):
    POOL = "pool"
    ROOM = "room"
    BATHROOM = "bathroom"
    LOBBY = "lobby"
    RESTAURANT = "restaurant"
    EXTERIOR = "exterior"
    SPA = "spa"
    BEACH = "beach"
    OTHER = "other"


class CameraMove(str, Enum):
    PUSH = "push"
    PULL = "pull"
    CIRCLE_AROUND = "circle around"
    MOVE_LEFT = "move left"
    MOVE_RIGHT = "move right"
    PAN_LEFT = "pan left"
    PAN_RIGHT = "pan right"
    RISE = "rise"
    TILT_UP = "tilt up"


class RewardDimension(str, Enum):
    """Three reward dimensions aligned with Seedance internal RM architecture.

    Seedance 1.0 tech report (arXiv:2506.09113) discloses 3 specialized
    Reward Models used in their RLHF pipeline:
    - Foundational RM: text-video alignment + structural stability
    - Motion RM: movement quality + temporal consistency + artifact detection
    - Aesthetic RM: visual appeal (evaluated on keyframes, image-space)
    """
    FOUNDATIONAL = "foundational"
    MOTION = "motion"
    AESTHETIC = "aesthetic"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

class QCResult(BaseModel):
    """QC result produced by the Gemini Video QC prompt."""

    qc_pass: bool = Field(alias="pass", default=True)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    human_present: bool = False
    auto_fail_triggered: List[str] = Field(default_factory=list)
    minor_issues: List[str] = Field(default_factory=list)
    summary: str = ""

    model_config = {"populate_by_name": True}


class HumanLabel(BaseModel):
    """Human annotation attached to a sample."""

    labeled_by: str = ""
    labeled_at: str = ""
    human_pass: bool = True
    human_score: int = Field(ge=1, le=10, default=5)
    human_issues: List[str] = Field(default_factory=list)
    human_notes: str = ""


class PromptInfo(BaseModel):
    """Prompt metadata for a sample."""

    system_prompt_version: str = ""
    system_prompt_hash: str = ""
    generated_prompt: str = ""


class InputInfo(BaseModel):
    """Input metadata for a sample."""

    image_url: str = ""
    scene_type: str = "other"
    poi_name: str = ""


class OutputInfo(BaseModel):
    """Output metadata for a sample."""

    video_url: str = ""
    video_duration_sec: float = 5.0
    generation_method: str = ""


class EvalSample(BaseModel):
    """A single evaluation sample combining input, prompt, output, QC, and
    optional human label."""

    sample_id: str
    input: InputInfo = Field(default_factory=InputInfo)
    prompt: PromptInfo = Field(default_factory=PromptInfo)
    output: OutputInfo = Field(default_factory=OutputInfo)
    qc_result: QCResult = Field(default_factory=lambda: QCResult(**{"pass": True}))
    human_label: Optional[HumanLabel] = None


# ---------------------------------------------------------------------------
# Result / report models
# ---------------------------------------------------------------------------

class DimensionScore(BaseModel):
    """Score for a single reward dimension."""

    dimension: str = ""
    score: float = Field(ge=0.0, le=1.0, default=0.0)
    weight: float = Field(ge=0.0, le=1.0, default=0.0)
    weighted_score: float = Field(ge=0.0, le=1.0, default=0.0)
    components: Dict[str, float] = Field(default_factory=dict)
    issues: List[str] = Field(default_factory=list)


class RewardBreakdown(BaseModel):
    """Detailed reward score breakdown.

    Supports both legacy single-score mode and the new 3-dimension mode
    aligned with Seedance's internal RM architecture.
    """

    total_score: float = Field(ge=0, le=100, default=0.0)
    breakdown: Dict[str, object] = Field(default_factory=dict)
    dimensions: Dict[str, DimensionScore] = Field(default_factory=dict)


class PromptFeatures(BaseModel):
    """Feature vector extracted from a generated prompt."""

    word_count: int = 0
    has_camera_move: bool = False
    camera_move_type: str = ""
    has_single_continuous_shot: bool = False
    has_sky_freeze: bool = False
    has_stable_element: bool = False
    stable_element: str = ""
    motion_elements: List[str] = Field(default_factory=list)
    has_speed_modifier: bool = False
    speed_modifier: str = ""
    shot_size: str = ""
    has_human_instruction: bool = False
    sentence_count: int = 0


class CorrelationResult(BaseModel):
    """Result of prompt-feature ↔ reward correlation analysis."""

    feature_importance: Dict[str, float] = Field(default_factory=dict)
    high_reward_patterns: List[str] = Field(default_factory=list)
    low_reward_patterns: List[str] = Field(default_factory=list)


class CalibrationReport(BaseModel):
    """QC calibration report against human labels."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    correlation: float = 0.0
    confusion_matrix: List[List[int]] = Field(
        default_factory=lambda: [[0, 0], [0, 0]]
    )
    false_positives: List[str] = Field(default_factory=list)
    false_negatives: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AutoFailRuleAnalysis(BaseModel):
    """Per-rule analysis of an auto-fail trigger."""

    rule_name: str = ""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    sample_count: int = 0
    false_positive_ids: List[str] = Field(default_factory=list)
