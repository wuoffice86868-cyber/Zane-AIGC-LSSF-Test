"""Prompt Evaluator — evaluation and optimization for AI video generation prompts.

Three-dimensional reward model aligned with Seedance RM architecture:
  - Foundational: structural stability + text-video alignment
  - Motion: movement quality + temporal consistency
  - Aesthetic: visual appeal + composition
"""

from .models import (
    QCResult,
    EvalSample,
    RewardBreakdown,
    DimensionScore,
    RewardDimension,
    HumanLabel,
    PromptInfo,
    InputInfo,
    OutputInfo,
    PromptFeatures,
    CorrelationResult,
    CalibrationReport,
    AutoFailRuleAnalysis,
    SceneType,
    CameraMove,
)
from .reward_calculator import RewardCalculator
from .prompt_analyzer import PromptAnalyzer
from .qc_client import QCClientProtocol, StubQCClient, GeminiQCClient
from .gemini_client import GeminiVideoQC, GeminiLLM, HOTEL_VIDEO_QC_PROMPT
from .calibration import QCCalibrator
from .kie_client import KieClient, TaskResult, ClientStats, KieAPIError, KieBudgetError, KieTimeoutError
from .pipeline import EvalPipeline, EvalResult, BatchResult, SceneSpec

__all__ = [
    "QCResult",
    "EvalSample",
    "RewardBreakdown",
    "DimensionScore",
    "RewardDimension",
    "HumanLabel",
    "PromptInfo",
    "InputInfo",
    "OutputInfo",
    "PromptFeatures",
    "CorrelationResult",
    "CalibrationReport",
    "AutoFailRuleAnalysis",
    "SceneType",
    "CameraMove",
    "RewardCalculator",
    "PromptAnalyzer",
    "QCCalibrator",
    "KieClient",
    "TaskResult",
    "ClientStats",
    "KieAPIError",
    "KieBudgetError",
    "KieTimeoutError",
    "EvalPipeline",
    "EvalResult",
    "BatchResult",
    "SceneSpec",
]

__version__ = "0.2.0"
