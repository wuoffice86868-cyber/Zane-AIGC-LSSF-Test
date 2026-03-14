"""Prompt Evaluator — evaluation and optimization for AI video generation prompts."""

from .models import (
    QCResult,
    EvalSample,
    RewardBreakdown,
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
from .optimizer import PromptOptimizer
from .qc_client import QCClientProtocol, StubQCClient, GeminiQCClient
from .gemini_client import GeminiVideoQC, GeminiLLM, HOTEL_VIDEO_QC_PROMPT
from .calibration import QCCalibrator
from .kie_client import KieClient, TaskResult, ClientStats, KieAPIError, KieBudgetError, KieTimeoutError
from .pipeline import EvalPipeline, EvalResult, BatchResult, SceneSpec

__all__ = [
    "QCResult",
    "EvalSample",
    "RewardBreakdown",
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
    "PromptOptimizer",
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

__version__ = "0.1.0"
