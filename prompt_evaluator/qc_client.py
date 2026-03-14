"""QC client interface for video quality checking.

Provides:
- QCClientProtocol: the interface any QC client must implement
- GeminiQCClient: real implementation using Gemini Vision API
- StubQCClient: deterministic stub for testing (no API needed)
- RandomQCClient: random scores for load testing

Usage:
    # Real Gemini client
    from prompt_evaluator.qc_client import GeminiQCClient
    qc = GeminiQCClient(api_key="...", qc_prompt="...")
    result = qc.evaluate("https://video-url.mp4")
    # result is a dict compatible with QCResult(**result)

    # Stub for testing
    from prompt_evaluator.qc_client import StubQCClient
    qc = StubQCClient(always_pass=True)
    result = qc.evaluate("https://video-url.mp4")
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class QCClientProtocol(Protocol):
    """Interface for QC video evaluation clients.

    Any object implementing ``evaluate(video_url: str) -> dict`` satisfies
    this protocol.  The returned dict must be compatible with ``QCResult``:

    {
        "pass": bool,
        "confidence": float (0.0-1.0),
        "human_present": bool,
        "auto_fail_triggered": List[str],
        "minor_issues": List[str],
        "summary": str,
    }
    """

    def evaluate(self, video_url: str, **kwargs: Any) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Stub clients (no API needed)
# ---------------------------------------------------------------------------

class StubQCClient:
    """Deterministic stub QC client for development and testing.

    Args:
        always_pass: If True, all videos pass QC.
        always_fail: If True, all videos fail QC.
        auto_fail_rules: List of rule names to always trigger.
        confidence: Fixed confidence level to return.
    """

    def __init__(
        self,
        *,
        always_pass: bool = False,
        always_fail: bool = False,
        auto_fail_rules: Optional[List[str]] = None,
        confidence: float = 0.9,
    ) -> None:
        if always_pass and always_fail:
            raise ValueError("Cannot set both always_pass and always_fail")
        # Default to pass if neither flag is set
        self.always_pass = always_pass or (not always_fail and not auto_fail_rules)
        self.always_fail = always_fail
        self.auto_fail_rules = auto_fail_rules or []
        self.confidence = confidence

    def evaluate(self, video_url: str, **kwargs: Any) -> Dict[str, Any]:
        triggered = list(self.auto_fail_rules)
        if self.always_fail and not triggered:
            triggered = ["stub_fail"]

        qc_pass = self.always_pass and not self.always_fail and not triggered

        return {
            "pass": qc_pass,
            "confidence": self.confidence,
            "human_present": False,
            "auto_fail_triggered": triggered,
            "minor_issues": [],
            "summary": f"StubQCClient: {'pass' if qc_pass else 'fail'} for {video_url[:50]}",
        }


# ---------------------------------------------------------------------------
# Gemini Vision QC client
# ---------------------------------------------------------------------------

_DEFAULT_QC_PROMPT = """You are a video quality control (QC) evaluator for hotel marketing video clips.

Evaluate the provided video and output a JSON object with these fields:
{
    "pass": true/false,
    "confidence": 0.0-1.0,
    "human_present": true/false,
    "auto_fail_triggered": ["rule_name", ...],
    "minor_issues": ["description", ...],
    "summary": "one sentence summary"
}

AUTO-FAIL RULES (any of these → pass=false):
- action_loop: video content loops visibly (water, fire, motion repeats)
- face_morphing: human faces distort or morph unnaturally
- object_morphing: objects change shape unnaturally
- sky_freeze_missing: outdoor scene with sky but sky/clouds are not frozen still
- scene_jump: abrupt scene cut or jump (not a single continuous shot)
- extreme_blur: motion blur that makes content unrecognizable
- watermark: visible watermark or text overlay on video

MINOR ISSUES (notable but not auto-fail):
- slight_flicker: minor lighting flicker
- edge_distortion: slight distortion at frame edges
- color_shift: subtle unintended color change
- speed_inconsistency: camera or motion speed changes unexpectedly

Return only the JSON object, no other text."""


class GeminiQCClient:
    """QC client using Google Gemini Vision API for video evaluation.

    Args:
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        qc_prompt: System prompt for QC evaluation.
            Defaults to the standard hotel video QC prompt.
        model: Gemini model to use. Default: gemini-1.5-pro.
        max_retries: Number of retries on API error.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        qc_prompt: str = _DEFAULT_QC_PROMPT,
        model: str = "gemini-1.5-pro",
        max_retries: int = 2,
    ) -> None:
        import os

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set GEMINI_API_KEY env var."
            )
        self.qc_prompt = qc_prompt
        self.model = model
        self.max_retries = max_retries
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Gemini client. Raises ImportError if google-genai not installed."""
        try:
            import google.generativeai as genai  # type: ignore[import]
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=self.qc_prompt,
            )
        except ImportError as e:
            raise ImportError(
                "google-generativeai package required for GeminiQCClient. "
                "Install with: pip install google-generativeai"
            ) from e

    def evaluate(self, video_url: str, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate a video URL using Gemini Vision.

        Args:
            video_url: Public URL of the video to evaluate.
            **kwargs: Extra kwargs forwarded to Gemini generate_content.

        Returns:
            Dict compatible with QCResult model.
        """
        import google.generativeai as genai  # type: ignore[import]

        for attempt in range(self.max_retries + 1):
            try:
                content = [
                    {"video_url": video_url},
                    "Evaluate this video for QC. Return only JSON.",
                ]
                response = self._client.generate_content(content, **kwargs)
                return self._parse_response(response.text, video_url)
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(
                        "Gemini QC attempt %d/%d failed: %s",
                        attempt + 1, self.max_retries + 1, e
                    )
                    continue
                logger.error("Gemini QC failed after %d attempts: %s", self.max_retries + 1, e)
                # Return a safe default on final failure
                return self._error_result(video_url, str(e))

    @staticmethod
    def _parse_response(text: str, video_url: str) -> Dict[str, Any]:
        """Parse Gemini JSON response, stripping any markdown fences."""
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (``` fences)
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse Gemini QC response as JSON: %s\nRaw: %s", e, text[:200])
            return GeminiQCClient._error_result(video_url, f"JSON parse error: {e}")

        # Normalize field names
        return {
            "pass": bool(data.get("pass", True)),
            "confidence": float(data.get("confidence", 0.5)),
            "human_present": bool(data.get("human_present", False)),
            "auto_fail_triggered": list(data.get("auto_fail_triggered", [])),
            "minor_issues": list(data.get("minor_issues", [])),
            "summary": str(data.get("summary", "")),
        }

    @staticmethod
    def _error_result(video_url: str, error: str) -> Dict[str, Any]:
        return {
            "pass": True,  # Safe default — don't fail on QC error
            "confidence": 0.0,
            "human_present": False,
            "auto_fail_triggered": [],
            "minor_issues": [],
            "summary": f"QC error (safe default pass): {error[:100]}",
        }
