"""Tests for QC client implementations."""

import pytest
from prompt_evaluator.qc_client import (
    QCClientProtocol,
    StubQCClient,
    GeminiQCClient,
)
from prompt_evaluator.models import QCResult


# ---------------------------------------------------------------------------
# StubQCClient tests
# ---------------------------------------------------------------------------

class TestStubQCClientPass:
    def test_always_pass_default(self):
        qc = StubQCClient(always_pass=True)
        result = qc.evaluate("https://example.com/video.mp4")
        assert result["pass"] is True
        assert result["confidence"] == 0.9
        assert result["auto_fail_triggered"] == []

    def test_always_fail(self):
        qc = StubQCClient(always_fail=True)
        result = qc.evaluate("https://example.com/video.mp4")
        assert result["pass"] is False
        assert len(result["auto_fail_triggered"]) > 0

    def test_custom_rules_triggered(self):
        qc = StubQCClient(always_fail=True, auto_fail_rules=["action_loop", "face_morphing"])
        result = qc.evaluate("https://example.com/video.mp4")
        assert "action_loop" in result["auto_fail_triggered"]
        assert "face_morphing" in result["auto_fail_triggered"]

    def test_custom_confidence(self):
        qc = StubQCClient(always_pass=True, confidence=0.42)
        result = qc.evaluate("https://example.com/video.mp4")
        assert result["confidence"] == 0.42

    def test_result_is_qcresult_compatible(self):
        qc = StubQCClient(always_pass=True)
        result = qc.evaluate("https://example.com/video.mp4")
        # Must be constructible as QCResult
        qcr = QCResult(**result)
        assert qcr.qc_pass is True

    def test_result_fail_is_qcresult_compatible(self):
        qc = StubQCClient(always_fail=True)
        result = qc.evaluate("https://example.com/video.mp4")
        qcr = QCResult(**result)
        assert qcr.qc_pass is False

    def test_both_pass_and_fail_raises(self):
        with pytest.raises(ValueError):
            StubQCClient(always_pass=True, always_fail=True)

    def test_summary_contains_url(self):
        qc = StubQCClient(always_pass=True)
        result = qc.evaluate("https://example.com/video.mp4")
        assert "example.com" in result["summary"]

    def test_human_present_false_by_default(self):
        qc = StubQCClient(always_pass=True)
        result = qc.evaluate("https://example.com/video.mp4")
        assert result["human_present"] is False


class TestStubQCClientProtocol:
    def test_satisfies_protocol(self):
        qc = StubQCClient(always_pass=True)
        assert isinstance(qc, QCClientProtocol)


# ---------------------------------------------------------------------------
# GeminiQCClient._parse_response tests (unit, no API)
# ---------------------------------------------------------------------------

class TestGeminiParseResponse:
    def test_clean_json(self):
        text = '{"pass": true, "confidence": 0.9, "human_present": false, "auto_fail_triggered": [], "minor_issues": [], "summary": "ok"}'
        result = GeminiQCClient._parse_response(text, "url")
        assert result["pass"] is True
        assert result["confidence"] == 0.9

    def test_json_with_markdown_fence(self):
        text = '```json\n{"pass": false, "confidence": 0.7, "auto_fail_triggered": ["action_loop"], "minor_issues": [], "summary": "loop", "human_present": false}\n```'
        result = GeminiQCClient._parse_response(text, "url")
        assert result["pass"] is False
        assert "action_loop" in result["auto_fail_triggered"]

    def test_json_parse_error_returns_safe_default(self):
        result = GeminiQCClient._parse_response("not valid json at all", "url")
        assert result["pass"] is True
        assert result["confidence"] == 0.0
        assert "error" in result["summary"].lower()

    def test_missing_fields_default_to_safe_values(self):
        result = GeminiQCClient._parse_response('{"pass": false}', "url")
        assert result["confidence"] == 0.5
        assert result["auto_fail_triggered"] == []
        assert result["minor_issues"] == []

    def test_error_result_is_qcresult_compatible(self):
        result = GeminiQCClient._error_result("url", "test error")
        qcr = QCResult(**result)
        assert qcr.qc_pass is True  # safe default

    def test_type_coercion(self):
        # Gemini might return string booleans
        text = '{"pass": false, "confidence": 0.85, "human_present": false, "auto_fail_triggered": ["sky_freeze_missing"], "minor_issues": ["slight_flicker"], "summary": "issues found"}'
        result = GeminiQCClient._parse_response(text, "url")
        assert isinstance(result["pass"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["auto_fail_triggered"], list)

    def test_no_api_key_raises(self):
        import os
        old = os.environ.pop("GEMINI_API_KEY", None)
        try:
            with pytest.raises((ValueError, ImportError)):
                GeminiQCClient(api_key="")
        finally:
            if old:
                os.environ["GEMINI_API_KEY"] = old
