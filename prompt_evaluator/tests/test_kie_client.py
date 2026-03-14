"""Tests for the KIE API client.

Uses monkeypatch / mocks exclusively — no real API calls.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from prompt_evaluator.kie_client import (
    KieClient,
    KieAPIError,
    KieBudgetError,
    KieTimeoutError,
    TaskResult,
    ClientStats,
    MODEL_SEEDREAM,
    MODEL_SEEDANCE,
    STATE_SUCCESS,
    STATE_FAIL,
    STATE_GENERATING,
    STATE_WAITING,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_api_key():
    return "test-api-key-12345"


@pytest.fixture
def client(mock_api_key):
    """Client with a test API key and high budget."""
    return KieClient(api_key=mock_api_key, max_requests=100)


def _make_create_response(task_id: str = "task_001") -> dict:
    return {"code": 200, "msg": "success", "data": {"taskId": task_id}}


def _make_query_response(
    task_id: str = "task_001",
    state: str = STATE_SUCCESS,
    result_urls: list | None = None,
    cost_time: int = 5000,
) -> dict:
    result_json = json.dumps({"resultUrls": result_urls or ["https://cdn.kie.ai/result.mp4"]})
    return {
        "code": 200,
        "msg": "success",
        "data": {
            "taskId": task_id,
            "model": MODEL_SEEDANCE,
            "state": state,
            "resultJson": result_json,
            "costTime": cost_time,
            "createTime": 1700000000000,
            "completeTime": 1700000005000,
        },
    }


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

class TestTaskResult:
    def test_success_property(self):
        r = TaskResult(task_id="t1", model="m", state=STATE_SUCCESS)
        assert r.success is True

    def test_fail_property(self):
        r = TaskResult(task_id="t1", model="m", state=STATE_FAIL)
        assert r.success is False

    def test_cost_time_sec(self):
        r = TaskResult(task_id="t1", model="m", state=STATE_SUCCESS, cost_time_ms=15000)
        assert r.cost_time_sec == 15.0


class TestClientStats:
    def test_summary(self):
        s = ClientStats(
            total_requests=10,
            successful_requests=8,
            failed_requests=2,
            total_cost_time_ms=60000,
            image_generations=3,
            video_generations=7,
        )
        summary = s.summary()
        assert "10 requests" in summary
        assert "8 ok" in summary
        assert "2 failed" in summary
        assert "Images: 3" in summary
        assert "Videos: 7" in summary
        assert "60.0s" in summary


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------

class TestClientInit:
    def test_explicit_key(self, mock_api_key):
        c = KieClient(api_key=mock_api_key)
        assert c.api_key == mock_api_key

    def test_env_var_key(self, monkeypatch):
        monkeypatch.setenv("KIE_API_KEY", "env-key-999")
        c = KieClient()
        assert c.api_key == "env-key-999"

    def test_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("KIE_API_KEY", raising=False)
        with patch.object(KieClient, "_resolve_api_key", return_value=None):
            with pytest.raises(KieAPIError, match="No API key"):
                KieClient()

    def test_default_budget(self, mock_api_key):
        c = KieClient(api_key=mock_api_key)
        assert c.max_requests == 50

    def test_custom_budget(self, mock_api_key):
        c = KieClient(api_key=mock_api_key, max_requests=10)
        assert c.max_requests == 10


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------

class TestBudgetGuard:
    def test_within_budget(self, client):
        client.stats.total_requests = 5
        client._check_budget()  # should not raise

    def test_at_limit_raises(self, mock_api_key):
        c = KieClient(api_key=mock_api_key, max_requests=3)
        c.stats.total_requests = 3
        with pytest.raises(KieBudgetError, match="Budget limit"):
            c._check_budget()

    def test_unlimited_budget(self, mock_api_key):
        c = KieClient(api_key=mock_api_key, max_requests=0)
        c.stats.total_requests = 99999
        c._check_budget()  # should not raise

    def test_reset_budget(self, mock_api_key):
        c = KieClient(api_key=mock_api_key, max_requests=5)
        c.stats.total_requests = 5
        c.reset_budget(new_max=10)
        c._check_budget()  # should not raise now


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------

class TestGenerateImage:
    def test_basic_image_gen(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response("img_001"),
                _make_query_response("img_001", result_urls=["https://cdn/img.png"]),
            ]
            result = client.generate_image("A hotel pool at sunset")

        assert result.success
        assert result.result_urls == ["https://cdn/img.png"]
        assert client.stats.image_generations == 1
        assert client.stats.total_requests == 1
        assert client.stats.successful_requests == 1

    def test_image_no_poll(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = _make_create_response("img_002")
            result = client.generate_image("Test", poll=False)

        assert result.task_id == "img_002"
        assert result.state == STATE_WAITING
        assert client.stats.image_generations == 1

    def test_image_params_forwarded(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response(),
                _make_query_response(),
            ]
            client.generate_image(
                "test", aspect_ratio="1:1", quality="hd"
            )

            # Check the create call payload
            call_args = mock_req.call_args_list[0]
            payload = call_args[0][2]  # positional: method, path, body
            assert payload["input"]["aspect_ratio"] == "1:1"
            assert payload["input"]["quality"] == "hd"
            assert payload["model"] == MODEL_SEEDREAM


# ---------------------------------------------------------------------------
# Video generation
# ---------------------------------------------------------------------------

class TestGenerateVideo:
    def test_basic_video_gen(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response("vid_001"),
                _make_query_response("vid_001"),
            ]
            result = client.generate_video(
                "Single continuous shot. Camera pushes forward.",
                image_url="https://example.com/img.jpg",
            )

        assert result.success
        assert client.stats.video_generations == 1

    def test_generate_audio_always_false(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response(),
                _make_query_response(),
            ]
            client.generate_video("test prompt")

            payload = mock_req.call_args_list[0][0][2]
            assert payload["input"]["generate_audio"] is False

    def test_multiple_image_urls(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response(),
                _make_query_response(),
            ]
            client.generate_video(
                "test",
                image_urls=["https://a.com/1.jpg", "https://a.com/2.jpg"],
            )

            payload = mock_req.call_args_list[0][0][2]
            assert payload["input"]["input_urls"] == [
                "https://a.com/1.jpg",
                "https://a.com/2.jpg",
            ]

    def test_image_urls_capped_at_2(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response(),
                _make_query_response(),
            ]
            client.generate_video(
                "test",
                image_urls=["https://a/1", "https://a/2", "https://a/3"],
            )

            payload = mock_req.call_args_list[0][0][2]
            assert len(payload["input"]["input_urls"]) == 2

    def test_no_image_no_input_urls(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response(),
                _make_query_response(),
            ]
            client.generate_video("test")

            payload = mock_req.call_args_list[0][0][2]
            assert "input_urls" not in payload["input"]

    def test_video_no_poll(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = _make_create_response("vid_np")
            result = client.generate_video("test", poll=False)

        assert result.state == STATE_WAITING
        assert result.task_id == "vid_np"

    def test_video_failed(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = [
                _make_create_response("vid_fail"),
                _make_query_response("vid_fail", state=STATE_FAIL, result_urls=[]),
            ]
            result = client.generate_video("test")

        assert not result.success
        assert client.stats.failed_requests == 1


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------

class TestPolling:
    def test_immediate_success(self, client):
        with patch.object(client, "_query_task") as mock_q:
            mock_q.return_value = TaskResult(
                task_id="t1", model="m", state=STATE_SUCCESS, cost_time_ms=3000
            )
            result = client._poll_task("t1")

        assert result.success
        mock_q.assert_called_once()

    def test_multiple_polls_before_success(self, client):
        with patch.object(client, "_query_task") as mock_q:
            mock_q.side_effect = [
                TaskResult(task_id="t1", model="m", state=STATE_GENERATING),
                TaskResult(task_id="t1", model="m", state=STATE_GENERATING),
                TaskResult(task_id="t1", model="m", state=STATE_SUCCESS, cost_time_ms=5000),
            ]
            with patch("prompt_evaluator.kie_client.time.sleep"):
                result = client._poll_task("t1", poll_interval=0.01)

        assert result.success
        assert mock_q.call_count == 3

    def test_timeout_raises(self, client):
        with patch.object(client, "_query_task") as mock_q:
            mock_q.return_value = TaskResult(
                task_id="t1", model="m", state=STATE_GENERATING
            )
            with patch("prompt_evaluator.kie_client.time.sleep"):
                with patch("prompt_evaluator.kie_client.time.monotonic") as mock_time:
                    # Simulate time passing beyond timeout
                    mock_time.side_effect = [0.0, 0.0, 999.0]
                    with pytest.raises(KieTimeoutError):
                        client._poll_task("t1", timeout=10.0)


# ---------------------------------------------------------------------------
# Task query
# ---------------------------------------------------------------------------

class TestQueryTask:
    def test_query_success(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = _make_query_response("t1")
            result = client.query_task("t1")

        assert result.task_id == "t1"
        assert result.success
        assert len(result.result_urls) == 1

    def test_query_empty_result_json(self, client):
        with patch.object(client, "_request") as mock_req:
            resp = _make_query_response("t1")
            resp["data"]["resultJson"] = ""
            mock_req.return_value = resp
            result = client.query_task("t1")

        assert result.result_urls == []

    def test_query_malformed_result_json(self, client):
        with patch.object(client, "_request") as mock_req:
            resp = _make_query_response("t1")
            resp["data"]["resultJson"] = "not-json"
            mock_req.return_value = resp
            result = client.query_task("t1")

        assert result.result_urls == []


# ---------------------------------------------------------------------------
# LLMClient protocol
# ---------------------------------------------------------------------------

class TestLLMClientProtocol:
    def test_generate_returns_url(self, client):
        with patch.object(client, "generate_video") as mock_gen:
            mock_gen.return_value = TaskResult(
                task_id="t1",
                model=MODEL_SEEDANCE,
                state=STATE_SUCCESS,
                result_urls=["https://cdn/vid.mp4"],
            )
            url = client.generate("Some prompt")

        assert url == "https://cdn/vid.mp4"

    def test_generate_empty_on_no_urls(self, client):
        with patch.object(client, "generate_video") as mock_gen:
            mock_gen.return_value = TaskResult(
                task_id="t1",
                model=MODEL_SEEDANCE,
                state=STATE_FAIL,
                result_urls=[],
            )
            url = client.generate("Some prompt")

        assert url == ""


# ---------------------------------------------------------------------------
# API error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_api_error_code(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.side_effect = KieAPIError("rate limited", status_code=429)
            with pytest.raises(KieAPIError, match="rate limited"):
                client.generate_image("test")

    def test_no_task_id_in_response(self, client):
        with patch.object(client, "_request") as mock_req:
            mock_req.return_value = {"code": 200, "data": {}}
            with pytest.raises(KieAPIError, match="No taskId"):
                client.generate_image("test")
