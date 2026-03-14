"""Tests for the end-to-end evaluation pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from prompt_evaluator.kie_client import KieClient, TaskResult, STATE_SUCCESS, STATE_FAIL
from prompt_evaluator.pipeline import (
    EvalPipeline,
    EvalResult,
    BatchResult,
    SceneSpec,
    DEFAULT_HOTEL_SCENES,
)
from prompt_evaluator.models import QCResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kie():
    """KieClient with mocked API calls."""
    client = KieClient(api_key="test-key", max_requests=100)
    return client


@pytest.fixture
def system_prompt():
    return "You are a cinematography director for hotel marketing videos."


def _img_result(task_id="img_001", url="https://cdn/img.jpg"):
    return TaskResult(
        task_id=task_id,
        model="seedream",
        state=STATE_SUCCESS,
        result_urls=[url],
        cost_time_ms=500,
    )


def _vid_result(task_id="vid_001", url="https://cdn/vid.mp4"):
    return TaskResult(
        task_id=task_id,
        model="seedance",
        state=STATE_SUCCESS,
        result_urls=[url],
        cost_time_ms=5000,
    )


def _failed_result(task_id="fail_001"):
    return TaskResult(
        task_id=task_id,
        model="seedance",
        state=STATE_FAIL,
        result_urls=[],
    )


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------

class TestEvalResult:
    def test_success_with_video(self):
        r = EvalResult(sample_id="t1", video_url="https://x.mp4")
        assert r.success is True

    def test_not_success_without_video(self):
        r = EvalResult(sample_id="t1")
        assert r.success is False

    def test_not_success_with_error(self):
        r = EvalResult(sample_id="t1", video_url="https://x.mp4", error="boom")
        assert r.success is False

    def test_to_eval_sample(self):
        r = EvalResult(
            sample_id="t1",
            scene_type="pool",
            scene_description="Test pool",
            generated_prompt="Camera pushes forward",
            image_url="https://img",
            video_url="https://vid",
            qc_result=QCResult(**{"pass": True, "confidence": 0.9}),
        )
        sample = r.to_eval_sample()
        assert sample.sample_id == "t1"
        assert sample.input.scene_type == "pool"
        assert sample.prompt.generated_prompt == "Camera pushes forward"
        assert sample.qc_result.qc_pass is True


# ---------------------------------------------------------------------------
# BatchResult
# ---------------------------------------------------------------------------

class TestBatchResult:
    def test_counts(self):
        b = BatchResult(results=[
            EvalResult(sample_id="1", video_url="a"),
            EvalResult(sample_id="2", video_url="b"),
            EvalResult(sample_id="3", error="fail"),
        ])
        assert b.success_count == 2
        assert b.fail_count == 1

    def test_avg_reward(self):
        from prompt_evaluator.models import RewardBreakdown
        b = BatchResult(results=[
            EvalResult(sample_id="1", reward=RewardBreakdown(total_score=80)),
            EvalResult(sample_id="2", reward=RewardBreakdown(total_score=60)),
        ])
        assert b.avg_reward == 70.0

    def test_summary(self):
        b = BatchResult(results=[
            EvalResult(sample_id="1", video_url="a"),
        ])
        assert "1 runs" in b.summary()


# ---------------------------------------------------------------------------
# SceneSpec
# ---------------------------------------------------------------------------

class TestSceneSpec:
    def test_default_scenes_exist(self):
        assert len(DEFAULT_HOTEL_SCENES) >= 4
        types = [s.scene_type for s in DEFAULT_HOTEL_SCENES]
        assert "pool" in types
        assert "room" in types


# ---------------------------------------------------------------------------
# Pipeline - evaluate_scene
# ---------------------------------------------------------------------------

class TestEvaluateScene:
    def test_full_success(self, mock_kie, system_prompt):
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(system_prompt, mock_kie)
                scene = SceneSpec(description="Test pool", scene_type="pool")
                result = pipeline.evaluate_scene(scene)

        assert result.success
        assert result.image_url == "https://cdn/img.jpg"
        assert result.video_url == "https://cdn/vid.mp4"
        assert result.reward is not None
        assert result.features is not None
        assert result.qc_result is not None

    def test_with_reference_image(self, mock_kie, system_prompt):
        with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
            pipeline = EvalPipeline(system_prompt, mock_kie)
            scene = SceneSpec(
                description="Test",
                scene_type="room",
                reference_image_url="https://existing/img.jpg",
            )
            result = pipeline.evaluate_scene(scene)

        assert result.success
        assert result.image_url == "https://existing/img.jpg"

    def test_image_gen_failure(self, mock_kie, system_prompt):
        with patch.object(mock_kie, "generate_image", return_value=_failed_result()):
            pipeline = EvalPipeline(system_prompt, mock_kie)
            scene = SceneSpec(description="Test", scene_type="pool")
            result = pipeline.evaluate_scene(scene)

        assert not result.success
        assert "Image generation failed" in result.error

    def test_video_gen_failure(self, mock_kie, system_prompt):
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_failed_result()):
                pipeline = EvalPipeline(system_prompt, mock_kie)
                scene = SceneSpec(description="Test", scene_type="pool")
                result = pipeline.evaluate_scene(scene)

        assert not result.success
        assert "Video generation failed" in result.error

    def test_with_custom_prompt(self, mock_kie, system_prompt):
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(system_prompt, mock_kie)
                scene = SceneSpec(description="Test", scene_type="pool")
                result = pipeline.evaluate_scene(
                    scene,
                    cinematography_prompt="Single continuous shot. Camera pushes forward slowly.",
                )

        assert result.generated_prompt == "Single continuous shot. Camera pushes forward slowly."

    def test_with_qc_client(self, mock_kie, system_prompt):
        qc = MagicMock()
        qc.evaluate.return_value = {
            "pass": False,
            "confidence": 0.9,
            "auto_fail_triggered": ["action_loop"],
            "minor_issues": [],
        }

        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(system_prompt, mock_kie, qc_client=qc)
                scene = SceneSpec(description="Test", scene_type="pool")
                result = pipeline.evaluate_scene(scene)

        assert result.qc_result is not None
        assert result.qc_result.qc_pass is False
        assert "action_loop" in result.qc_result.auto_fail_triggered

    def test_increments_counter(self, mock_kie, system_prompt):
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(system_prompt, mock_kie)
                r1 = pipeline.evaluate_scene(SceneSpec("A", "pool"))
                r2 = pipeline.evaluate_scene(SceneSpec("B", "room"))

        assert r1.sample_id == "eval_0001"
        assert r2.sample_id == "eval_0002"


# ---------------------------------------------------------------------------
# Pipeline - evaluate_batch
# ---------------------------------------------------------------------------

class TestEvaluateBatch:
    def test_batch_with_default_scenes(self, mock_kie, system_prompt):
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(system_prompt, mock_kie, output_dir="/tmp/test_eval")
                batch = pipeline.evaluate_batch(save=False)

        assert len(batch.results) == len(DEFAULT_HOTEL_SCENES)
        assert batch.success_count == len(DEFAULT_HOTEL_SCENES)

    def test_batch_with_custom_scenes(self, mock_kie, system_prompt):
        scenes = [
            SceneSpec("Pool", "pool"),
            SceneSpec("Room", "room"),
        ]
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(system_prompt, mock_kie)
                batch = pipeline.evaluate_batch(scenes, save=False)

        assert len(batch.results) == 2

    def test_batch_saves_results(self, mock_kie, system_prompt, tmp_path):
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(
                    system_prompt, mock_kie, output_dir=str(tmp_path)
                )
                pipeline.evaluate_batch(
                    [SceneSpec("Test", "pool")],
                    save=True,
                )

        json_files = list(tmp_path.glob("eval_*.json"))
        assert len(json_files) == 1


# ---------------------------------------------------------------------------
# Pipeline - report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_report_basic(self, mock_kie, system_prompt):
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(system_prompt, mock_kie)
                batch = pipeline.evaluate_batch(
                    [SceneSpec("Pool", "pool"), SceneSpec("Room", "room")],
                    save=False,
                )
                report = pipeline.generate_report(batch)

        assert "# Prompt Evaluation Report" in report
        assert "pool" in report
        assert "room" in report


# ---------------------------------------------------------------------------
# Pipeline - optimize
# ---------------------------------------------------------------------------

class TestOptimize:
    def test_optimize_without_llm_returns_meta_prompt(self, mock_kie, system_prompt):
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                # Force OPRO path by disabling DSPy
                pipeline = EvalPipeline(system_prompt, mock_kie, use_dspy=False)
                batch = pipeline.evaluate_batch(
                    [SceneSpec(f"Scene {i}", "pool") for i in range(5)],
                    save=False,
                )
                meta = pipeline.optimize(batch)

        assert "CURRENT SYSTEM PROMPT" in meta
        assert "HISTORICAL PROMPT VARIANTS" in meta

    def test_optimize_with_dspy_returns_improved_template(self, mock_kie, system_prompt):
        """When DSPy is available, optimize() returns an improved template string."""
        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(system_prompt, mock_kie)
                batch = pipeline.evaluate_batch(
                    [SceneSpec(f"Scene {i}", "pool") for i in range(5)],
                    save=False,
                )
                result = pipeline.optimize(batch)

        # DSPy should return a non-empty string (the improved template)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_optimize_with_llm_opro_fallback(self, mock_kie, system_prompt):
        """When DSPy is disabled but LLM is available, use OPRO path."""
        llm = MagicMock()
        llm.generate.return_value = "Improved system prompt here"

        with patch.object(mock_kie, "generate_image", return_value=_img_result()):
            with patch.object(mock_kie, "generate_video", return_value=_vid_result()):
                pipeline = EvalPipeline(
                    system_prompt, mock_kie, llm_client=llm, use_dspy=False,
                )
                batch = pipeline.evaluate_batch(
                    [SceneSpec(f"Scene {i}", "pool") for i in range(5)],
                    save=False,
                )
                improved = pipeline.optimize(batch)

        assert improved == "Improved system prompt here"


# ---------------------------------------------------------------------------
# Default prompts
# ---------------------------------------------------------------------------

class TestDefaultPrompts:
    def test_all_scene_types_have_defaults(self):
        pipeline = EvalPipeline.__new__(EvalPipeline)
        for scene_type in ["pool", "room", "lobby", "spa"]:
            prompt = pipeline._default_prompt(SceneSpec("test", scene_type))
            assert "Single continuous shot" in prompt
            assert len(prompt.split()) >= 20

    def test_unknown_scene_falls_back_to_room(self):
        pipeline = EvalPipeline.__new__(EvalPipeline)
        prompt = pipeline._default_prompt(SceneSpec("test", "unknown"))
        assert "hotel room" in prompt.lower()
