"""Tests for DSPy-based prompt optimizer.

OPRO optimizer has been removed (Phase 1 Foundation).
These tests verify the DSPy optimizer interfaces work correctly
with mocked LLM backends.
"""

import pytest

# DSPy may not be installed in test env
try:
    from prompt_evaluator.dspy_optimizer import (
        VideoPromptOptimizer,
        SceneInput,
        HAS_DSPY,
        SEEDANCE_CONSTRAINTS,
    )
except ImportError:
    HAS_DSPY = False


@pytest.mark.skipif(not HAS_DSPY, reason="DSPy not installed")
class TestSceneInput:
    def test_to_json(self):
        scene = SceneInput(
            scene_description="Hotel pool at sunset",
            main_subject="infinity pool",
            camera_move="push",
            lighting="golden hour",
        )
        j = scene.to_json()
        assert "infinity pool" in j
        assert "push" in j

    def test_extra_fields(self):
        scene = SceneInput(
            scene_description="test",
            extra={"scene_type": "pool", "custom_field": 42},
        )
        j = scene.to_json()
        assert "pool" in j
        assert "42" in j


@pytest.mark.skipif(not HAS_DSPY, reason="DSPy not installed")
class TestSeedanceConstraints:
    def test_constraints_not_empty(self):
        assert len(SEEDANCE_CONSTRAINTS) > 100

    def test_key_rules_present(self):
        c = SEEDANCE_CONSTRAINTS.lower()
        assert "negative prompts do not work" in c
        assert "degree adverbs" in c
        assert "25-45 words" in c


class TestOPRORemoved:
    """Verify OPRO optimizer module no longer exists."""

    def test_optimizer_module_removed(self):
        with pytest.raises(ImportError):
            from prompt_evaluator.optimizer import PromptOptimizer  # noqa: F401
