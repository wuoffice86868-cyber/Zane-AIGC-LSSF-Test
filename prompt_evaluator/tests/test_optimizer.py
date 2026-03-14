"""Tests for PromptOptimizer."""

import pytest

from prompt_evaluator.optimizer import PromptOptimizer

from .conftest import make_sample


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------

class MockLLM:
    """Deterministic mock that returns a fixed improved prompt."""

    def __init__(self, response: str = "Single continuous shot. Improved prompt."):
        self.response = response
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append(prompt)
        return self.response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CURRENT_PROMPT = """\
## Role
Convert scene analysis JSON into a Seedance video prompt.

## Rules
1. Single continuous shot
2. Use Seedance camera verbs
3. 30-45 words
"""

HISTORY_SAMPLES = [
    make_sample(
        f"h{i:02d}",
        human_score=score,
        qc_pass=(score >= 6),
        confidence=0.5 + score * 0.05,
        scene_type=scene,
        generated_prompt=prompt,
    )
    for i, (score, scene, prompt) in enumerate([
        (3, "pool", "Camera zooms into pool. Water moves. Things shift."),
        (4, "room", "The camera moves fast through the room. Bright light."),
        (5, "exterior", "Wide shot of building. Camera pans. Trees visible."),
        (7, "pool", "Single continuous shot. Wide shot of pool at sunset. "
                     "The camera pushes slowly forward. Water ripples softly. "
                     "Horizon holds steady."),
        (8, "room", "Single continuous shot. Medium shot of luxury suite. "
                     "The camera pushes slowly forward toward the bed. "
                     "Furniture stays perfectly still."),
        (9, "lobby", "Single continuous shot. Wide shot of grand lobby. "
                      "The camera moves right across marble floor. "
                      "Chandelier stays fixed. Light streams through windows."),
        (6, "pool", "Single continuous shot. The camera circles around the pool. "
                     "Water ripples. Palm fronds move gently."),
        (2, "bathroom", "Quick zoom on bathtub. Everything moves."),
    ])
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildMetaPrompt:
    def test_basic_structure(self):
        opt = PromptOptimizer()
        meta = opt.build_meta_prompt(CURRENT_PROMPT, HISTORY_SAMPLES)
        assert isinstance(meta, str)
        assert len(meta) > 100

        # Should contain key OPRO components
        assert "CURRENT SYSTEM PROMPT" in meta
        assert "HISTORICAL PROMPT VARIANTS" in meta
        assert "SCORING DIMENSIONS" in meta
        assert "DOMAIN CONSTRAINTS" in meta
        assert "TASK" in meta
        assert "Seedance" in meta

    def test_history_sorted_ascending(self):
        """History should be worst → best (OPRO convention)."""
        opt = PromptOptimizer()
        meta = opt.build_meta_prompt(CURRENT_PROMPT, HISTORY_SAMPLES, top_k=3, bottom_k=3)
        # Find score lines
        score_lines = [line for line in meta.split("\n") if line.startswith("Score ")]
        if len(score_lines) >= 2:
            scores = []
            for line in score_lines:
                s = float(line.split("|")[0].replace("Score", "").strip())
                scores.append(s)
            # Should be ascending
            assert scores == sorted(scores), f"Scores not ascending: {scores}"

    def test_top_k_bottom_k(self):
        opt = PromptOptimizer()
        meta = opt.build_meta_prompt(CURRENT_PROMPT, HISTORY_SAMPLES, top_k=2, bottom_k=2)
        score_lines = [line for line in meta.split("\n") if line.startswith("Score ")]
        # Should have at most 4 entries (may be less if overlap)
        assert len(score_lines) <= 4

    def test_empty_history(self):
        opt = PromptOptimizer()
        meta = opt.build_meta_prompt(CURRENT_PROMPT, [])
        assert "no history" in meta.lower()

    def test_includes_scene_stats(self):
        opt = PromptOptimizer()
        meta = opt.build_meta_prompt(
            CURRENT_PROMPT, HISTORY_SAMPLES,
            include_scene_breakdown=True
        )
        # With 8 samples, should include scene stats
        assert "pool" in meta.lower()

    def test_includes_feature_insights(self):
        opt = PromptOptimizer()
        meta = opt.build_meta_prompt(
            CURRENT_PROMPT, HISTORY_SAMPLES,
            include_feature_insights=True
        )
        # Should include correlation data
        assert "correlation" in meta.lower() or "FEATURE" in meta

    def test_no_scene_stats_when_disabled(self):
        opt = PromptOptimizer()
        meta = opt.build_meta_prompt(
            CURRENT_PROMPT, HISTORY_SAMPLES,
            include_scene_breakdown=False,
            include_feature_insights=False,
        )
        assert "SCENE-TYPE PERFORMANCE" not in meta
        assert "FEATURE INSIGHTS" not in meta


class TestSuggestImprovement:
    def test_with_llm(self):
        mock = MockLLM("Single continuous shot. Better prompt version.")
        opt = PromptOptimizer(llm_client=mock)
        result = opt.suggest_improvement(CURRENT_PROMPT, HISTORY_SAMPLES)
        assert result == "Single continuous shot. Better prompt version."
        assert len(mock.calls) == 1
        # The call should contain our meta-prompt
        assert "CURRENT SYSTEM PROMPT" in mock.calls[0]

    def test_without_llm_raises(self):
        opt = PromptOptimizer()
        with pytest.raises(RuntimeError, match="No llm_client"):
            opt.suggest_improvement(CURRENT_PROMPT, HISTORY_SAMPLES)

    def test_temperature_forwarded(self):
        call_kwargs = {}

        class TrackingLLM:
            def generate(self, prompt, **kwargs):
                call_kwargs.update(kwargs)
                return "improved"

        opt = PromptOptimizer(llm_client=TrackingLLM())
        opt.suggest_improvement(CURRENT_PROMPT, HISTORY_SAMPLES, temperature=0.8)
        assert call_kwargs.get("temperature") == 0.8


class TestOptimizeLoop:
    def test_loop_generates_candidates(self):
        mock = MockLLM("candidate prompt")
        opt = PromptOptimizer(llm_client=mock)
        candidates = opt.optimize_loop(
            CURRENT_PROMPT, HISTORY_SAMPLES,
            steps=2, candidates_per_step=3
        )
        assert len(candidates) == 6  # 2 steps * 3 candidates
        assert all(c == "candidate prompt" for c in candidates)
        assert len(mock.calls) == 6

    def test_loop_without_llm_raises(self):
        opt = PromptOptimizer()
        with pytest.raises(RuntimeError):
            opt.optimize_loop(CURRENT_PROMPT, HISTORY_SAMPLES)


class TestScoring:
    def test_human_score_preferred(self):
        """When human labels exist, use human_score * 10."""
        opt = PromptOptimizer()
        sample = make_sample("s1", human_score=8, qc_pass=False)
        scored = opt._score_samples([sample])
        assert scored[0][1] == 80.0  # 8 * 10

    def test_qc_fallback(self):
        """Without human labels, fall back to QC reward."""
        opt = PromptOptimizer()
        sample = make_sample("s1", has_human_label=False, qc_pass=True, confidence=0.95)
        scored = opt._score_samples([sample])
        assert scored[0][1] > 0  # Should have a QC-based score
