"""Tests for PromptAnalyzer."""

import pytest

from prompt_evaluator.models import PromptFeatures, CorrelationResult
from prompt_evaluator.prompt_analyzer import PromptAnalyzer

from .conftest import make_sample


class TestExtractFeatures:
    def setup_method(self):
        self.analyzer = PromptAnalyzer()

    def test_full_prompt(self):
        prompt = (
            "Single continuous shot. Wide shot of an infinity pool at golden hour. "
            "The camera pushes slowly forward over the water. "
            "Water ripples softly. The horizon holds steady. "
            "Sky and clouds remain completely still."
        )
        feat = self.analyzer.extract_features(prompt)
        assert isinstance(feat, PromptFeatures)
        assert feat.has_camera_move is True
        assert feat.camera_move_type == "push"
        assert feat.has_single_continuous_shot is True
        assert feat.has_sky_freeze is True
        assert feat.has_stable_element is True
        assert "horizon" in feat.stable_element.lower()
        assert feat.has_speed_modifier is True
        assert feat.speed_modifier == "slowly"
        assert feat.shot_size == "wide shot"
        assert feat.word_count > 20

    def test_circle_around(self):
        prompt = "Single continuous shot. The camera circles around the bathtub gently."
        feat = self.analyzer.extract_features(prompt)
        assert feat.has_camera_move is True
        assert feat.camera_move_type == "circle around"

    def test_no_camera_move(self):
        prompt = "A static view of the hotel room in warm light."
        feat = self.analyzer.extract_features(prompt)
        assert feat.has_camera_move is False
        assert feat.camera_move_type == ""

    def test_sky_freeze_detection(self):
        prompt = "The camera rises. Sky and clouds remain still."
        feat = self.analyzer.extract_features(prompt)
        assert feat.has_sky_freeze is True

    def test_no_sky_freeze(self):
        prompt = "The camera pushes forward over the pool."
        feat = self.analyzer.extract_features(prompt)
        assert feat.has_sky_freeze is False

    def test_motion_elements(self):
        prompt = "Water ripples softly. Palm fronds move gently in the breeze."
        feat = self.analyzer.extract_features(prompt)
        assert len(feat.motion_elements) >= 1

    def test_stable_element_variants(self):
        for phrase, expected_word in [
            ("Furniture stays perfectly still.", "furniture"),
            ("Tile walls remain fixed.", "tile walls"),
            ("Building stays anchored.", "building"),
        ]:
            feat = self.analyzer.extract_features(phrase)
            assert feat.has_stable_element is True, f"Failed for: {phrase}"

    def test_empty_prompt(self):
        feat = self.analyzer.extract_features("")
        assert feat.word_count == 0
        assert feat.has_camera_move is False

    def test_shot_sizes(self):
        for shot in ["close-up", "medium shot", "wide shot"]:
            feat = self.analyzer.extract_features(f"{shot} of the pool")
            assert feat.shot_size == shot

    def test_human_instruction(self):
        prompt = "Consistent human subject with natural features. Person remains natural."
        feat = self.analyzer.extract_features(prompt)
        assert feat.has_human_instruction is True

    def test_sentence_count(self):
        prompt = "First sentence. Second sentence. Third sentence!"
        feat = self.analyzer.extract_features(prompt)
        assert feat.sentence_count == 3


class TestBatchExtract:
    def test_batch(self):
        analyzer = PromptAnalyzer()
        prompts = [
            "The camera pushes forward slowly.",
            "The camera circles around the pool.",
        ]
        results = analyzer.batch_extract(prompts)
        assert len(results) == 2
        assert results[0].camera_move_type == "push"
        assert results[1].camera_move_type == "circle around"


class TestCorrelationAnalysis:
    def test_with_mixed_samples(self, mixed_samples):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze_correlation(mixed_samples)
        assert isinstance(result, CorrelationResult)
        assert len(result.feature_importance) > 0
        # All correlations should be in [-1, 1]
        for feat, r in result.feature_importance.items():
            assert -1.0 <= r <= 1.0, f"{feat} correlation {r} out of range"

    def test_too_few_samples(self):
        analyzer = PromptAnalyzer()
        samples = [make_sample("s0"), make_sample("s1")]
        result = analyzer.analyze_correlation(samples)
        # Should return empty result, not crash
        assert result.feature_importance == {}

    def test_uses_human_score(self, mixed_samples):
        analyzer = PromptAnalyzer()
        result = analyzer.analyze_correlation(mixed_samples, use_human_score=True)
        assert len(result.feature_importance) > 0

    def test_uses_qc_reward_fallback(self):
        """When no human labels, falls back to QC reward."""
        analyzer = PromptAnalyzer()
        samples = [
            make_sample(f"s{i}", has_human_label=False, qc_pass=(i > 2), confidence=0.5 + i * 0.1)
            for i in range(5)
        ]
        result = analyzer.analyze_correlation(samples, use_human_score=True)
        assert len(result.feature_importance) > 0
