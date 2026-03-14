"""End-to-end evaluation pipeline.

Orchestrates: image generation → video generation → QC evaluation →
reward scoring → prompt analysis → optimization suggestions.

This is the 闭环 (closed loop) that ties all prompt_evaluator components
together with the KIE API client.

Usage:
    from prompt_evaluator.pipeline import EvalPipeline

    pipeline = EvalPipeline(
        system_prompt="You are a cinematography director...",
        kie_client=KieClient(),
    )

    # Run a single evaluation
    result = pipeline.evaluate_prompt(
        scene_description="Luxury hotel pool at golden hour",
        scene_type="pool",
    )

    # Run batch evaluation
    results = pipeline.evaluate_batch(scenes)

    # Get optimization suggestions
    improved = pipeline.optimize(results)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .models import (
    EvalSample,
    InputInfo,
    OutputInfo,
    PromptFeatures,
    PromptInfo,
    QCResult,
    RewardBreakdown,
)
from .kie_client import KieClient, TaskResult
from .prompt_analyzer import PromptAnalyzer
from .reward_calculator import RewardCalculator
from .optimizer import PromptOptimizer, LLMClient

# DSPy optimizer (preferred when available)
try:
    from .dspy_optimizer import (
        VideoPromptOptimizer,
        SceneInput,
        HAS_DSPY,
    )
except ImportError:
    HAS_DSPY = False
    VideoPromptOptimizer = None  # type: ignore[misc, assignment]
    SceneInput = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _DummyScorer:
    """Placeholder scorer when no QC client is provided."""

    def evaluate(self, video_url: str, **kwargs: Any) -> Dict[str, Any]:
        return {
            "pass": True,
            "confidence": 0.5,
            "aesthetic_score": 5,
            "motion_score": 5,
            "prompt_adherence_score": 5,
            "issues": [],
            "auto_fail_triggered": [],
        }


# ---------------------------------------------------------------------------
# Pipeline result types
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of a single prompt evaluation cycle."""

    sample_id: str
    scene_type: str = ""
    scene_description: str = ""

    # Generated content
    generated_prompt: str = ""
    image_url: str = ""
    image_task_id: str = ""
    video_url: str = ""
    video_task_id: str = ""

    # Evaluation
    qc_result: Optional[QCResult] = None
    reward: Optional[RewardBreakdown] = None
    features: Optional[PromptFeatures] = None

    # Metadata
    system_prompt_version: str = ""
    timestamp: str = ""
    image_cost_ms: int = 0
    video_cost_ms: int = 0
    error: str = ""

    @property
    def success(self) -> bool:
        return bool(self.video_url) and self.error == ""

    def to_eval_sample(self) -> EvalSample:
        """Convert to an EvalSample for use with other prompt_evaluator modules."""
        return EvalSample(
            sample_id=self.sample_id,
            input=InputInfo(
                image_url=self.image_url,
                scene_type=self.scene_type,
                poi_name=self.scene_description,
            ),
            prompt=PromptInfo(
                system_prompt_version=self.system_prompt_version,
                generated_prompt=self.generated_prompt,
            ),
            output=OutputInfo(
                video_url=self.video_url,
                video_duration_sec=8.0,
                generation_method="Seedance 1.5 Pro",
            ),
            qc_result=self.qc_result or QCResult(**{"pass": True}),
        )


@dataclass
class BatchResult:
    """Result of a batch evaluation run."""

    results: List[EvalResult] = field(default_factory=list)
    system_prompt: str = ""
    started_at: str = ""
    completed_at: str = ""

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def avg_reward(self) -> float:
        scores = [r.reward.total_score for r in self.results if r.reward]
        return sum(scores) / len(scores) if scores else 0.0

    def to_eval_samples(self) -> List[EvalSample]:
        return [r.to_eval_sample() for r in self.results if r.success]

    def summary(self) -> str:
        return (
            f"Batch: {len(self.results)} runs, "
            f"{self.success_count} success, {self.fail_count} failed, "
            f"avg reward: {self.avg_reward:.1f}"
        )


# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------

@dataclass
class SceneSpec:
    """Specification for a scene to evaluate."""

    description: str
    scene_type: str = "other"
    image_prompt: str = ""  # Override for image generation (defaults to description)
    reference_image_url: str = ""  # Skip image gen, use this directly


# Default hotel scenes for evaluation
DEFAULT_HOTEL_SCENES: List[SceneSpec] = [
    SceneSpec(
        description="Luxury hotel swimming pool at golden hour with crystal clear water",
        scene_type="pool",
        image_prompt="A luxury hotel outdoor swimming pool at golden hour, crystal clear turquoise water, sun loungers, palm trees, warm golden light, professional hotel photography",
    ),
    SceneSpec(
        description="Modern hotel room with king bed and city view",
        scene_type="room",
        image_prompt="A modern luxury hotel room with king size bed, floor to ceiling windows showing city skyline, warm ambient lighting, crisp white sheets, professional hotel photography",
    ),
    SceneSpec(
        description="Hotel lobby with marble floors and chandelier",
        scene_type="lobby",
        image_prompt="A grand hotel lobby with polished marble floors, crystal chandelier, elegant furniture, warm lighting, fresh flowers on reception desk, professional hotel photography",
    ),
    SceneSpec(
        description="Hotel spa treatment room with candles",
        scene_type="spa",
        image_prompt="A serene hotel spa treatment room, warm candlelight, stone walls, fluffy white towels, bamboo accents, calming atmosphere, professional hotel photography",
    ),
]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class EvalPipeline:
    """End-to-end prompt evaluation pipeline.

    The pipeline generates images, then videos from those images using the
    system prompt, then evaluates the results. Optionally runs QC (when a
    qc_client is provided) and computes reward scores.

    Args:
        system_prompt: The cinematography system prompt to evaluate.
        kie_client: KIE API client for image + video generation.
        qc_client: Optional QC evaluator (any object with
            ``evaluate(video_url: str) -> dict`` method).
        llm_client: Optional LLM for prompt generation and optimization.
        reward_calculator: Reward calculator (default created if None).
        prompt_analyzer: Prompt analyzer (default created if None).
        output_dir: Directory to save results JSON files.
    """

    def __init__(
        self,
        system_prompt: str,
        kie_client: KieClient,
        *,
        qc_client: Any = None,
        llm_client: Optional[LLMClient] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        prompt_analyzer: Optional[PromptAnalyzer] = None,
        output_dir: str = "eval_results",
        use_dspy: bool = True,
        dspy_model: str = "gemini/gemini-2.5-flash",
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.kie = kie_client
        self.qc_client = qc_client
        self.llm = llm_client
        self.reward_calc = reward_calculator or RewardCalculator()
        self.analyzer = prompt_analyzer or PromptAnalyzer()

        # OPRO optimizer (legacy fallback)
        self.opro_optimizer = PromptOptimizer(
            llm_client=llm_client,
            reward_calculator=self.reward_calc,
            prompt_analyzer=self.analyzer,
        )

        # DSPy optimizer (preferred)
        self.dspy_optimizer: Optional[Any] = None
        self._use_dspy = use_dspy and HAS_DSPY and VideoPromptOptimizer is not None
        if self._use_dspy:
            try:
                self.dspy_optimizer = VideoPromptOptimizer(
                    generator=kie_client,
                    scorer=qc_client if qc_client else _DummyScorer(),
                    reward_calculator=self.reward_calc,
                    lm_model=dspy_model,
                    api_key=gemini_api_key,
                )
                # Load the system prompt template into DSPy optimizer
                self.dspy_optimizer.template = system_prompt
                logger.info("DSPy optimizer initialized (model=%s)", dspy_model)
            except Exception as e:
                logger.warning("DSPy optimizer init failed, falling back to OPRO: %s", e)
                self.dspy_optimizer = None
                self._use_dspy = False

        self.output_dir = Path(output_dir)
        self._counter = 0

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate_scene(
        self,
        scene: SceneSpec,
        *,
        cinematography_prompt: Optional[str] = None,
    ) -> EvalResult:
        """Evaluate a single scene through the full pipeline.

        Steps:
        1. Generate reference image (or use provided URL)
        2. Generate cinematography prompt (or use provided)
        3. Generate video from image + prompt
        4. Run QC (if qc_client available)
        5. Compute reward score
        6. Extract prompt features

        Args:
            scene: Scene specification.
            cinematography_prompt: Override the generated prompt.
                If None and llm_client is available, generates via LLM.
                If None and no llm_client, uses a default template.

        Returns:
            ``EvalResult`` with all pipeline outputs.
        """
        self._counter += 1
        sample_id = f"eval_{self._counter:04d}"
        now = datetime.now(timezone.utc).isoformat()

        result = EvalResult(
            sample_id=sample_id,
            scene_type=scene.scene_type,
            scene_description=scene.description,
            system_prompt_version=self._prompt_version(),
            timestamp=now,
        )

        try:
            # Step 1: Get reference image
            if scene.reference_image_url:
                result.image_url = scene.reference_image_url
                logger.info("[%s] Using provided reference image", sample_id)
            else:
                image_prompt = scene.image_prompt or scene.description
                logger.info("[%s] Generating image: %s", sample_id, image_prompt[:60])
                img_result = self.kie.generate_image(image_prompt)
                if not img_result.success or not img_result.result_urls:
                    result.error = f"Image generation failed: {img_result.state}"
                    return result
                result.image_url = img_result.result_urls[0]
                result.image_task_id = img_result.task_id
                result.image_cost_ms = img_result.cost_time_ms

            # Step 2: Get cinematography prompt
            if cinematography_prompt:
                result.generated_prompt = cinematography_prompt
            elif self.llm:
                result.generated_prompt = self._generate_prompt(scene)
            else:
                result.generated_prompt = self._default_prompt(scene)

            # Step 3: Generate video
            logger.info(
                "[%s] Generating video: %s",
                sample_id,
                result.generated_prompt[:60],
            )
            vid_result = self.kie.generate_video(
                prompt=result.generated_prompt,
                image_url=result.image_url,
            )
            if not vid_result.success or not vid_result.result_urls:
                result.error = f"Video generation failed: {vid_result.state}"
                return result
            result.video_url = vid_result.result_urls[0]
            result.video_task_id = vid_result.task_id
            result.video_cost_ms = vid_result.cost_time_ms

            # Step 4: QC evaluation (optional)
            if self.qc_client:
                try:
                    qc_dict = self.qc_client.evaluate(result.video_url)
                    result.qc_result = QCResult(**qc_dict)
                except Exception as e:
                    logger.warning("[%s] QC failed: %s", sample_id, e)
                    result.qc_result = QCResult(**{"pass": True, "confidence": 0.0})
            else:
                # Default pass with low confidence when no QC
                result.qc_result = QCResult(**{"pass": True, "confidence": 0.5})

            # Step 5: Reward
            result.reward = self.reward_calc.calculate(result.qc_result)

            # Step 6: Feature extraction
            result.features = self.analyzer.extract_features(result.generated_prompt)

        except Exception as e:
            result.error = str(e)
            logger.error("[%s] Pipeline error: %s", sample_id, e)

        return result

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        scenes: Optional[Sequence[SceneSpec]] = None,
        *,
        save: bool = True,
    ) -> BatchResult:
        """Evaluate multiple scenes.

        Args:
            scenes: Scene specifications. Uses DEFAULT_HOTEL_SCENES if None.
            save: Save results to output_dir as JSON.

        Returns:
            ``BatchResult`` with all individual results.
        """
        if scenes is None:
            scenes = DEFAULT_HOTEL_SCENES

        batch = BatchResult(
            system_prompt=self.system_prompt,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        for scene in scenes:
            logger.info("Evaluating scene: %s (%s)", scene.description[:40], scene.scene_type)
            result = self.evaluate_scene(scene)
            batch.results.append(result)

            if result.success:
                logger.info(
                    "  ✓ reward=%.1f, features: cam=%s, sky_freeze=%s",
                    result.reward.total_score if result.reward else 0,
                    result.features.camera_move_type if result.features else "?",
                    result.features.has_sky_freeze if result.features else "?",
                )
            else:
                logger.warning("  ✗ error: %s", result.error)

        batch.completed_at = datetime.now(timezone.utc).isoformat()

        if save:
            self._save_batch(batch)

        logger.info(batch.summary())
        return batch

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        batch: BatchResult,
        *,
        top_k: int = 5,
        bottom_k: int = 3,
    ) -> str:
        """Generate optimization suggestions based on batch results.

        Uses DSPy optimizer (critique → evidence-based template improvement)
        when available. Falls back to OPRO meta-prompt approach otherwise.

        Args:
            batch: Completed batch evaluation result.
            top_k: Number of top-scoring samples for meta-prompt.
            bottom_k: Number of bottom-scoring samples for meta-prompt.

        Returns:
            Improved system prompt string.
        """
        samples = batch.to_eval_samples()

        # --- DSPy path: critique each result, then improve template ---
        if self._use_dspy and self.dspy_optimizer is not None:
            logger.info("Running DSPy-based optimization (%d samples)", len(batch.results))
            try:
                # Collect examples with critiques
                examples: List[Dict[str, Any]] = []
                for r in batch.results:
                    if not r.success or not r.qc_result or not r.reward:
                        continue

                    qc_dict = r.qc_result.model_dump(by_alias=True)

                    # Build SceneInput from our SceneSpec data
                    scene_input = SceneInput(
                        scene_description=r.scene_description,
                        main_subject=r.scene_description.split(" with ")[0] if " with " in r.scene_description else r.scene_description,
                    )

                    # Get critique from DSPy
                    critique = self.dspy_optimizer.critique_result(
                        prompt=r.generated_prompt,
                        scene=scene_input,
                        qc_result=qc_dict,
                    )
                    logger.info(
                        "  Critique for %s (reward=%.1f): %s",
                        r.scene_type, r.reward.total_score, critique[:100],
                    )

                    examples.append({
                        "prompt": r.generated_prompt,
                        "scene": scene_input,
                        "qc_result": qc_dict,
                        "reward": r.reward.total_score,
                        "critique": critique,
                    })

                if not examples:
                    logger.warning("No successful results to optimize from")
                    return self.system_prompt

                # Improve template based on accumulated evidence
                improved = self.dspy_optimizer.improve_template(examples)
                logger.info(
                    "DSPy template improvement complete (%d chars → %d chars)",
                    len(self.system_prompt), len(improved),
                )
                return improved

            except Exception as e:
                logger.error("DSPy optimization failed, falling back to OPRO: %s", e)

        # --- OPRO fallback ---
        if self.llm:
            return self.opro_optimizer.suggest_improvement(
                self.system_prompt,
                samples,
                top_k=top_k,
                bottom_k=bottom_k,
            )

        return self.opro_optimizer.build_meta_prompt(
            self.system_prompt,
            samples,
            top_k=top_k,
            bottom_k=bottom_k,
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self, batch: BatchResult) -> str:
        """Generate a human-readable evaluation report.

        Args:
            batch: Completed batch results.

        Returns:
            Markdown-formatted report string.
        """
        lines: List[str] = [
            f"# Prompt Evaluation Report",
            f"",
            f"System prompt version: {self._prompt_version()}",
            f"Run time: {batch.started_at} → {batch.completed_at}",
            f"Total scenes: {len(batch.results)}",
            f"Success: {batch.success_count}, Failed: {batch.fail_count}",
            f"Average reward: {batch.avg_reward:.1f}/100",
            "",
        ]

        # Per-scene results
        lines.append("## Per-Scene Results\n")
        for r in batch.results:
            status = "✓" if r.success else "✗"
            reward_str = f"{r.reward.total_score:.1f}" if r.reward else "N/A"
            lines.append(f"### {status} {r.sample_id} — {r.scene_type}")
            lines.append(f"- Description: {r.scene_description}")
            lines.append(f"- Prompt: {r.generated_prompt[:100]}...")
            lines.append(f"- Reward: {reward_str}")
            if r.features:
                lines.append(
                    f"- Features: camera={r.features.camera_move_type or 'none'}, "
                    f"sky_freeze={r.features.has_sky_freeze}, "
                    f"stable={r.features.has_stable_element}, "
                    f"words={r.features.word_count}"
                )
            if r.video_url:
                lines.append(f"- Video: {r.video_url}")
            if r.error:
                lines.append(f"- Error: {r.error}")
            lines.append("")

        # Feature correlation (if enough data)
        samples = batch.to_eval_samples()
        if len(samples) >= 3:
            corr = self.analyzer.analyze_correlation(samples, use_human_score=False)
            if corr.feature_importance:
                lines.append("## Feature Correlation with Reward\n")
                for feat, r_val in list(corr.feature_importance.items())[:8]:
                    direction = "↑" if r_val > 0 else "↓"
                    lines.append(f"- {feat}: r={r_val:+.3f} {direction}")
                lines.append("")

        # Scene-type breakdown
        from collections import defaultdict
        scene_scores: Dict[str, List[float]] = defaultdict(list)
        for r in batch.results:
            if r.reward:
                scene_scores[r.scene_type].append(r.reward.total_score)

        if scene_scores:
            lines.append("## Scene-Type Performance\n")
            for scene, scores in sorted(scene_scores.items()):
                avg = sum(scores) / len(scores)
                lines.append(f"- {scene}: avg={avg:.1f}, n={len(scores)}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # DSPy full optimization loop
    # ------------------------------------------------------------------

    def run_optimization_loop(
        self,
        scenes: Optional[Sequence[SceneSpec]] = None,
        *,
        max_rounds: int = 5,
        target_reward: float = 85.0,
        save: bool = True,
    ) -> List[BatchResult]:
        """Run the full DSPy optimization loop autonomously.

        Each round:
        1. Generate + score videos using current system prompt
        2. Critique results with DSPy
        3. Improve system prompt based on evidence
        4. Repeat with improved prompt

        Stops when target_reward is reached or max_rounds exhausted.

        Args:
            scenes: Scenes to evaluate each round.
            max_rounds: Maximum optimization rounds.
            target_reward: Stop early if avg reward exceeds this.
            save: Save each round's results.

        Returns:
            List of BatchResult, one per round.
        """
        if scenes is None:
            scenes = DEFAULT_HOTEL_SCENES

        all_rounds: List[BatchResult] = []

        for round_num in range(1, max_rounds + 1):
            logger.info(
                "=== Optimization Round %d/%d (prompt version: %s) ===",
                round_num, max_rounds, self._prompt_version(),
            )

            # Step 1: Evaluate current prompt
            batch = self.evaluate_batch(scenes, save=save)
            all_rounds.append(batch)

            logger.info(
                "Round %d results: avg_reward=%.1f, pass=%d/%d",
                round_num, batch.avg_reward,
                batch.success_count, len(batch.results),
            )

            # Step 2: Check if we've hit the target
            if batch.avg_reward >= target_reward:
                logger.info(
                    "Target reward %.1f reached (got %.1f). Stopping.",
                    target_reward, batch.avg_reward,
                )
                break

            # Step 3: Optimize (DSPy critique → improve, or OPRO fallback)
            if round_num < max_rounds:
                improved_prompt = self.optimize(batch)
                if improved_prompt and improved_prompt != self.system_prompt:
                    old_version = self._prompt_version()
                    self.system_prompt = improved_prompt
                    # Update DSPy optimizer's template too
                    if self.dspy_optimizer is not None:
                        self.dspy_optimizer.template = improved_prompt
                    logger.info(
                        "System prompt updated: %s → %s (%d chars)",
                        old_version, self._prompt_version(), len(improved_prompt),
                    )
                else:
                    logger.warning("Optimizer returned no improvement. Stopping.")
                    break

        # Summary
        if all_rounds:
            first_avg = all_rounds[0].avg_reward
            last_avg = all_rounds[-1].avg_reward
            logger.info(
                "Optimization complete: %d rounds, %.1f → %.1f (%+.1f)",
                len(all_rounds), first_avg, last_avg, last_avg - first_avg,
            )

        return all_rounds

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _prompt_version(self) -> str:
        """Generate a short version hash from the system prompt."""
        import hashlib
        return hashlib.sha256(self.system_prompt.encode()).hexdigest()[:8]

    def _generate_prompt(self, scene: SceneSpec) -> str:
        """Generate a cinematography prompt using DSPy module or LLM.

        Prefers DSPy (optimizable module) over raw LLM call.
        """
        # DSPy path: use the trainable prompt module
        if self._use_dspy and self.dspy_optimizer is not None:
            try:
                scene_input = SceneInput(
                    scene_description=scene.description,
                    main_subject=scene.description.split(" with ")[0] if " with " in scene.description else scene.description,
                    extra={"scene_type": scene.scene_type},
                )
                prompt = self.dspy_optimizer.generate_prompt(scene_input)
                if prompt and len(prompt) > 10:
                    return prompt
                logger.warning("DSPy generated empty/short prompt, falling back to LLM")
            except Exception as e:
                logger.warning("DSPy prompt gen failed: %s, falling back to LLM", e)

        # LLM fallback
        if not self.llm:
            return self._default_prompt(scene)

        user_msg = (
            f"Generate a cinematography prompt for this scene: {scene.description}\n"
            f"Scene type: {scene.scene_type}"
        )
        full_prompt = f"{self.system_prompt}\n\n{user_msg}"
        return self.llm.generate(full_prompt).strip()

    @staticmethod
    def _default_prompt(scene: SceneSpec) -> str:
        """Generate a default prompt template when no LLM is available."""
        prompts = {
            "pool": (
                "Single continuous shot. Wide shot of the pool area. "
                "The camera pushes slowly forward. Water ripples softly "
                "in the warm light. Sky and clouds remain completely still. "
                "The pool edge stays anchored in frame."
            ),
            "room": (
                "Single continuous shot. Medium shot of the hotel room. "
                "The camera pushes slowly forward toward the window. "
                "Curtains sway gently. The bed frame stays fixed in place."
            ),
            "lobby": (
                "Single continuous shot. Wide shot of the hotel lobby. "
                "The camera pushes slowly forward across the marble floor. "
                "The chandelier stays anchored above. "
                "Reflections on the floor remain steady."
            ),
            "spa": (
                "Single continuous shot. Medium shot of the spa room. "
                "The camera pushes slowly forward. "
                "Candle flames flicker gently. "
                "Stone walls stay perfectly still."
            ),
        }
        return prompts.get(scene.scene_type, prompts["room"])

    def _save_batch(self, batch: BatchResult) -> None:
        """Save batch results to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"eval_{ts}.json"

        data = {
            "system_prompt": batch.system_prompt,
            "system_prompt_version": self._prompt_version(),
            "started_at": batch.started_at,
            "completed_at": batch.completed_at,
            "summary": {
                "total": len(batch.results),
                "success": batch.success_count,
                "failed": batch.fail_count,
                "avg_reward": batch.avg_reward,
            },
            "results": [],
        }

        for r in batch.results:
            entry: Dict[str, Any] = {
                "sample_id": r.sample_id,
                "scene_type": r.scene_type,
                "scene_description": r.scene_description,
                "generated_prompt": r.generated_prompt,
                "image_url": r.image_url,
                "video_url": r.video_url,
                "image_task_id": r.image_task_id,
                "video_task_id": r.video_task_id,
                "image_cost_ms": r.image_cost_ms,
                "video_cost_ms": r.video_cost_ms,
                "error": r.error,
                "timestamp": r.timestamp,
            }
            if r.reward:
                entry["reward"] = {
                    "total_score": r.reward.total_score,
                    "breakdown": r.reward.breakdown,
                }
            if r.qc_result:
                entry["qc_result"] = r.qc_result.model_dump(by_alias=True)
            if r.features:
                entry["features"] = r.features.model_dump()
            data["results"].append(entry)

        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Saved batch results to %s", path)
