"""DSPy-based prompt optimization for video generation.

Replaces the OPRO-style optimizer with DSPy's systematic framework.
Designed to be generalizable across different video generation pipelines
(hotel marketing, manga, voice service, etc.).

The key abstraction: optimize the MAPPING from structured scene input
to natural language generation prompt, using video quality scores as
the optimization signal.

Architecture:
    StructuredInput → DSPy Module → NaturalLanguagePrompt → Generator → Video → Scorer → Reward
                         ↑                                                              |
                         └──────────────── DSPy Optimizer ────────────────────────────────┘

Usage:
    from prompt_evaluator.dspy_optimizer import VideoPromptOptimizer

    optimizer = VideoPromptOptimizer(
        generator=kie_client,
        scorer=gemini_qc,
        template_path="prompts/gemini_cinematography_prompt.txt",
    )

    # Optimize with accumulated data
    improved_template = optimizer.optimize(
        training_examples=examples,
        num_trials=20,
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

logger = logging.getLogger(__name__)

# Add DSPy deps to path if needed
_DSPY_DEPS = "/tmp/dspy_deps"
if _DSPY_DEPS not in sys.path and os.path.isdir(_DSPY_DEPS):
    sys.path.insert(0, _DSPY_DEPS)

try:
    import dspy
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False
    logger.warning("DSPy not available. Install with: pip install dspy")


# ---------------------------------------------------------------------------
# Protocols (for generalizability)
# ---------------------------------------------------------------------------

class VideoGenerator(Protocol):
    """Protocol for any video generation backend."""
    def generate_video(self, prompt: str, *, image_url: str, **kwargs: Any) -> Any: ...


class VideoScorer(Protocol):
    """Protocol for any video quality scorer."""
    def evaluate(self, video_url: str, **kwargs: Any) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SceneInput:
    """Structured input describing a scene for video generation.
    
    Generic enough for hotel, manga, product, travel — any domain.
    The field mapping is domain-configurable.
    """
    scene_description: str = ""
    main_subject: str = ""
    foreground: str = ""
    background: str = ""
    camera_move: str = ""
    camera_direction: str = ""
    shot_size: str = ""
    lighting: str = ""
    subtle_motion: List[str] = field(default_factory=list)
    stable_element: str = ""
    human_detected: bool = False
    
    # Extensible: domain-specific fields
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Serialize to JSON for LLM consumption."""
        d = {
            "scene_description": self.scene_description,
            "main_subject": self.main_subject,
            "foreground": self.foreground,
            "background": self.background,
            "camera_move": self.camera_move,
            "camera_direction": self.camera_direction,
            "shot_size": self.shot_size,
            "lighting": self.lighting,
            "subtle_motion": self.subtle_motion,
            "stable_element": self.stable_element,
            "human_detected": self.human_detected,
        }
        if self.extra:
            d.update(self.extra)
        return json.dumps(d, indent=2)


@dataclass
class OptimizationResult:
    """Result of a DSPy optimization run."""
    improved_template: str = ""
    best_score: float = 0.0
    num_trials: int = 0
    score_history: List[float] = field(default_factory=list)
    best_examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------

if HAS_DSPY:

    class SceneToPrompt(dspy.Signature):
        """Convert structured scene analysis into a video generation prompt.
        
        Given scene analysis JSON and a system template, produce a natural
        language prompt optimized for the video generation model.
        The prompt should be 30-45 words, use the model's native camera
        vocabulary, and follow the subject+details+motion+camera structure.
        """
        scene_json: str = dspy.InputField(
            desc="Structured JSON describing the scene (subject, camera, lighting, etc.)"
        )
        template_instructions: str = dspy.InputField(
            desc="System instructions for how to structure the output prompt"
        )
        video_prompt: str = dspy.OutputField(
            desc="Natural language video generation prompt, 30-45 words"
        )

    class PromptCritique(dspy.Signature):
        """Analyze why a video prompt produced good or bad results.
        
        Given the prompt, the scene input, and quality scores, explain
        what aspects of the prompt contributed to the quality outcome.
        Provide specific, actionable feedback.
        """
        video_prompt: str = dspy.InputField(
            desc="The video generation prompt that was used"
        )
        scene_json: str = dspy.InputField(
            desc="The structured scene input"
        )
        quality_scores: str = dspy.InputField(
            desc="JSON with aesthetic, motion, adherence, and overall scores"
        )
        auto_fails: str = dspy.InputField(
            desc="List of auto-fail rules triggered, if any"
        )
        critique: str = dspy.OutputField(
            desc="Specific analysis of what worked/failed in the prompt and why"
        )

    class TemplateImprover(dspy.Signature):
        """Improve a video prompt template based on accumulated evidence.
        
        Given the current template and a set of critiques from multiple
        test runs, produce an improved template that addresses the most
        common failure patterns while preserving successful patterns.
        """
        current_template: str = dspy.InputField(
            desc="Current system template for video prompt generation"
        )
        success_patterns: str = dspy.InputField(
            desc="Patterns from high-scoring prompts (what worked)"
        )
        failure_patterns: str = dspy.InputField(
            desc="Patterns from low-scoring prompts (what failed)"
        )
        model_constraints: str = dspy.InputField(
            desc="Known constraints of the video generation model"
        )
        improved_template: str = dspy.OutputField(
            desc="Improved system template incorporating the evidence"
        )


# ---------------------------------------------------------------------------
# DSPy Modules
# ---------------------------------------------------------------------------

if HAS_DSPY:

    class VideoPromptModule(dspy.Module):
        """DSPy module for generating video prompts from structured input.
        
        This wraps the scene→prompt conversion as a trainable DSPy module.
        The optimizer can tune the few-shot examples, instructions, and
        chain-of-thought reasoning automatically.
        """

        def __init__(self, use_cot: bool = True):
            super().__init__()
            if use_cot:
                self.generate = dspy.ChainOfThought(SceneToPrompt)
            else:
                self.generate = dspy.Predict(SceneToPrompt)

        def forward(self, scene_json: str, template_instructions: str) -> dspy.Prediction:
            return self.generate(
                scene_json=scene_json,
                template_instructions=template_instructions,
            )

    class CritiqueModule(dspy.Module):
        """Analyzes prompt→video quality relationships."""

        def __init__(self):
            super().__init__()
            self.critique = dspy.ChainOfThought(PromptCritique)

        def forward(
            self,
            video_prompt: str,
            scene_json: str,
            quality_scores: str,
            auto_fails: str,
        ) -> dspy.Prediction:
            return self.critique(
                video_prompt=video_prompt,
                scene_json=scene_json,
                quality_scores=quality_scores,
                auto_fails=auto_fails,
            )

    class TemplateImproverModule(dspy.Module):
        """Improves system template based on evidence from test runs."""

        def __init__(self):
            super().__init__()
            self.improve = dspy.ChainOfThought(TemplateImprover)

        def forward(
            self,
            current_template: str,
            success_patterns: str,
            failure_patterns: str,
            model_constraints: str,
        ) -> dspy.Prediction:
            return self.improve(
                current_template=current_template,
                success_patterns=success_patterns,
                failure_patterns=failure_patterns,
                model_constraints=model_constraints,
            )


# ---------------------------------------------------------------------------
# Scoring function for DSPy optimizer
# ---------------------------------------------------------------------------

def make_video_quality_metric(
    generator: VideoGenerator,
    scorer: VideoScorer,
    reward_calculator: Any,
    image_provider: Optional[Callable[[str], str]] = None,
) -> Callable:
    """Create a DSPy-compatible metric function.
    
    The metric generates a video from the prompt, scores it with QC,
    and returns the reward as the optimization signal.
    
    Args:
        generator: Video generation client (e.g., KieClient).
        scorer: Video quality scorer (e.g., GeminiVideoQC).
        reward_calculator: RewardCalculator instance.
        image_provider: Optional function that returns an image URL
            given scene_json. If None, generates one.
    
    Returns:
        A function compatible with DSPy's metric interface.
    """
    def metric(example: Any, prediction: Any, trace: Any = None) -> float:
        """Score a predicted video prompt by actually generating + evaluating."""
        prompt = prediction.video_prompt
        scene_json = example.scene_json
        
        try:
            # Get image URL
            if image_provider:
                image_url = image_provider(scene_json)
            else:
                # Parse scene to get image prompt
                scene = json.loads(scene_json)
                img_prompt = scene.get("scene_description", "hotel scene")
                img_result = generator.generate_image(img_prompt, aspect_ratio="9:16")
                if not img_result.success:
                    return 0.0
                image_url = img_result.result_urls[0]
            
            # Generate video
            vid_result = generator.generate_video(
                prompt=prompt,
                image_url=image_url,
                duration=8,
                resolution="1080p",
                aspect_ratio="9:16",
            )
            if not vid_result.success or not vid_result.result_urls:
                return 0.0
            
            # Score video
            qc_result = scorer.evaluate(vid_result.result_urls[0])
            reward = reward_calculator.calculate(qc_result)
            
            return reward.total_score / 100.0  # Normalize to [0, 1]
            
        except Exception as e:
            logger.error("Metric evaluation failed: %s", e)
            return 0.0
    
    return metric


# ---------------------------------------------------------------------------
# Seedance Model Constraints (shared knowledge base)
# ---------------------------------------------------------------------------

SEEDANCE_CONSTRAINTS = """
Known constraints of Seedance 1.5 Pro (from official docs + testing):

1. NEGATIVE PROMPTS DO NOT WORK — "no warping", "avoid morphing" etc. are ignored
2. Only accepts duration=8 (seconds)
3. Degree adverbs matter: "slowly", "gently", "gradually" strongly affect output
4. Complex scenes (chandeliers, multiple reflections, crowds) cause morphing
5. Best results with: single subject, clear camera move, warm/ambient lighting
6. Camera vocabulary that works: pushes, pulls back, circles around, pans, rises, drifts, tilts
7. Camera vocabulary to avoid: gimbal, steadicam, drone, crane (equipment names)
8. Safe motion elements: water ripples, fabric sway, steam, candle flame, leaf movement
9. Dangerous motion elements: chandeliers, complex glass, crowds, beach with people
10. Optimal prompt length: 25-45 words (too short = uncontrolled, too long = selective ignoring)
11. Structure should be: subject+details+motion+camera (NOT "single continuous shot" opener)
12. 1080p supported but cannot do 10-second videos at 1080p
"""


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class VideoPromptOptimizer:
    """DSPy-powered video prompt optimization.
    
    Orchestrates the full optimization loop:
    1. Generate prompts using the current module
    2. Generate + score videos  
    3. Collect critiques
    4. Use DSPy optimizer to improve the module
    5. Optionally improve the system template itself
    
    Designed to be pipeline-agnostic. Swap generator/scorer for different
    video models or QC methods.
    
    Args:
        generator: Video generation client.
        scorer: Video quality scorer.
        reward_calculator: Reward computation.
        lm_model: LiteLLM model name for DSPy (e.g., "gemini/gemini-2.5-flash").
        template_path: Path to the system prompt template file.
        api_key: API key for the LM (if not in env).
    """

    def __init__(
        self,
        generator: VideoGenerator,
        scorer: VideoScorer,
        reward_calculator: Any,
        *,
        lm_model: str = "gemini/gemini-2.5-flash",
        template_path: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if not HAS_DSPY:
            raise ImportError("DSPy is required. Install with: pip install dspy")
        
        self.generator = generator
        self.scorer = scorer
        self.reward_calc = reward_calculator
        self.template_path = template_path
        
        # Load template
        self.template = ""
        if template_path and os.path.exists(template_path):
            self.template = Path(template_path).read_text()
        
        # Configure DSPy LM
        api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            # Try credentials file
            cred_paths = [
                os.path.expanduser("~/.openclaw/workspace/.gemini_credentials.json"),
                os.path.expanduser("~/.openclaw/workspace/.credentials/gemini.json"),
            ]
            for p in cred_paths:
                if os.path.exists(p):
                    with open(p) as f:
                        api_key = json.load(f).get("api_key", "")
                    if api_key:
                        break
        
        self.lm = dspy.LM(lm_model, api_key=api_key)
        dspy.configure(lm=self.lm)
        
        # Initialize modules
        self.prompt_module = VideoPromptModule(use_cot=True)
        self.critique_module = CritiqueModule()
        self.template_improver = TemplateImproverModule()
        
        # Results tracking
        self.history: List[Dict[str, Any]] = []

    def generate_prompt(self, scene: SceneInput) -> str:
        """Generate a video prompt from structured scene input.
        
        Uses the current DSPy module (which may have been optimized).
        """
        result = self.prompt_module(
            scene_json=scene.to_json(),
            template_instructions=self.template or "Generate a 30-45 word cinematic video prompt.",
        )
        return result.video_prompt

    def critique_result(
        self,
        prompt: str,
        scene: SceneInput,
        qc_result: Dict[str, Any],
    ) -> str:
        """Analyze why a prompt produced its quality result."""
        scores = {
            "aesthetic": qc_result.get("aesthetic_score", 5),
            "motion": qc_result.get("motion_score", 5),
            "adherence": qc_result.get("prompt_adherence_score", 5),
            "scroll_stop": qc_result.get("scroll_stop_score", 5),
            "pass": qc_result.get("pass", True),
        }
        fails = qc_result.get("auto_fail_triggered", [])
        
        result = self.critique_module(
            video_prompt=prompt,
            scene_json=scene.to_json(),
            quality_scores=json.dumps(scores),
            auto_fails=json.dumps(fails),
        )
        return result.critique

    def improve_template(
        self,
        examples: Sequence[Dict[str, Any]],
    ) -> str:
        """Improve the system template based on scored examples.
        
        Args:
            examples: List of dicts with keys:
                - prompt: the video prompt used
                - scene: SceneInput or dict
                - qc_result: QC result dict
                - reward: float score
        
        Returns:
            Improved template string.
        """
        # Separate good and bad examples
        sorted_ex = sorted(examples, key=lambda x: x.get("reward", 0), reverse=True)
        top_k = sorted_ex[:5]
        bottom_k = sorted_ex[-3:] if len(sorted_ex) >= 3 else []
        
        success_patterns = "\n".join([
            f"Score {e['reward']:.0f}: {e['prompt']}" for e in top_k
        ])
        failure_patterns = "\n".join([
            f"Score {e['reward']:.0f}: {e['prompt']} | Fails: {e.get('qc_result', {}).get('auto_fail_triggered', [])}"
            for e in bottom_k
        ])
        
        result = self.template_improver(
            current_template=self.template,
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            model_constraints=SEEDANCE_CONSTRAINTS,
        )
        
        return result.improved_template

    def optimize(
        self,
        training_scenes: Sequence[SceneInput],
        *,
        num_trials: int = 20,
        optimizer_type: str = "mipro",
        image_urls: Optional[Dict[str, str]] = None,
    ) -> OptimizationResult:
        """Run DSPy optimization on the prompt module.
        
        This is the main optimization entry point. It:
        1. Creates training examples from scenes
        2. Defines the evaluation metric (generate → score → reward)
        3. Runs the DSPy optimizer (MIPROv2 or BootstrapFewShot)
        4. Returns the optimized module and results
        
        Args:
            training_scenes: Scene inputs to use for optimization.
            num_trials: Number of optimization trials.
            optimizer_type: "mipro" (Bayesian) or "bootstrap" (few-shot).
            image_urls: Pre-generated image URLs keyed by scene description.
                Saves API calls during optimization.
        
        Returns:
            OptimizationResult with improved template and scores.
        """
        # Build training set
        trainset = []
        for scene in training_scenes:
            example = dspy.Example(
                scene_json=scene.to_json(),
                template_instructions=self.template or "Generate a 30-45 word cinematic video prompt.",
            ).with_inputs("scene_json", "template_instructions")
            trainset.append(example)
        
        # Build metric
        def image_provider(scene_json: str) -> str:
            if image_urls:
                scene = json.loads(scene_json)
                desc = scene.get("scene_description", "")
                if desc in image_urls:
                    return image_urls[desc]
            return ""
        
        metric = make_video_quality_metric(
            generator=self.generator,
            scorer=self.scorer,
            reward_calculator=self.reward_calc,
            image_provider=image_provider if image_urls else None,
        )
        
        # Select optimizer
        if optimizer_type == "mipro":
            optimizer = dspy.MIPROv2(
                metric=metric,
                auto="medium",
                num_threads=1,  # Sequential — API rate limits
            )
        elif optimizer_type == "bootstrap":
            optimizer = dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=4,
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Run optimization
        logger.info(
            "Starting DSPy optimization: %d scenes, %d trials, %s optimizer",
            len(trainset), num_trials, optimizer_type,
        )
        
        optimized_module = optimizer.compile(
            self.prompt_module,
            trainset=trainset,
            num_trials=num_trials if optimizer_type == "mipro" else None,
        )
        
        # Update our module with the optimized version
        self.prompt_module = optimized_module
        
        # Collect results
        result = OptimizationResult(
            improved_template=self.template,  # Template itself unchanged; module is optimized
            num_trials=num_trials,
            metadata={
                "optimizer": optimizer_type,
                "num_scenes": len(training_scenes),
                "lm_model": str(self.lm),
            },
        )
        
        return result

    def run_evaluation_round(
        self,
        scenes: Sequence[SceneInput],
        *,
        image_urls: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a full evaluation round: generate prompts → videos → score.
        
        Returns list of result dicts for each scene.
        """
        results = []
        
        for scene in scenes:
            entry: Dict[str, Any] = {
                "scene": scene.scene_description,
                "prompt": "",
                "video_url": "",
                "qc_result": {},
                "reward": 0.0,
                "error": "",
            }
            
            try:
                # Generate prompt
                prompt = self.generate_prompt(scene)
                entry["prompt"] = prompt
                logger.info("Generated prompt: %s", prompt[:80])
                
                # Get image
                image_url = ""
                if image_urls and scene.scene_description in image_urls:
                    image_url = image_urls[scene.scene_description]
                else:
                    img_result = self.generator.generate_image(
                        scene.scene_description,
                        aspect_ratio="9:16",
                    )
                    if img_result.success and img_result.result_urls:
                        image_url = img_result.result_urls[0]
                
                if not image_url:
                    entry["error"] = "No image available"
                    results.append(entry)
                    continue
                
                # Generate video
                vid_result = self.generator.generate_video(
                    prompt=prompt,
                    image_url=image_url,
                    duration=8,
                    resolution="1080p",
                    aspect_ratio="9:16",
                )
                if not vid_result.success or not vid_result.result_urls:
                    entry["error"] = f"Video gen failed: {vid_result.state}"
                    results.append(entry)
                    continue
                entry["video_url"] = vid_result.result_urls[0]
                
                # Score
                qc_result = self.scorer.evaluate(entry["video_url"])
                entry["qc_result"] = qc_result
                
                reward = self.reward_calc.calculate(qc_result)
                entry["reward"] = reward.total_score
                
                logger.info(
                    "  %s: reward=%.1f, pass=%s",
                    scene.scene_description[:30],
                    reward.total_score,
                    qc_result.get("pass", "?"),
                )
                
            except Exception as e:
                entry["error"] = str(e)
                logger.error("Evaluation failed for %s: %s", scene.scene_description[:30], e)
            
            results.append(entry)
            self.history.append(entry)
        
        return results

    def save_state(self, path: str) -> None:
        """Save the optimized module state."""
        if hasattr(self.prompt_module, 'save'):
            self.prompt_module.save(path)
        # Also save history
        history_path = Path(path).parent / "optimization_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

    def load_state(self, path: str) -> None:
        """Load a previously optimized module state."""
        if hasattr(self.prompt_module, 'load'):
            self.prompt_module.load(path)
