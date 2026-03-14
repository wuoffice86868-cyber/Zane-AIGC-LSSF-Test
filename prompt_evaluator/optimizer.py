"""OPRO-style prompt optimizer for video generation system prompts.

Implements the Optimization by PROmpting (OPRO) methodology from
Yang et al. (2023) — "Large Language Models as Optimizers" (ICLR 2024).

Core idea: use an LLM to generate improved prompt variants by feeding it
a *meta-prompt* containing (prompt, score) history sorted ascending.
The LLM identifies patterns in high-scoring prompts and proposes
improvements.

Adaptations for video generation:
- Domain-specific scoring dimensions (QC pass, aesthetic, motion)
- Seedance camera vocabulary constraints
- Scene-type aware optimization (pool vs room vs exterior)
- Prompt length constraints (30-45 words for cinematography)
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from .models import EvalSample, RewardBreakdown, SceneType
from .prompt_analyzer import PromptAnalyzer
from .reward_calculator import RewardCalculator


# ---------------------------------------------------------------------------
# LLM client protocol
# ---------------------------------------------------------------------------

class LLMClient(Protocol):
    """Minimal interface for an LLM text-generation client.

    Any object with a ``generate(prompt: str) -> str`` method satisfies
    this protocol.  Additional kwargs (temperature, max_tokens) may be
    passed through if the implementation supports them.
    """

    def generate(self, prompt: str, **kwargs: Any) -> str: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Meta-prompt templates
# ---------------------------------------------------------------------------

_META_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are a Prompt Optimizer for an AI video generation pipeline.

    Your task: improve the System Prompt used to generate cinematography
    instructions for hotel/travel video clips.  The generated prompts are
    fed to Seedance 1.0 (a text-to-video model) and then quality-checked.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CURRENT SYSTEM PROMPT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {current_prompt}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    HISTORICAL PROMPT VARIANTS AND SCORES
    (sorted ascending — last entries are highest-scoring)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {history_block}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    SCORING DIMENSIONS (total 0-100)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - QC pass rate (40%): no artifacts, morphing, loops, warping
    - Aesthetic quality (20%): visual appeal, composition, lighting
    - Prompt-video consistency (20%): video matches the prompt
    - Motion quality (20%): smooth, natural camera/object motion

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    DOMAIN CONSTRAINTS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    - Use exact Seedance camera verbs: pushes, pulls, circles around,
      moves left/right, pans left/right, rises, tilts up
    - Generated prompts should be 30-45 words
    - Always start with "Single continuous shot"
    - For outdoor scenes with sky: add "Sky and clouds remain completely still"
    - End with a stability anchor ("[element] stays fixed/still/anchored")
    - Speed modifiers: slowly, gradually, gently
    - No equipment names (gimbal, steadicam, drone)
    - No negative phrases ("no warping", "avoid morphing")
    {scene_context}
    {feature_insights}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TASK
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Analyse what high-scoring prompts have in common.
    2. Analyse what low-scoring prompts do wrong.
    3. Generate an IMPROVED version of the system prompt that should
       produce higher-scoring cinematography instructions.

    Return ONLY the improved system prompt. No commentary, no markdown
    fences, no explanation.
""")

_HISTORY_ENTRY = "Score {score:.1f} | {prompt}"

_SCENE_CONTEXT_BLOCK = textwrap.dedent("""\

    SCENE-TYPE PERFORMANCE
    {scene_stats}
""")

_FEATURE_INSIGHT_BLOCK = textwrap.dedent("""\

    FEATURE INSIGHTS FROM DATA
    {insights}
""")


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class PromptOptimizer:
    """OPRO-style prompt optimizer for video generation system prompts.

    The optimizer can operate in two modes:

    1. **With ``llm_client``**: call ``suggest_improvement()`` to get an
       optimized prompt directly.
    2. **Without ``llm_client``**: call ``build_meta_prompt()`` to get the
       meta-prompt string, then send it to your own LLM.

    The meta-prompt follows the OPRO structure:
        meta-instructions → solution-score pairs (ascending) →
        domain constraints → optimization task + output format

    References:
        Yang et al. "Large Language Models as Optimizers" (ICLR 2024)
        https://arxiv.org/abs/2309.03409
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        prompt_analyzer: Optional[PromptAnalyzer] = None,
    ) -> None:
        """
        Args:
            llm_client: Optional LLM client with a ``generate(prompt)``
                method.  If *None*, use ``build_meta_prompt`` manually.
            reward_calculator: Calculator for scoring samples.
                A default instance is created when *None*.
            prompt_analyzer: Analyzer for feature extraction.
                A default instance is created when *None*.
        """
        self.llm = llm_client
        self.reward = reward_calculator or RewardCalculator()
        self.analyzer = prompt_analyzer or PromptAnalyzer()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_meta_prompt(
        self,
        current_prompt: str,
        history: Sequence[EvalSample],
        *,
        top_k: int = 5,
        bottom_k: int = 3,
        include_feature_insights: bool = True,
        include_scene_breakdown: bool = True,
    ) -> str:
        """Build the OPRO meta-prompt from current prompt + scored history.

        The history is sorted by reward score ascending so the LLM sees
        worst → best, following the OPRO paper's finding that ascending
        order yields better optimization trajectories.

        Args:
            current_prompt: Current system prompt to optimize.
            history: Historical samples with QC results (and optionally
                human labels).
            top_k: Number of highest-scoring samples to include.
            bottom_k: Number of lowest-scoring samples to include.
            include_feature_insights: Inject prompt-feature correlation
                insights into the meta-prompt.
            include_scene_breakdown: Inject per-scene-type statistics.

        Returns:
            Complete meta-prompt string ready to send to an LLM.
        """
        # Score all history samples
        scored = self._score_samples(history)

        # Select top-k and bottom-k
        selected = self._select_examples(scored, top_k=top_k, bottom_k=bottom_k)

        # Format history block (ascending order per OPRO)
        history_block = self._format_history(selected)

        # Optional: scene context
        scene_context = ""
        if include_scene_breakdown and len(scored) >= 5:
            scene_context = self._scene_context_block(scored)

        # Optional: feature insights
        feature_insights = ""
        if include_feature_insights and len(history) >= 5:
            feature_insights = self._feature_insight_block(history)

        return _META_PROMPT_TEMPLATE.format(
            current_prompt=current_prompt.strip(),
            history_block=history_block,
            scene_context=scene_context,
            feature_insights=feature_insights,
        )

    def suggest_improvement(
        self,
        current_prompt: str,
        history: Sequence[EvalSample],
        *,
        top_k: int = 5,
        bottom_k: int = 3,
        temperature: float = 1.0,
        **llm_kwargs: Any,
    ) -> str:
        """Generate an improved prompt using the attached LLM client.

        Builds the meta-prompt internally and calls
        ``llm_client.generate()``.

        Args:
            current_prompt: Current system prompt.
            history: Historical samples.
            top_k: Top-k high-scoring samples.
            bottom_k: Bottom-k low-scoring samples.
            temperature: Sampling temperature for the optimizer LLM.
                OPRO paper recommends 1.0 for diversity.
            **llm_kwargs: Extra kwargs forwarded to ``llm_client.generate``.

        Returns:
            Optimized system prompt string.

        Raises:
            RuntimeError: If no ``llm_client`` was provided.
        """
        if self.llm is None:
            raise RuntimeError(
                "No llm_client provided. Use build_meta_prompt() to get "
                "the meta-prompt and call your LLM externally."
            )

        meta = self.build_meta_prompt(
            current_prompt,
            history,
            top_k=top_k,
            bottom_k=bottom_k,
        )
        return self.llm.generate(meta, temperature=temperature, **llm_kwargs)

    # ------------------------------------------------------------------
    # Multi-step optimization loop
    # ------------------------------------------------------------------

    def optimize_loop(
        self,
        current_prompt: str,
        history: Sequence[EvalSample],
        *,
        steps: int = 3,
        candidates_per_step: int = 4,
        top_k: int = 5,
        bottom_k: int = 3,
        temperature: float = 1.0,
    ) -> List[str]:
        """Run multiple OPRO optimization steps.

        At each step, generate ``candidates_per_step`` prompt variants.
        The variants are returned as a flat list (newest last).

        Note: Without an evaluation function for the generated prompts,
        this method collects candidates but cannot score them.  In a full
        pipeline integration you would score each candidate by running it
        through the video generation + QC loop, then feed the scored
        results back into the next step.

        Args:
            current_prompt: Starting system prompt.
            history: Initial scored history.
            steps: Number of optimization iterations.
            candidates_per_step: How many prompt variants to generate per
                step.
            top_k: Top-k for meta-prompt.
            bottom_k: Bottom-k for meta-prompt.
            temperature: LLM temperature.

        Returns:
            List of all generated prompt variants.

        Raises:
            RuntimeError: If no ``llm_client`` was provided.
        """
        if self.llm is None:
            raise RuntimeError(
                "optimize_loop requires an llm_client. Use "
                "build_meta_prompt() for manual optimization."
            )

        all_candidates: List[str] = []

        for _ in range(steps):
            meta = self.build_meta_prompt(
                current_prompt, history, top_k=top_k, bottom_k=bottom_k
            )
            for _ in range(candidates_per_step):
                candidate = self.llm.generate(meta, temperature=temperature)
                all_candidates.append(candidate.strip())

        return all_candidates

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _score_samples(
        self, samples: Sequence[EvalSample]
    ) -> List[Tuple[EvalSample, float]]:
        """Score samples and return (sample, score) tuples."""
        result: List[Tuple[EvalSample, float]] = []
        for s in samples:
            # Prefer human score if available
            if s.human_label is not None:
                score = float(s.human_label.human_score) * 10  # scale 1-10 → 10-100
            else:
                bd = self.reward.calculate(s.qc_result)
                score = bd.total_score
            result.append((s, score))
        return result

    def _select_examples(
        self,
        scored: List[Tuple[EvalSample, float]],
        *,
        top_k: int,
        bottom_k: int,
    ) -> List[Tuple[EvalSample, float]]:
        """Select top-k + bottom-k samples, deduplicated, sorted ascending."""
        if not scored:
            return []

        by_score = sorted(scored, key=lambda t: t[1])
        bottom = by_score[:bottom_k]
        top = by_score[-top_k:]

        # Deduplicate by sample_id, keep order
        seen: set[str] = set()
        selected: List[Tuple[EvalSample, float]] = []
        for item in bottom + top:
            sid = item[0].sample_id
            if sid not in seen:
                seen.add(sid)
                selected.append(item)

        # Re-sort ascending (worst → best) per OPRO convention
        selected.sort(key=lambda t: t[1])
        return selected

    @staticmethod
    def _format_history(
        selected: List[Tuple[EvalSample, float]],
    ) -> str:
        if not selected:
            return "(no history available yet)"

        lines: List[str] = []
        for sample, score in selected:
            prompt_text = sample.prompt.generated_prompt or "(empty)"
            # Truncate very long prompts
            if len(prompt_text) > 200:
                prompt_text = prompt_text[:197] + "..."
            lines.append(_HISTORY_ENTRY.format(score=score, prompt=prompt_text))
        return "\n".join(lines)

    def _scene_context_block(
        self, scored: List[Tuple[EvalSample, float]]
    ) -> str:
        """Build per-scene-type performance stats."""
        from collections import defaultdict

        buckets: Dict[str, List[float]] = defaultdict(list)
        for s, score in scored:
            scene = s.input.scene_type or "unknown"
            buckets[scene].append(score)

        if not buckets:
            return ""

        lines: List[str] = []
        for scene in sorted(buckets):
            scores = buckets[scene]
            avg = sum(scores) / len(scores)
            lines.append(f"  {scene}: avg_score={avg:.1f}, n={len(scores)}")

        return _SCENE_CONTEXT_BLOCK.format(scene_stats="\n".join(lines))

    def _feature_insight_block(
        self, history: Sequence[EvalSample]
    ) -> str:
        """Inject prompt-feature correlation insights."""
        corr = self.analyzer.analyze_correlation(history)
        if not corr.feature_importance:
            return ""

        lines: List[str] = []
        for feat, r in list(corr.feature_importance.items())[:6]:
            direction = "positive" if r > 0 else "negative"
            lines.append(f"  {feat}: r={r:+.3f} ({direction} correlation with score)")

        if corr.high_reward_patterns:
            lines.append("  High-score patterns: " + "; ".join(corr.high_reward_patterns[:3]))
        if corr.low_reward_patterns:
            lines.append("  Low-score patterns: " + "; ".join(corr.low_reward_patterns[:3]))

        return _FEATURE_INSIGHT_BLOCK.format(insights="\n".join(lines))
