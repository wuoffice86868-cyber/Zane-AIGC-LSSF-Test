"""Three-dimensional reward calculation aligned with Seedance RM architecture.

Seedance 1.0 tech report (arXiv:2506.09113) uses 3 specialized Reward Models:
  - Foundational RM: text-video alignment + structural stability (morphing, collapse)
  - Motion RM: movement quality, camera smoothness, temporal consistency, loops
  - Aesthetic RM: visual appeal, composition, lighting (keyframe-based, image-space)

This calculator maps QC results and quality scores to those 3 dimensions,
producing a weighted total reward in [0, 100].

Auto-fail categories are routed to the dimension they most affect:
  foundational: face_morphing, object_morphing, structural_collapse,
                reflection_error, text_distortion, hand_error
  motion:       action_loop, camera_jitter, temporal_flicker,
                unnatural_motion, freeze_frame
  aesthetic:    color_banding, overexposure, underexposure,
                low_resolution, compression_artifact
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .models import (
    DimensionScore,
    EvalSample,
    QCResult,
    RewardBreakdown,
    RewardDimension,
)


# ---------------------------------------------------------------------------
# Auto-fail → dimension routing
# ---------------------------------------------------------------------------

_FOUNDATIONAL_FAILS = frozenset({
    "face_morphing", "object_morphing", "structural_collapse",
    "reflection_error", "text_distortion", "hand_error",
    "background_warping", "penetration",
})

_MOTION_FAILS = frozenset({
    "action_loop", "camera_jitter", "temporal_flicker",
    "unnatural_motion", "freeze_frame", "speed_inconsistency",
})

_AESTHETIC_FAILS = frozenset({
    "color_banding", "overexposure", "underexposure",
    "low_resolution", "compression_artifact", "noise",
})


def _route_fail(fail_name: str) -> RewardDimension:
    """Route an auto-fail rule to its most relevant dimension."""
    name = fail_name.lower().strip()
    if name in _FOUNDATIONAL_FAILS:
        return RewardDimension.FOUNDATIONAL
    if name in _MOTION_FAILS:
        return RewardDimension.MOTION
    if name in _AESTHETIC_FAILS:
        return RewardDimension.AESTHETIC
    # Unknown → foundational (structural issues are most common)
    return RewardDimension.FOUNDATIONAL


# ---------------------------------------------------------------------------
# Default weights
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    RewardDimension.FOUNDATIONAL: 0.45,
    RewardDimension.MOTION: 0.30,
    RewardDimension.AESTHETIC: 0.25,
}

# Legacy 4-dim weights (backward compat for old callers)
LEGACY_WEIGHTS: Dict[str, float] = {
    "qc": 0.4,
    "aesthetic": 0.2,
    "clip": 0.2,
    "motion": 0.2,
}


class RewardCalculator:
    """Calculate reward scores using 3-dimension Seedance-aligned model.

    Each dimension produces a score in [0, 1]:
      - foundational: structural integrity + prompt adherence
      - motion: temporal quality + camera smoothness
      - aesthetic: visual appeal + composition

    Total reward = sum(dim_score * dim_weight) * 100, clamped to [0, 100].

    The calculator auto-extracts Gemini QC scores when available:
      aesthetic_score → aesthetic dimension
      motion_score → motion dimension
      prompt_adherence_score → foundational dimension
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            for k, v in weights.items():
                # Accept both RewardDimension enum and string keys
                key = k if isinstance(k, RewardDimension) else k
                self.weights[key] = v

        # Also keep legacy weights for backward compat
        self._legacy_weights = dict(LEGACY_WEIGHTS)

    # ------------------------------------------------------------------
    # Dimension scoring
    # ------------------------------------------------------------------

    def _score_foundational(
        self,
        qc: QCResult,
        *,
        adherence: Optional[float] = None,
    ) -> DimensionScore:
        """Score foundational dimension: structural stability + alignment.

        Components:
          - structural (0-1): penalized by foundational auto-fails
          - adherence (0-1): prompt-video alignment (from Gemini or default)
          - pass_bonus: +0.1 if QC overall pass
        """
        # Route fails to this dimension
        dim_fails = [f for f in qc.auto_fail_triggered if _route_fail(f) == RewardDimension.FOUNDATIONAL]
        dim_minors = [m for m in qc.minor_issues if _route_fail(m) == RewardDimension.FOUNDATIONAL]

        # Structural sub-score
        structural = 1.0
        structural -= len(dim_fails) * 0.25
        structural -= len(dim_minors) * 0.08
        structural = max(0.0, structural)

        # Adherence sub-score
        adh = adherence if adherence is not None else (0.7 if qc.qc_pass else 0.4)

        # Pass bonus
        pass_bonus = 0.1 if qc.qc_pass else 0.0

        # Combine: structural 50%, adherence 40%, pass bonus 10%
        raw = structural * 0.5 + adh * 0.4 + pass_bonus
        score = float(np.clip(raw, 0.0, 1.0))

        return DimensionScore(
            dimension=RewardDimension.FOUNDATIONAL,
            score=score,
            weight=self.weights.get(RewardDimension.FOUNDATIONAL, 0.45),
            weighted_score=score * self.weights.get(RewardDimension.FOUNDATIONAL, 0.45),
            components={
                "structural": round(structural, 4),
                "adherence": round(adh, 4),
                "pass_bonus": round(pass_bonus, 4),
                "auto_fail_count": len(dim_fails),
                "minor_issue_count": len(dim_minors),
            },
            issues=dim_fails + dim_minors,
        )

    def _score_motion(
        self,
        qc: QCResult,
        *,
        motion_score: Optional[float] = None,
    ) -> DimensionScore:
        """Score motion dimension: temporal quality + camera smoothness.

        Components:
          - temporal (0-1): penalized by motion auto-fails
          - smoothness (0-1): from Gemini motion_score or default
          - confidence_factor: higher QC confidence → slight boost
        """
        dim_fails = [f for f in qc.auto_fail_triggered if _route_fail(f) == RewardDimension.MOTION]
        dim_minors = [m for m in qc.minor_issues if _route_fail(m) == RewardDimension.MOTION]

        # Temporal sub-score
        temporal = 1.0
        temporal -= len(dim_fails) * 0.30  # Motion fails are severe
        temporal -= len(dim_minors) * 0.10
        temporal = max(0.0, temporal)

        # Smoothness from external scorer
        smooth = motion_score if motion_score is not None else (0.7 if qc.qc_pass else 0.35)

        # Confidence factor
        conf_factor = min(qc.confidence * 0.1, 0.1)

        # Combine: temporal 55%, smoothness 40%, confidence 5%
        raw = temporal * 0.55 + smooth * 0.40 + conf_factor * 0.5
        score = float(np.clip(raw, 0.0, 1.0))

        return DimensionScore(
            dimension=RewardDimension.MOTION,
            score=score,
            weight=self.weights.get(RewardDimension.MOTION, 0.30),
            weighted_score=score * self.weights.get(RewardDimension.MOTION, 0.30),
            components={
                "temporal": round(temporal, 4),
                "smoothness": round(smooth, 4),
                "confidence_factor": round(conf_factor, 4),
                "auto_fail_count": len(dim_fails),
                "minor_issue_count": len(dim_minors),
            },
            issues=dim_fails + dim_minors,
        )

    def _score_aesthetic(
        self,
        qc: QCResult,
        *,
        aesthetic: Optional[float] = None,
    ) -> DimensionScore:
        """Score aesthetic dimension: visual appeal + composition.

        Components:
          - visual (0-1): penalized by aesthetic auto-fails
          - appeal (0-1): from Gemini aesthetic_score or default
        """
        dim_fails = [f for f in qc.auto_fail_triggered if _route_fail(f) == RewardDimension.AESTHETIC]
        dim_minors = [m for m in qc.minor_issues if _route_fail(m) == RewardDimension.AESTHETIC]

        # Visual sub-score
        visual = 1.0
        visual -= len(dim_fails) * 0.20
        visual -= len(dim_minors) * 0.06
        visual = max(0.0, visual)

        # Appeal from external scorer
        appeal = aesthetic if aesthetic is not None else (0.75 if qc.qc_pass else 0.45)

        # Combine: visual 40%, appeal 60%
        raw = visual * 0.4 + appeal * 0.6
        score = float(np.clip(raw, 0.0, 1.0))

        return DimensionScore(
            dimension=RewardDimension.AESTHETIC,
            score=score,
            weight=self.weights.get(RewardDimension.AESTHETIC, 0.25),
            weighted_score=score * self.weights.get(RewardDimension.AESTHETIC, 0.25),
            components={
                "visual": round(visual, 4),
                "appeal": round(appeal, 4),
                "auto_fail_count": len(dim_fails),
                "minor_issue_count": len(dim_minors),
            },
            issues=dim_fails + dim_minors,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def calculate(
        self,
        qc_result: dict | QCResult,
        *,
        aesthetic: Optional[float] = None,
        clip_score: Optional[float] = None,
        motion_score: Optional[float] = None,
    ) -> RewardBreakdown:
        """Calculate 3-dimensional reward for a single sample.

        Automatically extracts Gemini QC extra scores (aesthetic_score,
        motion_score, prompt_adherence_score) from the dict if present
        and no explicit values are provided.

        For backward compat, ``clip_score`` is folded into the foundational
        dimension as ``adherence``.

        Args:
            qc_result: QC result dict or ``QCResult`` model.
            aesthetic: Optional aesthetic quality score [0, 1].
            clip_score: Optional CLIP-score / prompt adherence [0, 1].
                Mapped to foundational dimension's adherence component.
            motion_score: Optional motion smoothness score [0, 1].

        Returns:
            ``RewardBreakdown`` with ``total_score`` in [0, 100] and
            per-dimension breakdown in ``dimensions``.
        """
        raw_dict = qc_result if isinstance(qc_result, dict) else {}

        if isinstance(qc_result, dict):
            qc = QCResult(**{
                k: v for k, v in qc_result.items()
                if k in QCResult.model_fields or k == "pass"
            })
        else:
            qc = qc_result

        # Auto-extract Gemini extra scores
        if aesthetic is None and "aesthetic_score" in raw_dict:
            aesthetic = float(raw_dict["aesthetic_score"]) / 10.0
        if motion_score is None and "motion_score" in raw_dict:
            motion_score = float(raw_dict["motion_score"]) / 10.0
        if clip_score is None and "prompt_adherence_score" in raw_dict:
            clip_score = float(raw_dict["prompt_adherence_score"]) / 10.0

        # Map clip_score → adherence for foundational dimension
        adherence = clip_score

        # Score each dimension
        found = self._score_foundational(qc, adherence=adherence)
        motion = self._score_motion(qc, motion_score=motion_score)
        aesth = self._score_aesthetic(qc, aesthetic=aesthetic)

        # Weighted total
        total = (found.weighted_score + motion.weighted_score + aesth.weighted_score) * 100.0
        total = float(np.clip(total, 0.0, 100.0))

        return RewardBreakdown(
            total_score=total,
            breakdown={
                "foundational": found.score,
                "motion": motion.score,
                "aesthetic": aesth.score,
                "weights": {
                    RewardDimension.FOUNDATIONAL: found.weight,
                    RewardDimension.MOTION: motion.weight,
                    RewardDimension.AESTHETIC: aesth.weight,
                },
            },
            dimensions={
                RewardDimension.FOUNDATIONAL: found,
                RewardDimension.MOTION: motion,
                RewardDimension.AESTHETIC: aesth,
            },
        )

    def batch_calculate(
        self,
        samples: Sequence[EvalSample],
    ) -> List[RewardBreakdown]:
        """Calculate rewards for a list of ``EvalSample`` objects."""
        return [self.calculate(s.qc_result) for s in samples]

    # ------------------------------------------------------------------
    # Weight fitting
    # ------------------------------------------------------------------

    def fit_weights(
        self,
        samples: Sequence[EvalSample],
        *,
        extra_scores: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """Optimise 3-dim weights to maximise Pearson-r with human_score.

        Uses grid search over weight combinations (step 0.05).
        Only samples with a ``human_label`` are used.

        Args:
            samples: Evaluation samples (must have ``human_label``).
            extra_scores: Per-sample dicts with optional keys
                ``aesthetic``, ``motion``, ``adherence``.

        Returns:
            Best weight dict (also stored in ``self.weights``).
        """
        labeled = [
            (i, s)
            for i, s in enumerate(samples)
            if s.human_label is not None
        ]
        if len(labeled) < 3:
            return dict(self.weights)

        human_scores = np.array([
            s.human_label.human_score for _, s in labeled  # type: ignore[union-attr]
        ])

        if extra_scores is None:
            return dict(self.weights)

        # Build per-sample component vectors for 3 dimensions
        found_scores = np.array([
            self._score_foundational(
                s.qc_result,
                adherence=extra_scores[i].get("adherence"),
            ).score
            for i, (_, s) in enumerate(labeled)
        ])
        motion_scores = np.array([
            self._score_motion(
                s.qc_result,
                motion_score=extra_scores[i].get("motion"),
            ).score
            for i, (_, s) in enumerate(labeled)
        ])
        aesth_scores = np.array([
            self._score_aesthetic(
                s.qc_result,
                aesthetic=extra_scores[i].get("aesthetic"),
            ).score
            for i, (_, s) in enumerate(labeled)
        ])

        best_corr = -2.0
        best_w: Dict[str, float] = dict(self.weights)
        step = 0.05

        for wf in np.arange(0.1, 0.8 + step, step):
            for wm in np.arange(0.05, 1.0 - wf + step, step):
                wa = 1.0 - wf - wm
                if wa < -0.01:
                    continue
                wa = max(wa, 0.0)
                combined = (
                    found_scores * wf
                    + motion_scores * wm
                    + aesth_scores * wa
                ) * 100.0
                if np.std(combined) < 1e-9:
                    continue
                corr = float(np.corrcoef(combined, human_scores)[0, 1])
                if corr > best_corr:
                    best_corr = corr
                    best_w = {
                        RewardDimension.FOUNDATIONAL: round(float(wf), 2),
                        RewardDimension.MOTION: round(float(wm), 2),
                        RewardDimension.AESTHETIC: round(float(wa), 2),
                    }

        self.weights = best_w
        return best_w
