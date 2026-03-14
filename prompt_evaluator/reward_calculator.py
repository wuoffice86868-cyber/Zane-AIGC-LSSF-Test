"""Multi-dimensional reward calculation for prompt evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .models import EvalSample, QCResult, RewardBreakdown


# Default weight presets
DEFAULT_WEIGHTS: Dict[str, float] = {
    "qc": 0.4,
    "aesthetic": 0.2,
    "clip": 0.2,
    "motion": 0.2,
}


class RewardCalculator:
    """Calculate reward scores from QC results and optional quality metrics.

    The reward formula has two modes:

    **QC-only mode** (default when no aesthetic/clip/motion scores):
        ``REWARD = clamp(PASS_SCORE + ISSUE_PENALTY + CONFIDENCE_BOOST, 0, 100)``

    **Multi-dimensional mode** (when aesthetic/clip/motion provided):
        ``REWARD = (qc_norm * w_qc + aesthetic * w_a + clip * w_c + motion * w_m) * 100``
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        """
        Args:
            weights: Dimension weights (keys: qc, aesthetic, clip, motion).
                     Missing keys fall back to ``DEFAULT_WEIGHTS``.
        """
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    # ------------------------------------------------------------------
    # QC-only reward
    # ------------------------------------------------------------------

    @staticmethod
    def _pass_score(qc: QCResult) -> float:
        if qc.qc_pass:
            return 100.0
        total_issues = len(qc.auto_fail_triggered) + len(qc.minor_issues)
        return max(0.0, 100.0 - total_issues * 15.0)

    @staticmethod
    def _issue_penalty(qc: QCResult) -> float:
        auto_penalty = len(qc.auto_fail_triggered) * 8.0
        minor_penalty = len(qc.minor_issues) * 3.0
        return max(-50.0, -(auto_penalty + minor_penalty))

    @staticmethod
    def _confidence_boost(qc: QCResult) -> float:
        c = qc.confidence
        if c >= 0.95:
            return 10.0
        if c >= 0.80:
            return 5.0
        if c >= 0.60:
            return 2.0
        return 0.0

    def _qc_reward(self, qc: QCResult) -> RewardBreakdown:
        ps = self._pass_score(qc)
        ip = self._issue_penalty(qc)
        cb = self._confidence_boost(qc)
        total = float(np.clip(ps + ip + cb, 0.0, 100.0))
        return RewardBreakdown(
            total_score=total,
            breakdown={
                "pass_score": ps,
                "issue_penalty": ip,
                "confidence_boost": cb,
            },
        )

    # ------------------------------------------------------------------
    # Multi-dimensional reward
    # ------------------------------------------------------------------

    def _multi_reward(
        self,
        qc: QCResult,
        aesthetic: float,
        clip_score: float,
        motion_score: float,
    ) -> RewardBreakdown:
        qc_bd = self._qc_reward(qc)
        qc_norm = qc_bd.total_score / 100.0

        w = self.weights
        total = (
            qc_norm * w["qc"]
            + aesthetic * w["aesthetic"]
            + clip_score * w["clip"]
            + motion_score * w["motion"]
        ) * 100.0
        total = float(np.clip(total, 0.0, 100.0))

        return RewardBreakdown(
            total_score=total,
            breakdown={
                "qc_score": qc_bd.total_score,
                "aesthetic_score": aesthetic,
                "clip_score": clip_score,
                "motion_score": motion_score,
                "weights": dict(w),
            },
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
        """Calculate reward for a single sample.

        Automatically extracts Gemini QC extra scores (aesthetic_score,
        motion_score, prompt_adherence_score) from the dict if present
        and no explicit values are provided.

        Args:
            qc_result: QC result dict or ``QCResult`` model.
            aesthetic: Optional aesthetic quality score [0, 1].
            clip_score: Optional CLIP-score (prompt–video consistency) [0, 1].
            motion_score: Optional motion smoothness score [0, 1].

        Returns:
            ``RewardBreakdown`` with ``total_score`` in [0, 100].
        """
        raw_dict = qc_result if isinstance(qc_result, dict) else {}

        if isinstance(qc_result, dict):
            qc = QCResult(**{k: v for k, v in qc_result.items() if k in QCResult.model_fields or k == "pass"})
        else:
            qc = qc_result

        # Auto-extract Gemini extra scores if not explicitly provided
        if aesthetic is None and "aesthetic_score" in raw_dict:
            aesthetic = float(raw_dict["aesthetic_score"]) / 10.0
        if motion_score is None and "motion_score" in raw_dict:
            motion_score = float(raw_dict["motion_score"]) / 10.0
        if clip_score is None and "prompt_adherence_score" in raw_dict:
            clip_score = float(raw_dict["prompt_adherence_score"]) / 10.0

        has_extras = all(
            v is not None for v in (aesthetic, clip_score, motion_score)
        )
        if has_extras:
            return self._multi_reward(qc, aesthetic, clip_score, motion_score)  # type: ignore[arg-type]
        return self._qc_reward(qc)

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
        """Optimise weights to maximise Pearson correlation with ``human_score``.

        Uses a simple grid search over weight combinations (step 0.05) for the
        four dimensions.  Only samples with a ``human_label`` are used.

        Args:
            samples: Evaluation samples (must have ``human_label``).
            extra_scores: Per-sample dicts with optional keys
                ``aesthetic``, ``clip``, ``motion``.  If *None*, QC-only
                weights are returned unchanged.

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

        human_scores = np.array([s.human_label.human_score for _, s in labeled])  # type: ignore[union-attr]

        # If no extra scores, just return current weights (QC-only mode)
        if extra_scores is None:
            return dict(self.weights)

        # Build per-sample component vectors
        qc_scores = np.array([
            self._qc_reward(s.qc_result).total_score / 100.0
            for _, s in labeled
        ])
        aesthetics = np.array([
            extra_scores[i].get("aesthetic", 0.0) for i, _ in labeled
        ])
        clips = np.array([
            extra_scores[i].get("clip", 0.0) for i, _ in labeled
        ])
        motions = np.array([
            extra_scores[i].get("motion", 0.0) for i, _ in labeled
        ])

        best_corr = -2.0
        best_w: Dict[str, float] = dict(self.weights)
        step = 0.05

        for wq in np.arange(0.1, 0.8 + step, step):
            for wa in np.arange(0.0, 1.0 - wq + step, step):
                for wc in np.arange(0.0, 1.0 - wq - wa + step, step):
                    wm = 1.0 - wq - wa - wc
                    if wm < -0.01:
                        continue
                    wm = max(wm, 0.0)
                    combined = (
                        qc_scores * wq
                        + aesthetics * wa
                        + clips * wc
                        + motions * wm
                    ) * 100.0
                    if np.std(combined) < 1e-9:
                        continue
                    corr = float(np.corrcoef(combined, human_scores)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
                        best_w = {
                            "qc": round(float(wq), 2),
                            "aesthetic": round(float(wa), 2),
                            "clip": round(float(wc), 2),
                            "motion": round(float(wm), 2),
                        }

        self.weights = best_w
        return best_w
