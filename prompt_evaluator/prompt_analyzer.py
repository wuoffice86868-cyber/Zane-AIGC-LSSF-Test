"""Prompt feature extraction and correlation analysis."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .models import (
    CorrelationResult,
    EvalSample,
    PromptFeatures,
    RewardBreakdown,
)
from .reward_calculator import RewardCalculator


# Seedance camera verbs (canonical list)
_CAMERA_MOVES: List[Tuple[str, str]] = [
    ("circle around", "circle around"),
    ("circles around", "circle around"),
    ("move left", "move left"),
    ("moves left", "move left"),
    ("move right", "move right"),
    ("moves right", "move right"),
    ("pan left", "pan left"),
    ("pans left", "pan left"),
    ("pan right", "pan right"),
    ("pans right", "pan right"),
    ("tilt up", "tilt up"),
    ("tilts up", "tilt up"),
    ("push", "push"),
    ("pushes", "push"),
    ("pull", "pull"),
    ("pulls", "pull"),
    ("rise", "rise"),
    ("rises", "rise"),
]

_SPEED_MODIFIERS = ["slowly", "gradually", "gently", "very slowly"]

_SHOT_SIZES = ["close-up", "medium shot", "wide shot"]

_MOTION_PATTERNS = [
    re.compile(r"water ripples?\s+\w+", re.I),
    re.compile(r"palm fronds?\s+move\s+\w+", re.I),
    re.compile(r"curtains?\s+sway\s+\w+", re.I),
    re.compile(r"leaves?\s+rustle", re.I),
    re.compile(r"flag\s+flutter", re.I),
    re.compile(r"candle\s+flicker", re.I),
]

_STABLE_PATTERN = re.compile(
    r"(\w[\w\s]{0,25}?)\s+(?:stays?|remains?|holds?)\s+"
    r"(?:perfectly\s+)?(?:still|fixed|stable|anchored|steady)",
    re.I,
)


class PromptAnalyzer:
    """Extract features from generated prompts and analyse their correlation
    with reward scores."""

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, prompt: str) -> PromptFeatures:
        """Extract a feature vector from a single prompt string.

        Args:
            prompt: The generated cinematography prompt text.

        Returns:
            ``PromptFeatures`` with all fields populated.
        """
        lower = prompt.lower()
        words = prompt.split()

        # Camera move detection (longest match first — list is ordered)
        cam_type = ""
        has_cam = False
        for phrase, canonical in _CAMERA_MOVES:
            if phrase in lower:
                cam_type = canonical
                has_cam = True
                break

        # Speed modifier
        speed = ""
        has_speed = False
        for mod in _SPEED_MODIFIERS:
            if mod in lower:
                speed = mod
                has_speed = True
                break

        # Shot size
        shot = ""
        for s in _SHOT_SIZES:
            if s in lower:
                shot = s
                break

        # Motion elements
        motions: List[str] = []
        for pat in _MOTION_PATTERNS:
            m = pat.search(prompt)
            if m:
                motions.append(m.group(0).strip())

        # Stable element
        stable = ""
        has_stable = False
        sm = _STABLE_PATTERN.search(prompt)
        if sm:
            has_stable = True
            stable = sm.group(1).strip()

        # Sky freeze
        has_sky = ("sky" in lower) and any(
            w in lower for w in ("still", "frozen", "remain", "unchanged")
        )

        return PromptFeatures(
            word_count=len(words),
            has_camera_move=has_cam,
            camera_move_type=cam_type,
            has_single_continuous_shot="single continuous shot" in lower,
            has_sky_freeze=has_sky,
            has_stable_element=has_stable,
            stable_element=stable,
            motion_elements=motions,
            has_speed_modifier=has_speed,
            speed_modifier=speed,
            shot_size=shot,
            has_human_instruction=(
                "human subject" in lower or "person remains" in lower
            ),
            sentence_count=max(1, prompt.count(".") + prompt.count("!") + prompt.count("?")),
        )

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def batch_extract(
        self, prompts: Sequence[str]
    ) -> List[PromptFeatures]:
        """Extract features for a list of prompts."""
        return [self.extract_features(p) for p in prompts]

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    def analyze_correlation(
        self,
        samples: Sequence[EvalSample],
        *,
        reward_calculator: Optional[RewardCalculator] = None,
        use_human_score: bool = True,
    ) -> CorrelationResult:
        """Analyse feature–reward correlation across samples.

        For each boolean/numeric feature, compute Pearson *r* against the
        target score.  The target is ``human_score`` when available and
        ``use_human_score=True``; otherwise the QC reward score is used.

        Args:
            samples: Evaluation samples with prompts (and optionally
                ``human_label``).
            reward_calculator: Calculator for QC-based reward.  A default
                instance is created when *None*.
            use_human_score: Prefer ``human_label.human_score`` as target.

        Returns:
            ``CorrelationResult`` with sorted feature importances and
            pattern descriptions.
        """
        if reward_calculator is None:
            reward_calculator = RewardCalculator()

        # Gather features + scores
        rows: List[Dict[str, float]] = []
        scores: List[float] = []

        for s in samples:
            feat = self.extract_features(s.prompt.generated_prompt)

            target: Optional[float] = None
            if use_human_score and s.human_label is not None:
                target = float(s.human_label.human_score)
            if target is None:
                bd = reward_calculator.calculate(s.qc_result)
                target = bd.total_score

            row: Dict[str, float] = {
                "word_count": float(feat.word_count),
                "has_camera_move": float(feat.has_camera_move),
                "has_single_continuous_shot": float(feat.has_single_continuous_shot),
                "has_sky_freeze": float(feat.has_sky_freeze),
                "has_stable_element": float(feat.has_stable_element),
                "has_speed_modifier": float(feat.has_speed_modifier),
                "has_human_instruction": float(feat.has_human_instruction),
                "motion_element_count": float(len(feat.motion_elements)),
                "sentence_count": float(feat.sentence_count),
            }
            rows.append(row)
            scores.append(target)

        if len(rows) < 3:
            return CorrelationResult()

        df = pd.DataFrame(rows)
        score_arr = np.array(scores)

        # Pearson r per feature
        importances: Dict[str, float] = {}
        for col in df.columns:
            arr = df[col].values.astype(float)
            if np.std(arr) < 1e-9 or np.std(score_arr) < 1e-9:
                importances[col] = 0.0
            else:
                importances[col] = round(
                    float(np.corrcoef(arr, score_arr)[0, 1]), 4
                )

        # Sort descending by absolute value
        sorted_imp = dict(
            sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )

        # Identify high/low reward patterns
        median = float(np.median(score_arr))
        high_mask = score_arr >= np.percentile(score_arr, 75)
        low_mask = score_arr <= np.percentile(score_arr, 25)

        high_patterns: List[str] = []
        low_patterns: List[str] = []

        for col in sorted_imp:
            if abs(sorted_imp[col]) < 0.1:
                continue
            high_mean = float(df.loc[high_mask, col].mean()) if high_mask.any() else 0.0
            low_mean = float(df.loc[low_mask, col].mean()) if low_mask.any() else 0.0
            if high_mean > low_mean + 0.1:
                high_patterns.append(
                    f"{col} (high-reward avg={high_mean:.2f} vs low={low_mean:.2f})"
                )
            elif low_mean > high_mean + 0.1:
                low_patterns.append(
                    f"{col} (low-reward avg={low_mean:.2f} vs high={high_mean:.2f})"
                )

        return CorrelationResult(
            feature_importance=sorted_imp,
            high_reward_patterns=high_patterns[:10],
            low_reward_patterns=low_patterns[:10],
        )
