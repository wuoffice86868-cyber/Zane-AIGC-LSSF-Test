"""QC calibration tools — compare automated QC against human labels."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from .models import (
    AutoFailRuleAnalysis,
    CalibrationReport,
    EvalSample,
)


class QCCalibrator:
    """Calibrate automated QC judgments against human ground-truth labels.

    The calibrator treats the QC system as a binary classifier
    (pass / fail) and computes standard classification metrics using
    ``human_label.human_pass`` as ground truth.

    It also analyses individual auto-fail rules and produces
    actionable recommendations for QC prompt improvements.
    """

    # Human score threshold: scores >= this value count as "human pass"
    DEFAULT_PASS_THRESHOLD = 6

    def __init__(self, pass_threshold: int = DEFAULT_PASS_THRESHOLD) -> None:
        """
        Args:
            pass_threshold: ``human_score`` values **>=** this threshold
                are treated as *pass* when ``human_pass`` is not explicitly
                set.  Default **6** (on a 1-10 scale).
        """
        self.pass_threshold = pass_threshold

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    def _human_pass(self, sample: EvalSample) -> bool:
        """Derive human pass/fail from label, falling back to score threshold."""
        if sample.human_label is None:
            raise ValueError(f"Sample {sample.sample_id} has no human_label")
        return sample.human_label.human_pass

    # ------------------------------------------------------------------
    # Main calibration
    # ------------------------------------------------------------------

    def calibrate(self, samples: Sequence[EvalSample]) -> CalibrationReport:
        """Compute calibration metrics for QC vs human labels.

        Only samples with a ``human_label`` are used.

        Confusion matrix layout::

            [[TP, FP],
             [FN, TN]]

        Where *positive* = **fail** (i.e. detecting bad videos is the
        positive class, since that is the QC system's primary job).

        Args:
            samples: Evaluation samples.  Those without ``human_label``
                are silently skipped.

        Returns:
            ``CalibrationReport`` with accuracy, precision, recall, F1,
            Pearson correlation, confusion matrix, false-positive /
            false-negative sample IDs, and improvement recommendations.
        """
        labeled = [s for s in samples if s.human_label is not None]
        if not labeled:
            return CalibrationReport(recommendations=["No labeled samples provided."])

        # Build arrays
        # Positive class = FAIL (QC's job is to catch bad videos)
        tp = fp = fn = tn = 0
        false_pos_ids: List[str] = []  # QC says fail, human says pass
        false_neg_ids: List[str] = []  # QC says pass, human says fail

        qc_confs: List[float] = []
        human_scores: List[float] = []

        for s in labeled:
            qc_fail = not s.qc_result.qc_pass
            human_fail = not self._human_pass(s)

            qc_confs.append(s.qc_result.confidence)
            human_scores.append(float(s.human_label.human_score))  # type: ignore[union-attr]

            if qc_fail and human_fail:
                tp += 1
            elif qc_fail and not human_fail:
                fp += 1
                false_pos_ids.append(s.sample_id)
            elif not qc_fail and human_fail:
                fn += 1
                false_neg_ids.append(s.sample_id)
            else:
                tn += 1

        total = tp + fp + fn + tn
        accuracy = self._safe_div(tp + tn, total)
        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2 * precision * recall, precision + recall)

        # Pearson correlation: qc_confidence vs human_score
        corr = 0.0
        if len(qc_confs) >= 3:
            conf_arr = np.array(qc_confs)
            score_arr = np.array(human_scores)
            if np.std(conf_arr) > 1e-9 and np.std(score_arr) > 1e-9:
                corr = float(np.corrcoef(conf_arr, score_arr)[0, 1])

        # Recommendations
        recs = self._generate_recommendations(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            correlation=corr,
            fp_count=fp,
            fn_count=fn,
            total=total,
        )

        return CalibrationReport(
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            correlation=round(corr, 4),
            confusion_matrix=[[tp, fp], [fn, tn]],
            false_positives=false_pos_ids,
            false_negatives=false_neg_ids,
            recommendations=recs,
        )

    # ------------------------------------------------------------------
    # Auto-fail rule analysis
    # ------------------------------------------------------------------

    def analyze_auto_fail_rules(
        self, samples: Sequence[EvalSample]
    ) -> Dict[str, AutoFailRuleAnalysis]:
        """Evaluate each auto-fail rule's precision, recall, and F1.

        For a given rule *R*:

        - **TP**: rule fired AND human marked fail
        - **FP**: rule fired AND human marked pass
        - **FN**: human marked fail (with a matching issue keyword)
          AND rule did NOT fire

        Args:
            samples: Evaluation samples with ``human_label``.

        Returns:
            Dict mapping rule name → ``AutoFailRuleAnalysis``.
        """
        labeled = [s for s in samples if s.human_label is not None]
        if not labeled:
            return {}

        # Collect all rule names seen in data
        all_rules: set[str] = set()
        for s in labeled:
            all_rules.update(s.qc_result.auto_fail_triggered)

        results: Dict[str, AutoFailRuleAnalysis] = {}

        for rule in sorted(all_rules):
            tp = fp = fn = 0
            fp_ids: List[str] = []

            for s in labeled:
                rule_fired = rule in s.qc_result.auto_fail_triggered
                human_fail = not self._human_pass(s)

                # Check if human issues mention something related to this rule
                human_mentions_rule = self._human_mentions_rule(
                    rule, s.human_label.human_issues if s.human_label else []  # type: ignore[union-attr]
                )

                if rule_fired and human_fail:
                    tp += 1
                elif rule_fired and not human_fail:
                    fp += 1
                    fp_ids.append(s.sample_id)
                elif not rule_fired and human_fail and human_mentions_rule:
                    fn += 1

            prec = self._safe_div(tp, tp + fp)
            rec = self._safe_div(tp, tp + fn)
            f1 = self._safe_div(2 * prec * rec, prec + rec)

            results[rule] = AutoFailRuleAnalysis(
                rule_name=rule,
                precision=round(prec, 4),
                recall=round(rec, 4),
                f1=round(f1, 4),
                sample_count=tp + fp,
                false_positive_ids=fp_ids,
            )

        return results

    # ------------------------------------------------------------------
    # Scene-type breakdown
    # ------------------------------------------------------------------

    def calibrate_by_scene(
        self, samples: Sequence[EvalSample]
    ) -> Dict[str, CalibrationReport]:
        """Run ``calibrate`` separately for each ``scene_type``.

        Args:
            samples: Evaluation samples with ``human_label``.

        Returns:
            Dict mapping scene_type string → ``CalibrationReport``.
        """
        buckets: Dict[str, List[EvalSample]] = defaultdict(list)
        for s in samples:
            if s.human_label is not None:
                scene = s.input.scene_type or "unknown"
                buckets[scene].append(s)

        return {scene: self.calibrate(samps) for scene, samps in sorted(buckets.items())}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _human_mentions_rule(rule: str, human_issues: List[str]) -> bool:
        """Fuzzy check whether human issues mention the same concept as a rule."""
        # Normalize rule name: "face_morphing" → {"face", "morphing"}
        rule_tokens = set(rule.lower().replace("_", " ").split())
        for issue in human_issues:
            issue_lower = issue.lower()
            # If any rule token appears in the human issue text, count it
            if any(tok in issue_lower for tok in rule_tokens):
                return True
        return False

    @staticmethod
    def _generate_recommendations(
        *,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        correlation: float,
        fp_count: int,
        fn_count: int,
        total: int,
    ) -> List[str]:
        recs: List[str] = []

        if accuracy < 0.70:
            recs.append(
                f"Overall accuracy ({accuracy:.0%}) is below 70% — QC prompt "
                "needs significant revision before using for optimization."
            )

        if precision < 0.80 and fp_count > 0:
            recs.append(
                f"Precision ({precision:.0%}): {fp_count} false positives — "
                "QC is too aggressive. Consider relaxing thresholds for "
                "borderline cases or adding a confidence gate."
            )

        if recall < 0.75 and fn_count > 0:
            recs.append(
                f"Recall ({recall:.0%}): {fn_count} false negatives — "
                "QC is missing real issues. Review auto-fail rules and "
                "add detection for the missed failure modes."
            )

        if abs(correlation) < 0.4:
            recs.append(
                f"Confidence–score correlation ({correlation:.2f}) is weak — "
                "QC confidence does not track human quality perception. "
                "Consider recalibrating the confidence output or adding "
                "a multi-point scoring scale."
            )

        if f1 >= 0.85 and accuracy >= 0.85:
            recs.append(
                "QC performance is strong. Safe to proceed with reward-"
                "based optimization."
            )

        if not recs:
            recs.append("Collect more labeled samples for a reliable assessment.")

        return recs
