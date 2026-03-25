"""
evaluation.py — Face recognition evaluation metrics.

Definitions (binary: Employee vs Unknown)
──────────────────────────────────────────
  TP  True  Positive : model said "Employee X"  AND  it really was Employee X
  FP  False Positive : model said "Employee X"  BUT  it was a different person / unknown
  TN  True  Negative : model said "Unknown"     AND  it really was unknown
  FN  False Negative : model said "Unknown"     BUT  it was a known employee

Metrics
───────
  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  Precision = TP / (TP + FP)          — of all "employee" decisions, how many correct
  Recall    = TP / (TP + FN)          — of all real employees, how many caught
  FPR       = FP / (FP + TN)          — of all unknowns, how many were wrongly accepted
  F1        = 2 * P * R / (P + R)     — harmonic mean of Precision and Recall

Usage
─────
  from backend.evaluation import FaceRecognitionEvaluator

  ev = FaceRecognitionEvaluator()
  ev.add("EMP001", "EMP001")   # correct match
  ev.add("EMP002", "UNKNOWN")  # missed employee  → FN
  ev.add("UNKNOWN", "EMP003")  # wrong accept     → FP
  ev.add("UNKNOWN", "UNKNOWN") # correct reject   → TN

  report = ev.report()
  print(report)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ── Sentinel ──────────────────────────────────────────────────────────────────
UNKNOWN = "UNKNOWN"


# ── Per-sample result ─────────────────────────────────────────────────────────
@dataclass
class Sample:
    ground_truth: str   # employee_id or UNKNOWN
    predicted:    str   # employee_id or UNKNOWN
    confidence:   float = 0.0


# ── Metric snapshot ───────────────────────────────────────────────────────────
@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        """Fraction of all decisions that were correct."""
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        """Of every 'employee detected' call, how many were right."""
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Of every real employee, how many were correctly caught (sensitivity)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Of every unknown face, how many were wrongly accepted as employees."""
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def specificity(self) -> float:
        """True Negative Rate — complement of FPR."""
        return 1.0 - self.false_positive_rate

    def as_dict(self) -> dict:
        return {
            "TP":        self.tp,
            "FP":        self.fp,
            "TN":        self.tn,
            "FN":        self.fn,
            "Total":     self.total,
            "Accuracy":  round(self.accuracy,           4),
            "Precision": round(self.precision,          4),
            "Recall":    round(self.recall,             4),
            "FPR":       round(self.false_positive_rate,4),
            "F1":        round(self.f1,                 4),
            "Specificity": round(self.specificity,      4),
        }

    def __str__(self) -> str:
        d = self.as_dict()
        lines = [
            "┌─────────────────────────────────┐",
            "│     Face Recognition Metrics    │",
            "├──────────────┬──────────────────┤",
            f"│ Accuracy     │ {d['Accuracy']:.2%}          │",
            f"│ Precision    │ {d['Precision']:.2%}          │",
            f"│ Recall       │ {d['Recall']:.2%}          │",
            f"│ F1 Score     │ {d['F1']:.2%}          │",
            f"│ FPR          │ {d['FPR']:.2%}          │",
            f"│ Specificity  │ {d['Specificity']:.2%}          │",
            "├──────────────┴──────────────────┤",
            f"│ TP={d['TP']}  FP={d['FP']}  TN={d['TN']}  FN={d['FN']}  N={d['Total']} │",
            "└─────────────────────────────────┘",
        ]
        return "\n".join(lines)


# ── Main evaluator ────────────────────────────────────────────────────────────
class FaceRecognitionEvaluator:
    """
    Accumulates (ground_truth, predicted) pairs and computes all metrics.

    Binary mode  — treats every known employee as positive, UNKNOWN as negative.
    Per-employee — also breaks metrics down per employee_id (one-vs-rest).

    Parameters
    ----------
    confidence_threshold : If you pass confidence scores, samples below this
                           threshold are re-labelled UNKNOWN before scoring.
                           Set to 0.0 to use the predicted label as-is.
    """

    def __init__(self, confidence_threshold: float = 0.0) -> None:
        self._threshold = confidence_threshold
        self._samples: list[Sample] = []

    # ── Data ingestion ────────────────────────────────────────────────────────

    def add(
        self,
        ground_truth: str,
        predicted:    str,
        confidence:   float = 1.0,
    ) -> None:
        """
        Record one recognition decision.

        Parameters
        ----------
        ground_truth : Employee ID of the actual person, or UNKNOWN.
        predicted    : Employee ID the model output, or UNKNOWN.
        confidence   : Similarity score (0–1). If below threshold, predicted
                       is overridden to UNKNOWN.
        """
        if self._threshold and confidence < self._threshold:
            predicted = UNKNOWN
        self._samples.append(Sample(ground_truth, predicted, confidence))

    def add_batch(
        self,
        ground_truths: list[str],
        predictions:   list[str],
        confidences:   Optional[list[float]] = None,
    ) -> None:
        """Add multiple samples at once."""
        if confidences is None:
            confidences = [1.0] * len(ground_truths)
        for gt, pred, conf in zip(ground_truths, predictions, confidences):
            self.add(gt, pred, conf)

    def reset(self) -> None:
        self._samples.clear()

    # ── Confusion matrix (binary) ─────────────────────────────────────────────

    def confusion_matrix(self) -> Metrics:
        """
        Binary confusion matrix:
          Positive = predicted a known employee
          Negative = predicted UNKNOWN
        """
        m = Metrics()
        for s in self._samples:
            gt_known   = s.ground_truth != UNKNOWN
            pred_known = s.predicted    != UNKNOWN
            correct    = s.ground_truth == s.predicted

            if pred_known and correct:
                m.tp += 1   # said employee X, was employee X
            elif pred_known and not correct:
                m.fp += 1   # said employee X, was someone else / unknown
            elif not pred_known and not gt_known:
                m.tn += 1   # said unknown, was unknown
            else:
                m.fn += 1   # said unknown, but was a real employee

        return m

    # ── Global metrics ────────────────────────────────────────────────────────

    def report(self) -> Metrics:
        """Return the global binary Metrics object."""
        return self.confusion_matrix()

    # ── Per-employee breakdown ────────────────────────────────────────────────

    def per_employee_metrics(self) -> dict[str, Metrics]:
        """
        One-vs-rest metrics for every known employee_id.

        For employee E:
          TP = predicted E  AND  is E
          FP = predicted E  AND  is NOT E
          TN = not predicted E  AND  is NOT E
          FN = not predicted E  AND  is E
        """
        # Collect all known employee IDs
        all_ids = {s.ground_truth for s in self._samples} | {s.predicted for s in self._samples}
        all_ids.discard(UNKNOWN)

        results: dict[str, Metrics] = {}
        for emp_id in sorted(all_ids):
            m = Metrics()
            for s in self._samples:
                is_emp   = s.ground_truth == emp_id
                pred_emp = s.predicted    == emp_id
                if pred_emp and is_emp:
                    m.tp += 1
                elif pred_emp and not is_emp:
                    m.fp += 1
                elif not pred_emp and not is_emp:
                    m.tn += 1
                else:
                    m.fn += 1
            results[emp_id] = m
        return results

    # ── Threshold sweep (ROC data) ────────────────────────────────────────────

    def threshold_sweep(
        self, thresholds: Optional[list[float]] = None
    ) -> list[dict]:
        """
        Compute metrics at multiple confidence thresholds.
        Useful for plotting ROC / Precision-Recall curves.

        Returns a list of dicts sorted by threshold ascending.
        """
        if thresholds is None:
            thresholds = [i / 20 for i in range(21)]   # 0.00 … 1.00

        results = []
        original_threshold = self._threshold
        for t in thresholds:
            self._threshold = t
            m = self.confusion_matrix()
            row = m.as_dict()
            row["threshold"] = round(t, 3)
            results.append(row)
        self._threshold = original_threshold
        return results


# ── Standalone compute functions (no class needed) ────────────────────────────

def compute_metrics(
    ground_truths: list[str],
    predictions:   list[str],
    confidences:   Optional[list[float]] = None,
    threshold:     float = 0.0,
) -> Metrics:
    """
    One-shot metric computation.

    Parameters
    ----------
    ground_truths : List of actual employee IDs or "UNKNOWN".
    predictions   : List of predicted employee IDs or "UNKNOWN".
    confidences   : Optional list of similarity scores (0–1).
    threshold     : Predictions below this confidence become "UNKNOWN".

    Returns
    -------
    Metrics object with .accuracy, .precision, .recall, .false_positive_rate, .f1
    """
    ev = FaceRecognitionEvaluator(confidence_threshold=threshold)
    ev.add_batch(ground_truths, predictions, confidences)
    return ev.report()


def accuracy(ground_truths: list[str], predictions: list[str]) -> float:
    return compute_metrics(ground_truths, predictions).accuracy


def precision(ground_truths: list[str], predictions: list[str]) -> float:
    return compute_metrics(ground_truths, predictions).precision


def recall(ground_truths: list[str], predictions: list[str]) -> float:
    return compute_metrics(ground_truths, predictions).recall


def false_positive_rate(ground_truths: list[str], predictions: list[str]) -> float:
    return compute_metrics(ground_truths, predictions).false_positive_rate


def f1_score(ground_truths: list[str], predictions: list[str]) -> float:
    return compute_metrics(ground_truths, predictions).f1


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ev = FaceRecognitionEvaluator(confidence_threshold=0.5)

    # Simulated recognition session
    #              ground_truth   predicted      confidence
    ev.add("EMP001", "EMP001",  0.92)   # TP — correct
    ev.add("EMP002", "EMP002",  0.87)   # TP — correct
    ev.add("EMP003", "UNKNOWN", 0.40)   # FN — employee missed (low confidence)
    ev.add("EMP001", "EMP002",  0.78)   # FP — wrong employee
    ev.add("UNKNOWN","UNKNOWN", 0.00)   # TN — unknown correctly rejected
    ev.add("UNKNOWN","EMP003",  0.61)   # FP — unknown wrongly accepted
    ev.add("EMP004", "EMP004",  0.95)   # TP — correct
    ev.add("EMP005", "UNKNOWN", 0.30)   # FN — employee missed

    # ── Global metrics
    print(ev.report())

    # ── Per-employee breakdown
    print("\nPer-employee breakdown:")
    for emp_id, m in ev.per_employee_metrics().items():
        print(f"  {emp_id}: Acc={m.accuracy:.0%}  P={m.precision:.0%}  R={m.recall:.0%}  FPR={m.false_positive_rate:.0%}")

    # ── Threshold sweep
    print("\nThreshold sweep:")
    for row in ev.threshold_sweep([0.3, 0.5, 0.6, 0.7, 0.8, 0.9]):
        print(f"  t={row['threshold']}  Acc={row['Accuracy']:.2%}  P={row['Precision']:.2%}  R={row['Recall']:.2%}  FPR={row['FPR']:.2%}")
