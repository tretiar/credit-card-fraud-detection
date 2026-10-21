"""Metrics and threshold-selection helpers for imbalanced classification."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_report(
    model_name: str,
    y_true,
    y_pred,
    y_score=None,
    beta: float = 2,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Return a one-row DataFrame with classification metrics."""
    metrics = {"model": [model_name]}

    if threshold is not None:
        metrics["threshold"] = [threshold]

    metrics.update(
        {
            "precision": [precision_score(y_true, y_pred, zero_division=0)],
            "recall": [recall_score(y_true, y_pred, zero_division=0)],
            "f1": [f1_score(y_true, y_pred, zero_division=0)],
            "f2": [fbeta_score(y_true, y_pred, beta=beta, zero_division=0)],
        }
    )

    if y_score is not None:
        scores = np.asarray(y_score)
        if scores.ndim == 2:
            scores = scores[:, 1]

        try:
            metrics["pr_auc"] = [average_precision_score(y_true, scores)]
            metrics["roc_auc"] = [roc_auc_score(y_true, scores)]
        except ValueError:
            metrics["pr_auc"] = [np.nan]
            metrics["roc_auc"] = [np.nan]

    return pd.DataFrame(metrics)


def best_f2_threshold(y_true, y_score, beta: float = 2) -> float:
    """Select the decision threshold that maximises F-beta."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    beta_squared = beta**2

    f_beta_scores = (
        (1 + beta_squared) * precision[:-1] * recall[:-1]
        / (beta_squared * precision[:-1] + recall[:-1])
    )
    f_beta_scores = np.nan_to_num(f_beta_scores, nan=0.0, posinf=0.0, neginf=0.0)

    return float(thresholds[np.argmax(f_beta_scores)])
