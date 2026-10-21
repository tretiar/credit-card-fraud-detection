"""Unit tests for threshold selection."""

import numpy as np

from src.metrics import best_f2_threshold


def test_best_f2_threshold_returns_a_score_threshold():
    threshold = best_f2_threshold(
        y_true=np.array([0, 0, 1, 1]),
        y_score=np.array([0.1, 0.2, 0.8, 0.9]),
    )

    assert 0.1 <= threshold <= 0.9
