# -*- coding: utf-8 -*-
import numpy as np

from src.evaluation.compute_threshold_metrics import compute_threshold_metrics

def test_compute_threshold_metrics():
    X = np.array([[0, 1.0], [0, 1.1], [0, 5.0], [0, 5.2]])
    y = np.array([0, 0, 1, 1])
    cm, f1, acc = compute_threshold_metrics(X, y, threshold=3.0)
    assert acc == 1.0
    assert f1 == 1.0