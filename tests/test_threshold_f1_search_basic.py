# -*- coding: utf-8 -*-
import numpy as np

from src.analysis.threshold_f1_search import threshold_f1_search

def test_threshold_f1_search_basic():
    # std values claramente separáveis por classe
    X = np.array([[0, 1.0], [0, 1.1], [0, 5.0], [0, 5.2]])
    y = np.array([0, 0, 1, 1])
    t = threshold_f1_search(X, y, step=0.1)
    assert 1.1 < t < 5.0  # threshold deve cair entre os dois clusters