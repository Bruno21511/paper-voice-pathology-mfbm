# -*- coding: utf-8 -*-
import numpy as np

from src.visualization.plot_and_save_confusion_matrices import plot_and_save_confusion_matrices

def test_plot_and_save_confusion_matrices(tmp_path):
    results = {'a_vs_b': (np.array([[10, 2], [1, 15]]), 0.9, 0.92)}
    class_names = {'a_vs_b': ('A', 'B')}
    plot_and_save_confusion_matrices(results, class_names, output_dir=str(tmp_path))
    assert (tmp_path / "04_cm_a_vs_b.png").exists()