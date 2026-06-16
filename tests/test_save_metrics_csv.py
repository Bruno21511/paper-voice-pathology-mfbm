# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from src.evaluation.save_metrics_csv import save_metrics_csv

def test_save_metrics_csv(tmp_path):
    results = {'a_vs_b': (np.array([[10, 2], [1, 15]]), 0.9, 0.92)}
    output_path = tmp_path / "metrics.csv"
    save_metrics_csv(results, output_path=str(output_path))
    
    df = pd.read_csv(output_path, sep=';')
    assert df.iloc[0]['tp'] == 15
    assert df.iloc[0]['f1_score'] == 0.9