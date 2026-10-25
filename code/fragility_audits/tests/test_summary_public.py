"""Synthetic checks of denominators and strict majority-failure boundaries."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd


def test_summary():
    spec = importlib.util.spec_from_file_location(
        'builder', Path(__file__).resolve().parents[1] / 'build_master.py')
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    coefficients = pd.DataFrame({
        'paper_id': ['synthetic'] * 3,
        'perturbation_design': ['example'] * 3,
        'delta': [.02] * 3,
        'model_id': ['a', 'b', 'c'],
        'significant_5pct': [True, True, False],
        'inferential_failure_rate': [.5, .8, 1.],
    })
    claims = pd.DataFrame({
        'paper_id': ['synthetic'] * 2,
        'perturbation_design': ['example'] * 2,
        'delta': [.02] * 2,
        'claim_id': ['x', 'y'],
        'S_claim': [.5, .3],
    })
    row = builder.summarize(coefficients, claims).iloc[0]
    assert row.n_significant_coefficients == 2
    assert np.isclose(row.mean_inferential_failure, .65)
    assert row.n_majority_failing_coefficients == 1
    assert row.n_claims == 2
    assert row.n_majority_failing_claims == 1
    assert row.fraction_majority_failing_claims == .5


if __name__ == '__main__':
    test_summary()
    print('Synthetic aggregation checks: PASSED')
