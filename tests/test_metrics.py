import numpy as np
import pandas as pd

from uplift_modelling.metrics import ate_difference, balance_table, bootstrap_ate, uplift_curve


def test_ate_difference_matches_known_effect():
    outcome = np.array([0, 1, 0, 1])
    treatment = np.array([0, 0, 1, 1])
    assert ate_difference(outcome, treatment) == 0


def test_bootstrap_ate_is_reproducible():
    outcome = np.array([0, 1, 0, 1, 1, 1])
    treatment = np.array([0, 0, 0, 1, 1, 1])
    first = bootstrap_ate(outcome, treatment, n_bootstrap=100, random_state=7)
    second = bootstrap_ate(outcome, treatment, n_bootstrap=100, random_state=7)
    assert first == second


def test_balance_table_reports_absolute_smd():
    data = pd.DataFrame({"treatment": [0, 0, 1, 1], "feature": [0, 0, 2, 2]})
    result = balance_table(data, ["feature"])
    assert result.loc[0, "absolute_smd"] > 0


def test_uplift_curve_contains_all_rows():
    curve = uplift_curve([0, 1, 1, 0], [0, 1, 0, 1], [0.8, 0.7, 0.2, 0.1])
    assert len(curve) == 4
    assert np.isclose(curve["fraction_targeted"].iloc[-1], 1)
