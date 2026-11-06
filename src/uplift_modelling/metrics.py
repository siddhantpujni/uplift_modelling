"""Causal and uplift evaluation metrics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def ate_difference(outcome: Sequence[float], treatment: Sequence[int]) -> float:
    """Estimate the ATE by the treated-minus-control difference in means."""
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(treatment, dtype=int)
    return float(y[w == 1].mean() - y[w == 0].mean())


def bootstrap_ate(
    outcome: Sequence[float],
    treatment: Sequence[int],
    *,
    n_bootstrap: int = 1_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Return the ATE and percentile bootstrap confidence interval."""
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(treatment, dtype=int)

    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    while len(estimates) < n_bootstrap:
        indices = rng.integers(0, len(y), size=len(y))
        try:
            estimates.append(ate_difference(y[indices], w[indices]))
        except ValueError:
            continue
    interval = np.percentile(estimates, [2.5, 97.5])
    return {
        "estimate": ate_difference(y, w),
        "ci_lower": float(interval[0]),
        "ci_upper": float(interval[1]),
        "n_bootstrap": n_bootstrap,
    }


def balance_table(data: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Calculate standardised mean differences for every baseline feature."""
    treated = data["treatment"] == 1
    rows = []
    for feature in feature_columns:
        x_treated = data.loc[treated, feature].astype(float)
        x_control = data.loc[~treated, feature].astype(float)
        pooled_sd = np.sqrt((x_treated.var(ddof=1) + x_control.var(ddof=1)) / 2)
        mean_difference = x_treated.mean() - x_control.mean()
        smd = 0.0 if pooled_sd == 0 and mean_difference == 0 else (
            np.inf if pooled_sd == 0 else mean_difference / pooled_sd
        )
        rows.append(
            {
                "feature": feature,
                "treated_mean": x_treated.mean(),
                "control_mean": x_control.mean(),
                "smd": smd,
                "absolute_smd": abs(smd),
            }
        )
    return pd.DataFrame(rows)


def uplift_curve(
    outcome: Sequence[int],
    treatment: Sequence[int],
    uplift_score: Sequence[float],
) -> pd.DataFrame:
    """Build an incremental-gain curve by descending predicted uplift."""
    y = np.asarray(outcome, dtype=float)
    w = np.asarray(treatment, dtype=int)
    score = np.asarray(uplift_score, dtype=float)
    if not (len(y) == len(w) == len(score)):
        raise ValueError("All inputs must have the same length.")
    order = np.argsort(-score, kind="stable")
    ranked = pd.DataFrame({"outcome": y[order], "treatment": w[order]})
    treated = ranked["treatment"].eq(1)
    control = ~treated
    ranked["treated_n"] = treated.cumsum()
    ranked["control_n"] = control.cumsum()
    ranked["treated_outcomes"] = (ranked["outcome"] * treated).cumsum()
    ranked["control_outcomes"] = (ranked["outcome"] * control).cumsum()
    ranked["incremental_gain"] = (
        ranked["treated_outcomes"] / ranked["treated_n"].replace(0, np.nan)
        - ranked["control_outcomes"] / ranked["control_n"].replace(0, np.nan)
    ) * (ranked["treated_n"] + ranked["control_n"])
    ranked["fraction_targeted"] = (np.arange(len(ranked)) + 1) / len(ranked)
    return ranked.fillna(0)


def auuc(curve: pd.DataFrame) -> float:
    """Calculate area under the uplift curve using the trapezoid rule."""
    return float(np.trapz(curve["incremental_gain"], curve["fraction_targeted"]))
