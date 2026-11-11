"""Split-conformal intervals for conditional treatment effects."""

import numpy as np
import pandas as pd


class SplitConformalCATE:
    """Wrap a fitted CATE learner with a finite-sample calibration interval.

    Calibration uses the inverse-propensity transformed outcome. The interval
    reflects uncertainty in the noisy pseudo-outcome and is therefore intended
    as an empirical coverage diagnostic, not a claim of exact potential-outcome
    identification for each individual.
    """

    def __init__(self, learner, *, alpha: float = 0.1, propensity: float | None = None):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1.")
        self.learner = learner
        self.alpha = alpha
        self.propensity = propensity
        self.radius_: float | None = None

    def fit(
        self,
        X_calibration: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ) -> "SplitConformalCATE":
        if treatment.nunique() != 2:
            raise ValueError("Calibration data must contain both treatment groups.")
        p = self.propensity if self.propensity is not None else float(treatment.mean())
        if not 0 < p < 1:
            raise ValueError("propensity must be strictly between zero and one.")
        score = outcome.to_numpy() * (treatment.to_numpy() - p) / (p * (1 - p))
        prediction = self.learner.predict_uplift(X_calibration)
        residuals = np.abs(score - prediction)
        quantile = np.ceil((len(residuals) + 1) * (1 - self.alpha)) / len(residuals)
        self.radius_ = float(np.quantile(residuals, min(quantile, 1), method="higher"))
        return self

    def predict_interval(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.radius_ is None:
            raise RuntimeError("SplitConformalCATE must be fitted before prediction.")
        prediction = self.learner.predict_uplift(X)
        return prediction - self.radius_, prediction + self.radius_

    def empirical_coverage(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
    ) -> float:
        """Report coverage against the held-out transformed outcome."""
        p = self.propensity if self.propensity is not None else float(treatment.mean())
        pseudo = outcome.to_numpy() * (treatment.to_numpy() - p) / (p * (1 - p))
        lower, upper = self.predict_interval(X)
        return float(np.mean((pseudo >= lower) & (pseudo <= upper)))
