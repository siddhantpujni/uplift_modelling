"""Baseline heterogeneous treatment effect models."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class TLearner:
    """Two outcome models whose difference estimates conditional uplift."""

    def __init__(self, *, random_state: int = 42, max_iter: int = 100):
        self.random_state = random_state
        self.max_iter = max_iter
        self._treated = HistGradientBoostingClassifier(
            max_iter=max_iter, random_state=random_state
        )
        self._control = HistGradientBoostingClassifier(
            max_iter=max_iter, random_state=random_state
        )

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> "TLearner":
        if treatment.nunique() != 2:
            raise ValueError("T-learner requires both treatment groups.")
        self._treated.fit(X[treatment == 1], outcome[treatment == 1])
        self._control.fit(X[treatment == 0], outcome[treatment == 0])
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self._treated.predict_proba(X)[:, 1] - self._control.predict_proba(X)[:, 1]

    def predict_treated_response(self, X: pd.DataFrame) -> np.ndarray:
        """Predict response probability under treatment for response targeting."""
        return self._treated.predict_proba(X)[:, 1]
