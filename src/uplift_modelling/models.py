"""Heterogeneous treatment effect estimators used by the project."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _forest(random_state: int, n_estimators: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1,
    )


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


class XLearner:
    """X-learner using outcome models and treatment-specific effect models."""

    def __init__(self, *, random_state: int = 42, n_estimators: int = 100):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self._mu0 = HistGradientBoostingClassifier(random_state=random_state)
        self._mu1 = HistGradientBoostingClassifier(random_state=random_state)
        self._tau0 = _forest(random_state, n_estimators)
        self._tau1 = _forest(random_state + 1, n_estimators)
        self._propensity_model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, random_state=random_state)
        )

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> "XLearner":
        if treatment.nunique() != 2:
            raise ValueError("X-learner requires both treatment groups.")
        treated = treatment.to_numpy() == 1
        self._mu0.fit(X[~treated], outcome[~treated])
        self._mu1.fit(X[treated], outcome[treated])
        mu0 = self._mu0.predict_proba(X)[:, 1]
        mu1 = self._mu1.predict_proba(X)[:, 1]
        self._tau0.fit(X[~treated], mu1[~treated] - outcome[~treated])
        self._tau1.fit(X[treated], outcome[treated] - mu0[treated])
        self._propensity_model.fit(X, treatment)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        propensity = self._propensity_model.predict_proba(X)[:, 1]
        return (1 - propensity) * self._tau1.predict(X) + propensity * self._tau0.predict(X)


class RLearner:
    """R-learner that regresses residualised outcomes on residualised treatment."""

    def __init__(self, *, random_state: int = 42, n_estimators: int = 100):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self._outcome = HistGradientBoostingClassifier(random_state=random_state)
        self._propensity = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, random_state=random_state)
        )
        self._effect = _forest(random_state, n_estimators)

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> "RLearner":
        if treatment.nunique() != 2:
            raise ValueError("R-learner requires both treatment groups.")
        self._outcome.fit(X, outcome)
        self._propensity.fit(X, treatment)
        m = self._outcome.predict_proba(X)[:, 1]
        e = self._propensity.predict_proba(X)[:, 1]
        residual_treatment = treatment.to_numpy() - e
        pseudo_outcome = (outcome.to_numpy() - m) / np.where(
            np.abs(residual_treatment) > 1e-6, residual_treatment, np.nan
        )
        valid = np.isfinite(pseudo_outcome)
        self._effect.fit(
            X.loc[valid],
            pseudo_outcome[valid],
            sample_weight=residual_treatment[valid] ** 2,
        )
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self._effect.predict(X)


class DRLearner:
    """Doubly robust learner with cross-fitted nuisance predictions."""

    def __init__(
        self, *, random_state: int = 42, n_estimators: int = 100, n_splits: int = 2
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.n_splits = n_splits
        self._mu0 = HistGradientBoostingClassifier(random_state=random_state)
        self._mu1 = HistGradientBoostingClassifier(random_state=random_state + 1)
        self._propensity = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, random_state=random_state)
        )
        self._effect = _forest(random_state, n_estimators)

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> "DRLearner":
        if treatment.nunique() != 2:
            raise ValueError("DR-learner requires both treatment groups.")
        w = treatment.to_numpy()
        mu0 = np.zeros(len(X))
        mu1 = np.zeros(len(X))
        e = np.zeros(len(X))
        folds = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for train_indices, validation_indices in folds.split(X, treatment):
            mu0_model = clone(self._mu0).fit(
                X.iloc[train_indices][treatment.iloc[train_indices] == 0],
                outcome.iloc[train_indices][treatment.iloc[train_indices] == 0],
            )
            mu1_model = clone(self._mu1).fit(
                X.iloc[train_indices][treatment.iloc[train_indices] == 1],
                outcome.iloc[train_indices][treatment.iloc[train_indices] == 1],
            )
            propensity_model = clone(self._propensity).fit(
                X.iloc[train_indices], treatment.iloc[train_indices]
            )
            mu0[validation_indices] = mu0_model.predict_proba(
                X.iloc[validation_indices]
            )[:, 1]
            mu1[validation_indices] = mu1_model.predict_proba(
                X.iloc[validation_indices]
            )[:, 1]
            e[validation_indices] = propensity_model.predict_proba(
                X.iloc[validation_indices]
            )[:, 1]
        e = np.clip(e, 0.01, 0.99)
        pseudo = mu1 - mu0 + (w - e) / (e * (1 - e)) * (
            outcome.to_numpy() - w * mu1 - (1 - w) * mu0
        )
        self._effect.fit(X, pseudo)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self._effect.predict(X)


class CausalForest:
    """Forest estimator using an inverse-propensity transformed outcome.

    This is a transparent scikit-learn implementation of a causal forest
    baseline. It is not a substitute for an honest forest implementation from
    a specialised causal-inference library, but provides a dependency-light
    forest benchmark for the project.
    """

    def __init__(self, *, random_state: int = 42, n_estimators: int = 200):
        self.random_state = random_state
        self._propensity = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=500, random_state=random_state)
        )
        self._forest = _forest(random_state, n_estimators)

    def fit(self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series) -> "CausalForest":
        if treatment.nunique() != 2:
            raise ValueError("Causal forest requires both treatment groups.")
        self._propensity.fit(X, treatment)
        e = np.clip(self._propensity.predict_proba(X)[:, 1], 0.01, 0.99)
        transformed = outcome.to_numpy() * (treatment.to_numpy() - e) / (e * (1 - e))
        self._forest.fit(X, transformed)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self._forest.predict(X)


class TorchUpliftNetwork:
    """Shared-trunk neural network with treatment-specific outcome heads."""

    def __init__(
        self,
        *,
        hidden_dim: int = 32,
        epochs: int = 30,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._network = None

    def fit(
        self, X: pd.DataFrame, treatment: pd.Series, outcome: pd.Series
    ) -> "TorchUpliftNetwork":
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("Install the 'ml' extra to use TorchUpliftNetwork.") from exc
        torch.manual_seed(self.random_state)
        features = torch.tensor(X.to_numpy(dtype=np.float32))
        treatments = torch.tensor(treatment.to_numpy(dtype=np.float32)).reshape(-1, 1)
        outcomes = torch.tensor(outcome.to_numpy(dtype=np.float32)).reshape(-1, 1)
        self._network = nn.Sequential(
            nn.Linear(features.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            logits = self._network(features)
            selected = logits.gather(1, treatments.long())
            loss = loss_fn(selected, outcomes)
            loss.backward()
            optimizer.step()
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("TorchUpliftNetwork must be fitted before prediction.")
        import torch

        with torch.no_grad():
            logits = self._network(torch.tensor(X.to_numpy(dtype=np.float32)))
            probabilities = torch.sigmoid(logits).numpy()
        return probabilities[:, 1] - probabilities[:, 0]
