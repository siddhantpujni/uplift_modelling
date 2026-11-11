import json

import numpy as np
import pandas as pd
import pytest

from uplift_modelling.conformal import SplitConformalCATE
from uplift_modelling.constants import FEATURE_COLUMNS
from uplift_modelling.data import load_criteo_csv, validate_criteo_data
from uplift_modelling.models import CausalForest, DRLearner, RLearner, TLearner, XLearner


def synthetic_criteo_data(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    treatment = np.tile([0, 1], n_rows // 2)
    features = pd.DataFrame(
        rng.normal(size=(n_rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    probability = 0.15 + 0.04 * treatment + 0.03 * (features["f0"] > 0)
    visit = rng.binomial(1, probability)
    conversion = visit * rng.binomial(1, 0.2, size=n_rows)
    return features.assign(
        treatment=treatment,
        conversion=conversion,
        visit=visit,
        exposure=treatment,
    )


def test_synthetic_data_contract_and_t_learner():
    data = synthetic_criteo_data()
    validate_criteo_data(data)
    learner = TLearner(max_iter=10).fit(
        data[FEATURE_COLUMNS], data["treatment"], data["conversion"]
    )
    predictions = learner.predict_uplift(data[FEATURE_COLUMNS])
    assert predictions.shape == (len(data),)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("learner_class", [XLearner, RLearner, DRLearner, CausalForest])
def test_advanced_learners_predict_finite_uplift(learner_class):
    data = synthetic_criteo_data()
    learner = learner_class(n_estimators=10).fit(
        data[FEATURE_COLUMNS], data["treatment"], data["conversion"]
    )
    predictions = learner.predict_uplift(data[FEATURE_COLUMNS])
    assert predictions.shape == (len(data),)
    assert np.isfinite(predictions).all()


def test_split_conformal_interval_has_positive_width():
    data = synthetic_criteo_data()
    learner = TLearner(max_iter=10).fit(
        data[FEATURE_COLUMNS], data["treatment"], data["conversion"]
    )
    interval = SplitConformalCATE(learner).fit(
        data[FEATURE_COLUMNS], data["treatment"], data["conversion"]
    )
    lower, upper = interval.predict_interval(data[FEATURE_COLUMNS].head(5))
    assert np.all(upper >= lower)
    assert interval.radius_ is not None
    coverage = interval.empirical_coverage(
        data[FEATURE_COLUMNS], data["treatment"], data["conversion"]
    )
    assert 0 <= coverage <= 1


def test_torch_uplift_network_predicts_when_torch_is_available():
    pytest.importorskip("torch")
    from uplift_modelling.models import TorchUpliftNetwork

    data = synthetic_criteo_data()
    learner = TorchUpliftNetwork(epochs=2, hidden_dim=8).fit(
        data[FEATURE_COLUMNS], data["treatment"], data["conversion"]
    )
    predictions = learner.predict_uplift(data[FEATURE_COLUMNS].head(5))
    assert predictions.shape == (5,)
    assert np.isfinite(predictions).all()


def test_baseline_summary_is_json_serializable(tmp_path):
    summary = {"rows": 200, "auuc": {"t_learner": {"estimate": 1.2}}}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    assert json.loads(path.read_text()) == summary


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda data: data.drop(columns=["f0"]), "Missing required columns"),
        (lambda data: data.assign(treatment=2), "treatment must be binary"),
        (lambda data: data.assign(conversion=1, visit=0), "Conversion must not"),
        (lambda data: data.assign(treatment=0), "both treated and control"),
    ],
)
def test_data_contract_rejects_invalid_data(mutator, message):
    with pytest.raises(ValueError, match=message):
        validate_criteo_data(mutator(synthetic_criteo_data()))


def test_load_criteo_csv_applies_fraction(tmp_path):
    data = synthetic_criteo_data()
    path = tmp_path / "criteo.csv"
    data.to_csv(path, index=False)
    loaded = load_criteo_csv(path, sample_fraction=0.5, random_state=4)
    assert len(loaded) == 100
