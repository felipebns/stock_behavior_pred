import numpy as np
import pytest

from config.config import MODEL_PARAMS
from engine.models import MODELS, importance


def data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0.0, 0.5, n) > 0).astype(int)
    X[rng.random(X.shape) < 0.1] = np.nan
    return X, y


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_fits_and_predicts_probabilities_with_missing_values(name):
    X, y = data()
    model = MODELS[name][1](MODEL_PARAMS[name], len(y)).fit(X, y)
    prob = model.predict_proba(X)[:, 1]
    assert prob.shape == (len(y),) and ((prob >= 0) & (prob <= 1)).all()
    assert np.corrcoef(prob, y)[0, 1] > 0.3
    weights = importance(model, X.shape[1])
    assert weights.shape == (4,) and (weights.sum() == pytest.approx(1.0) or weights.sum() == 0)


def test_lightgbm_min_child_samples_grows_with_the_training_rows():
    make = MODELS["lightgbm"][1]
    assert make({**MODEL_PARAMS["lightgbm"], "min_child_share": 0.5, "min_child_floor": 2}, 8).min_child_samples == 4
    assert make(MODEL_PARAMS["lightgbm"], 100).min_child_samples == 20


@pytest.mark.parametrize("name", ["lightgbm", "logistic", "random_forest", "extra_trees"])
def test_importance_points_at_the_informative_feature(name):
    X, y = data()
    assert importance(MODELS[name][1](MODEL_PARAMS[name], len(y)).fit(X, y), 4).argmax() == 0


@pytest.mark.parametrize("name", list(MODELS))
def test_a_rows_probability_does_not_depend_on_the_batch_it_is_predicted_in(name):
    """One fit predicts several days at once and the batch shrinks when stocks leave the index: a row's probability must
    stay bitwise the same whatever else is in the batch, or later days would change earlier predictions."""
    rng = np.random.default_rng(0)
    scales = 10.0 ** rng.integers(-3, 3, 34)
    X = (rng.normal(size=(300, 34)) * scales).astype(np.float32)
    y = (X[:, 0] + rng.normal(0.0, 1.0, 300) > 0).astype(int)
    model = MODELS[name][1](MODEL_PARAMS[name], len(y)).fit(X, y)
    rows = (rng.normal(size=(200, 34)) * scales).astype(np.float32)
    alone = np.array([model.predict_proba(rows[i:i + 1])[0, 1] for i in range(len(rows))])
    np.testing.assert_array_equal(model.predict_proba(rows)[:, 1], alone)
