import dataclasses

import pandas as pd
import pytest

from config.config import CONFIG, LGBM_PARAMS, MODEL_PARAMS, TICKER_ALIASES


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.threshold = 0.6


def test_aliases_point_to_final_tickers():
    assert not set(TICKER_ALIASES.values()) & set(TICKER_ALIASES)


def test_aliases_are_only_well_known_pure_renames():
    assert TICKER_ALIASES["FB"] == "META" and TICKER_ALIASES["ANTM"] == "ELV"
    assert len(TICKER_ALIASES) == 21
    assert not {"DISCA", "MYL", "CBS", "VIAC", "PARA", "FI", "MMC", "BK", "SATS"} & set(TICKER_ALIASES)


def test_backtest_period_and_model_defaults():
    assert pd.Timestamp(CONFIG.price_start) < pd.Timestamp(CONFIG.backtest_start)
    assert CONFIG.benchmark_ticker == "^SP500TR" and CONFIG.risk_free_ticker == "^IRX"
    assert "min_child_samples" not in LGBM_PARAMS and LGBM_PARAMS["n_jobs"] == 1


def test_run_defaults():
    assert (CONFIG.model, CONFIG.scope, CONFIG.train_window_days, CONFIG.retrain_every) == ("lightgbm", "pooled", 21, 1)
    assert (CONFIG.per_stock_window_days, CONFIG.per_stock_retrain_every, CONFIG.per_stock_min_rows) == (252, 21, 63)
    assert (CONFIG.complete_history_days, CONFIG.selection_lookback_days, CONFIG.selection_step_days) == (252, 252, 63)
    assert set(CONFIG.model_params) == set(MODEL_PARAMS) == {"lightgbm", "logistic", "random_forest", "extra_trees", "naive_bayes"}
    assert CONFIG.model_params is not MODEL_PARAMS and CONFIG.model_params["lightgbm"]["min_child_floor"] == 20
