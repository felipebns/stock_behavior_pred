import dataclasses

import pytest

from config.config import CONFIG, TICKER_ALIASES


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.threshold = 0.6


def test_aliases_point_to_final_tickers():
    assert not set(TICKER_ALIASES.values()) & set(TICKER_ALIASES)


def test_aliases_include_renames_whose_old_ticker_was_recycled():
    assert TICKER_ALIASES["FB"] == "META"
    assert TICKER_ALIASES["LB"] == "BBWI"
    assert TICKER_ALIASES["SATS"] == "ECHO"


def test_lookahead_sensitive_defaults():
    assert CONFIG.gpr_lag_days == 7
    assert CONFIG.benchmark_ticker == "^SP500TR"
    assert CONFIG.risk_free_ticker == "^IRX"
