from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from engine.features import FEATURES
from engine.results import RunResults, save_results

APP = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"


def fake_results(n_days: int = 90, tickers=("AAA", "BBB", "CCC", "DDD", "EEE")) -> RunResults:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=n_days, name="date")
    predictions = pd.DataFrame([(d, t) for d in dates[:-1] for t in tickers], columns=["date", "ticker"])
    predictions["prob"] = rng.uniform(0.45, 0.65, len(predictions))
    predictions["fwd_ret"] = rng.normal(0.0005, 0.02, len(predictions))
    predictions["label"] = (predictions["fwd_ret"] > 0).astype(float)
    market = pd.DataFrame({
        "holding_date": pd.Series(dates, index=dates).shift(-1),
        "bench_fwd_ret": rng.normal(0.0004, 0.01, n_days),
        "rf_daily": 0.0002,
    }, index=pd.DatetimeIndex(dates, name="decision_date"))
    market.iloc[-2:, market.columns.get_loc("bench_fwd_ret")] = np.nan
    return RunResults(
        predictions=predictions,
        importance=pd.DataFrame(rng.dirichlet(np.ones(len(FEATURES)), n_days - 1), index=dates[:-1], columns=FEATURES),
        market=market,
        coverage=pd.DataFrame({"n_members": 5, "n_with_prices": 5, "n_eligible": 5}, index=dates),
        run_info={
            "config": {"train_window_days": 252}, "generated_at": "2026-09-26T12:00:00", "features": FEATURES,
            "data_quality": {"missing_tickers": ["GONE"], "n_extreme_returns": 0, "extreme_returns": [], "date_ranges": {}},
        },
    )


@pytest.fixture
def app(tmp_path, monkeypatch):
    save_results(fake_results(), tmp_path)
    monkeypatch.setenv("BACKTEST_RESULTS_DIR", str(tmp_path))
    return AppTest.from_file(str(APP), default_timeout=120)


def test_dashboard_renders_all_tabs(app):
    app.run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Visão geral", "Carteira por dia", "Recortes de período", "Modelo"]


def test_dashboard_survives_threshold_nobody_passes(app):
    app.run()
    app.sidebar.slider[0].set_value(0.8).run()
    assert not app.exception


def test_dashboard_without_results_shows_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_RESULTS_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert at.error


def test_dashboard_shows_new_results_after_a_rerun(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_RESULTS_DIR", str(tmp_path))
    save_results(fake_results(n_days=90), tmp_path)
    first = AppTest.from_file(str(APP), default_timeout=120).run()
    save_results(fake_results(n_days=60), tmp_path)
    second = AppTest.from_file(str(APP), default_timeout=120).run()
    assert any("88 dias" in caption.value for caption in first.caption)
    assert any("58 dias" in caption.value for caption in second.caption)


def test_turnover_is_labelled_as_buys_plus_sells(app):
    app.run()
    assert "Giro diário médio (compras + vendas)" in [metric.label for metric in app.metric]
