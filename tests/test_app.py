from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from synthetic import RUNS, make_market, synthetic_config

from engine import metrics
from engine.portfolio import Portfolio
from engine.prices import PriceData, clean_prices
from engine.project import Project, RunKey, RunResults
from engine.universe import Universe

APP = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"
TABS = ["Visão geral", "Previsto vs realizado", "Carteira por dia", "Recortes de período", "Runs", "Walk-forward", "Modelo"]


def open_app(project: Project, monkeypatch, key: RunKey = RUNS[0]) -> AppTest:
    """The app with the project's own config and the run key chosen in the sidebar."""
    monkeypatch.setattr("config.config.CONFIG", project.config)
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.selectbox(key="model").set_value(key.model)
    at.radio(key="scope").set_value(key.scope)
    at.run()
    at.number_input(key=f"window_{key.scope}").set_value(key.window)
    at.number_input(key=f"retrain_{key.scope}").set_value(key.retrain_every)
    return at.run()


def backtest(project: Project, key: RunKey, predictions: pd.DataFrame | None = None):
    market = project.features(with_dataset=False).market
    predictions = project.load_run(key).predictions if predictions is None else predictions
    return Portfolio(0.55).backtest(predictions, market)


@pytest.mark.parametrize("key", RUNS)
def test_dashboard_renders_a_saved_run(synthetic_project, monkeypatch, key):
    at = open_app(synthetic_project, monkeypatch, key)
    assert not at.exception
    assert [tab.label for tab in at.tabs] == TABS


def test_dashboard_survives_threshold_nobody_passes(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch, RUNS[1])
    at.sidebar.slider[0].set_value(0.8).run()
    assert not at.exception


def test_run_not_made_yet_offers_the_button(synthetic_project, monkeypatch):
    key = RunKey("lightgbm", "pooled", 21)
    at = open_app(synthetic_project, monkeypatch, key)
    assert not at.exception
    assert f"Rodar {key.label}" in [button.label for button in at.sidebar.button]
    assert any("ainda não existe" in info.value for info in at.info)


def test_missing_inputs_show_how_to_download(tmp_path, monkeypatch):
    monkeypatch.setattr("config.config.CONFIG", synthetic_config(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert at.error


def test_dashboard_shows_new_results_after_a_rerun(tmp_path, monkeypatch):
    raw = make_market()
    project = Project(synthetic_config(tmp_path))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    key = RUNS[0]
    run = project.run(key)
    first = open_app(project, monkeypatch, key)
    dates = sorted(run.predictions["date"].unique())
    cut = run.predictions[run.predictions["date"] <= dates[59]]
    project.save_run(RunResults(key, cut, run.importance.loc[:dates[59]], run.info))
    second = open_app(project, monkeypatch, key)
    n_full, n_cut = len(backtest(project, key, run.predictions).daily), len(backtest(project, key, cut).daily)
    assert n_full != n_cut
    assert any(f"{n_full} dias de carteira" in caption.value for caption in first.caption)
    assert any(f"{n_cut} dias de carteira" in caption.value for caption in second.caption)


def test_periods_tab_shows_model_metrics_per_period(synthetic_project, monkeypatch):
    key = RUNS[0]
    at = open_app(synthetic_project, monkeypatch, key)
    model = next(table.value for table in at.dataframe if "AUC" in table.value.index)
    assert list(model.columns) == ["1 mês", "3 meses", "6 meses", "12 meses", "Tudo"]
    kpis = {metric.label: metric.value for metric in at.metric}
    assert model["Tudo"].to_dict() == {label: kpis[label] for label in model.index}
    predictions = synthetic_project.load_run(key).predictions
    daily = backtest(synthetic_project, key).daily
    start = daily.loc[daily.index > daily.index.max() - pd.DateOffset(months=1), "decision_date"].min()
    month = predictions[predictions["date"] >= start].dropna(subset=["label"])
    assert model.loc["AUC", "1 mês"] == f"{metrics.model_metrics(month)['auc']:.3f}"


def test_signal_tab_shows_the_decile_table(synthetic_project, monkeypatch):
    key = RUNS[1]
    at = open_app(synthetic_project, monkeypatch, key)
    table = next(frame.value for frame in at.dataframe if "Subiu" in frame.value.columns)
    expected = metrics.deciles(synthetic_project.load_run(key).predictions)
    assert list(table.index) == list(expected.index)
    assert table["Prob. média"].to_numpy() == pytest.approx(expected["prob"].to_numpy())


def test_day_tab_shows_what_was_bought_and_what_happened(synthetic_project, monkeypatch):
    key = RUNS[1]
    result = backtest(synthetic_project, key)
    days = list(result.daily.index)
    at = open_app(synthetic_project, monkeypatch, key)
    at.selectbox(key="day").set_value(days.index(result.positions["holding_date"].iloc[-1])).run()
    held = next(frame.value for frame in at.dataframe if "Subiu?" in frame.value.columns)
    assert {"Prob. prevista", "Retorno realizado", "Contribuição"} <= set(held.columns)
    assert any("Soma das contribuições" in info.value for info in at.info)


def test_runs_tab_compares_every_run(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    table = next(frame.value for frame in at.dataframe if "Giro médio diário" in frame.value.columns)
    assert list(table.index) == [key.label for key in RUNS]


def test_run_without_any_realized_day_shows_a_warning(tmp_path, monkeypatch):
    raw = make_market()
    dates = sorted(raw["prices"]["date"].unique())
    project = Project(synthetic_config(tmp_path, backtest_start=str(dates[-1].date())))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    project.run(RunKey("lightgbm", "pooled", 40))
    at = open_app(project, monkeypatch, RunKey("lightgbm", "pooled", 40))
    assert not at.exception
    assert any("Nenhum dia de carteira" in warning.value for warning in at.warning)


def test_walk_forward_tab_shows_the_choices(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    choices = next(frame.value for frame in at.dataframe if "Limiar" in frame.value.columns and "Início" in frame.value.columns)
    assert {"Início", "Run", "Limiar", "Sharpe no período anterior", "Dias"} <= set(choices.columns)
    assert set(choices["Run"]) <= {key.label for key in RUNS}
