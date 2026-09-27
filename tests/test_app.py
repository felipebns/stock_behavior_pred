from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest
from synthetic import make_market, synthetic_config

from engine import metrics
from engine.portfolio import Portfolio
from engine.prices import PriceData, clean_prices
from engine.project import Project, RunResults
from engine.universe import Universe

APP = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"
TABS = ["Visão geral", "Carteira por dia", "Recortes de período", "Janelas", "Modelo"]


def open_app(project: Project, monkeypatch, window: int = 40) -> AppTest:
    """The app with the project's own config, as it would be if CONFIG were edited to match."""
    monkeypatch.setattr("config.config.CONFIG", project.config)
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.number_input[0].set_value(window).run()
    return at


def test_dashboard_renders_a_saved_run(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    assert not at.exception
    assert [tab.label for tab in at.tabs] == TABS


def test_dashboard_survives_threshold_nobody_passes(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    at.sidebar.slider[0].set_value(0.8).run()
    assert not at.exception


def test_window_not_run_yet_offers_the_button(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch, window=21)
    assert not at.exception
    assert "Rodar janela de 21 pregões" in [button.label for button in at.sidebar.button]
    assert any("ainda não foi rodada" in info.value for info in at.info)


def test_missing_inputs_show_how_to_download(tmp_path, monkeypatch):
    monkeypatch.setattr("config.config.CONFIG", synthetic_config(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert at.error


def test_dashboard_shows_new_results_after_a_rerun(tmp_path, monkeypatch):
    raw = make_market()
    project = Project(synthetic_config(tmp_path))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    run = project.run(40)
    market = project.features(with_dataset=False).market
    first = open_app(project, monkeypatch)
    dates = sorted(run.predictions["date"].unique())
    cut = run.predictions[run.predictions["date"] <= dates[59]]
    project.save_run(RunResults(40, cut, run.importance.loc[:dates[59]], run.info))
    second = open_app(project, monkeypatch)
    n_full = len(Portfolio(0.55).backtest(run.predictions, market).daily)
    n_cut = len(Portfolio(0.55).backtest(cut, market).daily)
    assert n_full != n_cut
    assert any(f"{n_full} dias de carteira" in caption.value for caption in first.caption)
    assert any(f"{n_cut} dias de carteira" in caption.value for caption in second.caption)


def test_periods_tab_shows_model_metrics_per_period(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    model = next(table.value for table in at.dataframe if "AUC" in table.value.index)
    assert list(model.columns) == ["1 mês", "3 meses", "6 meses", "12 meses", "Tudo"]
    kpis = {metric.label: metric.value for metric in at.metric}
    assert model["Tudo"].to_dict() == {label: kpis[label] for label in model.index}
    predictions = synthetic_project.load_run(40).predictions
    market = synthetic_project.features(with_dataset=False).market
    daily = Portfolio(0.55).backtest(predictions, market).daily
    start = daily.loc[daily.index > daily.index.max() - pd.DateOffset(months=1), "decision_date"].min()
    month = predictions[(predictions["date"] >= start) & predictions["date"].isin(market["bench_fwd_ret"].dropna().index)]
    assert model.loc["AUC", "1 mês"] == f"{metrics.model_metrics(month)['auc']:.3f}"
