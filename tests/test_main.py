import pytest
from synthetic import RUNS, make_market, synthetic_config

import main
from engine.prices import PriceData, clean_prices
from engine.project import Project, RunKey
from engine.universe import Universe


def test_report_prints_the_run_portfolio_benchmark_and_model(synthetic_project, capsys):
    main.report(synthetic_project, RUNS[1])
    out = capsys.readouterr().out
    assert RUNS[1].label in out
    assert "Carteira (líquida)" in out and "S&P 500 TR" in out and "sharpe" in out and "auc" in out


@pytest.mark.parametrize("option", ["--window", "--retrain"])
def test_run_options_must_be_positive(option, capsys):
    with pytest.raises(SystemExit):
        main.main(["run", option, "0"])
    assert "≥ 1" in capsys.readouterr().err


def test_expected_errors_print_a_message_instead_of_a_traceback(synthetic_project, monkeypatch):
    monkeypatch.setattr(main, "CONFIG", synthetic_project.config)
    with pytest.raises(SystemExit, match="erro: .*máximo"):
        main.main(["run", "--window", "500"])


def test_report_of_a_run_without_any_realized_day_says_so(tmp_path, capsys):
    raw = make_market()
    dates = sorted(raw["prices"]["date"].unique())
    project = Project(synthetic_config(tmp_path, backtest_start=str(dates[-1].date())))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    project.run(RunKey("lightgbm", "pooled", 40))
    main.report(project, RunKey("lightgbm", "pooled", 40))
    assert "Nenhum dia de carteira" in capsys.readouterr().out


def test_run_options_choose_the_model_and_scope(capsys):
    with pytest.raises(SystemExit):
        main.main(["run", "--model", "xgboost"])
    assert "invalid choice" in capsys.readouterr().err


def test_sweep_runs_missing_combinations_and_skips_existing_ones(tmp_path, capsys):
    raw = make_market()
    project = Project(synthetic_config(tmp_path))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    main.sweep(project, ["naive_bayes"], ["pooled", "per_stock"], [40], [5])
    main.sweep(project, ["naive_bayes"], ["pooled"], [40, 400], [5])
    out = capsys.readouterr().out
    assert len(project.runs()) == 2
    assert out.count("já existe") == 1 and "máximo" in out


def test_sweep_without_windows_uses_each_scopes_defaults(tmp_path, capsys):
    raw = make_market()
    project = Project(synthetic_config(tmp_path, per_stock_window_days=30, per_stock_retrain_every=10))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    main.sweep(project, ["naive_bayes"], ["pooled", "per_stock"])
    assert project.runs() == [RunKey("naive_bayes", "per_stock", 30, 10), RunKey("naive_bayes", "pooled", 40, 1)]
