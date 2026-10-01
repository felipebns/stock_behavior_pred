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
    project = Project(synthetic_config(tmp_path, backtest_start=str(dates[-2].date())))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    project.run(RunKey(40))
    main.report(project, RunKey(40))
    assert "Nenhum dia de carteira" in capsys.readouterr().out
