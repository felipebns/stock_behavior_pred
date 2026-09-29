from synthetic import RUNS

from main import report


def test_report_prints_the_run_portfolio_benchmark_and_model(synthetic_project, capsys):
    report(synthetic_project, RUNS[1])
    out = capsys.readouterr().out
    assert RUNS[1].label in out
    assert "Carteira (líquida)" in out and "S&P 500 TR" in out and "sharpe" in out and "auc" in out
