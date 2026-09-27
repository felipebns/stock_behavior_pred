from main import report


def test_report_prints_portfolio_benchmark_and_model(synthetic_project, capsys):
    report(synthetic_project, 40)
    out = capsys.readouterr().out
    assert "Carteira (líquida)" in out and "S&P 500 TR" in out and "sharpe" in out and "auc" in out
