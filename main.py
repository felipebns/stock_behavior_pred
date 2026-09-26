"""`python main.py download` baixa os preços; `python main.py run` roda features, walk-forward e backtest."""
import argparse
import dataclasses
import platform
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from config.config import CONFIG, TICKER_ALIASES, Config
from engine import metrics
from engine.data import prices as px
from engine.data import universe
from engine.pipeline import load_inputs, run_pipeline
from engine.portfolio import run_backtest
from engine.results import RunResults, save_results

PACKAGES = ("pandas", "numpy", "lightgbm", "scikit-learn", "yfinance", "streamlit")


def download(config: Config) -> None:
    snapshots = universe.load_constituents(config.data_in / config.constituents_file, TICKER_ALIASES)
    tickers = universe.tickers_since(snapshots, config.price_start) + [config.benchmark_ticker, config.risk_free_ticker]
    prices, failed = px.download_prices(tickers, start=config.price_start)
    path = config.data_in / config.prices_file
    px.save_prices(prices, path)
    print(f"{prices['ticker'].nunique()} tickers salvos em {path} "
          f"({prices['date'].min().date()} → {prices['date'].max().date()})")
    print(f"Sem dados no Yahoo ({len(failed)}): {', '.join(failed)}")


def output_dir(config: Config, start: str | None, end: str | None) -> Path:
    """A run limited by --start/--end goes to its own folder so it never replaces the full results the app reads."""
    return config.data_out if start is None and end is None else config.data_out / "partial"


def run(config: Config, start: str | None, end: str | None) -> None:
    started = time.perf_counter()
    results = run_pipeline(load_inputs(config), config, start=start, end=end)
    results.run_info.update({
        "config": dataclasses.asdict(config),
        "runtime_minutes": round((time.perf_counter() - started) / 60.0, 2),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "versions": {package: version(package) for package in PACKAGES},
    })
    out = output_dir(config, start, end)
    save_results(results, out)
    print(f"Resultados em {out}/ ({results.run_info['runtime_minutes']} min)")
    report(results, config)


def report(results: RunResults, config: Config) -> None:
    backtest = run_backtest(results.predictions, results.market, config.threshold, config.cost_bps)
    daily = backtest.daily
    invested = daily["invested"]
    table = pd.DataFrame({
        "Carteira (bruta)": metrics.performance_metrics(daily["gross"], daily["rf"], daily["bench"], invested),
        "Carteira (líquida)": metrics.performance_metrics(daily["net"], daily["rf"], daily["bench"], invested),
        "S&P 500 TR": metrics.performance_metrics(daily["bench"], daily["rf"]),
    })
    print(f"\nBacktest {daily.index.min().date()} → {daily.index.max().date()} · {len(daily)} dias · "
          f"limiar {config.threshold} · custo {config.cost_bps} bps")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"Giro médio (compras + vendas): {daily['turnover'].mean():.1%}/dia · custo total: {daily['cost'].sum():.2%}")
    print(f"Dias investido: {invested.mean():.1%} · posições por dia investido: "
          f"{daily.loc[invested, 'n_positions'].mean():.1f} · hit ratio das posições: "
          f"{metrics.position_hit_ratio(backtest.positions):.1%}")
    print("Modelo:", {name: round(value, 4) for name, value in metrics.model_metrics(results.predictions).items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML do S&P 500")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="baixa preços sem ajuste de dividendos (yfinance)")
    run_command = commands.add_parser("run", help="features, walk-forward e backtest; grava data/out")
    run_command.add_argument("--start", help="primeira data de decisão (YYYY-MM-DD); rodadas parciais gravam em data/out/partial")
    run_command.add_argument("--end", help="última data de decisão (YYYY-MM-DD)")
    args = parser.parse_args()
    if args.command == "download":
        download(CONFIG)
    else:
        run(CONFIG, args.start, args.end)


if __name__ == "__main__":
    main()


"""
Reorganizar em classes
Simplificar
garantir que não tem survivorship bias, somente as ações do instante
"""