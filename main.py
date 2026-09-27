"""`python main.py download` baixa constituintes e preços; `features` calcula as features; `run --window N` roda o walk-forward."""
import argparse

import pandas as pd

from config.config import CONFIG
from engine import metrics
from engine.portfolio import Portfolio
from engine.project import Project


def report(project: Project, window: int) -> None:
    config = project.config
    market = project.features(with_dataset=False).market
    run = project.load_run(window)
    daily = Portfolio(config.threshold, config.cost_bps).backtest(run.predictions, market).daily
    table = pd.DataFrame({
        "Carteira (bruta)": metrics.performance_metrics(daily["gross"], daily["rf"], daily["invested"]),
        "Carteira (líquida)": metrics.performance_metrics(daily["net"], daily["rf"], daily["invested"]),
        "S&P 500 TR": metrics.performance_metrics(daily["bench"], daily["rf"]),
    })
    print(f"\nJanela {window} pregões · {daily.index.min().date()} → {daily.index.max().date()} · "
          f"limiar {config.threshold} · custo {config.cost_bps} bps")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    realized = run.predictions[run.predictions["date"].isin(daily["decision_date"])]
    print("Modelo:", {name: round(value, 4) for name, value in metrics.model_metrics(realized).items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML do S&P 500")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="baixa os constituintes (GitHub) e os preços (yfinance) mais recentes")
    commands.add_parser("features", help="calcula as features uma vez (servem para todas as janelas)")
    run_command = commands.add_parser("run", help="walk-forward para uma janela de treino e relatório")
    run_command.add_argument("--window", type=int, default=CONFIG.train_window_days,
                             help="pregões de treino (5 = 1 semana, 21 = 1 mês)")
    args = parser.parse_args()
    project = Project(CONFIG)
    if args.command == "download":
        failed = project.download()
        print(f"Constituintes e preços atualizados. Sem dados no Yahoo ({len(failed)}): {', '.join(failed)}")
    elif args.command == "features":
        dataset = project.build_features().dataset
        dates = dataset.index.get_level_values("date")
        print(f"Features: {len(dataset):,} linhas, {dates.min().date()} → {dates.max().date()}")
    else:
        run = project.run(args.window, progress=lambda done: print(f"\rtreinando {done:.0%}", end="", flush=True))
        print(f"\n{len(run.predictions):,} previsões em {run.info['runtime_minutes']} min")
        report(project, args.window)


if __name__ == "__main__":
    main()

"""
Pergunta: deveria ser 1 por ação ou 1 por dia ? olhar todas as ações não gera ruido desnecessário na hora de prever para 1 única ação ? como que isso funciona ? 
último: tunning de parametros:
"""