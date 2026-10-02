"""`python main.py download` baixa constituintes e preços; `features` calcula as features; `run --model M --scope S --window X --retrain K` roda um walk-forward; `sweep` roda uma grade de combinações."""
import argparse

import pandas as pd

from config.config import CONFIG
from engine import metrics
from engine.models import MODELS
from engine.portfolio import Portfolio
from engine.project import Project, RunKey
from engine.walk_forward import SCOPES


def report(project: Project, key: RunKey) -> None:
    config = project.config
    market = project.features(with_dataset=False).market
    run = project.load_run(key)
    daily = Portfolio(config.threshold, config.cost_bps).backtest(run.predictions, market).daily
    if daily.empty:
        print(f"\n{key.label} · Nenhum dia de carteira com retorno realizado neste run.")
        return
    table = pd.DataFrame({
        "Carteira (bruta)": metrics.performance_metrics(daily["gross"], daily["rf"], daily["invested"]),
        "Carteira (líquida)": metrics.performance_metrics(daily["net"], daily["rf"], daily["invested"]),
        "S&P 500 TR": metrics.performance_metrics(daily["bench"], daily["rf"]),
    })
    print(f"\n{key.label} · {daily.index.min().date()} → {daily.index.max().date()} · "
          f"limiar {config.threshold} · custo {config.cost_bps} bps · giro médio diário {daily['turnover'].mean():.0%}")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    realized = run.predictions[run.predictions["date"].isin(daily["decision_date"])]
    print("Modelo:", {name: round(value, 4) for name, value in metrics.model_metrics(realized).items()})


def defaults(config, scope: str) -> tuple[int, int]:
    """Training window and refit interval used when none is given: a per-stock model needs a much longer window."""
    if scope == "pooled":
        return config.train_window_days, config.retrain_every
    return config.per_stock_window_days, config.per_stock_retrain_every


def sweep(project: Project, models: list[str], scopes: list[str], windows: list[int] | None = None,
          retrains: list[int] | None = None) -> None:
    """Runs every combination that does not exist yet, one after the other, printing one line per combination.
    Without windows or retrains, each scope uses its own defaults."""
    existing = set(project.runs())
    keys = [RunKey(m, s, w, k) for m in models for s in scopes
            for w in (windows or [defaults(project.config, s)[0]]) for k in (retrains or [defaults(project.config, s)[1]])]
    for key in keys:
        if key in existing:
            print(f"{key.label}: já existe")
            continue
        try:
            run = project.run(key)
        except ValueError as error:
            print(f"{key.label}: erro: {error}")
            continue
        summary = metrics.model_metrics(run.predictions)
        print(f"{key.label}: {run.info['runtime_minutes']} min · AUC {summary['auc']:.4f} · IC {summary['ic']:+.4f}",
              flush=True)


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"precisa ser um inteiro ≥ 1 (recebido: {text})")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Backtest ML do S&P 500")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="baixa os constituintes (GitHub) e os preços (yfinance) mais recentes")
    commands.add_parser("features", help="calcula as features uma vez (servem para todas as janelas)")
    run_command = commands.add_parser("run", help="walk-forward para uma combinação e relatório")
    run_command.add_argument("--model", choices=list(MODELS), default=CONFIG.model)
    run_command.add_argument("--scope", choices=list(SCOPES), default=CONFIG.scope,
                             help="pooled = um modelo para todas as ações; per_stock = um modelo por ação")
    run_command.add_argument("--window", type=positive, default=None,
                             help="pregões de exemplos de treino (padrão: 21 em pooled, 252 em per_stock)")
    run_command.add_argument("--retrain", type=positive, default=None,
                             help="retreina a cada k pregões (padrão: 1 em pooled, 21 em per_stock)")
    sweep_command = commands.add_parser("sweep", help="roda todas as combinações que ainda não existem")
    sweep_command.add_argument("--models", nargs="+", choices=list(MODELS), default=[CONFIG.model])
    sweep_command.add_argument("--scopes", nargs="+", choices=list(SCOPES), default=[CONFIG.scope])
    sweep_command.add_argument("--windows", nargs="+", type=positive, default=None,
                               help="padrão: 21 em pooled, 252 em per_stock")
    sweep_command.add_argument("--retrain", nargs="+", type=positive, default=None,
                               help="padrão: 1 em pooled, 21 em per_stock")
    args = parser.parse_args(argv)
    project = Project(CONFIG)
    try:
        if args.command == "download":
            failed = project.download()
            print(f"Constituintes e preços atualizados. Sem dados no Yahoo ({len(failed)}): {', '.join(failed)}")
        elif args.command == "features":
            dataset = project.build_features().dataset
            dates = dataset.index.get_level_values("date")
            print(f"Features: {len(dataset):,} linhas, {dates.min().date()} → {dates.max().date()}")
        elif args.command == "sweep":
            sweep(project, args.models, args.scopes, args.windows, args.retrain)
        else:
            window, retrain = defaults(CONFIG, args.scope)
            key = RunKey(args.model, args.scope, args.window or window, args.retrain or retrain)
            run = project.run(key, progress=lambda done: print(f"\rtreinando {done:.0%}", end="", flush=True))
            print(f"\n{len(run.predictions):,} previsões em {run.info['runtime_minutes']} min")
            report(project, key)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"erro: {error}")


if __name__ == "__main__":
    main()

"""
último: tunning de parametros
"""