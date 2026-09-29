"""`python main.py download` baixa constituintes e preços; `features` calcula as features; `run --horizon H --window X --retrain K` roda o walk-forward."""
import argparse

import pandas as pd

from config.config import CONFIG
from engine import metrics
from engine.portfolio import Portfolio
from engine.project import Project, RunKey


def report(project: Project, key: RunKey) -> None:
    config = project.config
    features = project.features(with_dataset=False)
    run = project.load_run(key)
    daily = Portfolio(config.threshold, config.cost_bps).backtest(
        run.predictions, features.market, features.returns, key.horizon).daily
    table = pd.DataFrame({
        "Carteira (bruta)": metrics.performance_metrics(daily["gross"], daily["rf"], daily["invested"]),
        "Carteira (líquida)": metrics.performance_metrics(daily["net"], daily["rf"], daily["invested"]),
        "S&P 500 TR": metrics.performance_metrics(daily["bench"], daily["rf"]),
    })
    turnover = daily.loc[daily["rebalance"], "turnover"].mean()
    print(f"\n{key.label} · {daily.index.min().date()} → {daily.index.max().date()} · "
          f"limiar {config.threshold} · custo {config.cost_bps} bps · giro médio por troca {turnover:.0%}")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    realized = run.predictions[run.predictions["date"].isin(daily["decision_date"])]
    print("Modelo:", {name: round(value, 4) for name, value in metrics.model_metrics(realized).items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML do S&P 500")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="baixa os constituintes (GitHub) e os preços (yfinance) mais recentes")
    commands.add_parser("features", help="calcula as features uma vez (servem para todas as janelas)")
    run_command = commands.add_parser("run", help="walk-forward para uma janela de treino e relatório")
    run_command.add_argument("--horizon", type=int, default=CONFIG.horizon_days,
                             help="pregões entre decisões e do rótulo (1 = diário, 5 = semanal, 21 = mensal)")
    run_command.add_argument("--window", type=int, default=CONFIG.train_window_days,
                             help="pregões de exemplos de treino (5 = 1 semana, 21 = 1 mês)")
    run_command.add_argument("--retrain", type=int, default=CONFIG.retrain_every, help="retreina a cada k decisões")
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
        key = RunKey(args.horizon, args.window, args.retrain)
        run = project.run(key, progress=lambda done: print(f"\rtreinando {done:.0%}", end="", flush=True))
        print(f"\n{len(run.predictions):,} previsões em {run.info['runtime_minutes']} min")
        report(project, key)


if __name__ == "__main__":
    main()

"""
Pergunta: deveria ser 1 por ação ou 1 por dia ? olhar todas as ações não gera ruido desnecessário na hora de prever para 1 única ação ? como que isso funciona ? 
último: tunning de parametros
Modelo atual não tem poder preditivo, IC muito baixo
Testar vários modelos, deixa o streamlit escolher qual
Pq tirou PARA e BBT ? 
Preciso explicar paralelização depois. Ta bem demorado, como posso otimizar mais ? 
Como que correlação entre ações é util ?  
Custos estão sendo calculados certos ? gap esta muito grande e entre liquido e bruto (ex: janela = 5 limiar=0.67, gap é grande de mais, não faz sentido)
usar material de algotrading para validar!!
Deveria deixar variável horizonte de previsão ? 1 dia, 1 semana, etc. Como esta agora ? 
Escrever mais testes para não perder funcionamento
Interpretar os dados para extrair sinal, recortes, carteira por dia, etc... Quero conseguir ver o que foi previsto vs o que aconteceu, preciso entender o que é o gap de erro
buscar outra fonte de dado para ação, sem ser yahoo (binance ? olhar algo trading )
usar framework de backtesting já pronto ? (material)
material em algo trading
"""