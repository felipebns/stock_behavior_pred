import pandas as pd
import numpy as np
from typing import Dict
from .model_metrics import ModelMetricsPlotter
from .strategy_plots import StrategyPlotter
from .threshold_plots import ThresholdPlotter
from .portfolio_plots import PortfolioPlotter

class PlotGenerator:
    """Generates all visualization plots: backtest results and model comparison.
    
    Responsibilities:
    - All matplotlib-based visualizations for the pipeline
    - Backtest equity curves, drawdowns, and strategy comparisons
    - Model selection metrics and comparison charts
    - Portfolio allocation and stock selection summaries
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.model_metrics_plotter = ModelMetricsPlotter()
        self.strategy_plotter = StrategyPlotter(initial_capital)
        self.threshold_plotter = ThresholdPlotter(initial_capital)
        self.portfolio_plotter = PortfolioPlotter(initial_capital)

    def _get_best_configs(self, results: Dict) -> Dict:
        """Helper to extract ONLY the best threshold for each ML Strategy + Threshold Only.
        Avoids repetition and clutter in the code and plots."""
        selected = {}
        ml_by_strategy = {}
        threshold_only_dict = {}
        
        # 1. Classify strategies
        for name, data in results.items():
            if name.startswith("ML "):
                parts = name.split()
                strategy_name = " ".join(parts[1:-1])  # Base name
                threshold = float(parts[-1])
                ml_by_strategy.setdefault(strategy_name, {})[threshold] = (name, data)
            elif name.startswith("Threshold Only "):
                threshold = float(name.split()[-1])
                threshold_only_dict[threshold] = (name, data)
            else:
                # Keep plain benchmarks
                if name not in ["Threshold Only (baseline)"]:
                    selected[name] = data
                    
        # 2. Add BEST ML strategy configurations
        for strategy_name, threshold_dict in ml_by_strategy.items():
            best_tuple = max(threshold_dict.items(), key=lambda x: x[1][1].get("total_return", -999))
            selected[strategy_name.replace("_", " ").title()] = best_tuple[1][1]
            
        # 3. Add BEST Threshold Only baseline
        if threshold_only_dict:
            best_base = max(threshold_only_dict.items(), key=lambda x: x[1][1].get("total_return", -999))
            selected["Threshold Only (baseline)"] = best_base[1][1]
            
        return selected

    def plot_model_metrics_comparison(self, model_results: Dict[str, Dict], output_dir: str) -> None:
        self.model_metrics_plotter.plot_model_metrics_comparison(model_results, output_dir)
        
    def plot_all(self, backtest_results: Dict, test_df: pd.DataFrame, output_dir: str) -> None:
        """Generate all backtest plots."""
        dates = np.sort(test_df["date"].unique())
        
        ml_strategies = {k: v for k, v in backtest_results.items() if k.startswith("ML ")}
        benchmark_strategies = {k: v for k, v in backtest_results.items() if "benchmark" in k.lower()}
        baseline_strategies = {k: v for k, v in backtest_results.items() 
                             if k.startswith("Threshold") or k.startswith("Buy & Hold") or k.startswith("Fixed")}
        
        self.strategy_plotter.plot_ml_strategies_by_threshold(ml_strategies, dates, output_dir)
        self.strategy_plotter.plot_all_strategies_comparison(backtest_results, dates, output_dir)
        self.strategy_plotter.plot_strategy_metrics_comparison(backtest_results, output_dir)
        self.threshold_plotter.plot_threshold_effect(ml_strategies, output_dir)
        self.threshold_plotter.plot_threshold_only_analysis(backtest_results, dates, output_dir)
        self.strategy_plotter.plot_drawdown_analysis(backtest_results, dates, output_dir)
        self.portfolio_plotter.plot_stock_selection_summary(backtest_results, test_df, output_dir)
        self.portfolio_plotter.plot_portfolio_allocation_summary(backtest_results, test_df, output_dir)
