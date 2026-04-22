import json
import pandas as pd
import numpy as np
import concurrent.futures
from typing import Dict, Optional
from concurrent.futures import ProcessPoolExecutor

from services.backtesting.metrics_calculator import MetricsCalculator
from services.backtesting.allocation_manager import AllocationManager
from services.backtesting.return_calculator import ReturnCalculator
from services.plotting.plot_generator import PlotGenerator
from services.log.logger_config import get_logger
from services.log.reporters import BacktestReporter
from services.strategies import (
    MomentumStrategy, MeanReversionStrategy, 
    VolatilityWeightedStrategy, EnsembleSmartStrategy,
    ThresholdStrategy, BuyAndHoldStrategy, FixedIncomeStrategy,
    SimpleReversalStrategy
)

logger = get_logger()

class Backtest:
    """Backtesting engine for multi-asset trading strategies."""
    
    def __init__(self, test_df: pd.DataFrame, initial_capital: float = 10000,
                 transaction_cost: float = 0.0005, slippage: float = 0.0005,
                 annual_rf_rate: float = 0.05, position_sizing: str = "equal_weight",
                 position_selection: str = "top_5",
                 allocation_mode: str = "full_deployment",
                 purchase_threshold: float = 0.50,
                 threshold_workers: int = 3,
                 output_dir: str = "output"):
        self.test_df = test_df.copy()
        self.initial_capital = initial_capital
        self.annual_rf_rate = annual_rf_rate
        self.position_sizing = position_sizing
        self.position_selection = position_selection
        self.allocation_mode = allocation_mode
        self.purchase_threshold = purchase_threshold
        self.threshold_workers = threshold_workers
        
        # Initialize components
        self.allocation_manager = AllocationManager(
            test_df,
            position_selection=position_selection,
            position_sizing=position_sizing,
            allocation_mode=allocation_mode,
            purchase_threshold=purchase_threshold
        )
        self.return_calculator = ReturnCalculator(transaction_cost, slippage, annual_rf_rate)
        self.metrics_calculator = MetricsCalculator(initial_capital, annual_rf_rate)
        self.plot_generator = PlotGenerator(initial_capital)
        self.reporter = BacktestReporter(output_dir)
        
        # Initialize strategies
        self.strategies = {
            # Smart strategies (ML + technical)
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "volatility_weighted": VolatilityWeightedStrategy(),
            "ensemble_smart": EnsembleSmartStrategy(),
            # Simple/baseline strategies
            "threshold": ThresholdStrategy(),
            "buy_and_hold": BuyAndHoldStrategy(),
            "fixed_income": FixedIncomeStrategy(),
            "simple_reversal": SimpleReversalStrategy()
        }
    
    def run_threshold_strategies(self, model_probabilities: np.ndarray, 
                                 thresholds: list[float] = None) -> Dict[str, Dict]:
        """Run all strategies with multiple probability thresholds in parallel."""
        if thresholds is None:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
        
        results = {}
        
        # All strategies to test (4 intelligent + 1 baseline + 3 benchmarks)
        threshold_strategies = ["ensemble_smart", "momentum", "mean_reversion", "volatility_weighted", "threshold"]
        
        # Test all strategies with all thresholds in parallel
        with ProcessPoolExecutor(max_workers=self.threshold_workers) as executor:
            futures = {}
            
            # Submit all strategy + threshold combinations for parallel execution
            # Note: The separation of submission (this loop) and retrieval (the next loop)
            # is necessary for proper asynchronous execution. If both were in the same loop,
            # execution would block on each iteration waiting for the result, destroying parallelism.
            for strategy_name in threshold_strategies:
                for threshold in thresholds:
                    future = executor.submit(
                        self._run_strategy, model_probabilities, threshold, strategy_name
                    )
                    futures[future] = (strategy_name, threshold)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                strategy_name, threshold = futures[future]
                
                # Format: "Threshold Only 0.50" for baseline, "ML ensemble_smart 0.50" for others
                if strategy_name == "threshold":
                    results[f"Threshold Only {threshold:.2f}"] = future.result()
                else:
                    results[f"ML {strategy_name} {threshold:.2f}"] = future.result()
        
        # Add benchmarks
        results["Buy & Hold (benchmark)"] = \
            self._run_strategy(model_probabilities, None, "buy_and_hold")
        results[f"Fixed Income {self.annual_rf_rate:.2%} (benchmark)"] = \
            self._run_strategy(model_probabilities, None, "fixed_income")
        results["Simple Reversal SMA20 (benchmark)"] = \
            self._run_strategy(model_probabilities, None, "simple_reversal")
        
        return results
    
    def _run_strategy(self, probabilities: np.ndarray, threshold: float, 
                      strategy_name: str) -> Dict:
        """Run a specific strategy by name."""
        strategy = self.strategies.get(strategy_name)
        
        if strategy is None:
            # Fallback to threshold strategy
            positions = (probabilities > threshold).astype(int)
            logger.debug(f"[{strategy_name} @ {threshold:.2f}] Fallback threshold selection")
        else:
            # Get per-ticker data for strategy application
            positions = np.zeros(len(self.test_df))
            tickers = self.test_df["ticker"].unique()
            
            for ticker in tickers:
                mask = self.test_df["ticker"] == ticker
                ticker_df = self.test_df[mask].reset_index(drop=True)
                ticker_probs = probabilities[mask]
                
                # Apply strategy
                ticker_positions = strategy.apply(ticker_df, ticker_probs, threshold)
                positions[mask] = ticker_positions
        
        # Consolidated allocation pipeline (top-k + weighting + normalization)
        positions = self.allocation_manager.allocate(positions, probabilities, strategy_name)
        
        daily_returns = self.return_calculator.calculate(positions, self.test_df)
        metrics = self.metrics_calculator.calculate(daily_returns, positions, self.test_df)
        
        # Store position information by ticker for visualization
        metrics["positions_by_ticker"] = self._aggregate_positions_by_ticker(positions)
        
        # Log strategy execution via reporter
        self.reporter.log_strategy_execution(
            strategy_name, threshold, 
            metrics['total_return'], metrics['sharpe_ratio'], metrics['max_drawdown']
        )
        
        return metrics
    
    def _aggregate_positions_by_ticker(self, positions: np.ndarray) -> Dict[str, Dict]:
        """Aggregate positions by ticker to show final weight and historical metrics.
        
        Returns:
            Dict mapping ticker → {final_weight, avg_weight, selection_days, selection_rate}
        """
        positions_df = self.test_df.copy()
        positions_df["_position"] = positions
        
        result = {}
        last_date = positions_df["date"].max()
        
        for ticker in positions_df["ticker"].unique():
            ticker_data = positions_df[positions_df["ticker"] == ticker]
            total_days = len(ticker_data)
            active_days = np.sum(ticker_data["_position"] > 0)
            avg_weight = ticker_data["_position"].mean()
            
            # Get final weight (last day of backtest)
            last_day_data = ticker_data[ticker_data["date"] == last_date]
            final_weight = float(last_day_data["_position"].values[0]) if len(last_day_data) > 0 else 0.0
            
            result[ticker] = {
                "final_weight": final_weight,  # Weight to use in production
                "avg_position_weight": float(avg_weight),  # Historical average
                "days_selected": int(active_days),
                "total_days": int(total_days),
                "selection_rate": float(active_days / total_days) if total_days > 0 else 0.0
            }
        
        return result
    
    
    def save_summary(self, backtest_results: Dict, output_dir: str) -> None:
        """Save backtest summary to JSON using reporter."""
        self.reporter.save_summary(backtest_results)
    
    def plot_results(self, backtest_results: Dict, output_dir: str) -> None:
        """Generate all plots."""
        self.plot_generator.plot_all(backtest_results, self.test_df, output_dir)
