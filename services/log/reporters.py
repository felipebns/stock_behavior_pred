import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List
from services.log.logger_config import get_logger


class BaseReporter:
    """Base class for all reporters.
    
    Responsibilities:
    - Manage output directory
    - Provide common logging utilities
    - Handle JSON serialization
    """
    
    def __init__(self, output_dir: Path):
        """Initialize reporter with output directory.
        
        Args:
            output_dir: Path to output directory for results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
    
    def _save_json(self, data: Dict, filename: str) -> Path:
        """Save dictionary to JSON file.
        
        Args:
            data: Dictionary to save
            filename: Name of the file (without extension)
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath


class ApplicationReporter(BaseReporter):
    """Handles application initialization and configuration logging.
    
    Responsibilities:
    - Log application startup
    - Log data initialization (Stock, Algorithms)
    - Log configuration parameters
    - Log parallelization settings
    """
    
    def log_startup(self) -> None:
        """Log application startup."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING CLASSIFICATION PIPELINE WITH ENSEMBLE LEARNING")
        self.logger.info("=" * 80)
    
    def log_data_initialization(self, n_tickers: int, start_date: str) -> None:
        """Log stock data initialization.
        
        Args:
            n_tickers: Number of tickers being fetched
            start_date: Start date for historical data
        """
        self.logger.info(f"Fetching historical data for {n_tickers} tickers since {start_date}...")
        self.logger.info("✓ Stock data initialized")
    
    def log_algorithms_initialization(self, algorithm_names: List[str]) -> None:
        """Log ML algorithms initialization.
        
        Args:
            algorithm_names: List of algorithm names
        """
        self.logger.info(f"Initializing {len(algorithm_names)} ML algorithms...")
        self.logger.info(f"✓ Algorithms initialized: {algorithm_names}")
    
    def log_configuration(self, config: Dict) -> None:
        """Log pipeline configuration parameters.
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("\nConfiguration:")
        self.logger.info(f"  Tickers: {config['tickers']} ({len(config['tickers'])} stocks)")
        self.logger.info(f"  Position Sizing: {config['position_sizing']}")
        self.logger.info(f"  Position Selection: {config['position_selection']} (top N stocks per day)")
        self.logger.info(f"  Allocation Mode: {config['allocation_mode']}")
        self.logger.info(f"  Purchase Threshold: {config['purchase_threshold']:.2%}")
        self.logger.info(f"  Transaction Cost: {config['transaction_cost']:.4%}")
        self.logger.info(f"  Initial Capital: ${config['initial_capital']}")
        self.logger.info(f"  WFV Windows: train={config['wfv_train_window']}, test={config['wfv_test_window']}")
        self.logger.info(f"  Probability Thresholds: {config['probability_thresholds']}")
    
    def log_backtesting_plan(self, n_strategies: int, n_thresholds: int) -> None:
        """Log backtesting plan.
        
        Args:
            n_strategies: Number of strategies to test
            n_thresholds: Number of probability thresholds
        """
        self.logger.info(f"\n  Backtesting will test ALL {n_strategies} STRATEGIES with {n_thresholds} thresholds each")
        self.logger.info(f"  Total combinations: {n_strategies} strategies × {n_thresholds} thresholds = {n_strategies * n_thresholds} tests")
    
    def log_parallelization(self, n_algorithms: int, n_strategies: int, n_thresholds: int, 
                           parallelization: Dict, n_cpu: int) -> None:
        """Log parallelization settings.
        
        Args:
            n_algorithms: Number of algorithms
            n_strategies: Number of strategies
            n_thresholds: Number of thresholds
            parallelization: Parallelization configuration dictionary
            n_cpu: Number of CPU cores available
        """
        self.logger.info(f"\n  Parallelization (CPU cores available: {n_cpu}):")
        self.logger.info(f"    - Algorithm selection: {parallelization['algorithm_selection']} workers (for {n_algorithms} algorithms)")
        self.logger.info(f"    - Fold evaluation: {parallelization['fold_evaluation']} workers (for 14 WFV folds)")
        self.logger.info(f"    - Threshold testing: {parallelization['threshold_testing']} workers (for {n_strategies * n_thresholds} combinations)")
    
    def log_completion(self) -> None:
        """Log successful completion."""
        self.logger.info("All results saved to 'output/' and 'logs/' directories")
    
    def save_summary(self, *args, **kwargs) -> None:
        """Not applicable for ApplicationReporter."""
        pass


class PipelineReporter(BaseReporter):
    """Handles pipeline logging and result saving (NOT plotting).
    
    Responsibilities:
    - Save evaluation metrics to JSON files
    - Log pipeline events and results to logger
    - Manage output directory structure
    
    Note: All plotting (model metrics, backtest results) is handled by PlotGenerator.
    This is purely for logging and non-visualization result archival.
    """
    
    def save_metrics(self, metrics: Dict) -> None:
        """Save classification metrics to JSON file.
        
        Args:
            metrics: Dictionary of metrics from best model evaluation
            Note: Only saves best model metrics (performance optimization)
        """
        try:
            filepath = self._save_json(metrics, "classification_metrics")
            self.logger.info(f"✓ Metrics saved to {filepath}")
            self.logger.debug(f"Metrics content: {metrics}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}", exc_info=True)
            raise
    
    def log_phase_start(self, phase_name: str, description: str = "") -> None:
        """Log the start of a pipeline phase.
        
        Args:
            phase_name: Name of the phase
            description: Optional phase description
        """
        message = f"[Phase: {phase_name}] Starting"
        if description:
            message += f" - {description}"
        self.logger.info("=" * 80)
        self.logger.info(message)
        self.logger.info("=" * 80)
    
    def log_phase_end(self, phase_name: str, status: str = "SUCCESS") -> None:
        """Log the end of a pipeline phase.
        
        Args:
            phase_name: Name of the phase
            status: Status of completion (SUCCESS, FAILED, etc.)
        """
        self.logger.info(f"[Phase: {phase_name}] Completed - Status: {status}")
    
    def log_model_selection(self, model_name: str, score: float, metrics: Dict) -> None:
        """Log model selection results.
        
        Args:
            model_name: Name of selected model
            score: Model score
            metrics: Model metrics dictionary
        """
        self.logger.info(f"✓ Best model selected: {model_name} (Score: {score:.6f})")
        self.logger.debug(f"Model metrics: {metrics}")
    
    def log_backtest_results(self, strategy_results: Dict) -> None:
        """Log backtest results for all strategies.
        
        Args:
            strategy_results: Dictionary of backtest results per strategy
        """
        self.logger.info(f"Backtest completed for {len(strategy_results)} strategies")
        for strategy_name, results in strategy_results.items():
            self.logger.info(
                f"  {strategy_name}: Return={results.get('total_return', 0):.2%}, "
                f"Sharpe={results.get('sharpe_ratio', 0):.4f}, "
                f"MaxDD={results.get('max_drawdown', 0):.2%}"
            )
    
    def log_timing_summary(self, phase_times: Dict[str, float], total_time: float) -> None:
        """Log a summary of execution times for all phases.
        
        Args:
            phase_times: Dictionary mapping phase names to their durations
            total_time: Total pipeline execution time
        """
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("EXECUTION TIME SUMMARY")
        self.logger.info("="*80)
        
        for phase, duration in phase_times.items():
            phase_name = phase.replace("_", " ").title()
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            self.logger.info(f"  {phase_name:.<50} {duration:>7.2f}s ({percentage:>5.1f}%)")
        
        self.logger.info("-"*80)
        self.logger.info(f"  {'Total Execution Time':.<50} {total_time:>7.2f}s (100.0%)")
        self.logger.info("="*80)


class BacktestReporter(BaseReporter):
    """Handles backtest logging and result saving.
    
    Responsibilities:
    - Log strategy execution and results
    - Save backtest summary to JSON
    - Print formatted backtest summary
    """
    
    def log_strategy_execution(self, strategy_name: str, threshold: float, 
                              total_return: float, sharpe_ratio: float, 
                              max_drawdown: float) -> None:
        """Log execution of a single strategy.
        
        Args:
            strategy_name: Name of the strategy
            threshold: Probability threshold used (None for benchmarks)
            total_return: Total return of strategy
            sharpe_ratio: Sharpe ratio of strategy
            max_drawdown: Maximum drawdown of strategy
        """
        if threshold is None:
            threshold_str = "N/A"
        else:
            threshold_str = f"{threshold:.2f}"
        
        self.logger.debug(
            f"[{strategy_name} @ {threshold_str}] "
            f"Return={total_return:.2%}, Sharpe={sharpe_ratio:.4f}, MaxDD={max_drawdown:.2%}"
        )
    
    def save_summary(self, backtest_results: Dict) -> None:
        """Save backtest summary to JSON.
        
        Args:
            backtest_results: Dictionary of backtest results per strategy
        """
        summary = {}
        for strategy_name, metrics in backtest_results.items():
            summary[strategy_name] = {
                "total_return": metrics["total_return"],
                "annualized_return": metrics["annualized_return"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "active_hit_rate": metrics["active_hit_rate"],
                "final_equity": metrics["final_equity"]
            }
            if "positions_by_ticker" in metrics:
                summary[strategy_name]["positions_by_ticker"] = metrics["positions_by_ticker"]
            if "_metadata" in metrics:
                summary[strategy_name]["_metadata"] = metrics["_metadata"]
        
        filepath = self._save_json(summary, "backtest_summary")
        self.logger.info(f"✓ Backtest summary saved to {filepath}")
        self._print_summary(summary)
    
    def _print_summary(self, summary: Dict) -> None:
        """Print backtest summary to console.
        
        Args:
            summary: Dictionary of backtest results
        """
        print("\n" + "="*90)
        print("BACKTEST SUMMARY - STRATEGY COMPARISON")
        print("="*90)
        
        threshold_results = {k: v for k, v in summary.items() if "Threshold" in k}
        if threshold_results:
            print("\nMULTIPLE PROBABILITY THRESHOLDS:")
            print("-" * 90)
            for name in sorted(threshold_results.keys()):
                self._print_strategy_metrics(name, threshold_results[name])
        
        benchmark_results = {k: v for k, v in summary.items() if "Threshold" not in k}
        if benchmark_results:
            print("\n\nBENCHMARK STRATEGIES:")
            print("-" * 90)
            for name, metrics in benchmark_results.items():
                self._print_strategy_metrics(name, metrics)
        
        print("\n" + "="*90 + "\n")
    
    def _print_strategy_metrics(self, name: str, metrics: Dict) -> None:
        """Print metrics for a single strategy.
        
        Args:
            name: Strategy name
            metrics: Dictionary of metrics for the strategy
        """
        print(f"\n{name}:")
        print(f"  Total Return:     {metrics['total_return']*100:>8.2f}%")
        print(f"  Annualized Ret:   {metrics['annualized_return']*100:>8.2f}%")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>8.4f}")
        print(f"  Max Drawdown:     {metrics['max_drawdown']*100:>8.2f}%")
        print(f"  Active Hit Rate:  {metrics['active_hit_rate']*100:>8.2f}%")
        print(f"  Final Equity:     ${metrics['final_equity']:>12,.2f}")
