from multiprocessing import cpu_count
from datetime import datetime, timezone
from engine.reproducibility import ReproducibilityManager

ReproducibilityManager.setup_reproducibility(seed=42)

from engine.stock.stock import Stock
from engine.pipeline import Pipeline
from engine.algorithms.lstm import LSTMAlgorithm
from engine.log.logger_config import setup_logging, get_logger
from engine.log.reporters import ApplicationReporter
from config.config import CONFIG

if __name__ == "__main__":
    # Setup logging system
    logger = setup_logging(log_dir="logs")
    app_reporter = ApplicationReporter(output_dir=CONFIG["output_dir"])

    # Log startup
    app_reporter.log_startup()
    
    end_date = datetime.now(timezone.utc).date().isoformat()
    
    # Log data initialization
    app_reporter.log_data_initialization(len(CONFIG['tickers']), CONFIG['start_date'])
    stock = Stock(tickers=CONFIG["tickers"], start=CONFIG["start_date"], end=end_date)

    # Log algorithms initialization
    model_names = ["LSTM"]
    app_reporter.log_algorithms_initialization(model_names)
    model = LSTMAlgorithm(
        lookback=CONFIG["lookback_period"],
        **CONFIG["lstm_params"],
    )

    # Log configuration
    app_reporter.log_configuration(CONFIG)
    
    # Calculate parallelization settings automatically based on CPU count
    n_strategies = 8
    n_thresholds = len(CONFIG['probability_thresholds'])
    n_cpu = cpu_count()

    parallelization = {
        "fold_evaluation": max(1, min(4, n_cpu)),
        "threshold_testing": max(1, n_cpu - 1),  # We can use all CPUs for the final backtest (leaving 1 for OS)
    }
    CONFIG["parallelization"] = parallelization
    
    # Log backtesting plan and parallelization
    app_reporter.log_backtesting_plan(n_strategies, n_thresholds)
    app_reporter.log_parallelization(n_strategies, n_thresholds, parallelization, n_cpu)

    pipeline = Pipeline(
        stock=stock, 
        algorithm=model, 
        output_dir=CONFIG["output_dir"],
        test_size=CONFIG["test_size"],
        wfv_train_window=CONFIG["wfv_train_window"],
        wfv_test_window=CONFIG["wfv_test_window"],
        initial_capital=CONFIG["initial_capital"],
        transaction_cost=CONFIG["transaction_cost"],
        slippage=CONFIG["slippage"],
        annual_rf_rate=CONFIG["annual_rf_rate"],
        probability_thresholds=CONFIG["probability_thresholds"],
        position_sizing=CONFIG["position_sizing"],
        position_selection=CONFIG["position_selection"],
        allocation_mode=CONFIG["allocation_mode"],
        purchase_threshold=CONFIG["purchase_threshold"],
        parallelization=CONFIG["parallelization"],
    )

    try:
        results = pipeline.run(save=True)
        app_reporter.log_completion()
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise

"""TODOs"""

"""Rewrite the code to my own vision"""
"""Should be deterministic, always needs to converge"""
"""Final purchase gate adjustments needs to work, strategies can change their threshold, it should be in a layer after that"""
"""Better test files, now is just random functions, need to centralize in a single testing framework, now is desorganized"""
"""Volatily weight funciona funciona com o full alocation ? ele ta pulando essa etapa ? """
"""Test other evaluation metrics, dont need to be just final backtesting results"""
"""Separamento de tickers correto ? estou analizando df corretamente com as features separadas ?"""
"""Log de linhas faltando no stock"""

"""Future testing"""

"""Test what is more effective, full deployment or cash fallbacks"""
"""Stocks selection, more than 5 causes overfitting, need to think of a way to get "similar" stocks to diversify"""
"""Explore more probabilities to chose best ML model, using strategy, etc. (not only IC)"""
"""Test more buy/sell strategies | Test more averages for mean reversion"""
"""Validation fine tunning, find right number of windows days..."""
"""Change strategies parameters to optimize"""
"""Models parameter tuning/more features ?"""

"""Future improvements"""

"""Possible signal decomposotion"""
"""Eigen portfolios pca ?"""
"""Encapsulating framework, finish .toml, put name, license, email, etc"""
