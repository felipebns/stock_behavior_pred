from multiprocessing import cpu_count
from datetime import datetime, timezone
from services.reproducibility import ReproducibilityManager

ReproducibilityManager.setup_reproducibility(seed=42)

from services.stock.stock import Stock
from services.pipeline import Pipeline
from services.algorithms.svc import SVCAlgorithm
from services.algorithms.random_forest import RandomForestAlgorithm
from services.algorithms.ensemble import EnsembleClassificationAlgorithm
from services.algorithms.logistic_regression import LogisticRegressionAlgorithm
from services.log.logger_config import setup_logging, get_logger
from services.log.reporters import ApplicationReporter
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
    model_names = ['Logistic Regression', 'Support Vector Classifier', 'Random Forest', 'Hybrid Ensemble']
    app_reporter.log_algorithms_initialization(model_names)
    models = [
        LogisticRegressionAlgorithm(**CONFIG["model_params"]["LogisticRegression"]),
        SVCAlgorithm(**CONFIG["model_params"]["SVC"]),
        RandomForestAlgorithm(**CONFIG["model_params"]["RandomForest"]),
        EnsembleClassificationAlgorithm(
            lr_params=CONFIG["model_params"]["LogisticRegression"],
            svc_params=CONFIG["model_params"]["SVC"],
            rf_params=CONFIG["model_params"]["RandomForest"],
        )
    ]

    # Log configuration
    app_reporter.log_configuration(CONFIG)
    
    # Calculate parallelization settings automatically based on CPU count
    n_algorithms = 4
    n_strategies = 8
    n_thresholds = len(CONFIG['probability_thresholds'])
    n_cpu = cpu_count()

    # Auto-calculate optimal workers to avoid CPU Over-subscription
    # For nested parallelization: algo_workers * fold_workers <= n_cpu
    algo_workers = min(n_algorithms, max(1, n_cpu // 2))
    fold_workers = max(1, n_cpu // algo_workers)

    parallelization = {
        "algorithm_selection": algo_workers, 
        "fold_evaluation": fold_workers,      
        "threshold_testing": max(1, n_cpu - 1),  # We can use all CPUs for the final backtest (leaving 1 for OS)
    }
    CONFIG["parallelization"] = parallelization
    
    # Log backtesting plan and parallelization
    app_reporter.log_backtesting_plan(n_strategies, n_thresholds)
    app_reporter.log_parallelization(n_algorithms, n_strategies, n_thresholds, parallelization, n_cpu)

    pipeline = Pipeline(
        stock=stock, 
        algorithms=models, 
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
"""Final purchase gate adjustments needs to work, strategies can change their threshold, it should be in a layer after that"""
"""Should be deterministic, always needs to converge"""
"""What is the difference between folder and package and library ?"""
"""Better test files, now is just random functions, need to centralize in a single testing framework, now is desorganized"""
"""Create output folder if it doesnt exist"""
"""Change README, this is now the quant fund engine"""
"""Volatily weight funciona funciona com o full alocation ? ele ta pulando essa etapa ? """
"""Try external backtest engine"""
"""Test other evaluation metrics, dont need to be just final backtesting results"""

"""Future testing"""

"""Test what is more effective, full deployment or cash fallbacks"""
"""Stocks selection, more than 5 causes overfitting, need to think of a way to get "similar" stocks to diversify"""
"""Test different algorithms (possible deep learning ? LSTM, CNN, GRU, XGBoost, etc.)"""
"""Explore more probabilities to chose best ML model, using strategy, etc. (not only IC)"""
"""Test more buy/sell strategies | Test more averages for mean reversion"""
"""Validation fine tunning, find right number of windows days..."""
"""Change strategies parameters to optimize"""
"""Models parameter tuning/more features ?"""
"""Compare again to paper, there could be usefull insights"""

"""Future improvements"""

"""Possible signal decomposotion"""
"""Eigen portfolios pca ?"""
"""Encapsulating framework"""