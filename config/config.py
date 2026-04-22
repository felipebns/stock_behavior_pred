CONFIG = {
    # Data parameters - 16 stocks
    # "tickers": [
    #     # Tech (5)
    #     "SPY", "AAPL", "MSFT", "GOOGL", "AMZN",
    #     # Financial (3)
    #     "JPM", "BAC", "GS",
    #     # Healthcare (3)
    #     "JNJ", "PFE", "UNH",
    #     # Energy (2)
    #     "XOM", "CVX",
    #     # Utilities (2)
    #     "NEE", "DUK",
    #     # Consumer (2)
    #     "WMT", "HD"
    # ],
    # testing with 5 stocks also... works better to reduce overfitting
    "tickers": ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"],
    "start_date": "2006-01-01",

    # Pipeline parameters
    "output_dir": "output",
    "test_size": 0.20,

    # Walk-Forward Validation parameters
    "wfv_train_window": 750,    # ~3 years of training data
    "wfv_test_window": 250,     # ~1 year of validation data

    # Backtesting parameters
    "initial_capital": 10000,
    "transaction_cost": 0.0005,  # 0.05% per trade
    "slippage": 0.0005,          # 0.05% slippage
    "annual_rf_rate": 0.05,      # 5% annual risk-free rate
    "probability_thresholds": [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60],  # Thresholds to test for position sizing
    "position_sizing": "probability_weighted",  # "equal_weight" or "probability_weighted"

    # Feature engineering parameters (placeholder for future extensions)
    "feature_profile": "classification_indicators",
    "lookback_period": 20,

    # ================== POSITION SELECTION ==================
    # NOTE: Backtest tests ALL 8 strategies and shows which performed best!
    #
    # Strategies tested automatically (no need to select one):
    #   "ensemble_smart"      - Consensus of 4 factors (RSI, momentum, mean reversion, vol)
    #   "momentum"            - Follows price trends
    #   "mean_reversion"      - Buys undervalued (price < SMA50)
    #   "volatility_weighted" - Reduces position size in high volatility
    #   "threshold"           - Baseline: probability threshold only
    #   "buy_and_hold"        - Benchmark: buy day 1, hold forever
    #   "fixed_income"        - Benchmark: 100% cash (conservative)
    #   "simple_reversal"     - Benchmark: SMA20 technical analysis only
    #
    # Position Selection (how many stocks to trade per day):
    #   "all"     - Trade all stocks beating threshold (diversified)
    #   "top_1"   - Trade only best stock of the day (concentrated)
    #   "top_3"   - Trade top 3 stocks (conservative)
    #   "top_5"   - Trade top 5 stocks (RECOMMENDED - balanced)
    #   "top_10"  - Trade top 10 stocks (diversified)
    # ============================================================================
    "position_selection": "top_5",           # How many stocks to select per day

    # ================== ALLOCATION LOGIC ==================
    # allocation_mode: How to deploy capital among selected stocks
    #   - "cash_allocation": Deploy only in selected, rest in CASH
    #   - "full_deployment": Deploy 100% equally among selected (RECOMMENDED)
    #
    # purchase_threshold: Global confidence gate (applied BEFORE normalization)
    #   - If best signal < threshold → 100% CASH (no position opened)
    #   - Default 0.50 = requires 50%+ confidence minimum
    # ============================================================================
    "allocation_mode": "full_deployment",
    "purchase_threshold": 0.50,

    # Model hyperparameters
    "model_params": {
        "LogisticRegression": {
            "random_state": 42,
            "max_iter": 1000,
            "class_weight": "balanced",
            "C": 1.0,
        },
        "SVC": {
            "kernel": "linear",
            "random_state": 42,
            "probability": True,
            "class_weight": "balanced",
            "C": 10.0,  # Increased regularization to improve convergence
            "max_iter": 2000,  # Standard limit; convergence improved by higher C
        },
        "RandomForest": {
            "n_estimators": 100,
            "random_state": 42,
            "class_weight": "balanced",
            "max_depth": None,
            "n_jobs": -1,
        },
        "Ensemble": {
            # Ensemble uses sub-models, so individual params not directly used
            "voting": "soft",
        }
    }
}