# QuantFund Engine

ML backtesting engine focused on model selection, strategy evaluation, portfolio allocation, and report generation. 

## Quick Start

```bash
python -m venv venv
pip install -r requirements.txt
python main.py
```

---

## What This Engine Does

End-to-end pipeline for daily, long-only portfolio research and backtesting:

1. **Data preparation and feature engineering**
2. **Model selection** via walk-forward validation
3. **Model training** on the best configuration
4. **Backtesting** with multiple strategies and probability thresholds
5. **Reporting and plotting** of results

This is designed to be modular so you can swap models, strategies, assets, and allocation policies without changing the pipeline.

---

## Pipeline Phases

### Phase 0: Data Preparation
- Fetch OHLCV data
- Engineer technical features (momentum, volatility, mean reversion, volume)
- Build a binary target for next-day direction
- Temporal split into train/test

### Phase 1: Model Selection (Walk-Forward Validation)
- Rolling window validation with fixed train/test sizes
- Selection metric: Information Coefficient (IC) on out-of-sample returns
- Best model chosen by robustness score (mean IC minus IC volatility)

### Phase 2: Full Training
- Train the selected model on the full training window

### Phase 3: Backtesting
- Run the trained model on the test period
- Evaluate multiple strategies and probability thresholds
- Apply transaction costs, slippage, and capital allocation rules

### Phase 4: Reporting and Plotting
- Persist model comparison and backtest summary JSON
- Generate equity curves and strategy comparison plots

---

## Core Components

### Models
Implement the `Algorithm` interface and provide:
- `fit`, `predict`, `predict_proba`
- `name` and `feature_profile`

Models live under [services/algorithms](services/algorithms).

### Strategies
Strategies modify the raw model signal (probability + threshold) with market conditions.
They live under [services/strategies](services/strategies) and implement `BaseStrategy.apply`.

### Allocation
Allocation is handled by a three-step pipeline:
1. Top-K selection per day
2. Probability weighting (optional)
3. Normalization with allocation mode and confidence gate

See [services/backtesting/allocation_manager.py](services/backtesting/allocation_manager.py) and
[services/backtesting/position_normalizer.py](services/backtesting/position_normalizer.py).

### Backtesting
The backtesting engine runs strategy/threshold grids and produces:
- Daily returns and equity curves
- Total return, Sharpe, max drawdown, hit rate
- Per-ticker position summaries

Core logic: [services/backtesting/backtest.py](services/backtesting/backtest.py),
[services/backtesting/metrics_calculator.py](services/backtesting/metrics_calculator.py),
[services/backtesting/return_calculator.py](services/backtesting/return_calculator.py).

### Plotting and Reporting
- JSON outputs in `output/`
- Plot generation for model and strategy comparisons

Plotting modules: [services/plotting](services/plotting).

---

## Configuration

Core parameters are defined in `config/config.py` and read by `main.py`.
Common tuning points:

- **Asset universe** (tickers)
- **Model list** and hyperparameters
- **Probability thresholds** for strategy grids
- **Allocation mode** (`full_deployment` or `cash_allocation`)
- **Top-K selection** (`top_1`, `top_5`, `all`)
- **Costs** (transaction cost, slippage)

---

## Outputs

- `output/models_comparison.json`
- `output/backtest_summary.json`
- Plots in `output/`

---

## Notes

- Engine is long-only.
- Focus is research and evaluation, not live execution.
- The application layer (portfolio deployment, UI, brokers) will be added separately.