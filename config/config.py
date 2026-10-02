import copy
from dataclasses import dataclass, field
from pathlib import Path

# Pure renames only (same company, new name/ticker), publicly known, and a same-day 1:1 swap in the constituents file.
TICKER_ALIASES: dict[str, str] = {
    "FB": "META", "ANTM": "ELV", "ABC": "COR", "FLT": "CPAY", "PKI": "RVTY", "RE": "EG",
    "BLL": "BALL", "WLTW": "WTW", "LB": "BBWI", "SYMC": "GEN", "NLOK": "GEN", "HCP": "DOC",
    "PEAK": "DOC", "JEC": "J", "TMK": "GL", "BHGE": "BKR", "CTL": "LUMN", "HRS": "LHX",
    "UTX": "RTX", "DWDP": "DD", "ARNC": "HWM",
}

# Yahoo now uses these tickers for other securities: those old index members are discarded (treated as unpriced).
REASSIGNED_TICKERS: frozenset[str] = frozenset({"PARA", "BBT"})

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)

LGBM_PARAMS: dict = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "importance_type": "gain",
    "random_state": 42,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}


# Default parameters per model (tuning starts here). sklearn models get median imputation; the logistic also standardization.
MODEL_PARAMS: dict[str, dict] = {
    "lightgbm": {**LGBM_PARAMS, "min_child_share": 0.004, "min_child_floor": 20},
    "logistic": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 20, "max_features": "sqrt",
                      "random_state": 42, "n_jobs": 1},
    "extra_trees": {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 20, "max_features": "sqrt",
                    "random_state": 42, "n_jobs": 1},
    "naive_bayes": {},
}


@dataclass(frozen=True)
class Config:
    data_in: Path = Path("data/in")
    data_out: Path = Path("data/out")
    constituents_url: str = CONSTITUENTS_URL
    constituents_file: str = "sp500_historical_constituents.csv"
    prices_file: str = "prices.parquet"
    price_start: str = "2018-01-02"
    backtest_start: str = "2020-01-02"
    benchmark_ticker: str = "^SP500TR"
    risk_free_ticker: str = "^IRX"
    train_window_days: int = 21
    retrain_every: int = 1
    model: str = "lightgbm"
    scope: str = "pooled"
    per_stock_window_days: int = 252
    per_stock_retrain_every: int = 21
    per_stock_min_rows: int = 63
    selection_lookback_days: int = 252
    selection_step_days: int = 63
    threshold: float = 0.55
    cost_bps: float = 5.0
    complete_history_days: int = 252
    peer_count: int = 10
    peer_lookback_days: int = 63
    risk_free_max_staleness_days: int = 5
    n_jobs: int = -1
    model_params: dict = field(default_factory=lambda: copy.deepcopy(MODEL_PARAMS))


CONFIG = Config()
