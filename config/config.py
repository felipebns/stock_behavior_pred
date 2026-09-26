from dataclasses import dataclass, field
from pathlib import Path

# Old ticker in the constituents file -> ticker Yahoo uses today for the same security.
# Only 1:1 renames visible in the constituents file itself (same-date swap, successor has history).
TICKER_ALIASES: dict[str, str] = {
    "HRS": "LHX", "TMK": "GL", "BHGE": "BKR", "JEC": "J", "CTL": "LUMN",
    "MYL": "VTRS", "WLTW": "WTW", "DISCA": "WBD", "BLL": "BALL", "ANTM": "ELV",
    "SYMC": "GEN", "NLOK": "GEN", "PKI": "RVTY", "RE": "EG", "ABC": "COR",
    "HCP": "DOC", "PEAK": "DOC", "FLT": "CPAY", "CBS": "PSKY", "VIAC": "PSKY",
    "PARA": "PSKY", "FI": "FISV", "MMC": "MRSH", "BK": "BNY", "UTX": "RTX",
    "ARNC": "HWM", "DWDP": "DD", "FB": "META", "LB": "BBWI", "SATS": "ECHO",
}

LGBM_PARAMS: dict = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "importance_type": "gain",
    "random_state": 42,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}


@dataclass(frozen=True)
class Config:
    data_in: Path = Path("data/in")
    data_out: Path = Path("data/out")
    constituents_file: str = "sp500_historical_constituents.csv"
    ff_futures_file: str = "ff_futures_daily.csv"
    gpr_file: str = "data_gpr_daily_recent.xls"
    prices_file: str = "prices.parquet"
    price_start: str = "2018-06-01"
    benchmark_ticker: str = "^SP500TR"
    risk_free_ticker: str = "^IRX"
    train_window_days: int = 252
    threshold: float = 0.55
    cost_bps: float = 5.0
    gpr_lag_days: int = 7
    macro_max_staleness_days: int = 5
    min_history_days: int = 63
    ff_horizon_months: int = 12
    ff_max_gap_days: int = 45
    ff_change_days: int = 21
    progress_every: int = 50
    lgbm_params: dict = field(default_factory=lambda: dict(LGBM_PARAMS))


CONFIG = Config()
