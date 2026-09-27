"""What the CLI and the app do: download inputs, build features once, run one walk-forward per window, load results."""
import dataclasses
import io
import json
import platform
import shutil
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from config.config import REASSIGNED_TICKERS, TICKER_ALIASES, Config
from engine.dataset import build_dataset
from engine.features import FEATURES, FeatureBuilder
from engine.prices import PriceData, daily_total_return, forward_open_return
from engine.universe import Universe
from engine.walk_forward import WalkForward

PACKAGES = ("pandas", "numpy", "lightgbm", "scikit-learn", "joblib", "yfinance", "streamlit")
# Config fields that decide what is saved: features or runs made with other values are not reused.
FEATURE_SETTINGS = ("price_start", "benchmark_ticker", "risk_free_ticker", "min_history_days", "peer_count",
                    "peer_lookback_days", "risk_free_max_staleness_days")
RUN_SETTINGS = ("backtest_start", "min_child_share", "min_child_floor", "lgbm_params")


@dataclass
class FeatureSet:
    dataset: pd.DataFrame | None
    market: pd.DataFrame
    coverage: pd.DataFrame
    quality: dict

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.market.index)


@dataclass
class RunResults:
    window: int
    predictions: pd.DataFrame
    importance: pd.DataFrame
    info: dict


class Project:
    def __init__(self, config: Config):
        self.config = config
        self.features_dir = config.data_out / "features"
        self.runs_dir = config.data_out / "runs"

    def download(self, price_downloader=None, fetch=None) -> list[str]:
        """Latest constituents and prices, saved only if Yahoo returned the indices and every current member;
        features and runs are deleted because they no longer match the data."""
        c = self.config
        content = Universe.fetch(c.constituents_url, fetch)
        universe = Universe.from_csv(io.BytesIO(content), TICKER_ALIASES)
        index_tickers = [c.benchmark_ticker, c.risk_free_ticker]
        prices, failed = PriceData.download(universe.tickers_since(c.price_start) + index_tickers, c.price_start,
                                            downloader=price_downloader)
        required = index_tickers + universe.members_on(universe.snapshots["date"].max())
        lost = [t for t in required if t in failed]
        if lost:
            raise RuntimeError(f"o Yahoo não devolveu {', '.join(lost)}; constituintes e preços antigos foram mantidos")
        c.data_in.mkdir(parents=True, exist_ok=True)
        (c.data_in / c.constituents_file).write_bytes(content)
        prices.save(c.data_in / c.prices_file)
        self._clear(self.features_dir, self.runs_dir)
        return failed

    def build_features(self, universe: Universe | None = None, prices: PriceData | None = None) -> FeatureSet:
        """Features do not depend on the training window: built once, shared by every run; old runs are deleted."""
        if universe is None or prices is None:
            universe, prices = self._load_inputs()
        features = self._make_features(universe, prices)
        features.quality.update(settings=self._settings(FEATURE_SETTINGS), built_at=time.time_ns())
        self._clear(self.features_dir, self.runs_dir)
        self.features_dir.mkdir(parents=True)
        features.dataset.to_parquet(self.features_dir / "dataset.parquet")
        features.market.to_parquet(self.features_dir / "market.parquet")
        features.coverage.to_parquet(self.features_dir / "coverage.parquet")
        (self.features_dir / "quality.json").write_text(json.dumps(features.quality, indent=2, default=str))
        return features

    def features(self, with_dataset: bool = True) -> FeatureSet:
        """The saved features, rebuilt first when missing or built with other settings."""
        quality = self._current_quality()
        if quality is None:
            return self.build_features()
        return FeatureSet(
            dataset=pd.read_parquet(self.features_dir / "dataset.parquet") if with_dataset else None,
            market=pd.read_parquet(self.features_dir / "market.parquet"),
            coverage=pd.read_parquet(self.features_dir / "coverage.parquet"),
            quality=quality,
        )

    def run(self, window: int, progress=None) -> RunResults:
        c = self.config
        features = self.features()
        started = time.perf_counter()
        result = WalkForward(window, c.lgbm_params, c.min_child_share, c.min_child_floor, c.n_jobs).run(
            features.dataset, FEATURES, features.calendar, start=c.backtest_start, progress=progress)
        results = RunResults(window, result.predictions, result.importance, info={
            "window": window,
            "features": FEATURES,
            "runtime_minutes": round((time.perf_counter() - started) / 60.0, 2),
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "versions": {package: version(package) for package in PACKAGES},
            "config": dataclasses.asdict(c),
            "settings": self._settings(RUN_SETTINGS),
            "features_built_at": features.quality["built_at"],
        })
        self.save_run(results)
        return results

    def save_run(self, results: RunResults) -> None:
        folder = self._run_dir(results.window)
        folder.mkdir(parents=True, exist_ok=True)
        results.predictions.to_parquet(folder / "predictions.parquet")
        results.importance.to_parquet(folder / "importance.parquet")
        (folder / "run_info.json").write_text(json.dumps(results.info, indent=2, default=str))

    def load_run(self, window: int) -> RunResults:
        folder = self._run_dir(window)
        return RunResults(window, pd.read_parquet(folder / "predictions.parquet"),
                          pd.read_parquet(folder / "importance.parquet"), json.loads((folder / "run_info.json").read_text()))

    def runs(self) -> list[int]:
        """Windows run with the current settings on the current features; any other run is stale."""
        quality = self._current_quality()
        if quality is None:
            return []
        expected = {"settings": self._settings(RUN_SETTINGS), "features_built_at": quality["built_at"]}
        windows = []
        for path in self.runs_dir.glob("w*/run_info.json"):
            info = json.loads(path.read_text())
            if all(info.get(key) == value for key, value in expected.items()):
                windows.append(int(path.parent.name[1:]))
        return sorted(windows)

    def stamp(self, window: int | None = None) -> int:
        """mtime of the file written last (quality.json / run_info.json); 0 when absent. Keys the app caches."""
        path = self.features_dir / "quality.json" if window is None else self._run_dir(window) / "run_info.json"
        return path.stat().st_mtime_ns if path.exists() else 0

    def _settings(self, names: tuple[str, ...]) -> dict:
        """Those config values as they read back from JSON, so saved and current settings compare equal."""
        return json.loads(json.dumps({name: getattr(self.config, name) for name in names}))

    def _current_quality(self) -> dict | None:
        """quality.json of the saved features, or None when absent or built with other settings."""
        path = self.features_dir / "quality.json"
        quality = json.loads(path.read_text()) if path.exists() else {}
        return quality if quality.get("settings") == self._settings(FEATURE_SETTINGS) else None

    def _run_dir(self, window: int) -> Path:
        return self.runs_dir / f"w{window:03d}"

    def _load_inputs(self) -> tuple[Universe, PriceData]:
        c = self.config
        return Universe.from_csv(c.data_in / c.constituents_file, TICKER_ALIASES), PriceData.load(c.data_in / c.prices_file)

    @staticmethod
    def _clear(*folders: Path) -> None:
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)

    def _make_features(self, universe: Universe, prices: PriceData) -> FeatureSet:
        c = self.config
        calendar = prices.calendar(c.benchmark_ticker, start=c.price_start)
        wanted = universe.tickers_since(c.price_start)
        tickers = sorted(set(wanted) & (prices.tickers - {c.benchmark_ticker, c.risk_free_ticker} - REASSIGNED_TICKERS))
        panel = prices.panel(calendar, tickers)
        membership = universe.membership(calendar, tickers)
        ticker_features, date_features = FeatureBuilder(c.peer_count, c.peer_lookback_days).build(
            panel, prices.series(c.benchmark_ticker, "close", calendar), membership)
        dataset = build_dataset(ticker_features, date_features, membership, panel.close,
                                forward_open_return(panel.open, panel.dividends), c.min_history_days)
        market = pd.DataFrame({
            "holding_date": pd.Series(calendar, index=calendar).shift(-1),
            "bench_fwd_ret": forward_open_return(prices.series(c.benchmark_ticker, "open", calendar)),
            "rf_daily": prices.series(c.risk_free_ticker, "close", calendar)
                              .ffill(limit=c.risk_free_max_staleness_days) / 100.0 / 252.0,
        }, index=calendar).rename_axis("decision_date")
        coverage = pd.DataFrame({
            "n_members": universe.sizes(calendar),
            "n_with_prices": (membership & panel.close.notna()).sum(axis=1).to_numpy(),
            "n_eligible": dataset.groupby(level="date").size().reindex(calendar, fill_value=0).to_numpy(),
        }, index=calendar)
        quality = _quality(universe, calendar, wanted, tickers, panel, membership, dataset, coverage)
        return FeatureSet(dataset, market, coverage, quality)


def _quality(universe, calendar, wanted, tickers, panel, membership, dataset, coverage) -> dict:
    returns = daily_total_return(panel.close, panel.dividends).where(membership)
    extreme = returns.where(returns.abs() > 0.5).stack().dropna()
    top = extreme.loc[extreme.abs().sort_values(ascending=False).index[:20]]
    missing = _members_without_history(universe, calendar, wanted, panel.close)
    in_dataset = coverage[coverage["n_eligible"] > 0]
    share = in_dataset["n_eligible"] / in_dataset["n_members"]
    dollar_volume = (panel.close * panel.volume).where(membership).median()
    return {
        "coverage_by_year": {str(year): round(float(value), 4) for year, value in share.groupby(share.index.year).mean().items()},
        "missing_tickers": [row["ticker"] for row in missing],
        "missing_detail": missing,
        "suspect_tickers": sorted(dollar_volume[dollar_volume < 1e6].index),
        "n_tickers_with_prices": len(tickers),
        "n_extreme_returns": len(extreme),
        "extreme_returns": [{"date": str(d.date()), "ticker": t, "ret": round(float(v), 4)} for (d, t), v in top.items()],
        "date_ranges": {
            "prices": _date_range(calendar),
            "constituents": _date_range(pd.DatetimeIndex(universe.snapshots["date"])),
            "dataset": _date_range(dataset.index.get_level_values("date")),
        },
    }


def _members_without_history(universe, calendar, wanted, close) -> list[dict]:
    """Members priced on under half of their member days: delisted, or a ticker that now names another company."""
    member = universe.membership(calendar, wanted)
    priced = close.notna().reindex(columns=wanted, fill_value=False)
    member_days, days_with_price = member.sum(), (member & priced).sum()
    poor = member_days[(member_days > 0) & (days_with_price < 0.5 * member_days)].index
    return [{"ticker": t, "member_days": int(member_days[t]), "days_with_price": int(days_with_price[t])} for t in sorted(poor)]


def _date_range(dates: pd.DatetimeIndex) -> list[str]:
    return [str(dates.min().date()), str(dates.max().date())]
