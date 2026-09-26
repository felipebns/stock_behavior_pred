"""Data → features → dataset → walk-forward, over in-memory inputs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.config import TICKER_ALIASES, Config
from engine.data import macro
from engine.data import prices as px
from engine.data import universe
from engine.dataset import build_dataset
from engine.features import FEATURES, cross_sectional_features, market_features, stock_features
from engine.model import make_model
from engine.results import RunResults
from engine.walk_forward import walk_forward


@dataclass(frozen=True)
class MarketInputs:
    prices: pd.DataFrame
    snapshots: pd.DataFrame
    ff_bars: pd.DataFrame
    gpr: pd.DataFrame


@dataclass
class FeatureSet:
    dataset: pd.DataFrame
    calendar: pd.DatetimeIndex
    market: pd.DataFrame
    coverage: pd.DataFrame
    data_quality: dict


def load_inputs(config: Config) -> MarketInputs:
    return MarketInputs(
        prices=px.load_prices(config.data_in / config.prices_file),
        snapshots=universe.load_constituents(config.data_in / config.constituents_file, TICKER_ALIASES),
        ff_bars=macro.load_ff_futures(config.data_in / config.ff_futures_file),
        gpr=macro.load_gpr(config.data_in / config.gpr_file),
    )


def build_feature_set(inputs: MarketInputs, config: Config) -> FeatureSet:
    calendar = px.trading_calendar(inputs.prices, config.benchmark_ticker, start=config.price_start)
    wanted = universe.tickers_since(inputs.snapshots, config.price_start)
    available = set(inputs.prices["ticker"]) - {config.benchmark_ticker, config.risk_free_ticker}
    tickers = sorted(set(wanted) & available)
    panel = px.build_panel(inputs.prices, calendar, tickers)
    membership = universe.membership_matrix(inputs.snapshots, calendar, tickers)
    bench = inputs.prices[inputs.prices["ticker"] == config.benchmark_ticker].set_index("date").sort_index()
    market_close = bench["close"].reindex(calendar)

    stock = stock_features(panel, market_close)
    ff = macro.ff_rates(inputs.ff_bars, config.ff_horizon_months, config.ff_max_gap_days, config.ff_change_days)
    date_features = pd.concat([
        market_features(market_close, stock, membership),
        macro.align_asof(ff[macro.FF_FEATURES], calendar, 0, config.macro_max_staleness_days),
        macro.align_asof(macro.gpr_features(inputs.gpr), calendar, config.gpr_lag_days, config.macro_max_staleness_days),
    ], axis=1)
    dataset = build_dataset(
        {**stock, **cross_sectional_features(stock, membership)}, date_features, membership,
        panel.close, px.forward_open_return(panel.open, panel.dividends), config.min_history_days,
    )
    market = pd.DataFrame({
        "holding_date": pd.Series(calendar, index=calendar).shift(-1),
        "bench_fwd_ret": px.forward_open_return(bench["open"].reindex(calendar)),
        "rf_daily": macro.risk_free_daily(inputs.prices, config.risk_free_ticker, calendar, config.macro_max_staleness_days),
    }, index=calendar).rename_axis("decision_date")
    coverage = _coverage(inputs.snapshots, calendar, membership, panel.close, dataset)
    return FeatureSet(
        dataset=dataset,
        calendar=calendar,
        market=market,
        coverage=coverage,
        data_quality=_data_quality(inputs, calendar, wanted, tickers, panel, membership, dataset, ff, coverage),
    )


def _coverage(snapshots, calendar, membership, close, dataset) -> pd.DataFrame:
    sizes = snapshots["tickers"].map(len).to_numpy()
    pos = snapshots["date"].searchsorted(calendar, side="right") - 1
    return pd.DataFrame({
        "n_members": np.where(pos >= 0, sizes[np.clip(pos, 0, None)], 0),
        "n_with_prices": (membership & close.notna()).sum(axis=1).to_numpy(),
        "n_eligible": dataset.groupby(level="date").size().reindex(calendar, fill_value=0).to_numpy(),
    }, index=calendar)


def _data_quality(inputs, calendar, wanted, tickers, panel, membership, dataset, ff, coverage) -> dict:
    returns = px.daily_total_return(panel.close, panel.dividends).where(membership)
    extreme = returns.where(returns.abs() > 0.5).stack().dropna()
    top = extreme.loc[extreme.abs().sort_values(ascending=False).index[:20]]
    missing = _members_without_history(inputs.snapshots, calendar, wanted, panel.close)
    in_dataset = coverage[coverage["n_eligible"] > 0]
    share = in_dataset["n_eligible"] / in_dataset["n_members"]
    return {
        "coverage_by_year": {str(year): round(float(value), 4) for year, value in share.groupby(share.index.year).mean().items()},
        "missing_tickers": [row["ticker"] for row in missing],
        "missing_detail": missing,
        "n_tickers_with_prices": len(tickers),
        "n_extreme_returns": len(extreme),
        "extreme_returns": [{"date": str(d.date()), "ticker": t, "ret": round(float(v), 4)} for (d, t), v in top.items()],
        "date_ranges": {
            "prices": _date_range(calendar),
            "ff_futures": _date_range(ff.index),
            "gpr": _date_range(inputs.gpr.index),
            "constituents": _date_range(pd.DatetimeIndex(inputs.snapshots["date"])),
            "dataset": _date_range(dataset.index.get_level_values("date")),
        },
    }


def _members_without_history(snapshots, calendar, wanted, close) -> list[dict]:
    """Members priced on under half of their member days: delisted, or a recycled ticker that now names another company."""
    member = universe.membership_matrix(snapshots, calendar, wanted)
    priced = close.notna().reindex(columns=wanted, fill_value=False)
    member_days, days_with_price = member.sum(), (member & priced).sum()
    poor = member_days[(member_days > 0) & (days_with_price < 0.5 * member_days)].index
    return [{"ticker": t, "member_days": int(member_days[t]), "days_with_price": int(days_with_price[t])} for t in sorted(poor)]


def _date_range(dates: pd.DatetimeIndex) -> list[str]:
    return [str(dates.min().date()), str(dates.max().date())]


def run_pipeline(inputs: MarketInputs, config: Config, start=None, end=None, log=print) -> RunResults:
    features = build_feature_set(inputs, config)
    dates = features.dataset.index.get_level_values("date")
    log(f"dataset: {len(features.dataset):,} linhas, {dates.min().date()} → {dates.max().date()}")
    result = walk_forward(
        features.dataset, FEATURES, features.calendar, config.train_window_days,
        lambda: make_model(config.lgbm_params), start=start, end=end,
        progress_every=config.progress_every, log=log,
    )
    return RunResults(
        predictions=result.predictions,
        importance=result.importance,
        market=features.market,
        coverage=features.coverage,
        run_info={"features": FEATURES, "data_quality": features.data_quality},
    )
