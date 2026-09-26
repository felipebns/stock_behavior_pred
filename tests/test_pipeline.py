import numpy as np
import pandas as pd
import pytest
from synthetic import BENCH, RF, make_market, perturb_after

from config.config import Config
from engine.data import universe
from engine.features import DATE_FEATURES, FEATURES
from engine.pipeline import MarketInputs, build_feature_set, run_pipeline

SMALL_LGBM = {
    "n_estimators": 10, "learning_rate": 0.1, "num_leaves": 4, "min_child_samples": 10,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "importance_type": "gain",
    "random_state": 0, "deterministic": True, "force_col_wise": True, "n_jobs": 1, "verbose": -1,
}


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def config():
    return Config(price_start="2019-01-01", train_window_days=40, progress_every=0, lgbm_params=SMALL_LGBM)


@pytest.fixture(scope="module")
def inputs():
    return MarketInputs(**make_market())


@pytest.fixture(scope="module")
def baseline(inputs, config):
    return run_pipeline(inputs, config, log=quiet)


def test_predictions_do_not_depend_on_future_data(config, baseline):
    decision_dates = baseline.predictions["date"].drop_duplicates().sort_values()
    cutoff = decision_dates.iloc[len(decision_dates) // 2]
    perturbed = run_pipeline(MarketInputs(**perturb_after(make_market(), cutoff)), config, log=quiet)
    columns = ["date", "ticker", "prob"]
    before = baseline.predictions.loc[baseline.predictions["date"] <= cutoff, columns].reset_index(drop=True)
    after = perturbed.predictions.loc[perturbed.predictions["date"] <= cutoff, columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    later = baseline.predictions[baseline.predictions["date"] > cutoff].merge(
        perturbed.predictions[perturbed.predictions["date"] > cutoff], on=["date", "ticker"], suffixes=("_base", "_pert"))
    assert len(later) > 0 and not np.allclose(later["prob_base"], later["prob_pert"])


def test_predictions_only_for_point_in_time_members(inputs, baseline):
    for date, group in baseline.predictions.groupby("date"):
        assert set(group["ticker"]) <= set(universe.members_on(inputs.snapshots, date))


def test_market_frame_aligns_benchmark_and_risk_free(inputs, config):
    features = build_feature_set(inputs, config)
    market, calendar = features.market, features.calendar
    bench_open = inputs.prices[inputs.prices["ticker"] == BENCH].set_index("date")["open"]
    irx = inputs.prices[inputs.prices["ticker"] == RF].set_index("date")["close"]
    t = calendar[100]
    assert market.loc[t, "holding_date"] == calendar[101]
    assert market.loc[t, "bench_fwd_ret"] == pytest.approx(bench_open[calendar[102]] / bench_open[calendar[101]] - 1)
    assert market.loc[t, "rf_daily"] == pytest.approx(irx[t] / 100 / 252)
    assert pd.isna(market["holding_date"].iloc[-1]) and pd.isna(market["bench_fwd_ret"].iloc[-2])


def test_dataset_has_all_features_and_known_date_features(inputs, config):
    dataset = build_feature_set(inputs, config).dataset
    assert list(dataset.columns) == FEATURES + ["fwd_ret", "label"]
    assert dataset[DATE_FEATURES].notna().all().all()


def test_dataset_stops_when_macro_data_goes_stale(config):
    raw = make_market()
    last_ff = pd.Timestamp(np.sort(raw["ff_bars"]["date"].unique())[280])
    raw["ff_bars"] = raw["ff_bars"][raw["ff_bars"]["date"] <= last_ff]
    dataset = build_feature_set(MarketInputs(**raw), config).dataset
    assert dataset.index.get_level_values("date").max() <= last_ff + pd.Timedelta(days=config.macro_max_staleness_days)


def test_data_quality_reports_members_without_usable_history(config):
    raw = make_market()
    dates = sorted(raw["prices"]["date"].unique())
    stub = raw["prices"][(raw["prices"]["ticker"] == "T05") & (raw["prices"]["date"] >= dates[-5])].assign(ticker="STUB")
    raw["prices"] = pd.concat([raw["prices"], stub], ignore_index=True)
    raw["snapshots"] = raw["snapshots"].assign(tickers=raw["snapshots"]["tickers"].map(lambda members: members + ["GONE", "STUB"]))
    quality = build_feature_set(MarketInputs(**raw), config).data_quality
    assert quality["missing_tickers"] == ["GONE", "STUB"]
    detail = {row["ticker"]: row for row in quality["missing_detail"]}
    assert detail["GONE"] == {"ticker": "GONE", "member_days": len(dates), "days_with_price": 0}
    assert detail["STUB"] == {"ticker": "STUB", "member_days": len(dates), "days_with_price": 5}


def test_quality_has_constituents_range_and_yearly_coverage(inputs, config):
    quality = build_feature_set(inputs, config).data_quality
    snapshot_dates = inputs.snapshots["date"]
    assert quality["date_ranges"]["constituents"] == [str(snapshot_dates.min().date()), str(snapshot_dates.max().date())]
    assert set(quality["coverage_by_year"]) == {"2019", "2020"}
    assert all(0.0 < share <= 1.0 for share in quality["coverage_by_year"].values())


def test_run_info_reports_quality_and_coverage(baseline):
    assert list(baseline.coverage.columns) == ["n_members", "n_with_prices", "n_eligible"]
    quality = baseline.run_info["data_quality"]
    assert {"missing_tickers", "n_tickers_with_prices", "n_extreme_returns", "extreme_returns", "date_ranges"} <= set(quality)
    assert baseline.run_info["features"] == FEATURES
    assert (baseline.coverage["n_eligible"] <= baseline.coverage["n_with_prices"]).all()
