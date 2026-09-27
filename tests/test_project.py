import dataclasses
import re

import numpy as np
import pandas as pd
import pytest
from synthetic import BENCH, RF, make_market, perturb_after, synthetic_config, yahoo_response

from engine.features import DATE_FEATURES, FEATURES
from engine.prices import PriceData, clean_prices
from engine.project import Project
from engine.universe import Universe

CONSTITUENTS = b'date,tickers\n2019-06-03,"AAA,BRK.B,GONE"\n2020-01-02,"AAA,BRK.B,NEWCO"\n'


def write_inputs(raw: dict, root) -> None:
    """The synthetic market as `download` would leave it in data/in."""
    PriceData(clean_prices(raw["prices"])).save(root / "in" / "prices.parquet")
    snapshots = raw["snapshots"].assign(tickers=raw["snapshots"]["tickers"].str.join(","))
    snapshots.to_csv(root / "in" / "sp500_historical_constituents.csv", index=False)


def build(raw: dict, root, **overrides) -> Project:
    project = Project(synthetic_config(root, **overrides))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    return project


def test_predictions_do_not_depend_on_future_data(synthetic_project, tmp_path):
    base = synthetic_project.load_run(40).predictions
    decision_dates = base["date"].drop_duplicates().sort_values()
    cutoff = decision_dates.iloc[len(decision_dates) // 2]
    perturbed = build(perturb_after(make_market(), cutoff), tmp_path, n_jobs=2).run(40).predictions
    columns = ["date", "ticker", "prob"]
    before = base.loc[base["date"] <= cutoff, columns].reset_index(drop=True)
    after = perturbed.loc[perturbed["date"] <= cutoff, columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    later = base[base["date"] > cutoff].merge(perturbed[perturbed["date"] > cutoff], on=["date", "ticker"], suffixes=("_base", "_pert"))
    assert len(later) > 0 and not np.allclose(later["prob_base"], later["prob_pert"])


def test_predictions_only_for_members_on_each_date(synthetic_project):
    universe = Universe(make_market()["snapshots"])
    for date, group in synthetic_project.load_run(40).predictions.groupby("date"):
        assert set(group["ticker"]) <= set(universe.members_on(date))


def test_decisions_start_at_backtest_start(synthetic_project):
    predictions = synthetic_project.load_run(40).predictions
    assert predictions["date"].min() == pd.Timestamp(synthetic_project.config.backtest_start)


def test_market_frame_aligns_benchmark_and_risk_free(synthetic_project):
    market = synthetic_project.features(with_dataset=False).market
    prices = make_market()["prices"]
    bench_open = prices[prices["ticker"] == BENCH].set_index("date")["open"]
    irx = prices[prices["ticker"] == RF].set_index("date")["close"]
    calendar, t = market.index, market.index[100]
    assert market.loc[t, "holding_date"] == calendar[101]
    assert market.loc[t, "bench_fwd_ret"] == pytest.approx(bench_open[calendar[102]] / bench_open[calendar[101]] - 1)
    assert market.loc[t, "rf_daily"] == pytest.approx(irx[t] / 100 / 252)
    assert pd.isna(market["holding_date"].iloc[-1]) and pd.isna(market["bench_fwd_ret"].iloc[-2])


def test_features_are_saved_with_every_column(synthetic_project):
    features = synthetic_project.features()
    assert list(features.dataset.columns) == FEATURES + ["fwd_ret", "label"]
    assert features.dataset[DATE_FEATURES].notna().all().all()
    assert synthetic_project.features(with_dataset=False).dataset is None


def test_quality_reports_members_without_usable_history(tmp_path):
    raw = make_market()
    dates = sorted(raw["prices"]["date"].unique())
    stub = raw["prices"][(raw["prices"]["ticker"] == "T05") & (raw["prices"]["date"] >= dates[-5])].assign(ticker="STUB")
    raw["prices"] = pd.concat([raw["prices"], stub], ignore_index=True)
    raw["snapshots"] = raw["snapshots"].assign(tickers=raw["snapshots"]["tickers"].map(lambda members: members + ["GONE", "STUB"]))
    quality = build(raw, tmp_path).features(with_dataset=False).quality
    assert quality["missing_tickers"] == ["GONE", "STUB"]
    detail = {row["ticker"]: row for row in quality["missing_detail"]}
    assert detail["GONE"] == {"ticker": "GONE", "member_days": len(dates), "days_with_price": 0}
    assert detail["STUB"] == {"ticker": "STUB", "member_days": len(dates), "days_with_price": 5}
    assert quality["date_ranges"]["constituents"] == [str(dates[0].date()), str(raw["snapshots"]["date"].max().date())]
    assert set(quality["coverage_by_year"]) == {"2019", "2020"}
    assert all(0 < share <= 1 for share in quality["coverage_by_year"].values())


def test_runs_are_saved_per_window_and_listed(synthetic_project):
    assert synthetic_project.runs() == [40]
    run = synthetic_project.load_run(40)
    assert run.info["window"] == 40
    assert list(run.importance.columns) == FEATURES
    assert synthetic_project.stamp(40) > 0 and synthetic_project.stamp(21) == 0


def test_rebuilding_features_deletes_old_runs(tmp_path):
    project = build(make_market(), tmp_path)
    project.run(40)
    assert project.runs() == [40]
    raw = make_market()
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    assert not project.runs_dir.exists()


def test_window_too_long_is_refused(synthetic_project):
    with pytest.raises(ValueError, match="máximo"):
        synthetic_project.run(100)
    assert synthetic_project.runs() == [40]


def test_download_saves_inputs_and_clears_derived_results(tmp_path):
    project = Project(synthetic_config(tmp_path))
    (project.runs_dir / "w021").mkdir(parents=True)
    (project.runs_dir / "w021" / "run_info.json").write_text("{}")
    failed = project.download(
        price_downloader=lambda tickers, **kwargs: yahoo_response(tickers, ["2024-01-02", "2024-01-03"], skip={"GONE"}),
        fetch=lambda url: CONSTITUENTS,
    )
    assert failed == ["GONE"]
    assert (tmp_path / "in" / "prices.parquet").exists()
    assert (tmp_path / "in" / "sp500_historical_constituents.csv").read_bytes() == CONSTITUENTS
    assert not project.runs_dir.exists()


@pytest.mark.parametrize("missing", ["^SP500TR", "NEWCO"])
def test_download_keeps_old_inputs_when_the_index_or_a_current_member_is_missing(tmp_path, missing):
    project = Project(synthetic_config(tmp_path))
    constituents = tmp_path / "in" / "sp500_historical_constituents.csv"
    constituents.parent.mkdir(parents=True)
    constituents.write_bytes(b"date,tickers\n2019-01-02,OLDONLY\n")
    with pytest.raises(RuntimeError, match=re.escape(missing)):
        project.download(
            price_downloader=lambda tickers, **kwargs: yahoo_response(tickers, ["2024-01-02"], skip={missing}),
            fetch=lambda url: CONSTITUENTS,
        )
    assert constituents.read_bytes() == b"date,tickers\n2019-01-02,OLDONLY\n"
    assert not (tmp_path / "in" / "prices.parquet").exists()


def test_reassigned_yahoo_tickers_are_discarded_and_illiquid_members_flagged(tmp_path):
    raw = make_market()
    reassigned = raw["prices"][raw["prices"]["ticker"] == "T05"].assign(ticker="PARA")
    thin = raw["prices"][raw["prices"]["ticker"] == "T06"].assign(ticker="THIN", volume=10.0)
    raw["prices"] = pd.concat([raw["prices"], reassigned, thin], ignore_index=True)
    raw["snapshots"] = raw["snapshots"].assign(tickers=raw["snapshots"]["tickers"].map(lambda members: members + ["PARA", "THIN"]))
    features = build(raw, tmp_path).features()
    tickers = set(features.dataset.index.get_level_values("ticker"))
    assert "PARA" not in tickers and "THIN" in tickers
    assert "PARA" in features.quality["missing_tickers"]
    assert features.quality["suspect_tickers"] == ["THIN"]


def test_features_are_rebuilt_when_their_settings_change(tmp_path):
    write_inputs(make_market(), tmp_path)
    project = Project(synthetic_config(tmp_path))
    project.features(with_dataset=False)
    built = project.stamp()
    Project(synthetic_config(tmp_path)).features(with_dataset=False)
    assert project.stamp() == built
    (project.runs_dir / "w040").mkdir(parents=True)
    (project.runs_dir / "w040" / "run_info.json").write_text("{}")
    quality = Project(synthetic_config(tmp_path, peer_count=2)).features(with_dataset=False).quality
    assert quality["settings"]["peer_count"] == 2
    assert project.stamp() != built and not project.runs_dir.exists()


def test_runs_made_with_other_training_settings_are_hidden(synthetic_project):
    assert Project(dataclasses.replace(synthetic_project.config, min_child_floor=5)).runs() == []
    assert synthetic_project.runs() == [40]


def test_a_run_saved_after_its_features_were_replaced_is_hidden(tmp_path):
    raw = make_market()
    project = build(raw, tmp_path)
    stale = project.run(5)
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    project.save_run(stale)
    assert project.runs() == []
