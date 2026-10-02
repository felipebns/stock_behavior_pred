import copy
import dataclasses
import json
import re

import numpy as np
import pandas as pd
import pytest
from synthetic import BENCH, RF, RUNS, make_market, perturb_after, synthetic_config, yahoo_response

from engine.features import DATE_FEATURES, FEATURES
from engine.prices import PriceData, clean_prices
from engine.project import FEATURES_FORMAT, Project, RunKey
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


@pytest.mark.parametrize("key", RUNS)
def test_predictions_do_not_depend_on_future_data(synthetic_project, tmp_path, key):
    base = synthetic_project.load_run(key)
    fits = base.importance.index
    cutoff = fits[len(fits) // 2]
    perturbed = build(perturb_after(make_market(), cutoff), tmp_path, n_jobs=2).run(key).predictions
    columns = ["date", "ticker", "prob"]
    predictions = base.predictions
    before = predictions.loc[predictions["date"] <= cutoff, columns].reset_index(drop=True)
    after = perturbed.loc[perturbed["date"] <= cutoff, columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    later = predictions[predictions["date"] > cutoff].merge(perturbed[perturbed["date"] > cutoff], on=["date", "ticker"],
                                                             suffixes=("_base", "_pert"))
    assert len(later) > 0 and not np.allclose(later["prob_base"], later["prob_pert"])


@pytest.mark.parametrize("key", RUNS)
def test_predictions_only_for_members_on_each_date(synthetic_project, key):
    universe = Universe(make_market()["snapshots"])
    for date, group in synthetic_project.load_run(key).predictions.groupby("date"):
        assert set(group["ticker"]) <= set(universe.members_on(date))


@pytest.mark.parametrize("key", RUNS)
def test_decisions_start_at_backtest_start_and_happen_every_session(synthetic_project, key):
    predictions = synthetic_project.load_run(key).predictions
    calendar = synthetic_project.features(with_dataset=False).calendar
    positions = calendar.get_indexer(sorted(predictions["date"].unique()))
    assert calendar[positions[0]] == pd.Timestamp(synthetic_project.config.backtest_start)
    assert set(np.diff(positions)) == {1}


def test_labels_say_whether_the_stock_went_up_during_the_next_day(synthetic_project):
    known = synthetic_project.load_run(RUNS[0]).predictions.dropna(subset=["label"])
    assert (known["label"] == (known["fwd_ret"] > 0)).all()
    prices = make_market()["prices"].set_index(["date", "ticker"])
    calendar = synthetic_project.features(with_dataset=False).calendar
    row = known.iloc[len(known) // 2]
    next_day = calendar[calendar.get_loc(row["date"]) + 1]
    bar = prices.loc[(next_day, row["ticker"])]
    assert row["fwd_ret"] == pytest.approx(bar["close"] / bar["open"] - 1)


def test_features_are_saved_with_every_column(synthetic_project):
    features = synthetic_project.features()
    assert list(features.dataset.columns) == FEATURES + ["fwd_ret", "label"]
    assert features.dataset[DATE_FEATURES].notna().all().all()
    assert features.quality["settings"]["format"] == FEATURES_FORMAT
    assert synthetic_project.features(with_dataset=False).dataset is None


def test_runs_are_saved_per_key_and_listed(synthetic_project):
    assert synthetic_project.runs() == list(RUNS)
    run = synthetic_project.load_run(RUNS[1])
    assert run.info["key"] == dataclasses.asdict(RUNS[1])
    assert list(run.importance.columns) == FEATURES
    assert (synthetic_project.runs_dir / "logistic-per_stock-w40-k3" / "run_info.json").exists()
    assert synthetic_project.stamp(RUNS[1]) > 0 and synthetic_project.stamp(RunKey("lightgbm", "pooled", 21)) == 0
    assert RUNS[1].label == "Logística regularizada · um modelo por ação · janela 40 · retreino a cada 3"


def test_rebuilding_features_deletes_old_runs(tmp_path):
    project = build(make_market(), tmp_path)
    project.run(RUNS[0])
    assert project.runs() == [RUNS[0]]
    raw = make_market()
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    assert not project.runs_dir.exists()


def test_window_too_long_is_refused(synthetic_project):
    with pytest.raises(ValueError, match="máximo"):
        synthetic_project.run(RunKey("lightgbm", "pooled", 100))
    assert synthetic_project.runs() == list(RUNS)


def test_features_saved_in_an_older_format_are_rebuilt(tmp_path):
    write_inputs(make_market(), tmp_path)
    project = Project(synthetic_config(tmp_path))
    project.features(with_dataset=False)
    path = project.features_dir / "quality.json"
    quality = json.loads(path.read_text())
    quality["settings"]["format"] = FEATURES_FORMAT - 1
    path.write_text(json.dumps(quality))
    assert project.features(with_dataset=False).quality["settings"]["format"] == FEATURES_FORMAT


def test_changing_a_models_parameters_hides_only_its_runs(synthetic_project):
    params = copy.deepcopy(synthetic_project.config.model_params)
    params["logistic"]["C"] = 0.5
    assert Project(dataclasses.replace(synthetic_project.config, model_params=params)).runs() == [RUNS[0]]
    assert Project(dataclasses.replace(synthetic_project.config, per_stock_min_rows=5)).runs() == []
    assert synthetic_project.runs() == list(RUNS)


def test_an_unknown_model_or_scope_is_refused():
    with pytest.raises(ValueError, match="modelo"):
        RunKey("xgboost", "pooled", 21)
    with pytest.raises(ValueError, match="escopo"):
        RunKey("lightgbm", "sector", 21)


def test_a_run_saved_after_its_features_were_replaced_is_hidden(tmp_path):
    raw = make_market()
    project = build(raw, tmp_path)
    stale = project.run(RunKey("lightgbm", "pooled", 5))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    project.save_run(stale)
    assert project.runs() == []


def test_market_frame_aligns_benchmark_and_risk_free(synthetic_project):
    market = synthetic_project.features(with_dataset=False).market
    prices = make_market()["prices"]
    bench = prices[prices["ticker"] == BENCH].set_index("date")
    irx = prices[prices["ticker"] == RF].set_index("date")["close"]
    calendar, t = market.index, market.index[100]
    assert market.loc[t, "holding_date"] == calendar[101]
    assert market.loc[t, "bench_fwd_ret"] == pytest.approx(bench["close"][calendar[101]] / bench["open"][calendar[101]] - 1)
    assert market.loc[t, "rf_daily"] == pytest.approx(irx[t] / 100 / 252)
    assert pd.isna(market["holding_date"].iloc[-1]) and pd.isna(market["bench_fwd_ret"].iloc[-1])
    assert pd.notna(market["bench_fwd_ret"].iloc[-2])


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


def test_download_saves_inputs_and_clears_derived_results(tmp_path):
    project = Project(synthetic_config(tmp_path))
    (project.runs_dir / "lightgbm-pooled-w21-k1").mkdir(parents=True)
    (project.runs_dir / "lightgbm-pooled-w21-k1" / "run_info.json").write_text("{}")
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
    (project.runs_dir / "lightgbm-pooled-w40-k1").mkdir(parents=True)
    (project.runs_dir / "lightgbm-pooled-w40-k1" / "run_info.json").write_text("{}")
    quality = Project(synthetic_config(tmp_path, peer_count=2)).features(with_dataset=False).quality
    assert quality["settings"]["peer_count"] == 2
    assert project.stamp() != built and not project.runs_dir.exists()