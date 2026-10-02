import numpy as np
import pandas as pd
import pytest

from engine.portfolio import Portfolio

D = pd.bdate_range("2022-01-03", periods=5, name="decision_date")


def preds(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["date", "ticker", "prob", "fwd_ret"])
    return frame.assign(label=(frame["fwd_ret"] > 0).astype(float).where(frame["fwd_ret"].notna()))


@pytest.fixture
def market():
    return pd.DataFrame({
        "holding_date": list(D[1:]) + [pd.NaT],
        "bench_fwd_ret": [0.01, -0.01, 0.002, 0.003, np.nan],
        "rf_daily": 0.0002,
    }, index=D)


@pytest.fixture
def predictions():
    return preds([
        (D[0], "A", 0.60, 0.010), (D[0], "B", 0.70, -0.020), (D[0], "C", 0.50, 0.050),
        (D[1], "A", 0.52, 0.030), (D[1], "B", 0.40, 0.010),
        (D[2], "A", 0.65, np.nan),
        (D[4], "A", 0.90, 0.100),
    ])


def test_selection_uses_threshold_and_probability_weights(predictions, market):
    positions = Portfolio(0.55).backtest(predictions, market).positions
    day0 = positions[positions["decision_date"] == D[0]].set_index("ticker")
    assert set(day0.index) == {"A", "B"}
    assert day0.loc["A", "weight"] == pytest.approx(0.6 / 1.3)
    assert day0.loc["B", "weight"] == pytest.approx(0.7 / 1.3)


def test_daily_returns_cash_and_missing(predictions, market):
    daily = Portfolio(0.55).backtest(predictions, market).daily
    assert list(daily.index) == list(D[1:5])
    assert daily["decision_date"].tolist() == list(D[:4])
    assert daily["gross"].iloc[0] == pytest.approx(0.6 / 1.3 * 0.01 + 0.7 / 1.3 * -0.02)
    assert daily["gross"].iloc[1] == pytest.approx(0.0002)
    assert daily["gross"].iloc[2] == pytest.approx(0.0)
    assert daily["gross"].iloc[3] == pytest.approx(0.0002)
    assert daily["n_missing"].tolist() == [0, 0, 1, 0]
    assert daily["invested"].tolist() == [True, False, True, False]
    assert daily["bench"].tolist() == pytest.approx([0.01, -0.01, 0.002, 0.003])


def test_selection_never_looks_at_realized_returns(predictions, market):
    shuffled = predictions.assign(fwd_ret=predictions["fwd_ret"].sample(frac=1.0, random_state=3).to_numpy())
    columns = ["decision_date", "ticker", "weight"]
    pd.testing.assert_frame_equal(
        Portfolio(0.55).backtest(predictions, market).positions[columns],
        Portfolio(0.55).backtest(shuffled, market).positions[columns],
    )


def test_threshold_never_met_is_all_cash(predictions, market):
    backtest = Portfolio(0.99).backtest(predictions, market)
    assert backtest.positions.empty
    assert not backtest.daily["invested"].any()
    assert backtest.daily["gross"].tolist() == pytest.approx([0.0002] * 4)


def test_positions_have_holding_dates_and_contributions(predictions, market):
    positions = Portfolio(0.55).backtest(predictions, market).positions
    assert list(positions.columns) == ["decision_date", "holding_date", "ticker", "prob", "weight", "fwd_ret", "contribution"]
    assert (positions["holding_date"] == positions["decision_date"].map(market["holding_date"])).all()
    day0 = positions[positions["decision_date"] == D[0]]
    assert day0["contribution"].sum() == pytest.approx(0.6 / 1.3 * 0.01 + 0.7 / 1.3 * -0.02)


def test_costs_are_a_full_round_trip_on_every_invested_day():
    dates = pd.bdate_range("2022-01-03", periods=6, name="decision_date")
    market = pd.DataFrame({
        "holding_date": list(dates[1:]) + [pd.NaT],
        "bench_fwd_ret": [0.0, 0.0, 0.0, 0.0, 0.0, np.nan],
        "rf_daily": 0.0001,
    }, index=dates)
    predictions = preds([
        (dates[0], "A", 0.6, 0.10), (dates[0], "B", 0.6, -0.10),
        (dates[1], "A", 0.6, 0.00), (dates[1], "B", 0.6, 0.00),
        (dates[2], "C", 0.7, 0.00),
        (dates[3], "A", 0.5, 0.00),
        (dates[4], "C", 0.7, 0.00),
    ])
    daily = Portfolio(0.55, cost_bps=10.0).backtest(predictions, market).daily
    round_trip = 1 - (1 - 0.001) ** 2
    assert daily["turnover"].tolist() == [2.0, 2.0, 2.0, 0.0, 2.0]
    assert daily["cost"].tolist() == pytest.approx([round_trip] * 3 + [0.0, round_trip])
    assert daily["net"].iloc[0] == pytest.approx((1 + daily["gross"].iloc[0]) * (1 - 0.001) ** 2 - 1)
    assert daily["net"].iloc[3] == pytest.approx(0.0001)


def test_zero_cost_means_net_equals_gross(predictions, market):
    daily = Portfolio(0.55).backtest(predictions, market).daily
    assert daily["net"].tolist() == pytest.approx(daily["gross"].tolist())
    assert (daily["cost"] == 0).all()


def test_benchmark_gap_inside_the_window_fails_loudly(predictions, market):
    broken = market.copy()
    broken.loc[D[1], "bench_fwd_ret"] = np.nan
    with pytest.raises(ValueError, match="2022-01-04"):
        Portfolio(0.55).backtest(predictions, broken)


def test_select_keeps_only_probabilities_above_the_threshold(predictions):
    chosen = Portfolio(0.6).select(predictions)
    assert set(zip(chosen["date"], chosen["ticker"])) == {(D[0], "B"), (D[2], "A"), (D[4], "A")}
    assert (chosen["weight"] == 1.0).all()


def test_no_realized_day_gives_an_empty_backtest(predictions, market):
    result = Portfolio(0.55).backtest(predictions, market.assign(bench_fwd_ret=np.nan))
    assert result.daily.empty and result.positions.empty
    assert list(result.positions.columns) == ["decision_date", "holding_date", "ticker", "prob", "weight", "fwd_ret", "contribution"]
