import numpy as np
import pandas as pd
import pytest

from engine.portfolio import Portfolio

D = pd.bdate_range("2022-01-03", periods=5, name="decision_date")


def preds(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["date", "ticker", "prob", "fwd_ret"])
    return frame.assign(label=(frame["fwd_ret"] > 0).astype(float).where(frame["fwd_ret"].notna()))


def returns_of(predictions: pd.DataFrame) -> pd.DataFrame:
    """One-session returns as the (date × ticker) panel, taken from the rows (horizon 1)."""
    return predictions.pivot(index="date", columns="ticker", values="fwd_ret")


def run(predictions, market, threshold=0.55, cost_bps=0.0, returns=None, horizon=1):
    returns = returns_of(predictions) if returns is None else returns
    return Portfolio(threshold, cost_bps).backtest(predictions, market, returns, horizon)


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
        (D[3], "A", 0.40, 0.020),
        (D[4], "A", 0.90, 0.100),
    ])


def test_selection_uses_threshold_and_probability_weights(predictions, market):
    positions = run(predictions, market).positions
    day0 = positions[positions["decision_date"] == D[0]].set_index("ticker")
    assert set(day0.index) == {"A", "B"}
    assert day0.loc["A", "weight"] == pytest.approx(0.6 / 1.3)
    assert day0.loc["B", "weight"] == pytest.approx(0.7 / 1.3)


def test_daily_returns_cash_and_missing(predictions, market):
    daily = run(predictions, market).daily
    assert list(daily.index) == list(D[1:5])
    assert daily["decision_date"].tolist() == list(D[:4])
    assert daily["gross"].iloc[0] == pytest.approx(0.6 / 1.3 * 0.01 + 0.7 / 1.3 * -0.02)
    assert daily["gross"].iloc[1] == pytest.approx(0.0002)
    assert daily["gross"].iloc[2] == pytest.approx(0.0)
    assert daily["gross"].iloc[3] == pytest.approx(0.0002)
    assert daily["n_missing"].tolist() == [0, 0, 1, 0]
    assert daily["invested"].tolist() == [True, False, True, False]
    assert daily["rebalance"].all()
    assert daily["bench"].tolist() == pytest.approx([0.01, -0.01, 0.002, 0.003])


def test_selection_never_looks_at_realized_returns(predictions, market):
    shuffled = predictions.assign(fwd_ret=predictions["fwd_ret"].sample(frac=1.0, random_state=3).to_numpy())
    columns = ["decision_date", "ticker", "weight"]
    pd.testing.assert_frame_equal(
        run(predictions, market).positions[columns],
        run(shuffled, market, returns=returns_of(predictions)).positions[columns],
    )


@pytest.mark.parametrize("horizon", [1, 3])
def test_threshold_never_met_is_all_cash(predictions, market, horizon):
    backtest = run(predictions, market, threshold=0.99, horizon=horizon)
    assert backtest.positions.empty
    assert not backtest.daily["invested"].any()
    assert backtest.daily["gross"].tolist() == pytest.approx([0.0002] * 4)


def test_positions_have_holding_dates_and_contributions(predictions, market):
    positions = run(predictions, market).positions
    assert list(positions.columns) == ["decision_date", "holding_date", "ticker", "prob", "weight", "realized", "contribution"]
    assert (positions["holding_date"] == positions["decision_date"].map(market["holding_date"])).all()
    day0 = positions[positions["decision_date"] == D[0]]
    assert day0["contribution"].sum() == pytest.approx(0.6 / 1.3 * 0.01 + 0.7 / 1.3 * -0.02)
    assert positions.loc[positions["decision_date"] == D[2], "realized"].tolist() == [0.0]


def test_costs_follow_turnover_against_drifted_weights():
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
    daily = run(predictions, market, cost_bps=10.0).daily
    assert daily["turnover"].tolist() == pytest.approx([1.0, 0.1, 2.0, 1.0, 1.0])
    assert daily["cost"].tolist() == pytest.approx([0.001, 0.0001, 0.002, 0.001, 0.001])
    assert daily["net"].iloc[0] == pytest.approx((1 + daily["gross"].iloc[0]) * (1 - 0.001) - 1)
    assert daily["net"].iloc[3] == pytest.approx((1 + 0.0001) * (1 - 0.001) - 1)


def holding_case(bench_days: int = 6):
    days = pd.bdate_range("2022-01-03", periods=7, name="decision_date")
    market = pd.DataFrame({
        "holding_date": list(days[1:]) + [pd.NaT],
        "bench_fwd_ret": [0.0] * bench_days + [np.nan] * (7 - bench_days),
        "rf_daily": 0.0001,
    }, index=days)
    returns = pd.DataFrame({"A": [0.10, 0.10, 0.0, 0.0, 0.0, 0.0, np.nan],
                            "B": [-0.10, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan]}, index=days)
    predictions = pd.DataFrame({"date": [days[0], days[0], days[3]], "ticker": ["A", "B", "A"],
                                "prob": [0.6, 0.6, 0.7], "fwd_ret": np.nan, "label": np.nan})
    return days, market, returns, predictions


def test_holding_for_the_horizon_lets_weights_drift_and_trades_only_at_decisions():
    days, market, returns, predictions = holding_case()
    result = run(predictions, market, cost_bps=10.0, returns=returns, horizon=3)
    daily = result.daily
    assert daily["decision_date"].tolist() == [days[0]] * 3 + [days[3]] * 3
    assert daily["rebalance"].tolist() == [True, False, False, True, False, False]
    assert daily["gross"].tolist() == pytest.approx([0.0, 0.055, 0.0, 0.0, 0.0, 0.0])
    assert daily["turnover"].tolist() == pytest.approx([1.0, 0.0, 0.0, 2 * 0.45 / 1.055, 0.0, 0.0])
    positions = result.positions.set_index(["decision_date", "ticker"])
    assert positions.loc[(days[0], "A"), "realized"] == pytest.approx(1.1 * 1.1 - 1)
    assert positions.loc[(days[0], "B"), "realized"] == pytest.approx(-0.1)
    period = (1 + daily["gross"].iloc[:3]).prod() - 1
    assert positions.loc[days[0], "contribution"].sum() == pytest.approx(period)


def test_last_holding_period_is_cut_at_the_last_realized_day():
    days, market, returns, predictions = holding_case(bench_days=5)
    result = run(predictions, market, returns=returns, horizon=3)
    assert list(result.daily.index) == list(days[1:6])
    assert result.daily["decision_date"].tolist() == [days[0]] * 3 + [days[3]] * 2
    assert result.positions.set_index("ticker").loc["A"].iloc[-1]["realized"] == pytest.approx(0.0)


def test_zero_cost_means_net_equals_gross(predictions, market):
    daily = run(predictions, market).daily
    assert daily["net"].tolist() == pytest.approx(daily["gross"].tolist())
    assert (daily["cost"] == 0).all()


def test_benchmark_gap_inside_the_window_fails_loudly(predictions, market):
    broken = market.copy()
    broken.loc[D[1], "bench_fwd_ret"] = np.nan
    with pytest.raises(ValueError, match="2022-01-04"):
        run(predictions, broken)


def test_select_keeps_only_probabilities_above_the_threshold(predictions):
    chosen = Portfolio(0.6).select(predictions)
    assert set(zip(chosen["date"], chosen["ticker"])) == {(D[0], "B"), (D[2], "A"), (D[4], "A")}
    assert (chosen["weight"] == 1.0).all()
