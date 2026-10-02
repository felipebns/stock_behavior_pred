import numpy as np
import pandas as pd
import pytest

from engine.selection import walk_forward_selection

DAYS = pd.bdate_range("2023-01-03", periods=12, name="holding_date")


def frame(net, rf=0.0) -> pd.DataFrame:
    return pd.DataFrame({"decision_date": DAYS - pd.offsets.BDay(1), "net": net, "rf": rf, "gross": net,
                         "bench": 0.0, "invested": True}, index=DAYS)


def test_picks_the_best_past_sharpe_and_follows_it_until_the_next_pick():
    steady, noisy = frame([0.010, 0.012] * 6), frame([0.03, -0.03] * 6)
    result = walk_forward_selection({"noisy": noisy, "steady": steady}, lookback=4, step=3)
    assert result.choices["candidate"].tolist() == ["steady"] * 3
    assert result.choices["start"].tolist() == [DAYS[4], DAYS[7], DAYS[10]]
    assert result.choices["days"].tolist() == [3, 3, 2]
    assert list(result.daily.index) == list(DAYS[4:])
    assert (result.daily["candidate"] == "steady").all()
    pd.testing.assert_series_equal(result.daily["net"], steady["net"].iloc[4:])


def test_a_pick_never_looks_at_the_days_it_will_trade():
    good, other = frame([0.010, 0.012, 0.011, 0.013] + [0.0] * 8), frame([0.0, 0.001] * 6)
    ruined = good.assign(net=[0.010, 0.012, 0.011, 0.013] + [-0.5] * 8)

    def first_pick(candidates):
        return walk_forward_selection(candidates, lookback=4, step=8).choices["candidate"].iloc[0]

    assert first_pick({"other": other, "good": good}) == first_pick({"other": other, "good": ruined}) == "good"


def test_cash_counts_as_zero_sharpe():
    cash, losing = frame([0.0001] * 12, rf=0.0001), frame([-0.010, -0.012] * 6)
    assert walk_forward_selection({"losing": losing, "cash": cash}, 4, 4).choices["candidate"].tolist() == ["cash"] * 2


def test_candidates_must_cover_the_same_days():
    with pytest.raises(ValueError, match="mesmos dias"):
        walk_forward_selection({"a": frame([0.0] * 12), "b": frame([0.0] * 12).iloc[1:]}, 4, 4)


def test_too_few_days_gives_an_empty_result():
    result = walk_forward_selection({"a": frame([0.01] * 12)}, lookback=12, step=4)
    assert result.daily.empty and result.choices.empty
    assert list(result.choices.columns) == ["start", "candidate", "lookback_sharpe", "days"]


def test_an_all_cash_portfolio_has_zero_excess_and_counts_as_zero_sharpe():
    from engine.metrics import performance_metrics
    from engine.portfolio import Portfolio
    dates = pd.bdate_range("2023-01-02", periods=13, name="decision_date")
    market = pd.DataFrame({"holding_date": list(dates[1:]) + [pd.NaT], "bench_fwd_ret": [0.001] * 12 + [np.nan],
                           "rf_daily": 0.00017 + 0.00001 * np.arange(13)}, index=dates)
    predictions = pd.DataFrame({"date": dates[:12], "ticker": "A", "prob": 0.3, "fwd_ret": 0.01, "label": 1.0})
    cash = Portfolio(0.55, cost_bps=5.0).backtest(predictions, market).daily
    assert np.isnan(performance_metrics(cash["net"], cash["rf"])["sharpe"])
    losing = cash.assign(net=-0.01, invested=True)
    picks = walk_forward_selection({"losing": losing, "cash": cash}, lookback=4, step=4).choices
    assert picks["candidate"].tolist() == ["cash"] * 2 and (picks["lookback_sharpe"] == 0).all()
