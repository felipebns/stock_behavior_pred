import pytest
from synthetic import BENCH, RF, make_market

from engine.data import prices as px
from engine.data import universe


@pytest.fixture(scope="session")
def synthetic_panel():
    raw = make_market()
    prices = raw["prices"]
    calendar = px.trading_calendar(prices, BENCH)
    tickers = sorted(set(prices["ticker"]) - {BENCH, RF})
    panel = px.build_panel(prices, calendar, tickers)
    membership = universe.membership_matrix(raw["snapshots"], calendar, tickers)
    market_close = prices[prices["ticker"] == BENCH].set_index("date")["close"].reindex(calendar)
    return panel, market_close, membership
