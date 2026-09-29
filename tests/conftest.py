import pytest
from synthetic import BENCH, RF, RUNS, make_market, synthetic_config

from engine.prices import PriceData, clean_prices
from engine.project import Project
from engine.universe import Universe


@pytest.fixture(scope="session")
def synthetic_panel():
    raw = make_market()
    prices = PriceData(clean_prices(raw["prices"]))
    calendar = prices.calendar(BENCH)
    tickers = sorted(prices.tickers - {BENCH, RF})
    membership = Universe(raw["snapshots"]).membership(calendar, tickers)
    return prices.panel(calendar, tickers), prices.series(BENCH, "close", calendar), membership


@pytest.fixture(scope="session")
def synthetic_project(tmp_path_factory):
    raw = make_market()
    project = Project(synthetic_config(tmp_path_factory.mktemp("project")))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    for key in RUNS:
        project.run(key)
    return project
