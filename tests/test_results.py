import numpy as np
import pandas as pd

from engine.results import RunResults, load_results, save_results


def test_save_and_load_roundtrip(tmp_path):
    dates = pd.bdate_range("2021-01-01", periods=3, name="date")
    results = RunResults(
        predictions=pd.DataFrame({"date": dates, "ticker": ["A", "B", "C"], "prob": [0.5, 0.6, 0.7],
                                  "fwd_ret": [0.01, np.nan, -0.02], "label": [1.0, np.nan, 0.0]}),
        importance=pd.DataFrame({"f1": [0.5, 0.4, 0.3]}, index=dates),
        market=pd.DataFrame({"holding_date": list(dates[1:]) + [pd.NaT], "bench_fwd_ret": [0.01, 0.0, np.nan],
                             "rf_daily": 0.0001}, index=pd.DatetimeIndex(dates, name="decision_date")),
        coverage=pd.DataFrame({"n_members": [3, 3, 3], "n_with_prices": [3, 2, 3], "n_eligible": [3, 2, 2]}, index=dates),
        run_info={"features": ["f1"], "config": {"threshold": 0.55}},
    )
    save_results(results, tmp_path / "out")
    loaded = load_results(tmp_path / "out")
    for name in ("predictions", "importance", "market", "coverage"):
        pd.testing.assert_frame_equal(getattr(loaded, name), getattr(results, name), check_freq=False)
    assert loaded.run_info == results.run_info
