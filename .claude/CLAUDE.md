# Stock Behavior Prediction

Daily S&P 500 direction model (LightGBM, rolling training window chosen per run) and a long-only backtest with a Streamlit app. See [README.md](../README.md).

## Status

Round 2 (2026-09-27, spec: `docs/superpowers/specs/2026-09-27-v2-price-only-oop-design.md`): price history only, simple classes, variable training window, parallel walk-forward.

## Working conventions

- **Never run `git commit` or `git push`.** This is blocked at the permission level (see `.claude/settings.json`, `permissions.deny`). Commits and pushes are entirely the user's responsibility — prepare and stage changes if asked, but leave committing to them.
- Use the project venv: `venv/bin/python -m pytest`, `venv/bin/python main.py download|features|run --window N`, `venv/bin/streamlit run app/dashboard.py`.
- All parameters live in `config/config.py` (`Config`, `TICKER_ALIASES`, `LGBM_PARAMS`). Aliases only for well-known pure renames; when in doubt, no alias.
- Lookahead invariants (keep `tests/test_project.py`, `tests/test_features.py` and `tests/test_walk_forward.py` green): decision after the close of t with data dated ≤ t; execution open t+1 → open t+2; training rows t−X−1 … t−2; never use Yahoo's `Adj Close`; the universe at t is the constituents snapshot in force at t.
- Keep the code as simple as possible and consistent; use the code-simplifier plugin after larger changes.

## Plugins available (see .claude/settings.json)

- **superpowers**: process/workflow skills (brainstorming, systematic-debugging, TDD, writing-plans, executing-plans, code review skills, etc). Given the scale of the coming refactor, prefer `superpowers:brainstorming` before redesigning a module and `superpowers:writing-plans` / `superpowers:executing-plans` for multi-step rework, rather than editing ad hoc.
- **code-simplifier**: use after implementing/refactoring a module to clean up clarity and consistency without changing behavior.

## Engine layout

- `engine/universe.py` — `Universe`: point-in-time constituents, fetch of the snapshots CSV.
- `engine/prices.py` — `PriceData`: yfinance download (split-adjusted, dividends separate), calendar, panel, returns.
- `engine/features.py` — `FeatureBuilder`: stock, cross-section, correlation peers and market features (34, causal).
- `engine/dataset.py` — the (date, ticker) training table.
- `engine/walk_forward.py` — `WalkForward`: daily retraining on the last X days, decision dates in parallel (joblib).
- `engine/portfolio.py`, `engine/metrics.py` — `Portfolio` (threshold, probability weights, cash at ^IRX, costs) and metrics.
- `engine/project.py` — `Project`: atomic download, cached features (`data/out/features`), one run per window (`data/out/runs/wNNN`); both are tied to the config values that produced them (`FEATURE_SETTINGS`, `RUN_SETTINGS`) and are rebuilt or hidden when those change.
