# Stock Behavior Prediction

Daily S&P 500 direction model (LightGBM, rolling training window chosen per run) and a long-only backtest with a Streamlit app. See [README.md](../README.md).

## Status

Round 2 (2026-09-27, spec: `docs/superpowers/specs/2026-09-27-v2-price-only-oop-design.md`): price history only, simple classes, variable training window, parallel walk-forward.

Round 3 (2026-09-29, spec: `docs/superpowers/specs/2026-09-29-v3-relative-horizon-design.md`): relative label (beats the day's median), horizon h with holding between decisions, refit every k decisions, predicted-vs-realized diagnostics.

## Working conventions

- **Never run `git commit` or `git push`.** This is blocked at the permission level (see `.claude/settings.json`, `permissions.deny`). Commits and pushes are entirely the user's responsibility — prepare and stage changes if asked, but leave committing to them.
- Use the project venv: `venv/bin/python -m pytest`, `venv/bin/python main.py download|features|run --horizon H --window X --retrain K`, `venv/bin/streamlit run app/dashboard.py`.
- All parameters live in `config/config.py` (`Config`, `TICKER_ALIASES`, `LGBM_PARAMS`). Aliases only for well-known pure renames; when in doubt, no alias.
- Lookahead invariants (keep `tests/test_project.py`, `tests/test_features.py`, `tests/test_walk_forward.py` and `tests/test_backtest_validation.py` green): decision every h sessions after the close of t with data dated ≤ t; buy at the open of t+1 and hold until the open after the next decision; label = the h-session return from the open of t+1 beats the median of that date's rows; training rows t−h−X … t−h−1; never use Yahoo's `Adj Close`; the universe at t is the constituents snapshot in force at t.
- Keep the code as simple as possible and consistent; use the code-simplifier plugin after larger changes.

## Plugins available (see .claude/settings.json)

- **superpowers**: process/workflow skills (brainstorming, systematic-debugging, TDD, writing-plans, executing-plans, code review skills, etc). Given the scale of the coming refactor, prefer `superpowers:brainstorming` before redesigning a module and `superpowers:writing-plans` / `superpowers:executing-plans` for multi-step rework, rather than editing ad hoc.
- **code-simplifier**: use after implementing/refactoring a module to clean up clarity and consistency without changing behavior.

## Engine layout

- `engine/universe.py` — `Universe`: point-in-time constituents, fetch of the snapshots CSV.
- `engine/prices.py` — `PriceData`: yfinance download (split-adjusted, dividends separate), calendar, panel, returns.
- `engine/features.py` — `FeatureBuilder`: stock, cross-section, correlation peers and market features (34, causal).
- `engine/dataset.py` — the (date, ticker) feature table and `horizon_target` (h-session return and relative label).
- `engine/walk_forward.py` — `WalkForward`: a decision every h sessions, a refit every k decisions on the last X sessions of known labels, fits in parallel (joblib); `MODELS`.
- `engine/portfolio.py`, `engine/metrics.py` — `Portfolio` (threshold, probability weights, holding h sessions with drifting weights, cash at ^IRX, costs at trades) and metrics (incl. deciles and top-minus-bottom spread).
- `engine/project.py` — `Project`: atomic download, cached features and daily returns panel (`data/out/features`), one run per `RunKey` (`data/out/runs/lightgbm-h5-w21-k1`); both are tied to the config values that produced them (`FEATURE_SETTINGS` + `FEATURES_FORMAT`, `RUN_SETTINGS`) and are rebuilt or hidden when those change.
