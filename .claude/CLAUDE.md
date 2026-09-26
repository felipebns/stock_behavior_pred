# QuantFund Engine

ML backtesting engine (walk-forward validation, strategy evaluation, portfolio allocation, reporting). See [README.md](README.md) for the pipeline overview.

## Status

Rebuilt on 2026-09-26 (spec: `docs/superpowers/specs/2026-09-26-ml-backtest-redesign-design.md`). Daily LightGBM direction model over the point-in-time S&P 500, long-only probability-weighted portfolio, Streamlit app.

## Working conventions

- **Never run `git commit` or `git push`.** This is blocked at the permission level (see `.claude/settings.json`, `permissions.deny`). Commits and pushes are entirely the user's responsibility — prepare and stage changes if asked, but leave committing to them.
- Use the project venv: `venv/bin/python -m pytest`, `venv/bin/python main.py download|run`, `venv/bin/streamlit run app/dashboard.py`.
- All parameters live in `config/config.py` (`Config`, `TICKER_ALIASES`, `LGBM_PARAMS`).
- Lookahead invariants (keep `tests/test_pipeline.py` and `tests/test_features.py` green): decision after the close of t with data dated ≤ t (GPR ≤ t−7); execution open t+1 → open t+2; training rows t−X−1 … t−2; never use Yahoo's `Adj Close`.
- No custom parallelization unless a full run gets too slow; `data/reference/` is not to be imported.

## Plugins available (see .claude/settings.json)

- **superpowers**: process/workflow skills (brainstorming, systematic-debugging, TDD, writing-plans, executing-plans, code review skills, etc). Given the scale of the coming refactor, prefer `superpowers:brainstorming` before redesigning a module and `superpowers:writing-plans` / `superpowers:executing-plans` for multi-step rework, rather than editing ad hoc.
- **code-simplifier**: use after implementing/refactoring a module to clean up clarity and consistency without changing behavior.

## Engine layout

- `engine/data/universe.py` — constituent snapshots → membership matrix (aliases applied).
- `engine/data/prices.py` — yfinance download (split-adjusted, dividends separate), calendar, wide panel, total and forward open-to-open returns.
- `engine/data/macro.py` — fed funds futures rates, GPR features, as-of alignment with lag/staleness, ^IRX risk-free.
- `engine/features.py`, `engine/dataset.py` — 36 causal features and the (date, ticker) training table.
- `engine/model.py`, `engine/walk_forward.py` — LightGBM factory and daily rolling retraining.
- `engine/pipeline.py`, `engine/results.py` — orchestration and `data/out` persistence.
- `engine/portfolio.py`, `engine/metrics.py` — portfolio (threshold, probability weights, cash at rf, costs) and metrics.
