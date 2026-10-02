# Stock Behavior Prediction

Intraday S&P 500 direction model (five classifier types, one model for all stocks or one per stock, rolling training window chosen per run), a long-only backtest and a walk-forward choice of run and threshold, with a Streamlit app. See [README.md](../README.md).

## Status

Round 2 (2026-09-27, spec: `docs/superpowers/specs/2026-09-27-v2-price-only-oop-design.md`): price history only, simple classes, variable training window, parallel walk-forward.

Round 3 (2026-09-29, spec: `docs/superpowers/specs/2026-09-29-v3-relative-horizon-design.md`): relative label (beats the day's median), horizon h with holding between decisions, refit every k decisions, predicted-vs-realized diagnostics.

Round 4 (2026-10-01, spec: `docs/superpowers/specs/2026-10-01-v4-next-day-direction-design.md`): back to next-day direction (label: the stock goes up) and a daily portfolio; horizon h removed (round 3 stays in commit 9512cfd); refit every k sessions kept; stability fixes (single-class windows, CLI validation, empty backtests).

Round 5 (2026-10-01, spec: `docs/superpowers/specs/2026-10-01-v5-intraday-models-walkforward-design.md`): intraday trading (buy at the open, sell at the close), universe = members with a close on each of the last 252 sessions, five models (`engine/models.py`) fit once for all stocks or once per stock, `main.py sweep`, and a walk-forward selection of run and threshold (`engine/selection.py`, app tab).

## Working conventions

- **Agent boundaries (since 2026-10-01).** Agents (the main session and every subagent) may only change files inside this repository.
  - Commits are allowed: small and meaningful, one per finished task, never with failing tests; end messages with the Co-Authored-By line Claude Code supplies.
  - `git push` is blocked; pushing (and anything that publishes outside this machine) is the owner's job.
  - Enforced in `.claude/settings.json`: `permissions.deny` (push, remote/config changes, history rewrites like `reset --hard`/`clean`/`rebase`, `sudo`, `ssh`/`scp`/`rsync`, `crontab`, `systemctl`, `docker`, reading `~/.ssh`/`~/.aws`/`~/.gnupg`/`~/.config/gh`); the Bash sandbox (writes only inside the repo and Claude's temp dir; network without prompts only to `*.yahoo.com`, `raw.githubusercontent.com`, PyPI); and the `PreToolUse` hook `.claude/hooks/restrict_to_repo.py`, which denies Edit/Write/NotebookEdit outside the repo (except Claude's scratchpad and this project's memory). Bypass-permissions mode is disabled.
  - Never weaken these rules, change global/user configuration (`~/.claude`, `~/.gitconfig`, shell rc files), install system packages, or touch other repositories; if a task needs that, stop and ask the owner.
- Use the project venv: `venv/bin/python -m pytest`, `venv/bin/python main.py download|features|run --model M --scope pooled|per_stock --window X --retrain K|sweep ...`, `venv/bin/streamlit run app/dashboard.py`.
- All parameters live in `config/config.py` (`Config`, `TICKER_ALIASES`, `MODEL_PARAMS`). Aliases only for well-known pure renames; when in doubt, no alias.
- Lookahead invariants (keep `tests/test_project.py`, `tests/test_features.py`, `tests/test_walk_forward.py`, `tests/test_models.py`, `tests/test_selection.py` and `tests/test_backtest_validation.py` green): decision after the close of t with data dated ≤ t; buy at the open of t+1, sell at the close of t+1; label = fwd_ret > 0 with fwd_ret = Close[t+1] / Open[t+1] − 1; training rows t−X … t−1, refit every k sessions at the group's first decision; per-stock models train only on that stock's rows; a row's probability must not depend on the other rows predicted with it (bitwise); the walk-forward selection at day i sees only rows < i; never use Yahoo's `Adj Close`; the universe at t is the constituents snapshot in force at t, restricted to members with a close on each of the last 252 sessions.
- Keep the code as simple as possible and consistent; use the code-simplifier plugin after larger changes.

## Plugins available (see .claude/settings.json)

- **superpowers**: process/workflow skills (brainstorming, systematic-debugging, TDD, writing-plans, executing-plans, code review skills, etc). Given the scale of the coming refactor, prefer `superpowers:brainstorming` before redesigning a module and `superpowers:writing-plans` / `superpowers:executing-plans` for multi-step rework, rather than editing ad hoc.
- **code-simplifier**: use after implementing/refactoring a module to clean up clarity and consistency without changing behavior.

## Engine layout

- `engine/universe.py` — `Universe`: point-in-time constituents, fetch of the snapshots CSV.
- `engine/prices.py` — `PriceData`: yfinance download (split-adjusted, dividends separate), calendar, panel, returns.
- `engine/features.py` — `FeatureBuilder`: stock, cross-section, correlation peers and market features (34, causal).
- `engine/dataset.py` — the (date, ticker) table (complete-history universe): features, intraday `fwd_ret` and `label`.
- `engine/models.py` — `MODELS` (LightGBM, row-wise logistic, random forest, extra trees, naive Bayes: name → (label, factory(params, n_rows))) and `importance`.
- `engine/walk_forward.py` — `WalkForward`: a decision every session, a refit every k sessions on the last X sessions of known labels, scope `pooled` or `per_stock` (`SCOPES`), small or single-class training sets predict their up-frequency, fits in parallel (joblib).
- `engine/portfolio.py`, `engine/metrics.py`, `engine/selection.py` — `Portfolio` (intraday: threshold, probability weights, cash at ^IRX, a fee on each buy and sell), metrics (incl. deciles and top-minus-bottom spread) and `walk_forward_selection` (run and threshold re-picked by past net Sharpe).
- `engine/project.py` — `Project`: atomic download, cached features (`data/out/features`), one run per `RunKey(model, scope, window, retrain_every)` (`data/out/runs/lightgbm-pooled-w21-k1`); features are tied to `FEATURE_SETTINGS` + `FEATURES_FORMAT`, runs to `RUN_SETTINGS` and their own model's parameters, and are rebuilt or hidden when those change.
