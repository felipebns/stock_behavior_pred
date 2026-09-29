# Rodada 3 — rótulo relativo, horizonte h, retreino a cada k, previsto × realizado — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trocar o rótulo "a ação sobe" pelo relativo "rende acima da mediana do índice", decidir a cada h pregões segurando a carteira entre decisões, retreinar a cada k decisões, validar a carteira e os custos com um simulador independente no estilo do curso e mostrar no app o previsto contra o realizado.

**Architecture:**
- O dataset salvo passa a ter só as features.
- O alvo de cada h é calculado na hora do run a partir de um painel de retornos diários salvo junto com as features.
- `WalkForward` ganha `horizon` e `retrain_every`. `Portfolio` passa a receber esse painel e segura as posições h pregões.
- `Project` identifica cada run por uma `RunKey(horizon, window, retrain_every, model)`.
- O app ganha três entradas na barra lateral e a aba "Previsto vs realizado".

**Tech Stack:** Python 3.14 (venv), pandas 3, numpy 2, pyarrow, scikit-learn, LightGBM, joblib, Streamlit + Altair, pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-v3-relative-horizon-design.md`. O que ele não muda continua valendo de `docs/superpowers/specs/2026-09-27-v2-price-only-oop-design.md`.

## Global Constraints

- **Nunca rodar `git commit` nem `git push`** (bloqueado); o usuário faz os commits. Este plano não tem passos de commit.
- Comandos pelo venv: `venv/bin/python -m pytest ...`, `venv/bin/python main.py ...`.
- **Convenção de tempo:**
  - a decisão acontece após o fechamento de *t*, só com dados ≤ *t*;
  - *r(s)* = (Open[*s*+2] + Div[*s*+2]) / Open[*s*+1] − 1;
  - *Rₕ(s)* = ∏ₖ₌₀^{h−1}(1 + *r*(*s*+k)) − 1;
  - rótulo = *Rₕ(s)* > mediana das linhas da data *s*;
  - o treino da decisão *t* usa as linhas *s* ∈ [*t*−h−X, *t*−h−1];
  - as decisões acontecem a cada h posições de calendário a partir de `backtest_start`;
  - a carteira compra na abertura de *t*+1 e segura até a abertura seguinte à próxima decisão.
- Paralelização só no `WalkForward` (processos `joblib`, LightGBM `n_jobs=1`), bit a bit igual ao sequencial.
- **Defaults novos:** `horizon_days=5`, `retrain_every=1`. Os demais continuam iguais: `train_window_days=21`, `threshold=0.55`, `cost_bps=5.0` etc.
- Métricas mantidas; novas: `deciles`, `decile_spread`.
- Código o mais simples possível. Sem comentários, a não ser quando o porquê não for óbvio. Identificadores e docstrings em inglês, UI do app em português, README e CLAUDE.md em inglês.
- As notas do usuário no fim do `main.py` (o bloco `"""` depois de `if __name__ == "__main__":`) são preservadas letra por letra: edite o `main.py` só com `Edit` pontual.

## Review Focus

1. **O último período passa do fim dos dados** (h grande): o backtest não quebra, as posições realizam só os dias disponíveis e o rótulo fica desconhecido. *(Task 4: `test_last_holding_period_is_cut_at_the_last_realized_day`)*
2. **Um recorte sem nenhum resultado conhecido** (por exemplo, "1 mês" com h = 21): as métricas do modelo saem NaN, sem exceção. *(Task 3: `test_model_metrics_without_known_outcomes_are_nan`)*
3. **Limiar que ninguém passa com h > 1**: a carteira fica toda em caixa, sem erro no pivot vazio. *(Task 4: `test_threshold_never_met_is_all_cash`, com h = 3)*
4. **Ação deslistada no meio de uma posição de h dias**: o valor congela, ela é vendida na próxima troca e o giro a conta. *(Task 4: simulador, `test_portfolio_matches_an_event_by_event_simulation`)*
5. **Janela + horizonte maior que o histórico antes de `backtest_start`**: o erro informa o máximo levando h em conta. *(Task 2: `test_the_maximum_window_accounts_for_the_horizon`)*

---

## File Structure

```
config/config.py                     + horizon_days, retrain_every
engine/dataset.py                    build_dataset (só features) + horizon_target
engine/walk_forward.py               MODELS, WalkForward(horizon, retrain_every)
engine/portfolio.py                  Portfolio.backtest(predictions, market, returns, horizon)
engine/metrics.py                    model_metrics robusto + deciles + decile_spread
engine/project.py                    RunKey, FeatureSet.returns, FEATURES_FORMAT, runs por chave
main.py                              run --horizon --window --retrain
app/dashboard.py                     barra lateral h/X/k, abas novas
tests/test_backtest_validation.py    simulador evento a evento (novo)
tests/synthetic.py, conftest.py      RUNS = (RunKey(1, 40), RunKey(5, 40, 2))
```

Ordem das tarefas: 1–3 são aditivas, e a suíte continua verde. A Task 4 muda a assinatura de `Portfolio.backtest`: `test_main` e `test_app` ficam vermelhos até as Tasks 5 e 6. Cada tarefa roda os testes dos módulos que toca, e a suíte inteira volta a ficar verde no fim da Task 6.

---

### Task 1: Config e alvo relativo de h pregões

**Files:**
- Modify: `config/config.py` (campos do `Config`)
- Modify: `engine/dataset.py` (nova função `horizon_target`)
- Test: `tests/test_config.py`, `tests/test_dataset.py`

**Interfaces:**
- Produces: `Config.horizon_days: int = 5`, `Config.retrain_every: int = 1`; `horizon_target(dataset: pd.DataFrame, returns: pd.DataFrame, horizon: int) -> pd.DataFrame` com colunas `fwd_ret`, `label`, mesmo índice (date, ticker) do dataset. `returns` é o painel largo (datas × tickers) de *r(s)*.

- [ ] **Step 1: Write the failing tests**

Em `tests/test_config.py`, acrescente:

```python
def test_run_defaults():
    assert (CONFIG.horizon_days, CONFIG.train_window_days, CONFIG.retrain_every) == (5, 21, 1)
```

Em `tests/test_dataset.py`, troque o import por `from engine.dataset import build_dataset, horizon_target` e acrescente no fim:

```python
def target_inputs():
    dates = pd.bdate_range("2021-01-01", periods=5, name="date")
    returns = pd.DataFrame({
        "A": [0.01, 0.02, -0.01, 0.03, np.nan],
        "B": [0.03, 0.00, 0.02, np.nan, np.nan],
        "C": [-0.02, 0.01, 0.05, 0.01, np.nan],
    }, index=dates)
    index = pd.MultiIndex.from_product([dates[:4], ["A", "B", "C"]], names=["date", "ticker"])
    return pd.DataFrame({"x": 0.0}, index=index), returns


def test_one_session_label_beats_the_median_of_the_date():
    dataset, returns = target_inputs()
    target = horizon_target(dataset, returns, 1)
    assert list(target.columns) == ["fwd_ret", "label"]
    assert target.index.equals(dataset.index)
    first = target.xs(returns.index[0], level="date")
    assert first["fwd_ret"].tolist() == pytest.approx([0.01, 0.03, -0.02])
    assert first["label"].tolist() == [0.0, 1.0, 0.0]


def test_horizon_compounds_the_next_sessions_and_ranks_against_the_date_median():
    dataset, returns = target_inputs()
    first = horizon_target(dataset, returns, 2).xs(returns.index[0], level="date")
    assert first["fwd_ret"].tolist() == pytest.approx([1.01 * 1.02 - 1, 1.03 * 1.00 - 1, 0.98 * 1.01 - 1])
    assert first["label"].tolist() == [1.0, 0.0, 0.0]


def test_a_missing_session_leaves_the_outcome_unknown():
    dataset, returns = target_inputs()
    target = horizon_target(dataset, returns, 2)
    third = target.xs(returns.index[2], level="date")
    assert np.isnan(third.loc["B", "fwd_ret"]) and np.isnan(third.loc["B", "label"])
    assert third.loc["A", "label"] == 0.0 and third.loc["C", "label"] == 1.0
    assert target.xs(returns.index[3], level="date")["label"].isna().all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_config.py tests/test_dataset.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'horizon_target'` (and `AttributeError: 'Config' object has no attribute 'horizon_days'`).

- [ ] **Step 3: Implement**

Em `config/config.py`, no `Config`, troque a linha `train_window_days: int = 21` por:

```python
    horizon_days: int = 5
    train_window_days: int = 21
    retrain_every: int = 1
```

Em `engine/dataset.py`, troque a docstring do módulo por `"""Long table indexed by (date, ticker): features known after the close of `date`; the target depends on the horizon."""` e acrescente no fim:

```python
def horizon_target(dataset: pd.DataFrame, returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """fwd_ret: open of t+1 to the open of t+1+horizon (dividends reinvested), NaN if a session has no return;
    label: 1 when fwd_ret beats the median of that date's rows. Both are known only after the open of t+1+horizon."""
    compounded = np.expm1(np.log1p(returns).rolling(horizon, min_periods=horizon).sum().shift(1 - horizon))
    rows = compounded.index.get_indexer(dataset.index.get_level_values("date"))
    cols = compounded.columns.get_indexer(dataset.index.get_level_values("ticker"))
    fwd_ret = pd.Series(compounded.to_numpy()[rows, cols], index=dataset.index)
    median = fwd_ret.groupby(level="date").transform("median")
    return pd.DataFrame({"fwd_ret": fwd_ret, "label": (fwd_ret > median).astype(float).where(fwd_ret.notna())})
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_config.py tests/test_dataset.py -q -p no:cacheprovider`
Expected: PASS (all).

---

### Task 2: `WalkForward` com horizonte e retreino a cada k

**Files:**
- Modify: `engine/walk_forward.py`
- Test: `tests/test_walk_forward.py`

**Interfaces:**
- Consumes: dataset com colunas de features + `fwd_ret`, `label` (Task 1).
- Produces: `MODELS = {"lightgbm": LGBMClassifier}`; `WalkForward(train_window, params, horizon=1, retrain_every=1, min_child_share=0.004, min_child_floor=20, n_jobs=1, model_class=LGBMClassifier)`; `run(...)` com a mesma assinatura e saída de antes; `importance` com uma linha por ajuste, indexada pela data do ajuste.

- [ ] **Step 1: Write the failing tests**

Em `tests/test_walk_forward.py`, acrescente depois de `test_window_too_long_for_the_start_date_is_refused`:

```python
def test_horizon_spaces_decisions_and_moves_the_training_rows_back():
    run_spy(spy_dataset(), window=3, horizon=2)
    fits = [days for kind, days, _ in Spy.log if kind == "fit"]
    predicts = [days for kind, days, _ in Spy.log if kind == "predict"]
    decisions = [7, 9, 11, 13]
    assert predicts == [[k] for k in decisions]
    assert fits == [list(range(k - 2 - 3, k - 2)) for k in decisions]


def test_one_fit_serves_retrain_every_consecutive_decisions():
    result = run_spy(spy_dataset(), window=3, horizon=2, retrain_every=3)
    assert [kind for kind, _, _ in Spy.log] == ["fit", "predict", "predict", "predict", "fit", "predict"]
    assert [days for kind, days, _ in Spy.log if kind == "fit"] == [[2, 3, 4], [8, 9, 10]]
    assert list(result.importance.index) == [DATES[7], DATES[13]]
    assert sorted(result.predictions["date"].unique()) == [DATES[k] for k in (7, 9, 11, 13)]


def test_the_maximum_window_accounts_for_the_horizon():
    with pytest.raises(ValueError, match="máximo é 4"):
        run_spy(spy_dataset(), window=5, horizon=2, start=DATES[8])


class Echo:
    """Predicts, for each row, its own first feature: every probability must land on its own (date, ticker) row."""

    def __init__(self, **params):
        pass

    def fit(self, X, y):
        self.feature_importances_ = np.ones(X.shape[1])
        return self

    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


def test_each_probability_lands_on_its_own_row():
    dataset, dates = random_dataset(n_days=30, n_tickers=7)
    dataset["a"] = np.random.default_rng(1).uniform(size=len(dataset)).astype(np.float32)
    predictions = WalkForward(5, {}, horizon=2, retrain_every=3, n_jobs=2, model_class=Echo).run(
        dataset, ["a", "b"], dates).predictions
    rows = pd.MultiIndex.from_frame(predictions[["date", "ticker"]])
    np.testing.assert_array_equal(predictions["prob"].to_numpy(), dataset.loc[rows, "a"].to_numpy())
```

E troque `test_parallel_run_is_identical_to_sequential` por:

```python
@pytest.mark.parametrize("horizon, retrain_every", [(1, 1), (3, 2)])
def test_parallel_run_is_identical_to_sequential(horizon, retrain_every):
    dataset, dates = random_dataset()
    params = {**LGBM_PARAMS, "n_estimators": 20}
    options = {"horizon": horizon, "retrain_every": retrain_every}
    sequential = WalkForward(10, params, n_jobs=1, **options).run(dataset, ["a", "b", "c"], dates)
    parallel = WalkForward(10, params, n_jobs=2, **options).run(dataset, ["a", "b", "c"], dates)
    pd.testing.assert_frame_equal(sequential.predictions, parallel.predictions, check_exact=True)
    pd.testing.assert_frame_equal(sequential.importance, parallel.importance, check_exact=True)
    assert ((sequential.predictions["prob"] > 0) & (sequential.predictions["prob"] < 1)).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_walk_forward.py -q -p no:cacheprovider`
Expected: FAIL — `TypeError: WalkForward.__init__() got an unexpected keyword argument 'horizon'`.

- [ ] **Step 3: Implement**

Substitua `engine/walk_forward.py` inteiro por:

```python
"""Walk-forward: a decision every `horizon` sessions, each predicted by a model fit only on labels already known;
decision dates run in parallel."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from lightgbm import LGBMClassifier

MODELS = {"lightgbm": LGBMClassifier}


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    importance: pd.DataFrame


class WalkForward:
    """Row s's label needs the open of s+1+horizon, so a fit at decision t uses exactly the rows dated
    t-horizon-X .. t-horizon-1; that model also predicts the next retrain_every-1 decisions, all after t."""

    def __init__(self, train_window: int, params: dict, horizon: int = 1, retrain_every: int = 1,
                 min_child_share: float = 0.004, min_child_floor: int = 20, n_jobs: int = 1, model_class=LGBMClassifier):
        self.train_window = train_window
        self.params = params
        self.horizon = horizon
        self.retrain_every = retrain_every
        self.min_child_share = min_child_share
        self.min_child_floor = min_child_floor
        self.n_jobs = n_jobs
        self.model_class = model_class

    def decisions(self, row_pos: np.ndarray, calendar: pd.DatetimeIndex, start=None) -> np.ndarray:
        first = row_pos[0] + self.train_window + self.horizon
        begin = first if start is None else calendar.searchsorted(pd.Timestamp(start))
        if begin < first:
            raise ValueError(f"a janela de {self.train_window} pregões com horizonte de {self.horizon} precisa de "
                             f"{self.train_window + self.horizon} pregões de dados antes de {pd.Timestamp(start).date()}; "
                             f"o máximo é {begin - row_pos[0] - self.horizon}")
        with_rows = np.unique(row_pos)
        schedule = np.arange(begin, with_rows[-1] + 1, self.horizon)
        decisions = schedule[np.isin(schedule, with_rows)]
        if len(decisions) == 0:
            raise ValueError("nenhuma data de decisão depois do início pedido")
        return decisions

    def run(self, dataset: pd.DataFrame, features: list[str], calendar: pd.DatetimeIndex, start=None,
            progress=None) -> WalkForwardResult:
        row_pos = calendar.get_indexer(dataset.index.get_level_values("date"))
        X = dataset[features].to_numpy(dtype=np.float32)
        y = dataset["label"].to_numpy(dtype=float)
        decisions = self.decisions(row_pos, calendar, start)
        fits = [decisions[i:i + self.retrain_every] for i in range(0, len(decisions), self.retrain_every)]
        blocks = np.array_split(np.arange(len(fits)), min(len(fits), 4 * effective_n_jobs(self.n_jobs)))
        jobs = (delayed(_fit_block)(X, y, row_pos, [fits[i] for i in block], self.train_window, self.horizon,
                                    self.params, self.min_child_share, self.min_child_floor, self.model_class)
                for block in blocks)
        results = []
        for done, result in enumerate(Parallel(n_jobs=self.n_jobs, return_as="generator")(jobs), start=1):
            results.append(result)
            if progress is not None:
                progress(done / len(blocks))
        rows = dataset[np.isin(row_pos, decisions)]
        predictions = pd.DataFrame({
            "date": rows.index.get_level_values("date"),
            "ticker": rows.index.get_level_values("ticker"),
            "prob": np.concatenate([probs for probs, _ in results]),
            "fwd_ret": rows["fwd_ret"].to_numpy(),
            "label": rows["label"].to_numpy(),
        })
        importance = pd.DataFrame(np.vstack([gains for _, gains in results]),
                                  index=pd.DatetimeIndex(calendar[[group[0] for group in fits]], name="date"),
                                  columns=features)
        return WalkForwardResult(predictions, importance)


def _fit_block(X, y, row_pos, fits, window, horizon, params, min_child_share, min_child_floor, model_class):
    probs, gains = [], []
    for group in fits:
        lo = np.searchsorted(row_pos, group[0] - horizon - window, side="left")
        hi = np.searchsorted(row_pos, group[0] - horizon - 1, side="right")
        known = ~np.isnan(y[lo:hi])
        model = model_class(**params, min_child_samples=max(min_child_floor, round(min_child_share * known.sum())))
        model.fit(X[lo:hi][known], y[lo:hi][known].astype(int))
        for t in group:
            first, last = np.searchsorted(row_pos, t, side="left"), np.searchsorted(row_pos, t, side="right")
            probs.append(model.predict_proba(X[first:last])[:, 1])
        gain = np.asarray(model.feature_importances_, dtype=float)
        gains.append(gain / gain.sum() if gain.sum() > 0 else gain)
    return np.concatenate(probs), np.vstack(gains)
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_walk_forward.py -q -p no:cacheprovider`
Expected: PASS (all). Se o `Echo` com `n_jobs=2` não puder ser importado pelos processos do joblib, use `n_jobs=1` nesse teste. O teste de paralelo cobre a remontagem dos blocos. Nesse caso, registre uma ruling.

---

### Task 3: Métricas robustas e diagnóstico por decis

**Files:**
- Modify: `engine/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: predictions com `date`, `prob`, `fwd_ret` (retorno de h pregões), `label` (relativo).
- Produces: `model_metrics` devolve NaN, sem erro, quando não há resultado conhecido ou só uma classe; `deciles(predictions) -> DataFrame` (índice `decile` 1..10; colunas `prob`, `label`, `relative`); `decile_spread(predictions) -> Series` (índice `date`).

- [ ] **Step 1: Write the failing tests**

Em `tests/test_metrics.py`, troque o import por `from engine.metrics import decile_spread, deciles, drawdown, model_metrics, monthly_auc, performance_metrics` e acrescente:

```python
def test_model_metrics_without_known_outcomes_are_nan():
    empty = pd.DataFrame({"date": pd.to_datetime(["2022-01-03"] * 2), "prob": [0.4, 0.6],
                          "fwd_ret": [np.nan, np.nan], "label": [np.nan, np.nan]})
    assert all(np.isnan(value) for value in model_metrics(empty).values())
    one_class = empty.assign(fwd_ret=[0.01, 0.02], label=[1.0, 1.0])
    assert np.isnan(model_metrics(one_class)["auc"])


def ranked_predictions(n_stocks: int = 10) -> pd.DataFrame:
    """Two dates; prob rises with i; returns rise with i on the first date and fall on the second."""
    rows = [
        {"date": date, "ticker": f"T{i}", "prob": (i + 0.5) / n_stocks, "fwd_ret": sign * 0.01 * i}
        for date, sign in zip(pd.to_datetime(["2022-01-03", "2022-01-10"]), [1, -1])
        for i in range(n_stocks)
    ]
    frame = pd.DataFrame(rows)
    return frame.assign(label=(frame["fwd_ret"] > frame.groupby("date")["fwd_ret"].transform("median")).astype(float))


def test_deciles_average_within_each_date_then_across_dates():
    table = deciles(ranked_predictions())
    assert list(table.index) == list(range(1, 11))
    assert list(table.columns) == ["prob", "label", "relative"]
    assert table["prob"].tolist() == pytest.approx([(i + 0.5) / 10 for i in range(10)])
    assert table["label"].tolist() == pytest.approx([0.5] * 10)
    assert table["relative"].tolist() == pytest.approx([0.0] * 10)


def test_decile_spread_is_top_minus_bottom_per_date():
    spread = decile_spread(ranked_predictions())
    assert list(spread.index) == list(pd.to_datetime(["2022-01-03", "2022-01-10"]))
    assert spread.tolist() == pytest.approx([0.09, -0.09])


def test_decile_spread_uses_at_least_one_stock_per_side():
    assert decile_spread(ranked_predictions(5)).tolist() == pytest.approx([0.04, -0.04])


def test_rows_with_unknown_outcome_are_left_out_of_the_deciles():
    predictions = ranked_predictions()
    predictions.loc[predictions["ticker"] == "T9", ["fwd_ret", "label"]] = np.nan
    assert decile_spread(predictions).tolist() == pytest.approx([0.08, -0.08])
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_metrics.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'decile_spread'`.

- [ ] **Step 3: Implement**

Em `engine/metrics.py`, troque `model_metrics` por:

```python
def model_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """NaN where there is nothing to measure: no known outcome yet, or a single class."""
    known = predictions.dropna(subset=["label"])
    label, prob = known["label"].astype(int), known["prob"]
    return {
        "auc": float(roc_auc_score(label, prob)) if label.nunique() == 2 else float("nan"),
        "accuracy": float(((prob > 0.5).astype(int) == label).mean()) if len(known) else float("nan"),
        "ic": float(daily_ic(known).mean()) if len(known) else float("nan"),
    }
```

e acrescente no fim:

```python
def _ranked(predictions: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known outcome, with the return relative to the day's median and the probability rank and decile that day."""
    known = predictions.dropna(subset=["label"])
    by_date = known.groupby("date")
    rank = by_date["prob"].rank(method="first")
    size = by_date["prob"].transform("size")
    return known.assign(
        relative=known["fwd_ret"] - by_date["fwd_ret"].transform("median"),
        rank=rank, size=size, decile=np.ceil(10 * rank / size).astype(int),
    )


def deciles(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per probability decile (1 = lowest that day): mean prob, share that beat the median and return relative to it —
    averaged within each date first, then across dates."""
    per_day = _ranked(predictions).groupby(["date", "decile"])[["prob", "label", "relative"]].mean()
    return per_day.groupby(level="decile").mean()


def decile_spread(predictions: pd.DataFrame) -> pd.Series:
    """Per decision date: mean return of the top 10% by probability minus the bottom 10% (at least one stock each)."""
    ranked = _ranked(predictions)
    side = np.maximum(1, ranked["size"] // 10)
    top = ranked[ranked["rank"] > ranked["size"] - side].groupby("date")["fwd_ret"].mean()
    bottom = ranked[ranked["rank"] <= side].groupby("date")["fwd_ret"].mean()
    return top - bottom
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_metrics.py -q -p no:cacheprovider`
Expected: PASS (all).

---

### Task 4: `Portfolio` que segura h pregões e simulador estilo curso

**Files:**
- Modify: `engine/portfolio.py` (inteiro)
- Modify: `tests/test_portfolio.py`
- Create: `tests/test_backtest_validation.py`

**Interfaces:**
- Consumes: predictions (Task 2); `market` (índice `decision_date`; `holding_date`, `bench_fwd_ret`, `rf_daily`); `returns` (painel largo de *r(s)*).
- Produces: `Portfolio.backtest(predictions, market, returns, horizon=1) -> BacktestResult`.
  - `daily`: índice `holding_date`; colunas `decision_date` (a decisão em vigor), `gross`, `net`, `bench`, `rf`, `n_positions`, `invested`, `rebalance`, `turnover`, `cost`, `n_missing`.
  - `positions`: `POSITION_COLUMNS = ["decision_date", "holding_date", "ticker", "prob", "weight", "realized", "contribution"]`.
  - `select(predictions)` devolve `date, ticker, prob, weight`.

- [ ] **Step 1: Write the failing tests**

Substitua `tests/test_portfolio.py` inteiro por:

```python
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
```

Crie `tests/test_backtest_validation.py`:

```python
"""Independent check of Portfolio against an event-by-event simulation in the style of the Algotrading course
(Aulas 4 e 5): cash and share counts, marking to market at each open, a fee on the traded value, idle cash earning
the risk-free rate (carry), dividends reinvested at the open they go ex. It shares no code with engine.portfolio."""
import numpy as np
import pandas as pd
import pytest

from engine.portfolio import Portfolio
from engine.prices import forward_open_return

CAPITAL = 1_000_000.0
THRESHOLD = 0.55
COST_BPS = 10.0


def market_data(n_days: int = 40, seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days, name="date")
    tickers = ["A", "B", "C", "D", "E"]
    opens = pd.DataFrame(50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (n_days, len(tickers))), axis=0)),
                         index=dates, columns=tickers)
    opens.loc[dates[17]:, "E"] = np.nan
    dividends = pd.DataFrame(0.0, index=dates, columns=tickers)
    dividends.loc[dates[[6, 12, 23]], ["A", "C"]] = 0.4
    rf_daily = pd.Series(0.0002 + 0.0001 * np.sin(np.arange(n_days)), index=dates)
    return opens, dividends, rf_daily


def model_output(opens: pd.DataFrame, horizon: int, seed: int = 5) -> pd.DataFrame:
    """Random probabilities at each decision (every `horizon` sessions); every 4th decision nobody passes."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, date in enumerate(opens.index[2:-3:horizon]):
        high = 0.5 if k % 4 == 3 else 0.8
        rows += [{"date": date, "ticker": t, "prob": rng.uniform(0.3, high)}
                 for t in opens.columns if not np.isnan(opens.at[date, t])]
    return pd.DataFrame(rows)


def market_frame(opens: pd.DataFrame, rf_daily: pd.Series) -> pd.DataFrame:
    dates = opens.index
    return pd.DataFrame({
        "holding_date": pd.Series(dates, index=dates).shift(-1),
        "bench_fwd_ret": forward_open_return(opens.mean(axis=1)),
        "rf_daily": rf_daily,
    }, index=dates)


def simulate(opens, dividends, rf_daily, predictions) -> pd.Series:
    """Portfolio value at each open, before trading, keyed by the holding day that ends at that open."""
    targets = {}
    for date, day in predictions.groupby("date"):
        chosen = day[day["prob"] > THRESHOLD]
        targets[date] = dict(zip(chosen["ticker"], chosen["prob"] / chosen["prob"].sum()))
    dates, price = opens.index, opens.ffill()
    cash, shares, equity = CAPITAL, {}, {}
    for i in range(dates.get_loc(min(targets)), len(dates) - 2):
        decided, trade_open, next_open = dates[i], dates[i + 1], dates[i + 2]
        if decided in targets:
            holdings = {t: q * price.at[trade_open, t] for t, q in shares.items()}
            value = cash + sum(holdings.values())
            target = targets[decided]
            traded = sum(abs(target.get(t, 0.0) * value - holdings.get(t, 0.0)) for t in set(target) | set(holdings))
            value -= traded * COST_BPS / 10_000
            shares = {t: w * value / price.at[trade_open, t] for t, w in target.items()}
            cash = 0.0 if target else value
        cash *= 1.0 + rf_daily.at[decided]
        for t in shares:
            if not np.isnan(opens.at[next_open, t]):
                shares[t] *= 1.0 + dividends.at[next_open, t] / opens.at[next_open, t]
        equity[trade_open] = cash + sum(q * price.at[next_open, t] for t, q in shares.items())
    return pd.Series(equity)


@pytest.mark.parametrize("horizon", [1, 5])
def test_portfolio_matches_an_event_by_event_simulation(horizon):
    opens, dividends, rf_daily = market_data()
    predictions = model_output(opens, horizon)
    daily = Portfolio(THRESHOLD, COST_BPS).backtest(
        predictions, market_frame(opens, rf_daily), forward_open_return(opens, dividends), horizon).daily
    assert daily["invested"].any() and not daily["invested"].all()
    assert daily["n_missing"].sum() > 0 and (daily["turnover"] > 0).sum() > 3
    expected = CAPITAL * (1.0 + daily["net"]).cumprod()
    simulated = simulate(opens, dividends, rf_daily, predictions).reindex(daily.index)
    np.testing.assert_allclose(simulated.to_numpy(), expected.to_numpy(), rtol=1e-10)
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_portfolio.py tests/test_backtest_validation.py -q -p no:cacheprovider`
Expected: FAIL — `TypeError: Portfolio.backtest() takes 3 positional arguments but 5 were given`.

- [ ] **Step 3: Implement**

Substitua `engine/portfolio.py` inteiro por:

```python
"""Long-only portfolio: at each decision, buy at the next open the stocks with prob > threshold (weights ∝ prob) and
hold them until the next decision, their weights drifting with prices."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

POSITION_COLUMNS = ["decision_date", "holding_date", "ticker", "prob", "weight", "realized", "contribution"]


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame


class Portfolio:
    """Buys at the open after each decision the stocks with prob > threshold, weighted by prob; otherwise cash at the risk-free rate."""

    def __init__(self, threshold: float, cost_bps: float = 0.0):
        self.threshold = threshold
        self.cost_bps = cost_bps

    def select(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Only `prob` drives the choice."""
        chosen = predictions.loc[predictions["prob"] > self.threshold, ["date", "ticker", "prob"]].copy()
        chosen["weight"] = chosen["prob"] / chosen.groupby("date")["prob"].transform("sum")
        return chosen

    def backtest(self, predictions: pd.DataFrame, market: pd.DataFrame, returns: pd.DataFrame,
                 horizon: int = 1) -> BacktestResult:
        """Day s earns `returns` row s (open of s+1 to open of s+2) with the decision in force at s. Trades, and their
        cost, happen only at the open after a decision; the fee is charged on the value traded to reach the target
        weights and comes out of every position. A held stock without a return that day (delisting, halt) counts as 0:
        dropping it would use future information."""
        decisions = pd.DatetimeIndex(np.sort(predictions["date"].unique()))
        window = market.iloc[market.index.get_loc(decisions[0]):market.index.get_loc(decisions[-1]) + horizon]
        last_realized = window["bench_fwd_ret"].last_valid_index()
        window = window.iloc[:0] if last_realized is None else window.loc[:last_realized]
        gaps = window.index[window["bench_fwd_ret"].isna()]
        if len(gaps):
            raise ValueError(f"benchmark sem retorno realizado dentro do período ({', '.join(str(d.date()) for d in gaps)}); "
                             "baixe os preços de novo")
        days = window.index
        in_force = decisions[decisions.searchsorted(days, side="right") - 1]
        period, periods = pd.factorize(in_force)
        chosen = self.select(predictions[predictions["date"].isin(periods)])
        weights = chosen.pivot(index="date", columns="ticker", values="weight").reindex(in_force).fillna(0.0)
        raw = returns.reindex(index=days, columns=weights.columns).to_numpy()
        held = weights.to_numpy()
        growth = pd.DataFrame(1.0 + np.nan_to_num(raw)).groupby(period).cumprod()
        value_start = held * growth.groupby(period).shift(1, fill_value=1.0).to_numpy()
        value_end = held * growth.to_numpy()
        invested = value_start.sum(axis=1) > 0
        stock_return = np.divide(value_end.sum(axis=1), value_start.sum(axis=1), out=np.ones(len(days)), where=invested) - 1.0
        gross = np.where(invested, stock_return, window["rf_daily"].to_numpy())
        drifted = np.divide(value_end, value_end.sum(axis=1, keepdims=True), out=np.zeros_like(value_end),
                            where=invested[:, None])
        before = np.vstack([np.zeros((1, held.shape[1])), drifted[:-1]])
        rebalance = days.isin(decisions)
        turnover = np.where(rebalance, np.abs(held - before).sum(axis=1), 0.0)
        cost = turnover * self.cost_bps / 10_000.0
        daily = pd.DataFrame({
            "decision_date": in_force,
            "gross": gross,
            "net": (1.0 + gross) * (1.0 - cost) - 1.0,
            "bench": window["bench_fwd_ret"].to_numpy(),
            "rf": window["rf_daily"].to_numpy(),
            "n_positions": (held > 0).sum(axis=1),
            "invested": invested,
            "rebalance": rebalance,
            "turnover": turnover,
            "cost": cost,
            "n_missing": (np.isnan(raw) & (held > 0)).sum(axis=1),
        }, index=pd.DatetimeIndex(window["holding_date"], name="holding_date"))
        last_day = pd.Series(np.arange(len(days))).groupby(period).max().to_numpy()
        realized = growth.to_numpy()[last_day] - 1.0
        positions = chosen.rename(columns={"date": "decision_date"})
        positions["holding_date"] = positions["decision_date"].map(market["holding_date"])
        positions["realized"] = realized[periods.get_indexer(positions["decision_date"]),
                                         weights.columns.get_indexer(positions["ticker"])]
        positions["contribution"] = positions["weight"] * positions["realized"]
        return BacktestResult(daily=daily, positions=positions[POSITION_COLUMNS].reset_index(drop=True))
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_portfolio.py tests/test_backtest_validation.py -q -p no:cacheprovider`
Expected: PASS (all). As asserções-guarda do simulador (há caixa, há ausente, há trocas) dependem das sementes. Se alguma falhar por acaso, troque a semente de `model_output` (e registre uma ruling); a comparação das curvas em si não muda.

- [ ] **Step 5: Mutation check**

Troque temporariamente `before = np.vstack([...drifted[:-1]])` por `before = np.vstack([np.zeros((1, held.shape[1])), held[:-1]])` (esquece a deriva). Rode `tests/test_backtest_validation.py`: tem que falhar. Desfaça a troca e rode de novo: PASS.

---

### Task 5: `Project` com `RunKey`, painel de retornos e CLI

**Files:**
- Modify: `engine/dataset.py` (`build_dataset` sem alvo)
- Modify: `engine/project.py`
- Modify: `main.py` (só a parte de cima; as notas no fim ficam)
- Modify: `tests/synthetic.py`, `tests/conftest.py`, `tests/test_dataset.py`, `tests/test_project.py`, `tests/test_main.py`

**Interfaces:**
- Consumes: `horizon_target` (Task 1), `WalkForward`/`MODELS` (Task 2), `Portfolio.backtest(..., returns, horizon)` (Task 4).
- Produces:
  - `RunKey(horizon, window, retrain_every=1, model="lightgbm")` — congelado e ordenável, com `.name` = `"lightgbm-h5-w21-k1"` e `.label` = `"h=5 · janela 21 · retreino a cada 1"`.
  - `FeatureSet(dataset, market, returns, coverage, quality)` e `RunResults(key, predictions, importance, info)`.
  - `Project.run(key, progress=None)`, `save_run`, `load_run(key)`, `runs() -> list[RunKey]`, `stamp(key=None)`.
  - `FEATURES_FORMAT = 2`.
  - `build_dataset(ticker_features, date_features, membership, close, min_history_days)` (só `FEATURES`).
  - `main.report(project, key)`.

- [ ] **Step 1: Write the failing tests**

Em `tests/synthetic.py`, acrescente `from engine.project import RunKey` nos imports. Depois de `SMALL_LGBM`, acrescente:

```python
RUNS = (RunKey(1, 40), RunKey(5, 40, 2))
```

No `settings` de `synthetic_config`, acrescente `"horizon_days": 1, "retrain_every": 1,`.

Em `tests/conftest.py`, troque o import por `from synthetic import BENCH, RF, RUNS, make_market, synthetic_config`. Em `synthetic_project`, troque `project.run(40)` por:

```python
    for key in RUNS:
        project.run(key)
```

Em `tests/test_dataset.py`:
- no fixture `parts`, apague a linha `fwd = ...` e devolva `return ticker_features, date_features, membership, close`;
- troque em `test_columns_and_values` a asserção das colunas por `assert list(dataset.columns) == FEATURES`;
- apague `test_label_is_positive_forward_return`.

Substitua `tests/test_main.py` por:

```python
from synthetic import RUNS

from main import report


def test_report_prints_the_run_portfolio_benchmark_and_model(synthetic_project, capsys):
    report(synthetic_project, RUNS[1])
    out = capsys.readouterr().out
    assert RUNS[1].label in out
    assert "Carteira (líquida)" in out and "S&P 500 TR" in out and "sharpe" in out and "auc" in out
```

Em `tests/test_project.py`:
- troque o import de `synthetic` por `from synthetic import BENCH, RF, RUNS, make_market, perturb_after, synthetic_config, yahoo_response`;
- troque o de `engine.project` por `from engine.project import FEATURES_FORMAT, Project, RunKey`;
- substitua os testes abaixo (mantenha os demais como estão).

```python
@pytest.mark.parametrize("key", RUNS)
def test_predictions_do_not_depend_on_future_data(synthetic_project, tmp_path, key):
    base = synthetic_project.load_run(key)
    fits = base.importance.index
    cutoff = fits[len(fits) // 2]
    perturbed = build(perturb_after(make_market(), cutoff), tmp_path, n_jobs=2).run(key).predictions
    columns = ["date", "ticker", "prob"]
    predictions = base.predictions
    before = predictions.loc[predictions["date"] <= cutoff, columns].reset_index(drop=True)
    after = perturbed.loc[perturbed["date"] <= cutoff, columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    later = predictions[predictions["date"] > cutoff].merge(perturbed[perturbed["date"] > cutoff], on=["date", "ticker"],
                                                             suffixes=("_base", "_pert"))
    assert len(later) > 0 and not np.allclose(later["prob_base"], later["prob_pert"])


@pytest.mark.parametrize("key", RUNS)
def test_predictions_only_for_members_on_each_date(synthetic_project, key):
    universe = Universe(make_market()["snapshots"])
    for date, group in synthetic_project.load_run(key).predictions.groupby("date"):
        assert set(group["ticker"]) <= set(universe.members_on(date))


@pytest.mark.parametrize("key", RUNS)
def test_decisions_start_at_backtest_start_and_follow_the_horizon(synthetic_project, key):
    predictions = synthetic_project.load_run(key).predictions
    calendar = synthetic_project.features(with_dataset=False).calendar
    positions = calendar.get_indexer(sorted(predictions["date"].unique()))
    assert calendar[positions[0]] == pd.Timestamp(synthetic_project.config.backtest_start)
    assert set(np.diff(positions)) == {key.horizon}


@pytest.mark.parametrize("key", RUNS)
def test_labels_are_relative_to_the_median_of_each_date(synthetic_project, key):
    known = synthetic_project.load_run(key).predictions.dropna(subset=["label"])
    share = known.groupby("date")["label"].mean()
    assert share.between(0.4, 0.5).all()
    median = known.groupby("date")["fwd_ret"].transform("median")
    assert (known["label"] == (known["fwd_ret"] > median)).all()


def test_features_are_saved_with_every_column(synthetic_project):
    features = synthetic_project.features()
    assert list(features.dataset.columns) == FEATURES
    assert features.dataset[DATE_FEATURES].notna().all().all()
    assert features.returns.index.equals(features.calendar)
    assert set(features.dataset.index.get_level_values("ticker")) <= set(features.returns.columns)
    assert features.quality["settings"]["format"] == FEATURES_FORMAT
    assert synthetic_project.features(with_dataset=False).dataset is None


def test_runs_are_saved_per_key_and_listed(synthetic_project):
    assert synthetic_project.runs() == list(RUNS)
    run = synthetic_project.load_run(RUNS[1])
    assert run.info["key"] == dataclasses.asdict(RUNS[1])
    assert list(run.importance.columns) == FEATURES
    assert (synthetic_project.runs_dir / "lightgbm-h5-w40-k2" / "run_info.json").exists()
    assert synthetic_project.stamp(RUNS[1]) > 0 and synthetic_project.stamp(RunKey(1, 21)) == 0
    assert RUNS[1].label == "h=5 · janela 40 · retreino a cada 2"


def test_rebuilding_features_deletes_old_runs(tmp_path):
    project = build(make_market(), tmp_path)
    project.run(RUNS[0])
    assert project.runs() == [RUNS[0]]
    raw = make_market()
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    assert not project.runs_dir.exists()


def test_window_too_long_is_refused(synthetic_project):
    with pytest.raises(ValueError, match="máximo"):
        synthetic_project.run(RunKey(1, 100))
    assert synthetic_project.runs() == list(RUNS)


def test_features_saved_in_an_older_format_are_rebuilt(tmp_path):
    write_inputs(make_market(), tmp_path)
    project = Project(synthetic_config(tmp_path))
    project.features(with_dataset=False)
    path = project.features_dir / "quality.json"
    quality = json.loads(path.read_text())
    quality["settings"]["format"] = FEATURES_FORMAT - 1
    path.write_text(json.dumps(quality))
    assert project.features(with_dataset=False).quality["settings"]["format"] == FEATURES_FORMAT


def test_runs_made_with_other_training_settings_are_hidden(synthetic_project):
    assert Project(dataclasses.replace(synthetic_project.config, min_child_floor=5)).runs() == []
    assert synthetic_project.runs() == list(RUNS)


def test_a_run_saved_after_its_features_were_replaced_is_hidden(tmp_path):
    raw = make_market()
    project = build(raw, tmp_path)
    stale = project.run(RunKey(1, 5))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    project.save_run(stale)
    assert project.runs() == []
```

Acrescente `import json` ao topo de `tests/test_project.py`. Em `test_features_are_rebuilt_when_their_settings_change`, troque as duas ocorrências de `"w040"` por `"lightgbm-h1-w40-k1"`. Em `test_download_saves_inputs_and_clears_derived_results`, troque as duas de `"w021"` por `"lightgbm-h5-w21-k1"`.

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_dataset.py tests/test_project.py tests/test_main.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'FEATURES_FORMAT'` (e `RunKey`).

- [ ] **Step 3: Implement**

**3a.** Em `engine/dataset.py`, mude a assinatura e o fim de `build_dataset`:

```python
def build_dataset(ticker_features: dict[str, pd.DataFrame], date_features: pd.DataFrame, membership: pd.DataFrame,
                  close: pd.DataFrame, min_history_days: int) -> pd.DataFrame:
```

e troque as últimas linhas (de `dataset = pd.DataFrame(data, index=index)[FEATURES]` até o `return`) por:

```python
    return pd.DataFrame(data, index=index)[FEATURES]
```

**3b.** Em `engine/project.py`:
- troque a docstring do módulo por `"""What the CLI and the app do: download inputs, build features once, run one walk-forward per run key, load results."""`;
- troque `from engine.dataset import build_dataset` por `from engine.dataset import build_dataset, horizon_target`;
- troque `from engine.walk_forward import WalkForward` por `from engine.walk_forward import MODELS, WalkForward`.

Depois de `RUN_SETTINGS`, acrescente:

```python
FEATURES_FORMAT = 2  # bump when the saved features change shape: features saved in another format are rebuilt


@dataclass(frozen=True, order=True)
class RunKey:
    """A run: decide every `horizon` sessions, train on the last `window` sessions of rows, refit every
    `retrain_every` decisions."""
    horizon: int
    window: int
    retrain_every: int = 1
    model: str = "lightgbm"

    @property
    def name(self) -> str:
        return f"{self.model}-h{self.horizon}-w{self.window}-k{self.retrain_every}"

    @property
    def label(self) -> str:
        return f"h={self.horizon} · janela {self.window} · retreino a cada {self.retrain_every}"
```

Em `FeatureSet`, acrescente o campo `returns: pd.DataFrame` entre `market` e `coverage`. Em `RunResults`, troque `window: int` por `key: RunKey`.

Em `build_features`:
- troque `features.quality.update(settings=self._settings(FEATURE_SETTINGS), built_at=time.time_ns())` por `features.quality.update(settings=self._feature_settings(), built_at=time.time_ns())`;
- depois de gravar `market.parquet`, acrescente `features.returns.to_parquet(self.features_dir / "returns.parquet")`.

Em `features()`, depois de `market=...`, acrescente `returns=pd.read_parquet(self.features_dir / "returns.parquet"),`.

Substitua `run`, `save_run`, `load_run`, `runs`, `stamp` e `_run_dir` por:

```python
    def run(self, key: RunKey, progress=None) -> RunResults:
        c = self.config
        features = self.features()
        dataset = features.dataset.join(horizon_target(features.dataset, features.returns, key.horizon))
        started = time.perf_counter()
        result = WalkForward(key.window, c.lgbm_params, key.horizon, key.retrain_every, c.min_child_share,
                             c.min_child_floor, c.n_jobs, MODELS[key.model]).run(
            dataset, FEATURES, features.calendar, start=c.backtest_start, progress=progress)
        results = RunResults(key, result.predictions, result.importance, info={
            "key": dataclasses.asdict(key),
            "features": FEATURES,
            "runtime_minutes": round((time.perf_counter() - started) / 60.0, 2),
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "versions": {package: version(package) for package in PACKAGES},
            "config": dataclasses.asdict(c),
            "settings": self._settings(RUN_SETTINGS),
            "features_built_at": features.quality["built_at"],
        })
        self.save_run(results)
        return results

    def save_run(self, results: RunResults) -> None:
        folder = self._run_dir(results.key)
        folder.mkdir(parents=True, exist_ok=True)
        results.predictions.to_parquet(folder / "predictions.parquet")
        results.importance.to_parquet(folder / "importance.parquet")
        (folder / "run_info.json").write_text(json.dumps(results.info, indent=2, default=str))

    def load_run(self, key: RunKey) -> RunResults:
        folder = self._run_dir(key)
        return RunResults(key, pd.read_parquet(folder / "predictions.parquet"),
                          pd.read_parquet(folder / "importance.parquet"), json.loads((folder / "run_info.json").read_text()))

    def runs(self) -> list[RunKey]:
        """Runs made with the current settings on the current features; any other run is stale."""
        quality = self._current_quality()
        if quality is None:
            return []
        expected = {"settings": self._settings(RUN_SETTINGS), "features_built_at": quality["built_at"]}
        keys = []
        for path in self.runs_dir.glob("*/run_info.json"):
            info = json.loads(path.read_text())
            if "key" in info and all(info.get(name) == value for name, value in expected.items()):
                keys.append(RunKey(**info["key"]))
        return sorted(keys)

    def stamp(self, key: RunKey | None = None) -> int:
        """mtime of the file written last (quality.json / run_info.json); 0 when absent. Keys the app caches."""
        path = self.features_dir / "quality.json" if key is None else self._run_dir(key) / "run_info.json"
        return path.stat().st_mtime_ns if path.exists() else 0
```

```python
    def _feature_settings(self) -> dict:
        return {"format": FEATURES_FORMAT, **self._settings(FEATURE_SETTINGS)}
```

(logo depois de `_settings`).

Em `_current_quality`, troque `self._settings(FEATURE_SETTINGS)` por `self._feature_settings()`.

```python
    def _run_dir(self, key: RunKey) -> Path:
        return self.runs_dir / key.name
```

Em `_make_features`, troque as duas linhas do `dataset = build_dataset(...)` por:

```python
        returns = forward_open_return(panel.open, panel.dividends)
        dataset = build_dataset(ticker_features, date_features, membership, panel.close, c.min_history_days)
```

Troque também `return FeatureSet(dataset, market, coverage, quality)` por `return FeatureSet(dataset, market, returns, coverage, quality)`.

**3c.** Em `main.py`, com `Edit` (as notas do fim ficam):
- troque a docstring da linha 1 por `"""`python main.py download` baixa constituintes e preços; `features` calcula as features; `run --horizon H --window X --retrain K` roda o walk-forward."""`;
- troque `from engine.project import Project` por `from engine.project import Project, RunKey`;
- substitua a função `report` por:

```python
def report(project: Project, key: RunKey) -> None:
    config = project.config
    features = project.features(with_dataset=False)
    run = project.load_run(key)
    daily = Portfolio(config.threshold, config.cost_bps).backtest(
        run.predictions, features.market, features.returns, key.horizon).daily
    table = pd.DataFrame({
        "Carteira (bruta)": metrics.performance_metrics(daily["gross"], daily["rf"], daily["invested"]),
        "Carteira (líquida)": metrics.performance_metrics(daily["net"], daily["rf"], daily["invested"]),
        "S&P 500 TR": metrics.performance_metrics(daily["bench"], daily["rf"]),
    })
    print(f"\n{key.label} · {daily.index.min().date()} → {daily.index.max().date()} · "
          f"limiar {config.threshold} · custo {config.cost_bps} bps · giro médio por troca "
          f"{daily.loc[daily['rebalance'], 'turnover'].mean():.0%}")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    realized = run.predictions[run.predictions["date"].isin(daily["decision_date"])]
    print("Modelo:", {name: round(value, 4) for name, value in metrics.model_metrics(realized).items()})
```

Em `main()`, troque as duas linhas do `run_command.add_argument("--window", ...)` por:

```python
    run_command.add_argument("--horizon", type=int, default=CONFIG.horizon_days,
                             help="pregões entre decisões e do rótulo (1 = diário, 5 = semanal, 21 = mensal)")
    run_command.add_argument("--window", type=int, default=CONFIG.train_window_days,
                             help="pregões de exemplos de treino (5 = 1 semana, 21 = 1 mês)")
    run_command.add_argument("--retrain", type=int, default=CONFIG.retrain_every, help="retreina a cada k decisões")
```

e o ramo `else:` por:

```python
    else:
        key = RunKey(args.horizon, args.window, args.retrain)
        run = project.run(key, progress=lambda done: print(f"\rtreinando {done:.0%}", end="", flush=True))
        print(f"\n{len(run.predictions):,} previsões em {run.info['runtime_minutes']} min")
        report(project, key)
```

- [ ] **Step 4: Run to verify they pass**

Run: `venv/bin/python -m pytest tests/test_dataset.py tests/test_project.py tests/test_main.py tests/test_walk_forward.py tests/test_portfolio.py tests/test_backtest_validation.py tests/test_metrics.py tests/test_features.py tests/test_prices.py tests/test_universe.py tests/test_config.py -q -p no:cacheprovider`
Expected: PASS (all). O `test_app` continua vermelho até a Task 6.

- [ ] **Step 5: Mutation check (lookahead)**

Em `_fit_block`, troque temporariamente `group[0] - horizon - 1` por `group[0] - horizon` (treina com um dia a mais) e rode `venv/bin/python -m pytest tests/test_project.py -q -p no:cacheprovider -k future`: os dois casos têm que falhar. Desfaça a troca e rode de novo: PASS.

---

### Task 6: App

**Files:**
- Modify: `app/dashboard.py` (inteiro)
- Modify: `tests/test_app.py` (inteiro)

**Interfaces:**
- Consumes: `RunKey`, `Project.runs/run/load_run/stamp/features` (Task 5); `Portfolio.backtest(..., returns, horizon)` (Task 4); `metrics.deciles`, `metrics.decile_spread` (Task 3); `MODELS` (Task 2).
- Produces:
  - a barra lateral tem `number_input` 0 = h, 1 = X, 2 = k, o slider do limiar e o custo;
  - as abas são `["Visão geral", "Previsto vs realizado", "Carteira por decisão", "Recortes de período", "Runs", "Modelo"]`;
  - tabela de decis com colunas `Prob. média`, `Bateu a mediana`, `Retorno acima da mediana`;
  - tabela da decisão com `Bateu a mediana?`;
  - tabela de runs com `Giro médio por troca`, indexada por `RunKey.label`.

- [ ] **Step 1: Write the failing tests**

Substitua `tests/test_app.py` inteiro por:

```python
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from synthetic import RUNS, make_market, synthetic_config

from engine import metrics
from engine.portfolio import Portfolio
from engine.prices import PriceData, clean_prices
from engine.project import Project, RunKey, RunResults
from engine.universe import Universe

APP = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"
TABS = ["Visão geral", "Previsto vs realizado", "Carteira por decisão", "Recortes de período", "Runs", "Modelo"]


def open_app(project: Project, monkeypatch, key: RunKey = RUNS[0]) -> AppTest:
    """The app with the project's own config and the run key typed in the sidebar."""
    monkeypatch.setattr("config.config.CONFIG", project.config)
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    for field, value in zip(at.sidebar.number_input, (key.horizon, key.window, key.retrain_every)):
        field.set_value(value)
    return at.run()


def backtest(project: Project, key: RunKey, predictions: pd.DataFrame | None = None):
    features = project.features(with_dataset=False)
    predictions = project.load_run(key).predictions if predictions is None else predictions
    return Portfolio(0.55).backtest(predictions, features.market, features.returns, key.horizon)


@pytest.mark.parametrize("key", RUNS)
def test_dashboard_renders_a_saved_run(synthetic_project, monkeypatch, key):
    at = open_app(synthetic_project, monkeypatch, key)
    assert not at.exception
    assert [tab.label for tab in at.tabs] == TABS


def test_dashboard_survives_threshold_nobody_passes(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch, RUNS[1])
    at.sidebar.slider[0].set_value(0.8).run()
    assert not at.exception


def test_run_not_made_yet_offers_the_button(synthetic_project, monkeypatch):
    key = RunKey(1, 21)
    at = open_app(synthetic_project, monkeypatch, key)
    assert not at.exception
    assert f"Rodar {key.label}" in [button.label for button in at.sidebar.button]
    assert any("ainda não existe" in info.value for info in at.info)


def test_missing_inputs_show_how_to_download(tmp_path, monkeypatch):
    monkeypatch.setattr("config.config.CONFIG", synthetic_config(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert at.error


def test_dashboard_shows_new_results_after_a_rerun(tmp_path, monkeypatch):
    raw = make_market()
    project = Project(synthetic_config(tmp_path))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    key = RUNS[0]
    run = project.run(key)
    first = open_app(project, monkeypatch, key)
    dates = sorted(run.predictions["date"].unique())
    cut = run.predictions[run.predictions["date"] <= dates[59]]
    project.save_run(RunResults(key, cut, run.importance.loc[:dates[59]], run.info))
    second = open_app(project, monkeypatch, key)
    n_full, n_cut = len(backtest(project, key, run.predictions).daily), len(backtest(project, key, cut).daily)
    assert n_full != n_cut
    assert any(f"{n_full} dias de carteira" in caption.value for caption in first.caption)
    assert any(f"{n_cut} dias de carteira" in caption.value for caption in second.caption)


def test_periods_tab_shows_model_metrics_per_period(synthetic_project, monkeypatch):
    key = RUNS[0]
    at = open_app(synthetic_project, monkeypatch, key)
    model = next(table.value for table in at.dataframe if "AUC" in table.value.index)
    assert list(model.columns) == ["1 mês", "3 meses", "6 meses", "12 meses", "Tudo"]
    kpis = {metric.label: metric.value for metric in at.metric}
    assert model["Tudo"].to_dict() == {label: kpis[label] for label in model.index}
    predictions = synthetic_project.load_run(key).predictions
    daily = backtest(synthetic_project, key).daily
    start = daily.loc[daily.index > daily.index.max() - pd.DateOffset(months=1), "decision_date"].min()
    month = predictions[predictions["date"] >= start].dropna(subset=["label"])
    assert model.loc["AUC", "1 mês"] == f"{metrics.model_metrics(month)['auc']:.3f}"


def test_signal_tab_shows_the_decile_table(synthetic_project, monkeypatch):
    key = RUNS[1]
    at = open_app(synthetic_project, monkeypatch, key)
    table = next(frame.value for frame in at.dataframe if "Bateu a mediana" in frame.value.columns)
    expected = metrics.deciles(synthetic_project.load_run(key).predictions)
    assert list(table.index) == list(expected.index)
    assert table["Prob. média"].to_numpy() == pytest.approx(expected["prob"].to_numpy())


def test_decision_tab_shows_what_was_bought_and_what_happened(synthetic_project, monkeypatch):
    key = RUNS[1]
    result = backtest(synthetic_project, key)
    decisions = list(result.daily["decision_date"].unique())
    at = open_app(synthetic_project, monkeypatch, key)
    at.selectbox[0].set_value(decisions.index(result.positions["decision_date"].iloc[-1])).run()
    held = next(frame.value for frame in at.dataframe if "Bateu a mediana?" in frame.value.columns)
    assert {"Prob. prevista", "Retorno no período", "Acima da mediana", "Contribuição"} <= set(held.columns)
    assert any("Soma das contribuições" in info.value for info in at.info)


def test_runs_tab_compares_every_run(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    table = next(frame.value for frame in at.dataframe if "Giro médio por troca" in frame.value.columns)
    assert list(table.index) == [key.label for key in RUNS]
```

- [ ] **Step 2: Run to verify they fail**

Run: `venv/bin/python -m pytest tests/test_app.py -q -p no:cacheprovider`
Expected: FAIL (o app ainda usa `PROJECT.runs()` como janelas inteiras e `Portfolio.backtest` com três argumentos).

- [ ] **Step 3: Implement**

Substitua `app/dashboard.py` inteiro por:

```python
"""Resultados do backtest. Rode com: streamlit run app/dashboard.py

Escolha na barra lateral o horizonte h, a janela de treino X e o retreino k: se esse run ainda não existe, o botão
treina (em paralelo) e salva; os runs já feitos abrem na hora e são comparados na aba "Runs".
"""
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from config.config import CONFIG
from engine import metrics
from engine.portfolio import Portfolio
from engine.project import Project, RunKey
from engine.walk_forward import MODELS

PROJECT = Project(dataclasses.replace(CONFIG, data_in=ROOT / CONFIG.data_in, data_out=ROOT / CONFIG.data_out))
OUT = str(PROJECT.config.data_out)
BENCH = "S&P 500 TR"
LEVELS = {"gross": "Carteira (bruta)", "net": "Carteira (líquida)"}
CHART_WIDTH = 980
METRIC_FORMATS = {
    "total_return": ("Retorno acumulado", "{:+.2%}"),
    "volatility": ("Volatilidade anualizada", "{:.2%}"),
    "sharpe": ("Sharpe", "{:.2f}"),
    "max_drawdown": ("Máximo drawdown", "{:.2%}"),
    "hit_ratio": ("Hit ratio (dias positivos)", "{:.1%}"),
}
MODEL_FORMATS = {"auc": ("AUC", "{:.3f}"), "accuracy": ("Acurácia (corte 0,5)", "{:.1%}"), "ic": ("IC médio", "{:+.4f}")}
BEAT = {1.0: "bateu", 0.0: "não bateu"}

st.set_page_config(page_title="Backtest ML — S&P 500", layout="wide")


@st.cache_resource(show_spinner="Carregando as features…")
def load_features(data_out: str, stamp: int):
    return PROJECT.features(with_dataset=False)


@st.cache_resource(show_spinner=False)
def load_run(data_out: str, key: RunKey, stamp: int):
    return PROJECT.load_run(key)


@st.cache_data(show_spinner="Recalculando a carteira…", max_entries=64)
def backtest(data_out: str, key: RunKey, run_stamp: int, features_stamp: int, threshold: float, cost_bps: float):
    features = load_features(data_out, features_stamp)
    predictions = load_run(data_out, key, run_stamp).predictions
    return Portfolio(threshold, cost_bps).backtest(predictions, features.market, features.returns, key.horizon)


def known(data_out: str, key: RunKey, run_stamp: int) -> pd.DataFrame:
    """Predictions whose outcome (the h-session return) is already known: the ones the model is evaluated on."""
    return load_run(data_out, key, run_stamp).predictions.dropna(subset=["label"])


@st.cache_data(show_spinner=False)
def evaluate(data_out: str, key: RunKey, run_stamp: int) -> dict:
    evaluated = known(data_out, key, run_stamp)
    return {
        "summary": metrics.model_metrics(evaluated),
        "monthly_auc": metrics.monthly_auc(evaluated),
        "ic": metrics.daily_ic(evaluated),
        "deciles": metrics.deciles(evaluated),
        "spread": metrics.decile_spread(evaluated),
    }


@st.cache_data(show_spinner=False)
def evaluate_since(data_out: str, key: RunKey, run_stamp: int, start: pd.Timestamp) -> dict:
    evaluated = known(data_out, key, run_stamp)
    return metrics.model_metrics(evaluated[evaluated["date"] >= start])


def fmt(value: float, pattern: str) -> str:
    return pattern.format(value) if pd.notna(value) else "—"


def growth(returns: pd.Series) -> pd.Series:
    curve = (1.0 + returns).cumprod()
    return pd.concat([pd.Series([1.0], index=[curve.index[0] - pd.Timedelta(days=1)]), curve])


def line_chart(frame: pd.DataFrame, y_title: str, height: int = 320, area: bool = False, y_format: str = ".4f") -> None:
    data = frame.rename_axis("Data").reset_index().melt("Data", var_name="Série", value_name="valor").dropna()
    mark = alt.Chart(data).mark_area(opacity=0.35) if area else alt.Chart(data).mark_line(strokeWidth=1.8)
    chart = (
        mark.encode(
            x=alt.X("Data:T", title=None),
            y=alt.Y("valor:Q", title=y_title, scale=alt.Scale(zero=False)),
            color=alt.Color("Série:N", title=None, legend=alt.Legend(orient="bottom")),
            tooltip=[alt.Tooltip("Data:T"), alt.Tooltip("Série:N"), alt.Tooltip("valor:Q", format=y_format)],
        )
        .properties(width=CHART_WIDTH, height=height)
        .add_params(alt.selection_interval(bind="scales", encodings=["x"]))
    )
    st.altair_chart(chart, width="content")


def labeled(values: dict[str, float], formats: dict[str, tuple[str, str]]) -> dict[str, str]:
    return {label: fmt(values[key], pattern) for key, (label, pattern) in formats.items()}


def metrics_table(columns: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame({name: labeled(values, METRIC_FORMATS) for name, values in columns.items()})


def compare(part: pd.DataFrame) -> dict[str, dict]:
    columns = {label: metrics.performance_metrics(part[level], part["rf"], part["invested"]) for level, label in LEVELS.items()}
    columns[BENCH] = metrics.performance_metrics(part["bench"], part["rf"])
    return columns


def curves(part: pd.DataFrame, transform) -> pd.DataFrame:
    frame = pd.DataFrame({label: transform(part[level]) for level, label in LEVELS.items()})
    frame[BENCH] = transform(part["bench"])
    return frame


def shade(value: float) -> str:
    if pd.isna(value):
        return ""
    return "background-color: rgba(38,166,91,0.15)" if value > 0 else "background-color: rgba(214,69,65,0.15)"


try:
    features_stamp = PROJECT.stamp()
    features = load_features(OUT, features_stamp)
    features_stamp = PROJECT.stamp()
except FileNotFoundError:
    st.error("Sem dados em data/in. Rode `python main.py download` e depois `python main.py features`.")
    st.stop()

with st.sidebar:
    st.header("Run")
    model = st.selectbox("Modelo", list(MODELS)) if len(MODELS) > 1 else next(iter(MODELS))
    horizon = int(st.number_input("Horizonte h: pregões entre decisões e do rótulo (1 = diário, 5 = semanal, 21 = mensal)",
                                  min_value=1, max_value=63, value=CONFIG.horizon_days, step=1))
    window = int(st.number_input("Janela de treino X: pregões de exemplos (5 = 1 semana, 21 = 1 mês, 63 = 3 meses)",
                                 min_value=1, max_value=1000, value=CONFIG.train_window_days, step=1))
    retrain = int(st.number_input("Retreinar a cada k decisões (1 = em toda decisão)",
                                  min_value=1, max_value=100, value=CONFIG.retrain_every, step=1))
    key = RunKey(horizon, window, retrain, model)
    runs = PROJECT.runs()
    if key not in runs and st.button(f"Rodar {key.label}", type="primary"):
        bar = st.progress(0.0, text="Treinando…")
        try:
            PROJECT.run(key, progress=lambda done: bar.progress(done, text=f"Treinando… {done:.0%}"))
        except ValueError as error:
            st.error(str(error))
        else:
            st.rerun()
    st.caption("Runs já feitos: " + ("; ".join(other.label for other in runs) or "nenhum"))
    st.header("Carteira")
    threshold = st.slider("Limiar: probabilidade mínima de bater a mediana", 0.40, 0.80, float(CONFIG.threshold), 0.005,
                          format="%.3f")
    cost_bps = st.number_input("Custo de transação (bps por lado, em cada compra ou venda)", 0.0, 50.0, float(CONFIG.cost_bps), 0.5)
    st.caption(
        "A cada decisão, compra as ações cuja probabilidade prevista de render acima da mediana do índice passa do limiar, "
        "com peso proporcional à probabilidade, e segura até a próxima decisão; se nenhuma passar, fica em caixa rendendo "
        "a T-bill (^IRX). Escolher limiar, h, X ou k olhando este resultado é otimizar dentro do próprio backtest "
        "(data snooping)."
    )

if key not in runs:
    st.info(f"O run {key.label} ainda não existe: use o botão na barra lateral.")
    st.stop()

run_stamp = PROJECT.stamp(key)
run = load_run(OUT, key, run_stamp)
bt = backtest(OUT, key, run_stamp, features_stamp, threshold, cost_bps)
daily, positions = bt.daily, bt.positions
if daily.empty:
    st.warning("Nenhum dia de carteira com retorno realizado neste run.")
    st.stop()
evaluation = evaluate(OUT, key, run_stamp)
window_coverage = features.coverage.loc[daily["decision_date"].min():daily["decision_date"].max()]

st.title("Backtest ML — S&P 500")
st.caption(f"{key.label} · {int(daily['rebalance'].sum())} decisões · {len(daily)} dias de carteira · "
           f"{daily.index.min().date()} → {daily.index.max().date()} · gerado em {run.info.get('generated_at', '?')}")
tab_overview, tab_signal, tab_decision, tab_periods, tab_runs, tab_model = st.tabs(
    ["Visão geral", "Previsto vs realizado", "Carteira por decisão", "Recortes de período", "Runs", "Modelo"])

with tab_overview:
    for col, (label, text) in zip(st.columns(3), labeled(evaluation["summary"], MODEL_FORMATS).items()):
        col.metric(label, text)
    st.caption(
        f"{model} retreinado a cada {retrain} decisão(ões) com os exemplos dos últimos {window} pregões; a cada {horizon} "
        f"pregão(ões), estima a probabilidade de cada ação do índice render acima da mediana do índice nos {horizon} "
        f"pregões seguintes · em média {window_coverage['n_eligible'].mean():.0f} ações elegíveis por dia, de "
        f"{window_coverage['n_members'].mean():.0f} membros do índice."
    )
    st.subheader("Carteira vs S&P 500 Total Return")
    st.dataframe(metrics_table(compare(daily)))
    line_chart(curves(daily, growth), "Capital (1,0 no início)")
    st.caption("Decisão após o fechamento de t, compra na abertura de t+1 e troca só na abertura seguinte à próxima "
               "decisão; o custo incide só nas trocas (retornos abertura → abertura, com dividendos). Arraste ou use a "
               "roda do mouse para aproximar.")
    st.markdown("**Drawdown**")
    line_chart(curves(daily, metrics.drawdown), "Drawdown", height=220, area=True)
    st.markdown("**Nº de ações na carteira**")
    line_chart(daily[["n_positions"]].rename(columns={"n_positions": "Ações"}), "Ações", height=220, y_format=".0f")

with tab_signal:
    st.caption(
        "Em cada decisão, as ações com resultado já conhecido são divididas em 10 grupos pela probabilidade prevista "
        "(decil 1 = menor, 10 = maior). Se o modelo tem sinal, o retorno acima da mediana cresce do decil 1 ao 10 e a "
        "fração que bateu a mediana acompanha a probabilidade prevista (pontos perto da diagonal). AUC: chance de uma "
        "ação que bateu a mediana ter recebido probabilidade maior que uma que não bateu (0,5 = sorteio). IC: correlação "
        "de postos entre probabilidade e retorno em cada decisão."
    )
    table = evaluation["deciles"].rename_axis("decil").reset_index()
    if table.empty:
        st.info("Ainda não há decisão com resultado conhecido neste run.")
    else:
        bars = alt.Chart(table).mark_bar().encode(
            x=alt.X("decil:O", title="Decil de probabilidade prevista"),
            y=alt.Y("relative:Q", title="Retorno médio acima da mediana do dia", axis=alt.Axis(format="%")),
            color=alt.condition(alt.datum.relative > 0, alt.value("#26a65b"), alt.value("#d64541")),
            tooltip=[alt.Tooltip("decil:O"), alt.Tooltip("relative:Q", format="+.3%")],
        )
        low = float(min(table["prob"].min(), table["label"].min())) - 0.02
        high = float(max(table["prob"].max(), table["label"].max())) + 0.02
        scale = alt.Scale(domain=[low, high])
        diagonal = alt.Chart(pd.DataFrame({"x": [low, high]})).mark_line(color="gray", strokeDash=[4, 4]).encode(
            x=alt.X("x:Q", scale=scale), y=alt.Y("x:Q", scale=scale))
        calibration = alt.Chart(table).mark_line(point=True).encode(
            x=alt.X("prob:Q", title="Probabilidade média prevista", scale=scale),
            y=alt.Y("label:Q", title="Fração que bateu a mediana", scale=scale),
            tooltip=[alt.Tooltip("decil:O"), alt.Tooltip("prob:Q", format=".3f"), alt.Tooltip("label:Q", format=".1%")],
        )
        left, right = st.columns(2)
        left.markdown("**Retorno por decil**")
        left.altair_chart(bars.properties(width=CHART_WIDTH // 2 - 20, height=260), width="content")
        right.markdown("**Calibração**")
        right.altair_chart((diagonal + calibration).properties(width=CHART_WIDTH // 2 - 20, height=260), width="content")
        st.dataframe(
            table.set_index("decil").rename(columns={
                "prob": "Prob. média", "label": "Bateu a mediana", "relative": "Retorno acima da mediana"})
            .style.format({"Prob. média": "{:.3f}", "Bateu a mediana": "{:.1%}", "Retorno acima da mediana": "{:+.3%}"}),
        )
        st.markdown("**Spread acumulado: 10% de maior probabilidade − 10% de menor**")
        line_chart(evaluation["spread"].cumsum().to_frame("Spread acumulado"), "Soma dos spreads", height=240)
        st.caption("Em cada decisão, retorno médio das ações com as 10% maiores probabilidades menos o das 10% menores, "
                   "somado ao longo do tempo. Sobe quando a ordenação acerta; não depende do limiar e não inclui custos.")

with tab_decision:
    period_return = daily.groupby("decision_date")[["gross", "net", "bench"]].apply(lambda part: (1.0 + part).prod() - 1.0)
    decisions = list(period_return.index)
    choice = st.selectbox(
        "Decisão",
        options=range(len(decisions)),
        index=len(decisions) - 1,
        format_func=lambda i: (f"{decisions[i].date()}  ·  carteira {period_return['gross'].iloc[i]:+.2%}  ·  "
                               f"S&P {period_return['bench'].iloc[i]:+.2%}"),
    )
    decision = decisions[choice]
    days = daily[daily["decision_date"] == decision]
    trade = days.iloc[0]
    cols = st.columns(6)
    cols[0].metric("Retorno bruto", f"{period_return.loc[decision, 'gross']:+.2%}")
    cols[1].metric("Retorno líquido", f"{period_return.loc[decision, 'net']:+.2%}")
    cols[2].metric(BENCH, f"{period_return.loc[decision, 'bench']:+.2%}")
    cols[3].metric("Ações", int(trade["n_positions"]))
    cols[4].metric("Giro na troca", f"{trade['turnover']:.0%}")
    cols[5].metric("Custo", f"{trade['cost']:.3%}")
    st.caption(f"Sinal calculado após o fechamento de {decision.date()}; compra na abertura de {days.index[0].date()} e "
               f"segura por {len(days)} pregão(ões), até a abertura do pregão seguinte a {days.index[-1].date()}.")
    day_predictions = run.predictions[run.predictions["date"] == decision]
    median = day_predictions["fwd_ret"].median()
    held = positions[positions["decision_date"] == decision].merge(day_predictions[["ticker", "label"]], on="ticker")
    if held.empty:
        st.info("Nenhuma ação passou do limiar: carteira em caixa rendendo a T-bill (^IRX) no período.")
    else:
        table = pd.DataFrame({
            "Ticker": held["ticker"], "Prob. prevista": held["prob"], "Peso": held["weight"],
            "Retorno no período": held["realized"], "Acima da mediana": held["realized"] - median,
            "Bateu a mediana?": held["label"].map(BEAT), "Contribuição": held["contribution"],
        }).sort_values("Peso", ascending=False)
        st.dataframe(
            table.style.format({"Prob. prevista": "{:.3f}", "Peso": "{:.2%}", "Retorno no período": "{:+.2%}",
                                "Acima da mediana": "{:+.2%}", "Contribuição": "{:+.3%}"}, na_rep="—")
            .map(shade, subset=["Acima da mediana", "Contribuição"]),
            hide_index=True, height=min(38 * (len(table) + 1), 520),
        )
        st.info(f"Soma das contribuições = {held['contribution'].sum():+.4%} = retorno bruto do período "
                f"({period_return.loc[decision, 'gross']:+.4%}).")
        if days["n_missing"].any():
            st.warning("Há ação sem retorno em algum dia do período (deslistagem/halt): o valor dela ficou congelado nesses dias.")
    outcome = day_predictions.assign(relative=day_predictions["fwd_ret"] - median,
                                     beat=day_predictions["label"].map(BEAT)).dropna(subset=["relative"])
    st.markdown(f"**Previsto × realizado em {decision.date()}** ({len(day_predictions)} ações do índice)")
    if outcome.empty:
        st.info("O resultado desta decisão ainda não é conhecido: o horizonte passa do último dado.")
    else:
        dots = alt.Chart(outcome).mark_circle(size=45, opacity=0.7).encode(
            x=alt.X("prob:Q", title="Probabilidade prevista de bater a mediana", scale=alt.Scale(zero=False)),
            y=alt.Y("relative:Q", title="Retorno realizado menos a mediana do dia", axis=alt.Axis(format="%")),
            color=alt.Color("beat:N", title=None, scale=alt.Scale(domain=list(BEAT.values()), range=["#26a65b", "#d64541"])),
            tooltip=["ticker", alt.Tooltip("prob:Q", format=".3f"), alt.Tooltip("relative:Q", format="+.2%")],
        )
        rule = alt.Chart(pd.DataFrame({"limiar": [threshold]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="limiar:Q")
        st.altair_chart((dots + rule).properties(width=CHART_WIDTH, height=320), width="content")
        st.caption("Cada ponto é uma ação do índice nesta decisão: à direita da linha tracejada, as que passaram do limiar "
                   "e foram compradas; acima de zero, as que renderam mais que a mediana no período.")
    near = day_predictions[day_predictions["prob"] <= threshold].nlargest(10, "prob")
    if len(near):
        st.markdown("**Quase entraram** (maiores probabilidades abaixo do limiar)")
        st.dataframe(
            pd.DataFrame({"Ticker": near["ticker"], "Prob. prevista": near["prob"], "Retorno no período": near["fwd_ret"],
                          "Acima da mediana": near["fwd_ret"] - median})
            .style.format({"Prob. prevista": "{:.3f}", "Retorno no período": "{:+.2%}", "Acima da mediana": "{:+.2%}"},
                          na_rep="—"),
            hide_index=True,
        )

with tab_periods:
    st.caption("As mesmas métricas, recalculadas só no trecho final de cada período.")
    periods = {"1 mês": 1, "3 meses": 3, "6 meses": 6, "12 meses": 12, "Tudo": None}
    last = daily.index.max()

    def cut(months):
        return daily if months is None else daily[daily.index > last - pd.DateOffset(months=months)]

    blocks, model_blocks = {}, {}
    for label, months in periods.items():
        part = cut(months)
        if len(part) >= 2:
            blocks.update({f"{label} · {name}": values for name, values in compare(part).items()})
            model_blocks[label] = evaluate_since(OUT, key, run_stamp, part["decision_date"].min())
    st.dataframe(metrics_table(blocks))
    st.markdown("**Modelo** (não depende do limiar nem do custo)")
    st.dataframe(pd.DataFrame({label: labeled(values, MODEL_FORMATS) for label, values in model_blocks.items()}))
    selected = st.radio("Período do gráfico", list(periods), horizontal=True, index=len(periods) - 1)
    part = cut(periods[selected])
    if len(part) >= 2:
        line_chart(curves(part, growth), "Capital (1,0 no início do período)")

with tab_runs:
    st.caption("Todos os runs, no mesmo período, com o limiar e o custo da barra lateral (carteira líquida).")
    rows = {}
    for other in runs:
        other_stamp = PROJECT.stamp(other)
        other_daily = backtest(OUT, other, other_stamp, features_stamp, threshold, cost_bps).daily
        performance = metrics.performance_metrics(other_daily["net"], other_daily["rf"], other_daily["invested"])
        rows[other.label] = {
            **labeled(performance, METRIC_FORMATS),
            **labeled(evaluate(OUT, other, other_stamp)["summary"], MODEL_FORMATS),
            "Giro médio por troca": f"{other_daily.loc[other_daily['rebalance'], 'turnover'].mean():.0%}",
            "Tempo de treino (min)": fmt(load_run(OUT, other, other_stamp).info.get("runtime_minutes", float("nan")), "{:.1f}"),
        }
    st.dataframe(pd.DataFrame(rows).T)

with tab_model:
    st.markdown("**AUC por mês**")
    line_chart(evaluation["monthly_auc"].to_frame("AUC"), "AUC", height=220)
    span = max(5, 63 // horizon)
    st.markdown(f"**IC por decisão (média móvel de {span} decisões)**")
    line_chart(evaluation["ic"].rolling(span, min_periods=max(2, span // 3)).mean().to_frame("IC"), "IC", height=220)
    st.markdown("**Importância das features** (média do ganho normalizado nos ajustes)")
    importance = run.importance.mean().sort_values(ascending=False).rename_axis("feature").reset_index(name="importância")
    st.altair_chart(
        alt.Chart(importance).mark_bar().encode(
            x=alt.X("importância:Q", title="Participação no ganho"), y=alt.Y("feature:N", sort="-x", title=None),
        ).properties(width=CHART_WIDTH, height=18 * len(importance)),
        width="content",
    )
    st.markdown("**Cobertura do universo**: membros do índice, com preço no Yahoo e elegíveis ao modelo")
    line_chart(window_coverage.rename(columns={
        "n_members": "Membros do índice", "n_with_prices": "Com preço", "n_eligible": "Elegíveis"}),
        "Ações", height=240, y_format=".0f")
    quality = features.quality
    with st.expander("Qualidade dos dados"):
        st.write(f"Membros do índice sem histórico utilizável no Yahoo (preço em menos da metade dos dias em que eram "
                 f"membros — deslistados ou ticker hoje de outra empresa): {len(quality['missing_tickers'])}")
        if quality["missing_detail"]:
            st.dataframe(pd.DataFrame(quality["missing_detail"]).rename(columns={
                "ticker": "Ticker", "member_days": "Dias como membro", "days_with_price": "Dias com preço"}), hide_index=True)
        st.write("Membros com volume financeiro mediano abaixo de US$ 1 milhão/dia (possível ticker reatribuído a outra "
                 f"empresa no Yahoo): {', '.join(quality['suspect_tickers']) or '—'}")
        st.write(f"Retornos diários com |r| > 50% entre membros (erro de dado ou spin-off): {quality['n_extreme_returns']}")
        if quality["extreme_returns"]:
            st.dataframe(pd.DataFrame(quality["extreme_returns"]), hide_index=True)
        st.write(f"Dias de posição sem retorno (contados como 0%): {int(daily['n_missing'].sum())}")
        st.json(quality["date_ranges"])
```

- [ ] **Step 4: Run to verify they pass, then the full suite**

Run: `venv/bin/python -m pytest tests/test_app.py -q -p no:cacheprovider`
Expected: PASS (all).
Run: `venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: PASS (all).

---

### Task 7: Documentação, simplificação e verificação

**Files:**
- Modify: `README.md`, `.claude/CLAUDE.md`
- Revisão com o plugin code-simplifier nos arquivos tocados

- [ ] **Step 1: README**

Substitua o primeiro parágrafo e as seções "Objective and scope", "Benchmark" e "How to run" do `README.md` por:

````markdown
Every *h* trading days, after the close, a LightGBM classifier estimates for each stock in the S&P 500 **on that day** the probability that it beats the median of the index members over the next *h* days. A long-only portfolio buys, at the next open, the stocks whose probability is above a threshold, weighted by that probability, and holds them until the next decision; if none qualifies, it holds cash at the 13-week T-bill rate.

## Objective and scope

- **Objective:** find out whether stock selection from price history alone beats the S&P 500 after costs, iterating quickly over horizons and training windows.
- **Run parameters:** horizon *h* (1 = daily, 5 = weekly, 21 = monthly decisions and label), training window *X* (sessions of examples) and retraining every *k* decisions. All runs cover the same period (from 2020-01-02).
- **Label:** relative. It is 1 if the stock's return over the next *h* sessions beats the median of the index members that day, so the model ranks stocks instead of guessing the market.
- **Data:** Yahoo Finance prices (split-adjusted, dividends added separately) and point-in-time S&P 500 constituents ([fja05680/sp500](https://github.com/fja05680/sp500)). Price history only, no macro data.
- **No lookahead:** a decision after the close of *t* uses only data dated on or before *t*; the model trains only on labels already known then (rows up to *t* − *h* − 1). The tests enforce it. An independent event-by-event simulation (cash, shares, fees, carry) checks the portfolio and its costs.
- **Out of scope:** live trading, hyperparameter tuning, and delisted companies (Yahoo has no data for them; the app lists them).

## Benchmark

S&P 500 Total Return (`^SP500TR`), open to open, on the same days as the portfolio.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python main.py download                                   # latest constituents and prices
venv/bin/python main.py features                                   # features, once per download
venv/bin/python main.py run --horizon 5 --window 21 --retrain 1    # one walk-forward run
venv/bin/streamlit run app/dashboard.py                            # results; pick h, X and k in the sidebar and click "Rodar"
venv/bin/python -m pytest
```

Defaults (h, X, k, threshold, transaction cost, backtest start) live in `config/config.py`.
````

- [ ] **Step 2: CLAUDE.md**

Em `.claude/CLAUDE.md`, na seção "Status", acrescente depois da linha da rodada 2:

```markdown
Round 3 (2026-09-29, spec: `docs/superpowers/specs/2026-09-29-v3-relative-horizon-design.md`): relative label (beats the day's median), horizon h with holding between decisions, refit every k decisions, predicted-vs-realized diagnostics.
```

Troque o item "Lookahead invariants" por:

```markdown
- Lookahead invariants (keep `tests/test_project.py`, `tests/test_features.py`, `tests/test_walk_forward.py` and `tests/test_backtest_validation.py` green): decision every h sessions after the close of t with data dated ≤ t; buy at the open of t+1 and hold until the open after the next decision; label = the h-session return from the open of t+1 beats the median of that date's rows; training rows t−h−X … t−h−1; never use Yahoo's `Adj Close`; the universe at t is the constituents snapshot in force at t.
```

Em "Engine layout", troque as linhas de `dataset.py`, `walk_forward.py`, `portfolio.py/metrics.py` e `project.py` por:

```markdown
- `engine/dataset.py` — the (date, ticker) feature table and `horizon_target` (h-session return and relative label).
- `engine/walk_forward.py` — `WalkForward`: a decision every h sessions, a refit every k decisions on the last X sessions of known labels, fits in parallel (joblib); `MODELS`.
- `engine/portfolio.py`, `engine/metrics.py` — `Portfolio` (threshold, probability weights, holding h sessions with drifting weights, cash at ^IRX, costs at trades) and metrics (incl. deciles and top-minus-bottom spread).
- `engine/project.py` — `Project`: atomic download, cached features and daily returns panel (`data/out/features`), one run per `RunKey` (`data/out/runs/lightgbm-h5-w21-k1`); both are tied to the config values that produced them (`FEATURE_SETTINGS` + `FEATURES_FORMAT`, `RUN_SETTINGS`) and are rebuilt or hidden when those change.
```

- [ ] **Step 3: code-simplifier**

Rode o agente `code-simplifier:code-simplifier` sobre os arquivos alterados: `engine/dataset.py`, `engine/walk_forward.py`, `engine/portfolio.py`, `engine/metrics.py`, `engine/project.py`, `main.py` (sem tocar nas notas do fim) e `app/dashboard.py`. A instrução é: sem mudar comportamento nem interfaces. Depois, rode a suíte.

Run: `venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: PASS (all).

---

### Task 8: Dados reais e medições

**Files:** nenhum arquivo do projeto; scripts no scratchpad.

- [ ] **Step 1: Refazer as features e rodar os runs**

Run (em segundo plano, com log):
- `venv/bin/python main.py features`
- depois `venv/bin/python main.py run --horizon H --window X --retrain K`, para (1, 21, 1), (5, 21, 1), (21, 63, 1) e (1, 21, 5).

Expected: cada run imprime a tabela, o giro médio por troca e as métricas do modelo. `Project(CONFIG).runs()` lista as quatro chaves.

- [ ] **Step 2: Efeito das features de pares**

Script no scratchpad: roda `WalkForward(21, LGBM_PARAMS, 5, 1, ..., n_jobs=-1)` sobre o dataset com alvo h = 5 duas vezes, com `FEATURES` e com `[f for f in FEATURES if f not in PEER_FEATURES]`. Compara AUC, IC e spread médio decil 10 − decil 1.

- [ ] **Step 3: Smoke test do app nos dados reais**

`AppTest` do `app/dashboard.py` com o `CONFIG` real, na chave (5, 21, 1): sem exceção, seis abas, tabela de decis presente.
