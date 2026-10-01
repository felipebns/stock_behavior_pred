# Rodada 5 — intraday, universo completo, vários modelos, varredura e walk-forward de parâmetros — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prever e operar intraday; aplicar o universo "completo nos últimos 252 pregões"; oferecer cinco tipos de modelo, um para todas as ações ou um por ação; rodar varreduras pelo CLI; escolher run e limiar por walk-forward de Sharpe líquido no app.

**Architecture:**
- `engine/models.py` (novo) concentra as fábricas de modelos.
- `WalkForward` recebe uma fábrica `make_model(params, n_rows)` e um escopo (`pooled`/`per_stock`).
- `Portfolio` passa a cobrar ida e volta em todo dia investido.
- `engine/selection.py` (novo) costura os melhores candidatos do passado.
- `Project`/`RunKey` ganham modelo e escopo, e o fingerprint passa a ser por modelo.
- O CLI ganha `sweep`; o app ganha a aba Walk-forward.

**Tech Stack:** Python 3.14 (venv), pandas 3, numpy 2, scikit-learn, LightGBM, joblib, Streamlit + Altair, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-v5-intraday-models-walkforward-design.md` (onde omisso, valem as specs das rodadas 4, 3 e 2).

## Global Constraints

- **Commits:** liberados, um por tarefa concluída e verde, com o trailer `Co-Authored-By` do Claude Code. `git push` é proibido. Só mexer em arquivos deste repositório (ver `.claude/CLAUDE.md`).
- Comandos pelo venv: `venv/bin/python -m pytest ...`, `venv/bin/python main.py ...`.
- **Tempo:**
  - decisão após o fechamento de *t*, só com dados ≤ *t*;
  - `fwd_ret = Close[t+1] / Open[t+1] − 1` e `label = fwd_ret > 0`;
  - treino nas linhas [*t*−X, *t*−1];
  - retreino a cada k decisões, com o ajuste na primeira decisão do grupo;
  - no escopo por ação, cada ação treina só com as próprias linhas.
- yfinance com `auto_adjust=False`; o `Adj Close` nunca é usado.
- Paralelismo só no `WalkForward` (processos joblib, modelos com `n_jobs=1`), bit a bit igual ao sequencial.
- Código o mais simples possível; identificadores e docstrings em inglês, UI em português, README e CLAUDE.md em inglês. As notas do dono no fim do `main.py` ficam intactas (só `Edit` pontual).

## Review Focus

1. **Ação com poucas linhas ou uma classe só no escopo por ação:** prevê a fração de altas, sem ajustar modelo. *(Task 3: `test_per_stock_without_enough_rows_predicts_its_up_frequency`)*
2. **Features com NaN em modelos do scikit-learn:** a imputação acontece só com as linhas de treino. *(Task 2: `test_every_model_fits_and_predicts_probabilities_with_missing_values`)*
3. **Walk-forward com período menor que o olhar para trás:** resultado vazio, e o app avisa. *(Task 5: `test_too_few_days_gives_an_empty_result`)*
4. **A escolha nunca usa o dia que vai operar.** *(Task 5: `test_a_pick_never_looks_at_the_days_it_will_trade`)*
5. **Mudar os parâmetros de um modelo esconde só os runs dele.** *(Task 6: `test_changing_a_models_parameters_hides_only_its_runs`)*

---

### Task 1: Retorno intraday, universo completo e config

**Files:** `engine/prices.py`, `engine/dataset.py`, `config/config.py`, `tests/test_prices.py`, `tests/test_dataset.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `forward_intraday_return(open_, close)` (substitui `forward_open_return`);
  - `build_dataset(ticker_features, date_features, membership, close, fwd_ret, history_days)`;
  - `MODEL_PARAMS`;
  - `Config` com os campos `complete_history_days=252`, `model="lightgbm"`, `scope="pooled"`, `per_stock_window_days=252`, `per_stock_retrain_every=21`, `per_stock_min_rows=63`, `selection_lookback_days=252`, `selection_step_days=63` e `model_params`. Saem `min_history_days`, `min_child_share`, `min_child_floor` e `lgbm_params`.

- [ ] **Step 1: Testes**

Em `tests/test_prices.py`, troque `forward_open_return` por `forward_intraday_return` no import e substitua os dois testes de `forward_open_return` por:

```python
def test_forward_intraday_return_is_next_open_to_next_close():
    open_ = pd.DataFrame({"A": [10.0, 11.0, 12.0]})
    close = pd.DataFrame({"A": [10.5, 11.55, 11.4]})
    fwd = forward_intraday_return(open_, close)
    assert fwd.loc[0, "A"] == pytest.approx(11.55 / 11.0 - 1.0)
    assert fwd.loc[1, "A"] == pytest.approx(11.4 / 12.0 - 1.0)
    assert np.isnan(fwd.loc[2, "A"])


def test_forward_intraday_return_is_nan_when_a_price_is_missing():
    assert np.isnan(forward_intraday_return(pd.Series([10.0, np.nan]), pd.Series([10.0, 11.0])).iloc[0])
```

Em `tests/test_dataset.py`:
- troque `min_history_days=2` por `history_days=2` nas três chamadas;
- a lista esperada em `test_rows_are_members_with_close_history_and_date_features` passa a ser `[(DATES[1], "A"), (DATES[2], "A"), (DATES[2], "C"), (DATES[3], "B"), (DATES[3], "C")]` (B em DATES[2] sai: faltou preço em DATES[1]);
- renomeie esse teste para `test_rows_are_members_with_a_complete_recent_history_and_date_features`;
- em `test_columns_and_values`, troque `(DATES[2], "B")` por `(DATES[3], "B")` e os valores esperados `7.0`/`10.0` por `10.0`/`13.0`;
- em `test_label_is_positive_forward_return`, os vetores esperados passam a ser `[0.02, 0.0, 0.03, np.nan, np.nan]` e `[1.0, 0.0, 1.0, np.nan, np.nan]`.

Em `tests/test_config.py`, troque o import por `from config.config import CONFIG, LGBM_PARAMS, MODEL_PARAMS, TICKER_ALIASES` e troque `test_run_defaults` por:

```python
def test_run_defaults():
    assert (CONFIG.model, CONFIG.scope, CONFIG.train_window_days, CONFIG.retrain_every) == ("lightgbm", "pooled", 21, 1)
    assert (CONFIG.per_stock_window_days, CONFIG.per_stock_retrain_every, CONFIG.per_stock_min_rows) == (252, 21, 63)
    assert (CONFIG.complete_history_days, CONFIG.selection_lookback_days, CONFIG.selection_step_days) == (252, 252, 63)
    assert set(CONFIG.model_params) == set(MODEL_PARAMS) == {"lightgbm", "logistic", "random_forest", "extra_trees", "naive_bayes"}
    assert CONFIG.model_params is not MODEL_PARAMS and CONFIG.model_params["lightgbm"]["min_child_floor"] == 20
```

- [ ] **Step 2: Rodar e ver falhar** — `venv/bin/python -m pytest tests/test_prices.py tests/test_dataset.py tests/test_config.py -q -p no:cacheprovider` → FAIL (import de `forward_intraday_return`, argumento `history_days`, campos de config).

- [ ] **Step 3: Implementar**

Em `engine/prices.py`, troque `forward_open_return` por:

```python
def forward_intraday_return(open_, close):
    """Row t: buy at the open of t+1 and sell at the close of t+1 (a dividend going ex at t+1 is already out of that open)."""
    return close.shift(-1) / open_.shift(-1) - 1.0
```

Em `engine/dataset.py`, troque a assinatura, a docstring e o cálculo de elegibilidade de `build_dataset` por:

```python
def build_dataset(ticker_features: dict[str, pd.DataFrame], date_features: pd.DataFrame, membership: pd.DataFrame,
                  close: pd.DataFrame, fwd_ret: pd.DataFrame, history_days: int) -> pd.DataFrame:
    """A row per stock that, on that date, is in the index, has a close on each of the last history_days sessions
    (that date included) and all date features: a gap or a recent listing keeps it out until the window is complete."""
    counts = np.cumsum(close.notna().to_numpy(), axis=0)
    previous = np.zeros_like(counts)
    previous[history_days:] = counts[:-history_days]
    eligible = (
        membership.to_numpy()
        & (counts - previous >= history_days)
        & date_features[DATE_FEATURES].notna().all(axis=1).to_numpy()[:, None]
    )
```

(o resto da função fica igual).

Em `config/config.py`:
- acrescente `import copy`;
- depois de `LGBM_PARAMS`, acrescente:

```python
# Default parameters per model (tuning starts here). sklearn models get median imputation; logistic also standardization.
MODEL_PARAMS: dict[str, dict] = {
    "lightgbm": {**LGBM_PARAMS, "min_child_share": 0.004, "min_child_floor": 20},
    "logistic": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 20, "max_features": "sqrt",
                      "random_state": 42, "n_jobs": 1},
    "extra_trees": {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 20, "max_features": "sqrt",
                    "random_state": 42, "n_jobs": 1},
    "naive_bayes": {},
}
```

- no `Config`:
  - troque `min_history_days: int = 63` por `complete_history_days: int = 252`;
  - apague `min_child_share`, `min_child_floor` e `lgbm_params`;
  - depois de `retrain_every`, acrescente:

```python
    model: str = "lightgbm"
    scope: str = "pooled"
    per_stock_window_days: int = 252
    per_stock_retrain_every: int = 21
    per_stock_min_rows: int = 63
    selection_lookback_days: int = 252
    selection_step_days: int = 63
```

  - no lugar do antigo `lgbm_params`, acrescente `model_params: dict = field(default_factory=lambda: copy.deepcopy(MODEL_PARAMS))`.

- [ ] **Step 4: Rodar e ver passar** — mesmo comando → PASS. Daqui até a Task 6, `test_project`, `test_main` e `test_app` ficam vermelhos.

- [ ] **Step 5: Commit** — `git add` dos arquivos da tarefa e `git commit -m "Intraday next-day return, complete-history universe and model parameters"` (com o trailer).

---

### Task 2: Fábricas de modelos

**Files:** Create `engine/models.py`, `tests/test_models.py`

**Interfaces:** Produces `MODELS: dict[str, tuple[str, Callable[[dict, int], estimator]]]` e `importance(model, n_features) -> np.ndarray`.

- [ ] **Step 1: Testes** — crie `tests/test_models.py`:

```python
import numpy as np
import pytest

from config.config import MODEL_PARAMS
from engine.models import MODELS, importance


def data(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0.0, 0.5, n) > 0).astype(int)
    X[rng.random(X.shape) < 0.1] = np.nan
    return X, y


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_fits_and_predicts_probabilities_with_missing_values(name):
    X, y = data()
    model = MODELS[name][1](MODEL_PARAMS[name], len(y)).fit(X, y)
    prob = model.predict_proba(X)[:, 1]
    assert prob.shape == (len(y),) and ((prob >= 0) & (prob <= 1)).all()
    assert np.corrcoef(prob, y)[0, 1] > 0.3
    weights = importance(model, X.shape[1])
    assert weights.shape == (4,) and (weights.sum() == pytest.approx(1.0) or weights.sum() == 0)


def test_lightgbm_min_child_samples_grows_with_the_training_rows():
    make = MODELS["lightgbm"][1]
    assert make({**MODEL_PARAMS["lightgbm"], "min_child_share": 0.5, "min_child_floor": 2}, 8).min_child_samples == 4
    assert make(MODEL_PARAMS["lightgbm"], 100).min_child_samples == 20


@pytest.mark.parametrize("name", ["lightgbm", "logistic", "random_forest", "extra_trees"])
def test_importance_points_at_the_informative_feature(name):
    X, y = data()
    assert importance(MODELS[name][1](MODEL_PARAMS[name], len(y)).fit(X, y), 4).argmax() == 0
```

- [ ] **Step 2: Rodar e ver falhar** — `venv/bin/python -m pytest tests/test_models.py -q -p no:cacheprovider` → FAIL (`No module named 'engine.models'`).

- [ ] **Step 3: Implementar** — crie `engine/models.py`:

```python
"""Classifiers the walk-forward can fit: name → (label shown in the app, factory(params, n_rows))."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _lightgbm(params: dict, n_rows: int):
    """min_child_samples grows with the training rows: share × rows, never below the floor."""
    params = dict(params)
    share, floor = params.pop("min_child_share"), params.pop("min_child_floor")
    return LGBMClassifier(**params, min_child_samples=max(floor, round(share * n_rows)))


def _sklearn(estimator, scale: bool = False):
    """sklearn models don't accept NaN: missing features get the training rows' median (and, for scale, the training
    rows' mean and spread) — fit inside the pipeline, so only on the training window."""
    def make(params: dict, n_rows: int):
        steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
        return make_pipeline(*steps, *([StandardScaler()] if scale else []), estimator(**params))
    return make


MODELS = {
    "lightgbm": ("LightGBM", _lightgbm),
    "logistic": ("Logística regularizada", _sklearn(LogisticRegression, scale=True)),
    "random_forest": ("Random Forest", _sklearn(RandomForestClassifier)),
    "extra_trees": ("Extra Trees", _sklearn(ExtraTreesClassifier)),
    "naive_bayes": ("Naive Bayes gaussiano", _sklearn(GaussianNB)),
}


def importance(model, n_features: int) -> np.ndarray:
    """Normalized importance: tree gain/impurity, |coefficient| for the logistic, zeros when the model has neither."""
    final = model[-1] if hasattr(model, "steps") else model
    if hasattr(final, "feature_importances_"):
        values = np.asarray(final.feature_importances_, dtype=float)
    elif hasattr(final, "coef_"):
        values = np.abs(np.asarray(final.coef_, dtype=float)).ravel()
    else:
        values = np.zeros(n_features)
    return values / values.sum() if values.sum() > 0 else values
```

- [ ] **Step 4: Rodar e ver passar** — mesmo comando → PASS.

- [ ] **Step 5: Commit** — `git commit -m "Model factories: LightGBM, logistic, random forest, extra trees, naive Bayes"`.

---

### Task 3: `WalkForward` intraday, com fábrica de modelo e escopo

**Files:** `engine/walk_forward.py` (inteiro), `tests/test_walk_forward.py` (inteiro)

**Interfaces:**
- Consumes: `MODELS`, `importance` (Task 2).
- Produces:
  - `SCOPES = {"pooled": "um modelo para todas", "per_stock": "um modelo por ação"}`;
  - `WalkForward(train_window, make_model, params, retrain_every=1, scope="pooled", min_stock_rows=63, n_jobs=1)`;
  - `run(dataset, features, calendar, start=None, progress=None) -> WalkForwardResult(predictions, importance)`.

- [ ] **Step 1: Testes** — substitua `tests/test_walk_forward.py` por:

```python
import numpy as np
import pandas as pd
import pytest

from config.config import MODEL_PARAMS
from engine.models import MODELS
from engine.walk_forward import WalkForward

DATES = pd.bdate_range("2021-01-01", periods=15, name="date")


def spy_dataset(tickers=("A", "B")) -> pd.DataFrame:
    rows = [
        {"date": day, "ticker": ticker, "day": float(k), "stock": float(i), "fwd_ret": 0.01, "label": float((k + i) % 2)}
        for k, day in enumerate(DATES) if k >= 2
        for i, ticker in enumerate(tickers)
    ]
    return pd.DataFrame(rows).set_index(["date", "ticker"])


class Spy:
    """Records which days and stocks (features 'day', 'stock') each fit and predict sees, and the rows it was built for."""
    log: list = []

    def __init__(self, n_rows):
        self.n_rows = n_rows

    def fit(self, X, y):
        Spy.log.append(("fit", sorted({int(v) for v in X[:, 0]}), sorted({int(v) for v in X[:, 1]}), self.n_rows))
        self.feature_importances_ = np.array([3.0, 1.0])
        return self

    def predict_proba(self, X):
        Spy.log.append(("predict", sorted({int(v) for v in X[:, 0]}), sorted({int(v) for v in X[:, 1]}), None))
        p = np.full(len(X), 0.7)
        return np.column_stack([1 - p, p])


def run_spy(dataset, window=4, start=None, **options):
    Spy.log = []
    return WalkForward(window, lambda params, n_rows: Spy(n_rows), {}, n_jobs=1, **options).run(
        dataset, ["day", "stock"], DATES, start=start)


def logged(kind):
    return [entry[1:] for entry in Spy.log if entry[0] == kind]


def random_dataset(n_days=60, n_tickers=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days, name="date")
    index = pd.MultiIndex.from_product([dates, [f"T{i}" for i in range(n_tickers)]], names=["date", "ticker"])
    X = rng.normal(size=(len(index), 3))
    realized = 0.01 * X[:, 0] + rng.normal(0.0, 0.02, len(index))
    dataset = pd.DataFrame(X, index=index, columns=["a", "b", "c"]).astype(np.float32)
    return dataset.assign(fwd_ret=realized, label=(realized > 0).astype(float)), dates


def test_trains_on_exactly_the_labels_known_at_decision_time():
    run_spy(spy_dataset())
    decisions = range(6, 15)
    assert [days for days, _, _ in logged("predict")] == [[k] for k in decisions]
    assert [days for days, _, _ in logged("fit")] == [list(range(k - 4, k)) for k in decisions]


def test_rows_with_unknown_label_are_not_used_for_training():
    dataset = spy_dataset()
    dataset.loc[dataset.index.get_level_values("date") == DATES[5], "label"] = np.nan
    run_spy(dataset)
    fits = [days for days, _, _ in logged("fit")]
    assert fits[0] == [2, 3, 4]
    assert all(5 not in days for days in fits)


def test_the_model_factory_gets_the_number_of_training_rows():
    run_spy(spy_dataset())
    assert logged("fit")[0][2] == 8


def test_outputs_predictions_and_importance():
    result = run_spy(spy_dataset())
    predictions = result.predictions
    assert list(predictions.columns) == ["date", "ticker", "prob", "fwd_ret", "label"]
    assert predictions["date"].min() == DATES[6]
    assert len(predictions) == 2 * (len(DATES) - 6)
    assert (predictions["prob"] == 0.7).all()
    assert list(result.importance.columns) == ["day", "stock"]
    assert result.importance.iloc[0].tolist() == [0.75, 0.25]
    assert list(result.importance.index) == list(DATES[6:])


def test_start_sets_the_first_decision():
    assert run_spy(spy_dataset(), start=DATES[9]).predictions["date"].min() == DATES[9]


def test_window_too_long_for_the_start_date_is_refused():
    with pytest.raises(ValueError, match="máximo é 6"):
        run_spy(spy_dataset(), window=7, start=DATES[8])


def test_a_start_with_no_room_for_any_window_says_so():
    with pytest.raises(ValueError, match="não há pregões"):
        run_spy(spy_dataset(), window=4, start=DATES[2])


def test_progress_reaches_one():
    seen = []
    Spy.log = []
    WalkForward(4, lambda params, n_rows: Spy(n_rows), {}).run(spy_dataset(), ["day", "stock"], DATES, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


def test_one_fit_serves_retrain_every_consecutive_decisions():
    result = run_spy(spy_dataset(), window=3, retrain_every=3)
    assert [entry[0] for entry in Spy.log] == ["fit", "predict", "predict", "predict"] * 3 + ["fit", "predict"]
    assert [days for days, _, _ in logged("fit")] == [[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]
    assert list(result.importance.index) == [DATES[5], DATES[8], DATES[11], DATES[14]]


def test_per_stock_fits_one_model_per_stock_on_its_own_rows():
    run_spy(spy_dataset(("A", "B", "C")), window=3, scope="per_stock", min_stock_rows=1)
    assert logged("fit")[:3] == [([2, 3, 4], [i], 3) for i in range(3)]
    assert logged("predict")[:3] == [([5], [i], None) for i in range(3)]


def test_per_stock_without_enough_rows_predicts_its_up_frequency():
    predictions = run_spy(spy_dataset(), window=3, scope="per_stock", min_stock_rows=4).predictions
    assert logged("fit") == []
    first = predictions[predictions["date"] == DATES[5]].set_index("ticker")["prob"]
    assert first.tolist() == pytest.approx([1 / 3, 2 / 3])


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_a_window_with_a_single_class_predicts_that_class(value):
    params = {**MODEL_PARAMS["lightgbm"], "n_estimators": 5}
    result = WalkForward(1, MODELS["lightgbm"][1], params).run(spy_dataset().assign(label=value), ["day", "stock"], DATES)
    assert (result.predictions["prob"] == value).all()
    assert (result.importance.to_numpy() == 0).all()


class Echo:
    """Predicts, for each row, its own first feature: every probability must land on its own (date, ticker) row."""

    def fit(self, X, y):
        self.feature_importances_ = np.ones(X.shape[1])
        return self

    def predict_proba(self, X):
        return np.column_stack([1 - X[:, 0], X[:, 0]])


@pytest.mark.parametrize("scope", ["pooled", "per_stock"])
def test_each_probability_lands_on_its_own_row(scope):
    dataset, dates = random_dataset(n_days=30, n_tickers=7)
    dataset["a"] = np.random.default_rng(1).uniform(size=len(dataset)).astype(np.float32)
    predictions = WalkForward(5, lambda params, n_rows: Echo(), {}, retrain_every=3, scope=scope, min_stock_rows=1,
                              n_jobs=2).run(dataset, ["a", "b"], dates).predictions
    rows = pd.MultiIndex.from_frame(predictions[["date", "ticker"]])
    np.testing.assert_array_equal(predictions["prob"].to_numpy(), dataset.loc[rows, "a"].to_numpy())


@pytest.mark.parametrize("model, scope, retrain_every", [("lightgbm", "pooled", 1), ("lightgbm", "pooled", 2),
                                                          ("logistic", "per_stock", 3)])
def test_parallel_run_is_identical_to_sequential(model, scope, retrain_every):
    dataset, dates = random_dataset()
    params = {**MODEL_PARAMS[model], "n_estimators": 20} if model == "lightgbm" else MODEL_PARAMS[model]
    options = {"retrain_every": retrain_every, "scope": scope, "min_stock_rows": 5}
    sequential = WalkForward(10, MODELS[model][1], params, n_jobs=1, **options).run(dataset, ["a", "b", "c"], dates)
    parallel = WalkForward(10, MODELS[model][1], params, n_jobs=2, **options).run(dataset, ["a", "b", "c"], dates)
    pd.testing.assert_frame_equal(sequential.predictions, parallel.predictions, check_exact=True)
    pd.testing.assert_frame_equal(sequential.importance, parallel.importance, check_exact=True)
    assert ((sequential.predictions["prob"] > 0) & (sequential.predictions["prob"] < 1)).all()
```

- [ ] **Step 2: Rodar e ver falhar** — `venv/bin/python -m pytest tests/test_walk_forward.py -q -p no:cacheprovider` → FAIL (assinatura antiga do `WalkForward`).

- [ ] **Step 3: Implementar** — substitua `engine/walk_forward.py` por:

```python
"""Walk-forward: after each close, predict the next day's open-to-close move with a model fit only on labels already
known; decision dates run in parallel."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs

from engine.models import importance

SCOPES = {"pooled": "um modelo para todas", "per_stock": "um modelo por ação"}


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    importance: pd.DataFrame


class WalkForward:
    """Row s's label (open to close of s+1) is known at the close of s+1, so a fit at decision t uses exactly the rows
    dated t-X .. t-1; that fit also predicts the next retrain_every-1 decisions, all after t. "pooled" fits one model on
    every stock's rows; "per_stock" fits one model per stock on its own rows."""

    def __init__(self, train_window: int, make_model, params: dict, retrain_every: int = 1, scope: str = "pooled",
                 min_stock_rows: int = 63, n_jobs: int = 1):
        self.train_window = train_window
        self.make_model = make_model
        self.params = params
        self.retrain_every = retrain_every
        self.scope = scope
        self.min_stock_rows = min_stock_rows
        self.n_jobs = n_jobs

    def decisions(self, row_pos: np.ndarray, calendar: pd.DatetimeIndex, start=None) -> np.ndarray:
        first = row_pos[0] + self.train_window
        begin = first if start is None else calendar.searchsorted(pd.Timestamp(start))
        if begin < first:
            room = begin - row_pos[0]
            if room < 1:
                raise ValueError(f"não há pregões de dados antes de {pd.Timestamp(start).date()} para treinar")
            raise ValueError(f"a janela de {self.train_window} pregões precisa de {self.train_window} pregões de dados "
                             f"antes de {pd.Timestamp(start).date()}; o máximo é {room}")
        decisions = np.unique(row_pos)
        decisions = decisions[decisions >= begin]
        if len(decisions) == 0:
            raise ValueError("nenhuma data de decisão depois do início pedido")
        return decisions

    def run(self, dataset: pd.DataFrame, features: list[str], calendar: pd.DatetimeIndex, start=None,
            progress=None) -> WalkForwardResult:
        row_pos = calendar.get_indexer(dataset.index.get_level_values("date"))
        stocks = pd.factorize(dataset.index.get_level_values("ticker"))[0]
        X = dataset[features].to_numpy(dtype=np.float32)
        y = dataset["label"].to_numpy(dtype=float)
        decisions = self.decisions(row_pos, calendar, start)
        fits = [decisions[i:i + self.retrain_every] for i in range(0, len(decisions), self.retrain_every)]
        blocks = np.array_split(np.arange(len(fits)), min(len(fits), 4 * effective_n_jobs(self.n_jobs)))
        jobs = (delayed(_fit_block)(X, y, row_pos, stocks, [fits[i] for i in block], self.train_window, self.make_model,
                                    self.params, self.scope, self.min_stock_rows) for block in blocks)
        prob = np.full(len(dataset), np.nan)
        gains = []
        for done, (rows, probs, block_gains) in enumerate(Parallel(n_jobs=self.n_jobs, return_as="generator")(jobs), start=1):
            prob[rows] = probs
            gains.append(block_gains)
            if progress is not None:
                progress(done / len(blocks))
        decided = np.flatnonzero(np.isin(row_pos, decisions))
        predictions = pd.DataFrame({
            "date": dataset.index.get_level_values("date")[decided],
            "ticker": dataset.index.get_level_values("ticker")[decided],
            "prob": prob[decided],
            "fwd_ret": dataset["fwd_ret"].to_numpy()[decided],
            "label": y[decided],
        })
        importance_ = pd.DataFrame(np.vstack(gains), index=pd.DatetimeIndex(calendar[[group[0] for group in fits]], name="date"),
                                   columns=features)
        return WalkForwardResult(predictions, importance_)


def _fit(X, y, make_model, params, min_rows=1):
    """P(up) for new rows and the normalized importance. Fewer than min_rows examples or a single class predict the
    up-frequency of those examples (0.5 with none): a classifier fit on one class would put it in the wrong column."""
    if len(y) < min_rows or len(np.unique(y)) < 2:
        constant = float(y.mean()) if len(y) else 0.5
        return (lambda rows: np.full(len(rows), constant)), np.zeros(X.shape[1])
    model = make_model(params, len(y)).fit(X, y.astype(int))
    return (lambda rows: model.predict_proba(rows)[:, 1]), importance(model, X.shape[1])


def _fit_block(X, y, row_pos, stocks, fits, window, make_model, params, scope, min_stock_rows):
    rows_out, probs_out, gains = [], [], []
    for group in fits:
        train = np.arange(np.searchsorted(row_pos, group[0] - window, side="left"),
                          np.searchsorted(row_pos, group[0] - 1, side="right"))
        train = train[~np.isnan(y[train])]
        target = np.arange(np.searchsorted(row_pos, group[0], side="left"), np.searchsorted(row_pos, group[-1], side="right"))
        if scope == "pooled":
            predict, gain = _fit(X[train], y[train], make_model, params)
            rows_out.append(target)
            probs_out.append(predict(X[target]))
            gains.append(gain)
            continue
        order = train[np.argsort(stocks[train], kind="stable")]
        sorted_stocks = stocks[order]
        stock_gains = []
        for stock in np.unique(stocks[target]):
            mine = order[np.searchsorted(sorted_stocks, stock, side="left"):np.searchsorted(sorted_stocks, stock, side="right")]
            predict, gain = _fit(X[mine], y[mine], make_model, params, min_stock_rows)
            rows = target[stocks[target] == stock]
            rows_out.append(rows)
            probs_out.append(predict(X[rows]))
            stock_gains.append(gain)
        gains.append(np.mean(stock_gains, axis=0))
    return np.concatenate(rows_out), np.concatenate(probs_out), np.vstack(gains)
```

- [ ] **Step 4: Rodar e ver passar** — mesmo comando → PASS.

- [ ] **Step 5: Mutação** — troque `group[0] - 1, side="right"` por `group[0], side="right"` e rode `-k trains_on_exactly`: deve falhar; desfaça e rode de novo (PASS).

- [ ] **Step 6: Commit** — `git commit -m "Walk-forward: intraday training slice, model factories, pooled or per-stock scope"`.

---

### Task 4: `Portfolio` intraday e simulador

**Files:** `engine/portfolio.py`, `tests/test_portfolio.py`, `tests/test_backtest_validation.py` (inteiro)

**Interfaces:** Produces `Portfolio.backtest(predictions, market)` com `turnover = 2` e `cost = 1 − (1 − c)²` nos dias investidos (0 em caixa) e `net = (1 + gross)(1 − cost) − 1`. As colunas não mudam.

- [ ] **Step 1: Testes**

Em `tests/test_portfolio.py`, substitua `test_costs_follow_turnover_against_drifted_weights` por:

```python
def test_costs_are_a_full_round_trip_on_every_invested_day():
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
    daily = Portfolio(0.55, cost_bps=10.0).backtest(predictions, market).daily
    round_trip = 1 - (1 - 0.001) ** 2
    assert daily["turnover"].tolist() == [2.0, 2.0, 2.0, 0.0, 2.0]
    assert daily["cost"].tolist() == pytest.approx([round_trip] * 3 + [0.0, round_trip])
    assert daily["net"].iloc[0] == pytest.approx((1 + daily["gross"].iloc[0]) * (1 - 0.001) ** 2 - 1)
    assert daily["net"].iloc[3] == pytest.approx(0.0001)
```

Substitua `tests/test_backtest_validation.py` por:

```python
"""Independent check of Portfolio against an event-by-event simulation in the style of the Algotrading course (Aulas 4
e 5): cash and share counts, a fee on the traded value of each buy and sell, idle cash earning the risk-free rate.
Intraday: buy at the open after the decision, sell at that day's close. It shares no code with engine.portfolio."""
import numpy as np
import pandas as pd

from engine.portfolio import Portfolio
from engine.prices import forward_intraday_return

CAPITAL = 1_000_000.0
THRESHOLD = 0.55
FEE = 0.001


def market_data(n_days: int = 40, seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days, name="date")
    tickers = ["A", "B", "C", "D", "E"]
    opens = pd.DataFrame(50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (n_days, len(tickers))), axis=0)),
                         index=dates, columns=tickers)
    closes = opens * np.exp(rng.normal(0.0, 0.015, opens.shape))
    opens.loc[dates[17]:, "E"] = np.nan
    closes.loc[dates[16]:, "E"] = np.nan
    rf_daily = pd.Series(0.0002 + 0.0001 * np.sin(np.arange(n_days)), index=dates)
    return opens, closes, rf_daily


def model_output(opens, closes, seed: int = 5) -> pd.DataFrame:
    """Random probabilities every session (with the realized next-day return); every 4th day nobody passes."""
    rng = np.random.default_rng(seed)
    fwd_ret = forward_intraday_return(opens, closes)
    rows = []
    for k, date in enumerate(opens.index[2:-3]):
        high = 0.5 if k % 4 == 3 else 0.8
        rows += [{"date": date, "ticker": t, "prob": rng.uniform(0.3, high), "fwd_ret": fwd_ret.at[date, t]}
                 for t in opens.columns if not np.isnan(closes.at[date, t])]
    return pd.DataFrame(rows)


def market_frame(opens, closes, rf_daily) -> pd.DataFrame:
    dates = opens.index
    return pd.DataFrame({
        "holding_date": pd.Series(dates, index=dates).shift(-1),
        "bench_fwd_ret": forward_intraday_return(opens.mean(axis=1), closes.mean(axis=1)),
        "rf_daily": rf_daily,
    }, index=dates)


def simulate(opens, closes, rf_daily, predictions) -> pd.Series:
    """Portfolio value after each trading day's close, keyed by that day."""
    targets = {}
    for date, day in predictions.groupby("date"):
        chosen = day[day["prob"] > THRESHOLD]
        targets[date] = dict(zip(chosen["ticker"], chosen["prob"] / chosen["prob"].sum()))
    dates, cash, equity = opens.index, CAPITAL, {}
    for i in range(dates.get_loc(min(targets)), dates.get_loc(max(targets)) + 1):
        decided, day = dates[i], dates[i + 1]
        target = targets.get(decided, {})
        if target:
            invested = cash * (1 - FEE)
            proceeds = 0.0
            for ticker, weight in target.items():
                buy, sell = opens.at[day, ticker], closes.at[day, ticker]
                money = weight * invested
                proceeds += money if np.isnan(buy) or np.isnan(sell) else money / buy * sell
            cash = proceeds * (1 - FEE)
        else:
            cash *= 1 + rf_daily.at[decided]
        equity[day] = cash
    return pd.Series(equity)


def test_portfolio_matches_an_event_by_event_simulation():
    opens, closes, rf_daily = market_data()
    predictions = model_output(opens, closes)
    daily = Portfolio(THRESHOLD, FEE * 10_000).backtest(predictions, market_frame(opens, closes, rf_daily)).daily
    assert daily["invested"].any() and not daily["invested"].all()
    assert daily["n_missing"].sum() > 0
    expected = CAPITAL * (1.0 + daily["net"]).cumprod()
    simulated = simulate(opens, closes, rf_daily, predictions).reindex(daily.index)
    np.testing.assert_allclose(simulated.to_numpy(), expected.to_numpy(), rtol=1e-10)
```

- [ ] **Step 2: Rodar e ver falhar** — `venv/bin/python -m pytest tests/test_portfolio.py tests/test_backtest_validation.py -q -p no:cacheprovider` → FAIL (o giro ainda considera a deriva).

- [ ] **Step 3: Implementar**

Em `engine/portfolio.py`:
- a docstring do módulo passa a ser `"""Long-only intraday portfolio: after each close, buy at the next open the stocks with prob > threshold (weights ∝ prob) and sell them at that day's close."""`;
- a docstring de `backtest` passa a ser `"""Every invested day buys at the open and sells at the close, paying the fee on both trades: net = (1 + gross)(1 − fee)² − 1. A held stock without a return that day (delisting, halt) counts as 0: dropping it would use future information."""`;
- apague `_turnover`;
- troque as linhas `turnover = self._turnover(...)` e `cost = turnover * ...` por:

```python
        fee = self.cost_bps / 10_000.0
        turnover = 2.0 * invested.astype(float)
        cost = invested * (1.0 - (1.0 - fee) ** 2)
```

(`net` continua `((1.0 + gross) * (1.0 - cost) - 1.0)`).

- [ ] **Step 4: Rodar e ver passar** — mesmo comando → PASS.

- [ ] **Step 5: Mutação** — troque `** 2` por `** 1` e rode o simulador (deve falhar); desfaça e rode de novo (PASS).

- [ ] **Step 6: Commit** — `git commit -m "Intraday portfolio: round-trip cost on every invested day, simulator updated"`.

---

### Task 5: Walk-forward de parâmetros

**Files:** Create `engine/selection.py`, `tests/test_selection.py`

**Interfaces:** Produces `walk_forward_selection(candidates: dict[str, DataFrame], lookback: int, step: int) -> SelectionResult(daily, choices)`.
- `daily`: as linhas dos candidatos escolhidos, mais a coluna `candidate`.
- `choices`: colunas `start, candidate, lookback_sharpe, days`.

- [ ] **Step 1: Testes** — crie `tests/test_selection.py`:

```python
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
```

- [ ] **Step 2: Rodar e ver falhar** — `venv/bin/python -m pytest tests/test_selection.py -q -p no:cacheprovider` → FAIL (`No module named 'engine.selection'`).

- [ ] **Step 3: Implementar** — crie `engine/selection.py`:

```python
"""Walk-forward choice of run and threshold: every `step` days, pick the candidate with the best net Sharpe over the
previous `lookback` days and follow it until the next pick. Nothing is refit: candidates are out-of-sample already."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.metrics import TRADING_DAYS

CHOICE_COLUMNS = ["start", "candidate", "lookback_sharpe", "days"]


@dataclass
class SelectionResult:
    daily: pd.DataFrame
    choices: pd.DataFrame


def walk_forward_selection(candidates: dict[str, pd.DataFrame], lookback: int, step: int) -> SelectionResult:
    """`candidates`: name → Portfolio daily frame, all on the same days. Row i is the day traded after decision i and is
    realized at that day's close, so the pick made at decision i sees rows i-lookback .. i-1 only. An undefined Sharpe
    (no variation, e.g. all cash) counts as 0; ties go to the first candidate."""
    names = list(candidates)
    index = candidates[names[0]].index
    if any(not frame.index.equals(index) for frame in candidates.values()):
        raise ValueError("os candidatos precisam cobrir os mesmos dias")
    excess = pd.DataFrame({name: frame["net"] - frame["rf"] for name, frame in candidates.items()})
    pieces, choices = [], []
    for start in range(lookback, len(index), step):
        past = excess.iloc[start - lookback:start]
        sharpe = (past.mean() / past.std(ddof=1) * np.sqrt(TRADING_DAYS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        best = sharpe.idxmax()
        end = min(start + step, len(index))
        pieces.append(candidates[best].iloc[start:end].assign(candidate=best))
        choices.append({"start": index[start], "candidate": best, "lookback_sharpe": float(sharpe[best]), "days": end - start})
    daily = pd.concat(pieces) if pieces else candidates[names[0]].iloc[:0].assign(candidate=pd.Series(dtype=object))
    return SelectionResult(daily, pd.DataFrame(choices, columns=CHOICE_COLUMNS))
```

- [ ] **Step 4: Rodar e ver passar** — mesmo comando → PASS.

- [ ] **Step 5: Mutação** — troque `past = excess.iloc[start - lookback:start]` por `past = excess.iloc[start - lookback + 1:start + 1]` e rode `-k never_looks` (deve falhar); desfaça e rode de novo (PASS).

- [ ] **Step 6: Commit** — `git commit -m "Walk-forward selection of run and threshold by past net Sharpe"`.

---

### Task 6: `Project`, `RunKey` com modelo e escopo, CLI com `sweep`

**Files:** `engine/project.py`, `main.py`, `tests/synthetic.py`, `tests/test_project.py`, `tests/test_main.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `RunKey(model, scope, window, retrain_every=1)`. Valida o modelo e o escopo (`ValueError`). `.name` = `"logistic-per_stock-w252-k21"`; `.label` = `"Logística regularizada · um modelo por ação · janela 252 · retreino a cada 21"`.
  - `RUN_SETTINGS = ("backtest_start", "per_stock_min_rows")`, com `info["params"]` = parâmetros do modelo do run.
  - `FEATURES_FORMAT = 4`.
  - `main.sweep(project, models, scopes, windows, retrains)`; `run --model --scope --window --retrain`, em que janela e k padrão dependem do escopo.

- [ ] **Step 1: Testes**

`tests/synthetic.py`:
- `from config.config import MODEL_PARAMS, Config`;
- `SMALL_LGBM` ganha `"min_child_share": 0.004, "min_child_floor": 20`;
- `RUNS = (RunKey("lightgbm", "pooled", 40), RunKey("logistic", "per_stock", 40, 3))`;
- em `synthetic_config`, troque `"lgbm_params": SMALL_LGBM` por:

```python
"model_params": {**copy.deepcopy(MODEL_PARAMS), "lightgbm": dict(SMALL_LGBM)}, "complete_history_days": 63, "per_stock_min_rows": 20, "selection_lookback_days": 40, "selection_step_days": 20,
```

  (e acrescente `import copy`).

`tests/test_project.py`:
- `test_labels_say_whether_the_stock_went_up` acrescenta a conferência do retorno intraday:

```python
def test_labels_say_whether_the_stock_went_up_during_the_next_day(synthetic_project):
    known = synthetic_project.load_run(RUNS[0]).predictions.dropna(subset=["label"])
    assert (known["label"] == (known["fwd_ret"] > 0)).all()
    prices = make_market()["prices"].set_index(["date", "ticker"])
    calendar = synthetic_project.features(with_dataset=False).calendar
    row = known.iloc[len(known) // 2]
    next_day = calendar[calendar.get_loc(row["date"]) + 1]
    bar = prices.loc[(next_day, row["ticker"])]
    assert row["fwd_ret"] == pytest.approx(bar["close"] / bar["open"] - 1)
```

  (substitui o antigo).
- Em `test_runs_are_saved_per_key_and_listed`:
  - `"lightgbm-w40-k3"` → `"logistic-per_stock-w40-k3"`;
  - `RunKey(21)` → `RunKey("lightgbm", "pooled", 21)`;
  - o rótulo esperado vira `"Logística regularizada · um modelo por ação · janela 40 · retreino a cada 3"`.
- Também:
  - `RunKey(100)` → `RunKey("lightgbm", "pooled", 100)`;
  - `RunKey(5)` → `RunKey("lightgbm", "pooled", 5)`;
  - `"lightgbm-w40-k1"` → `"lightgbm-pooled-w40-k1"`;
  - `"lightgbm-w21-k1"` → `"lightgbm-pooled-w21-k1"`.
- Troque `test_runs_made_with_other_training_settings_are_hidden` por:

```python
def test_changing_a_models_parameters_hides_only_its_runs(synthetic_project):
    params = copy.deepcopy(synthetic_project.config.model_params)
    params["logistic"]["C"] = 0.5
    assert Project(dataclasses.replace(synthetic_project.config, model_params=params)).runs() == [RUNS[0]]
    assert Project(dataclasses.replace(synthetic_project.config, per_stock_min_rows=5)).runs() == []
    assert synthetic_project.runs() == list(RUNS)


def test_an_unknown_model_or_scope_is_refused():
    with pytest.raises(ValueError, match="modelo"):
        RunKey("xgboost", "pooled", 21)
    with pytest.raises(ValueError, match="escopo"):
        RunKey("lightgbm", "sector", 21)
```

- Em `test_market_frame_aligns_benchmark_and_risk_free`, troque a linha de `bench_open` por `bench = prices[prices["ticker"] == BENCH].set_index("date")`. A conferência do benchmark vira `assert market.loc[t, "bench_fwd_ret"] == pytest.approx(bench["close"][calendar[101]] / bench["open"][calendar[101]] - 1)`, e a última linha vira `assert pd.isna(market["holding_date"].iloc[-1]) and pd.isna(market["bench_fwd_ret"].iloc[-1]) and pd.notna(market["bench_fwd_ret"].iloc[-2])`.
- Acrescente `import copy` no topo.

`tests/test_main.py`:
- o teste de opções positivas vira `@pytest.mark.parametrize("option", ["--window", "--retrain"])`, igual;
- acrescente:

```python
def test_run_options_choose_the_model_and_scope(capsys):
    with pytest.raises(SystemExit):
        main.main(["run", "--model", "xgboost"])
    assert "invalid choice" in capsys.readouterr().err


def test_sweep_runs_missing_combinations_and_skips_existing_ones(tmp_path, capsys):
    raw = make_market()
    project = Project(synthetic_config(tmp_path))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    main.sweep(project, ["naive_bayes"], ["pooled", "per_stock"], [40], [5])
    main.sweep(project, ["naive_bayes"], ["pooled"], [40, 400], [5])
    out = capsys.readouterr().out
    assert len(project.runs()) == 2
    assert out.count("já existe") == 1 and "máximo" in out
```

  - `test_report_of_a_run_without_any_realized_day_says_so` usa `RunKey("lightgbm", "pooled", 40)`.

- [ ] **Step 2: Rodar e ver falhar** — `venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_app.py` → FAIL (`RunKey` antigo).

- [ ] **Step 3: Implementar**

Em `engine/project.py`:
- imports:
  - `from engine.models import MODELS`;
  - `from engine.prices import PriceData, daily_total_return, forward_intraday_return`;
  - `from engine.walk_forward import SCOPES, WalkForward`;
- `FEATURE_SETTINGS` troca `"min_history_days"` por `"complete_history_days"`;
- `RUN_SETTINGS = ("backtest_start", "per_stock_min_rows")`;
- `FEATURES_FORMAT = 4`;
- `RunKey`:

```python
@dataclass(frozen=True, order=True)
class RunKey:
    """A run: which model, one model for all stocks or one per stock, the last `window` sessions of examples, and a
    refit every `retrain_every` sessions."""
    model: str
    scope: str
    window: int
    retrain_every: int = 1

    def __post_init__(self):
        if self.model not in MODELS:
            raise ValueError(f"modelo desconhecido: {self.model} (opções: {', '.join(MODELS)})")
        if self.scope not in SCOPES:
            raise ValueError(f"escopo desconhecido: {self.scope} (opções: {', '.join(SCOPES)})")

    @property
    def name(self) -> str:
        return f"{self.model}-{self.scope}-w{self.window}-k{self.retrain_every}"

    @property
    def label(self) -> str:
        return f"{MODELS[self.model][0]} · {SCOPES[self.scope]} · janela {self.window} · retreino a cada {self.retrain_every}"
```

- em `run`, o `WalkForward` vira `WalkForward(key.window, MODELS[key.model][1], c.model_params[key.model], key.retrain_every, key.scope, c.per_stock_min_rows, c.n_jobs)`;
- `info` ganha `"params": self._model_params(key.model)`;
- `runs()`:

```python
    def runs(self) -> list[RunKey]:
        """Runs made on the current features with the current settings and their model's current parameters."""
        quality = self._current_quality()
        if quality is None:
            return []
        settings = self._settings(RUN_SETTINGS)
        keys = []
        for path in self.runs_dir.glob("*/run_info.json"):
            info = json.loads(path.read_text())
            try:
                key = RunKey(**info["key"])
            except (KeyError, TypeError, ValueError):
                continue
            if (info.get("settings") == settings and info.get("features_built_at") == quality["built_at"]
                    and info.get("params") == self._model_params(key.model)):
                keys.append(key)
        return sorted(keys)
```

- `_model_params(self, model) -> dict`: `return json.loads(json.dumps(self.config.model_params[model]))`;
- em `_make_features`:
  - o `build_dataset` recebe `forward_intraday_return(panel.open, panel.close)` e `c.complete_history_days`;
  - `bench_fwd_ret` passa a ser `forward_intraday_return(prices.series(c.benchmark_ticker, "open", calendar), prices.series(c.benchmark_ticker, "close", calendar))`.

Em `main.py` (com `Edit` pontual):
- importe `from engine.models import MODELS` e `from engine.walk_forward import SCOPES`;
- crie `sweep`:

```python
def sweep(project: Project, models: list[str], scopes: list[str], windows: list[int], retrains: list[int]) -> None:
    """Runs every combination that does not exist yet, one after the other, printing one line per combination."""
    existing = set(project.runs())
    for key in [RunKey(m, s, w, k) for m in models for s in scopes for w in windows for k in retrains]:
        if key in existing:
            print(f"{key.label}: já existe")
            continue
        try:
            run = project.run(key)
        except ValueError as error:
            print(f"{key.label}: erro: {error}")
            continue
        summary = metrics.model_metrics(run.predictions)
        print(f"{key.label}: {run.info['runtime_minutes']} min · AUC {summary['auc']:.4f} · IC {summary['ic']:+.4f}", flush=True)
```

- no `run`:
  - `--model` com `choices=list(MODELS)` e `default=CONFIG.model`;
  - `--scope` com `choices=list(SCOPES)` e `default=CONFIG.scope`;
  - `--window` e `--retrain` com `default=None`;
  - a chave fica:

```python
            pooled = args.scope == "pooled"
            key = RunKey(args.model, args.scope,
                         args.window or (CONFIG.train_window_days if pooled else CONFIG.per_stock_window_days),
                         args.retrain or (CONFIG.retrain_every if pooled else CONFIG.per_stock_retrain_every))
```

- um subparser `sweep` com `--models` (`nargs="+"`, `choices`, padrão `[CONFIG.model]`), `--scopes` (idem, `[CONFIG.scope]`), `--windows` (`type=positive`, `[CONFIG.train_window_days]`) e `--retrain` (`type=positive`, `[CONFIG.retrain_every]`), que chama `sweep(...)`;
- a docstring da linha 1 menciona `sweep`.

- [ ] **Step 4: Rodar e ver passar** — `venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_app.py` → PASS.

- [ ] **Step 5: Mutação (lookahead ponta a ponta)** — em `_fit_block`, troque `group[0] - 1, side="right"` por `group[0], side="right"` e rode `tests/test_project.py -k future` (os dois casos devem falhar); desfaça e rode de novo (PASS).

- [ ] **Step 6: Commit** — `git commit -m "Run keys with model and scope, per-model fingerprints, intraday market frame, sweep command"`.

---

### Task 7: App

**Files:** `app/dashboard.py`, `tests/test_app.py`

- [ ] **Step 1: Testes** — em `tests/test_app.py`:
- `TABS = ["Visão geral", "Previsto vs realizado", "Carteira por dia", "Recortes de período", "Runs", "Walk-forward", "Modelo"]`;
- `open_app`:

```python
def open_app(project: Project, monkeypatch, key: RunKey = RUNS[0]) -> AppTest:
    """The app with the project's own config and the run key chosen in the sidebar."""
    monkeypatch.setattr("config.config.CONFIG", project.config)
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.selectbox(key="model").set_value(key.model)
    at.radio(key="scope").set_value(key.scope)
    at.run()
    at.number_input(key=f"window_{key.scope}").set_value(key.window)
    at.number_input(key=f"retrain_{key.scope}").set_value(key.retrain_every)
    return at.run()
```

- `RunKey(21)` → `RunKey("lightgbm", "pooled", 21)`; `RunKey(40)` → `RunKey("lightgbm", "pooled", 40)`;
- no teste da aba do dia, `at.selectbox[0]` → `at.selectbox(key="day")`;
- acrescente:

```python
def test_walk_forward_tab_shows_the_choices(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    choices = next(frame.value for frame in at.dataframe if "Limiar" in frame.value.columns)
    assert {"Início", "Run", "Limiar", "Sharpe no período anterior", "Dias"} <= set(choices.columns)
    assert set(choices["Run"]) <= {key.label for key in RUNS}
```

- [ ] **Step 2: Rodar e ver falhar** — `venv/bin/python -m pytest tests/test_app.py -q -p no:cacheprovider` → FAIL.

- [ ] **Step 3: Implementar** (a partir do `app/dashboard.py` atual)
- **Imports:** `from engine.models import MODELS`, `from engine.selection import walk_forward_selection`, `from engine.walk_forward import SCOPES`, `import numpy as np`.
- **Benchmark:** `BENCH = "S&P 500 TR (intraday)"`.
- **Barra lateral:**

```python
    st.header("Run")
    model = st.selectbox("Modelo", list(MODELS), index=list(MODELS).index(CONFIG.model),
                         format_func=lambda name: MODELS[name][0], key="model")
    scope = st.radio("Escopo", list(SCOPES), index=list(SCOPES).index(CONFIG.scope), format_func=SCOPES.get, key="scope")
    pooled = scope == "pooled"
    window = int(st.number_input("Janela de treino X: pregões de exemplos", min_value=1, max_value=1000, step=1,
                                 value=CONFIG.train_window_days if pooled else CONFIG.per_stock_window_days,
                                 key=f"window_{scope}"))
    retrain = int(st.number_input("Retreinar a cada k pregões (1 = todo dia)", min_value=1, max_value=252, step=1,
                                  value=CONFIG.retrain_every if pooled else CONFIG.per_stock_retrain_every,
                                  key=f"retrain_{scope}"))
    key = RunKey(model, scope, window, retrain)
```

- **Textos da carteira** (barra lateral e Visão geral): "Todo dia, compra na abertura as ações cuja probabilidade prevista de subir durante o dia passa do limiar, com peso proporcional à probabilidade, e vende no fechamento do mesmo dia; se nenhuma passar, fica em caixa rendendo a T-bill (^IRX)…".
- **Legenda da curva:** "Decisão após o fechamento de t, compra na abertura de t+1 e venda no fechamento de t+1 (retorno abertura → fechamento); o custo incide na compra e na venda de cada dia investido."
- **Legenda do modelo:** `f"{MODELS[model][0]} ({SCOPES[scope]}) retreinado a cada {retrain} pregão(ões) com os exemplos dos últimos {window} pregões; após cada fechamento, estima a probabilidade de cada ação do índice subir da abertura ao fechamento do dia seguinte · …"`.
- **Carteira por dia:** `selectbox` com `key="day"`; legenda "compra na abertura de {dia} e vende no fechamento do mesmo dia"; y da dispersão "Retorno da abertura ao fechamento".
- **Previsto vs realizado:** a barra mostra "Retorno médio da abertura ao fechamento do dia seguinte".
- **Nova aba "Walk-forward"** entre "Runs" e "Modelo", com a função em cache:

```python
@st.cache_data(show_spinner="Escolhendo runs e limiares…", max_entries=16)
def select(data_out: str, keys: tuple, stamps: tuple, features_stamp: int, thresholds: tuple, cost_bps: float,
           lookback: int, step: int):
    candidates, names = {}, {}
    for key, stamp in zip(keys, stamps):
        for threshold in thresholds:
            name = f"{key.name}@{threshold:.3f}"
            candidates[name] = backtest(data_out, key, stamp, features_stamp, threshold, cost_bps).daily
            names[name] = (key.label, threshold)
    return walk_forward_selection(candidates, lookback, step), names
```

  e o conteúdo da aba:

```python
with tab_walk:
    st.caption("A cada período, escolhe o run e o limiar com o maior Sharpe líquido no período anterior (só com dias já "
               "realizados) e segue essa escolha até a próxima. Nada é retreinado: os candidatos são runs já feitos, cujas "
               "previsões já são fora da amostra; o custo é o da barra lateral.")
    candidates = st.multiselect("Runs candidatos", runs, default=runs, format_func=lambda other: other.label, key="wf_runs")
    low, high = st.slider("Limiares candidatos", 0.40, 0.80, (0.50, 0.70), 0.025, key="wf_thresholds")
    thresholds = tuple(float(v) for v in np.round(np.arange(low, high + 1e-9, 0.025), 3))
    left, right = st.columns(2)
    lookback = int(left.number_input("Olhar para trás (pregões)", 5, 1000, CONFIG.selection_lookback_days, key="wf_lookback"))
    step = int(right.number_input("Reescolher a cada (pregões)", 1, 504, CONFIG.selection_step_days, key="wf_step"))
    if not candidates:
        st.info("Escolha ao menos um run.")
    else:
        selection, names = select(OUT, tuple(candidates), tuple(PROJECT.stamp(other) for other in candidates),
                                  features_stamp, thresholds, cost_bps, lookback, step)
        if selection.choices.empty:
            st.info("O período do backtest é curto demais para esse olhar para trás.")
        else:
            chosen = selection.daily
            st.dataframe(metrics_table(compare(chosen)))
            line_chart(curves(chosen, growth), "Capital (1,0 no início do walk-forward)")
            choices = selection.choices.assign(
                Run=lambda frame: frame["candidate"].map(lambda name: names[name][0]),
                Limiar=lambda frame: frame["candidate"].map(lambda name: names[name][1]),
            ).rename(columns={"start": "Início", "lookback_sharpe": "Sharpe no período anterior", "days": "Dias"})
            st.markdown("**Escolhas**")
            st.dataframe(choices[["Início", "Run", "Limiar", "Sharpe no período anterior", "Dias"]]
                         .style.format({"Limiar": "{:.3f}", "Sharpe no período anterior": "{:.2f}"}), hide_index=True)
            st.markdown("**Quanto tempo cada escolha ficou valendo**")
            st.dataframe(choices.groupby(["Run", "Limiar"])["Dias"].sum().sort_values(ascending=False).reset_index(),
                         hide_index=True)
```

- [ ] **Step 4: Rodar e ver passar** — `venv/bin/python -m pytest -q -p no:cacheprovider` → PASS (a suíte inteira).

- [ ] **Step 5: Commit** — `git commit -m "App: model and scope choice, intraday texts, walk-forward tab"`.

---

### Task 8: Documentação e simplificação

- [ ] **README:**
  - a operação intraday, com a linha do tempo `close t → open t+1 (buy) → close t+1 (sell)`, o rótulo, o custo de ida e volta e o fato noite × dia;
  - o universo completo em 252 pregões;
  - modelos e escopo;
  - `sweep`;
  - a aba Walk-forward e como ler as escolhas;
  - a tabela de parâmetros com modelo, escopo, X, k, limiar, custo, olhar para trás e passo;
  - "How to run" com `run --model … --scope …` e `sweep …`.
- **CLAUDE.md:**
  - linha de status da rodada 5;
  - invariantes: `fwd_ret` intraday, linhas t−X … t−1, escopo por ação só com as próprias linhas;
  - layout com `engine/models.py`, `engine/selection.py`, `RunKey(model, scope, window, retrain_every)` e o fingerprint por modelo;
  - comandos (`run --model --scope --window --retrain`, `sweep`).
- **code-simplifier** nos arquivos de `engine/`, `main.py` (notas intactas) e `app/dashboard.py`, sem mudar comportamento; depois, a suíte.
- **Commit:** `git commit -m "Docs for round 5; simplification pass"`.

---

### Task 9: Dados reais

- [ ] `venv/bin/python main.py features` (formato 4).
- [ ] **Varredura inicial em segundo plano, com log:**
  - `sweep --models lightgbm logistic naive_bayes --scopes pooled --windows 21 63 --retrain 5`;
  - `sweep --models logistic naive_bayes lightgbm --scopes per_stock --windows 252 --retrain 21` (se o máximo for menor que 252, usar o máximo informado pelo erro).
- [ ] Walk-forward de parâmetros nos dados reais, com todos os runs e limiares de 0,50 a 0,70: curva, métricas contra o S&P intraday, escolhas e estabilidade.
- [ ] Smoke test do app nos dados reais.
