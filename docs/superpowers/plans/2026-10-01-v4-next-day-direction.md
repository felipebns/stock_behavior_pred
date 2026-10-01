# Rodada 4 — volta ao "sobe ou desce" no dia seguinte — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Voltar ao rótulo "a ação sobe no dia seguinte" e à carteira diária. O horizonte h sai e o retreino a cada k fica. A tarefa inclui deixar o pipeline estável (janela com uma classe só, CLI e caches) e conferir nos dados reais que a rodada 2 é reproduzida.

**Architecture:** `engine/dataset.py` e `engine/portfolio.py` voltam às versões da rodada 2 (commit b428777). A única mudança é que a carteira passa a aceitar uma janela sem retorno realizado. `WalkForward` perde o `horizon` e ganha a previsão constante para janelas com uma classe só. `Project`/`RunKey` perdem o `horizon` e o painel de retornos. O app e o CLI se ajustam.

**Tech Stack:** Python 3.14 (venv), pandas 3, numpy 2, LightGBM, joblib, Streamlit + Altair, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-v4-next-day-direction-design.md`. Onde ela é omissa, valem as specs das rodadas 3 e 2.

## Global Constraints

- **Nunca rodar `git commit` nem `git push`.** Os commits ficam com o dono.
- Comandos via venv: `venv/bin/python -m pytest ...`, `venv/bin/python main.py ...`.
- **Convenção de tempo:**
  - a decisão é tomada após o fechamento de *t*, só com dados ≤ *t*;
  - `fwd_ret = (Open[t+2] + Div[t+2]) / Open[t+1] − 1` e `label = fwd_ret > 0`;
  - o treino usa as linhas *s* ∈ [*t*−X−1, *t*−2];
  - o retreino acontece a cada k decisões, e o ajuste fica na primeira decisão do grupo.
- yfinance com `auto_adjust=False`; o `Adj Close` nunca é usado.
- Paralelismo só no `WalkForward` (processos joblib, LightGBM `n_jobs=1`), bit a bit igual ao sequencial.
- **Defaults:** `train_window_days=21`, `retrain_every=1`, `threshold=0.55`, `cost_bps=5.0`. Os demais não mudam.
- Código o mais simples possível. Identificadores e docstrings em inglês, UI em português, README e CLAUDE.md em inglês.
- As notas do dono no fim do `main.py` (o bloco `"""` depois de `if __name__ == "__main__":`) ficam byte a byte; o `main.py` só se edita com `Edit` pontual.

## Review Focus

1. **Janela de treino com uma classe só** (X = 1 num dia de queda geral): a previsão tem de ser a frequência da classe, não a probabilidade invertida. *(Task 2: `test_a_window_with_a_single_class_predicts_that_class`)*
2. **Nenhum dia com retorno realizado** (início do backtest perto do fim dos dados): tabelas vazias, e o app mostra o aviso sem erro. *(Task 3: `test_no_realized_day_gives_an_empty_backtest`; Task 6: `test_run_without_any_realized_day_shows_a_warning`)*
3. **Opções inválidas no CLI** (`--window 0`, `--retrain 0`): mensagem do argparse, sem traceback. *(Task 5: `test_run_options_must_be_positive`)*
4. **Erro esperado no CLI** (janela grande demais): `erro: …` em uma linha. *(Task 5: `test_expected_errors_print_a_message_instead_of_a_traceback`)*
5. **Features salvas no formato da rodada 3** (sem `label`, com `returns.parquet`): são refeitas automaticamente. *(Task 5: `test_features_saved_in_an_older_format_are_rebuilt`)*

---

### Task 1: Config e dataset com o rótulo "subiu"

**Files:** `config/config.py`, `engine/dataset.py`, `tests/test_config.py`, `tests/test_dataset.py`

**Interfaces:**
- Produces: `build_dataset(ticker_features, date_features, membership, close, fwd_ret, min_history_days)`, com colunas `FEATURES + ["fwd_ret", "label"]` e `label = fwd_ret > 0` (versão b428777). `Config` sem `horizon_days`.

- [ ] **Step 1: Testes**

Em `tests/test_config.py`, troque `test_run_defaults` por:

```python
def test_run_defaults():
    assert (CONFIG.train_window_days, CONFIG.retrain_every) == (21, 1)
    assert not hasattr(CONFIG, "horizon_days")
```

Restaure o teste da rodada 2: `git show b428777:tests/test_dataset.py > tests/test_dataset.py`.

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_config.py tests/test_dataset.py -q -p no:cacheprovider`
Expected: FAIL. `build_dataset() takes 5 positional arguments but 6 were given` e `horizon_days` ainda existe.

- [ ] **Step 3: Implementar**

Em `config/config.py`, apague a linha `    horizon_days: int = 5`. Restaure `git show b428777:engine/dataset.py > engine/dataset.py`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_config.py tests/test_dataset.py -q -p no:cacheprovider`
Expected: PASS. A partir daqui o `test_project` fica vermelho até a Task 5.

---

### Task 2: `WalkForward` diário, retreino a cada k e janela com uma classe só

**Files:** `engine/walk_forward.py` (inteiro), `tests/test_walk_forward.py`

**Interfaces:**
- Produces: `WalkForward(train_window, params, retrain_every=1, min_child_share=0.004, min_child_floor=20, n_jobs=1, model_class=LGBMClassifier)`.
  - `run(dataset, features, calendar, start=None, progress=None)`: as decisões são todas as datas com linhas a partir do início, e `importance` tem uma linha por ajuste.
  - `MODELS = {"lightgbm": LGBMClassifier}`.

- [ ] **Step 1: Testes**

Em `tests/test_walk_forward.py`:
- apague `test_horizon_spaces_decisions_and_moves_the_training_rows_back` e `test_the_maximum_window_accounts_for_the_horizon`;
- troque `test_one_fit_serves_retrain_every_consecutive_decisions`, `test_each_probability_lands_on_its_own_row` e `test_parallel_run_is_identical_to_sequential` por:

```python
def test_one_fit_serves_retrain_every_consecutive_decisions():
    result = run_spy(spy_dataset(), window=3, retrain_every=3)
    assert [kind for kind, _, _ in Spy.log] == ["fit", "predict", "predict", "predict"] * 3
    assert [days for kind, days, _ in Spy.log if kind == "fit"] == [[2, 3, 4], [5, 6, 7], [8, 9, 10]]
    assert list(result.importance.index) == [DATES[6], DATES[9], DATES[12]]
    assert sorted(result.predictions["date"].unique()) == list(DATES[6:])


def test_each_probability_lands_on_its_own_row():
    dataset, dates = random_dataset(n_days=30, n_tickers=7)
    dataset["a"] = np.random.default_rng(1).uniform(size=len(dataset)).astype(np.float32)
    predictions = WalkForward(5, {}, retrain_every=3, n_jobs=2, model_class=Echo).run(dataset, ["a", "b"], dates).predictions
    rows = pd.MultiIndex.from_frame(predictions[["date", "ticker"]])
    np.testing.assert_array_equal(predictions["prob"].to_numpy(), dataset.loc[rows, "a"].to_numpy())


@pytest.mark.parametrize("retrain_every", [1, 2])
def test_parallel_run_is_identical_to_sequential(retrain_every):
    dataset, dates = random_dataset()
    params = {**LGBM_PARAMS, "n_estimators": 20}
    sequential = WalkForward(10, params, retrain_every=retrain_every, n_jobs=1).run(dataset, ["a", "b", "c"], dates)
    parallel = WalkForward(10, params, retrain_every=retrain_every, n_jobs=2).run(dataset, ["a", "b", "c"], dates)
    pd.testing.assert_frame_equal(sequential.predictions, parallel.predictions, check_exact=True)
    pd.testing.assert_frame_equal(sequential.importance, parallel.importance, check_exact=True)
    assert ((sequential.predictions["prob"] > 0) & (sequential.predictions["prob"] < 1)).all()


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_a_window_with_a_single_class_predicts_that_class(value):
    result = WalkForward(1, {**LGBM_PARAMS, "n_estimators": 5}).run(spy_dataset().assign(label=value), ["day", "noise"], DATES)
    assert (result.predictions["prob"] == value).all()
    assert (result.importance.to_numpy() == 0).all()


def test_a_start_with_no_room_for_any_window_says_so():
    with pytest.raises(ValueError, match="não há pregões"):
        run_spy(spy_dataset(), window=4, start=DATES[2])
```

(A classe `Echo` fica como está.)

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_walk_forward.py -q -p no:cacheprovider`
Expected: FAIL. O primeiro teste falha porque o `retrain_every` ainda passa como `horizon` por posição; o da classe única e o do início sem espaço falham pelas suas próprias asserções.

- [ ] **Step 3: Implementar**

Substitua `engine/walk_forward.py` por:

```python
"""Walk-forward: after each close, predict the next day with a model fit only on labels already known; decision
dates run in parallel."""
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
    """Row s's label needs the open of s+2, so a fit at decision t uses exactly the rows dated t-X-1 .. t-2; that model
    also predicts the next retrain_every-1 decisions, all after t."""

    def __init__(self, train_window: int, params: dict, retrain_every: int = 1, min_child_share: float = 0.004,
                 min_child_floor: int = 20, n_jobs: int = 1, model_class=LGBMClassifier):
        self.train_window = train_window
        self.params = params
        self.retrain_every = retrain_every
        self.min_child_share = min_child_share
        self.min_child_floor = min_child_floor
        self.n_jobs = n_jobs
        self.model_class = model_class

    def decisions(self, row_pos: np.ndarray, calendar: pd.DatetimeIndex, start=None) -> np.ndarray:
        first = row_pos[0] + self.train_window + 1
        begin = first if start is None else calendar.searchsorted(pd.Timestamp(start))
        if begin < first:
            room = begin - row_pos[0] - 1
            if room < 1:
                raise ValueError(f"não há pregões de dados antes de {pd.Timestamp(start).date()} para treinar")
            raise ValueError(f"a janela de {self.train_window} pregões precisa de {self.train_window + 1} pregões de dados "
                             f"antes de {pd.Timestamp(start).date()}; o máximo é {room}")
        decisions = np.unique(row_pos)
        decisions = decisions[decisions >= begin]
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
        jobs = (delayed(_fit_block)(X, y, row_pos, [fits[i] for i in block], self.train_window, self.params,
                                    self.min_child_share, self.min_child_floor, self.model_class) for block in blocks)
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


def _fit(X, y, params, min_child_share, min_child_floor, model_class):
    """P(up) for new rows and the normalized gains. A window with a single class (say X = 1 on a day every stock fell)
    predicts that class's frequency: a classifier fit on one class would put it in the wrong column."""
    if len(np.unique(y)) < 2:
        constant = float(y.mean()) if len(y) else 0.5
        return (lambda rows: np.full(len(rows), constant)), np.zeros(X.shape[1])
    model = model_class(**params, min_child_samples=max(min_child_floor, round(min_child_share * len(y))))
    model.fit(X, y)
    gain = np.asarray(model.feature_importances_, dtype=float)
    return (lambda rows: model.predict_proba(rows)[:, 1]), (gain / gain.sum() if gain.sum() > 0 else gain)


def _fit_block(X, y, row_pos, fits, window, params, min_child_share, min_child_floor, model_class):
    probs, gains = [], []
    for group in fits:
        lo = np.searchsorted(row_pos, group[0] - window - 1, side="left")
        hi = np.searchsorted(row_pos, group[0] - 2, side="right")
        known = ~np.isnan(y[lo:hi])
        predict, gain = _fit(X[lo:hi][known], y[lo:hi][known].astype(int), params, min_child_share, min_child_floor,
                             model_class)
        for t in group:
            first, last = np.searchsorted(row_pos, t, side="left"), np.searchsorted(row_pos, t, side="right")
            probs.append(predict(X[first:last]))
        gains.append(gain)
    return np.concatenate(probs), np.vstack(gains)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_walk_forward.py -q -p no:cacheprovider`
Expected: PASS.

---

### Task 3: `Portfolio` diário (rodada 2), janela vazia e simulador

**Files:** `engine/portfolio.py`, `tests/test_portfolio.py`, `tests/test_backtest_validation.py`

**Interfaces:**
- Produces: `Portfolio(threshold, cost_bps).backtest(predictions, market) -> BacktestResult`.
  - `daily`: `decision_date, gross, net, bench, rf, n_positions, invested, turnover, cost, n_missing`, indexado por `holding_date`.
  - `positions`: `decision_date, holding_date, ticker, prob, weight, fwd_ret, contribution`.
  - Uma janela sem nenhum retorno realizado do benchmark devolve as duas tabelas vazias, com essas colunas.

- [ ] **Step 1: Testes**

Restaure `git show b428777:tests/test_portfolio.py > tests/test_portfolio.py` e acrescente no fim:

```python
def test_no_realized_day_gives_an_empty_backtest(predictions, market):
    result = Portfolio(0.55).backtest(predictions, market.assign(bench_fwd_ret=np.nan))
    assert result.daily.empty and result.positions.empty
    assert list(result.positions.columns) == ["decision_date", "holding_date", "ticker", "prob", "weight", "fwd_ret", "contribution"]
```

Em `tests/test_backtest_validation.py`:
- troque `model_output` por:

```python
def model_output(opens: pd.DataFrame, dividends: pd.DataFrame, seed: int = 5) -> pd.DataFrame:
    """Random probabilities every session (with the realized next-day return); every 4th day nobody passes."""
    rng = np.random.default_rng(seed)
    fwd_ret = forward_open_return(opens, dividends)
    rows = []
    for k, date in enumerate(opens.index[2:-3]):
        high = 0.5 if k % 4 == 3 else 0.8
        rows += [{"date": date, "ticker": t, "prob": rng.uniform(0.3, high), "fwd_ret": fwd_ret.at[date, t]}
                 for t in opens.columns if not np.isnan(opens.at[date, t])]
    return pd.DataFrame(rows)
```

- e o teste final por:

```python
def test_portfolio_matches_an_event_by_event_simulation():
    opens, dividends, rf_daily = market_data()
    predictions = model_output(opens, dividends)
    daily = Portfolio(THRESHOLD, COST_BPS).backtest(predictions, market_frame(opens, rf_daily)).daily
    assert daily["invested"].any() and not daily["invested"].all()
    assert daily["n_missing"].sum() > 0 and (daily["turnover"] > 0).sum() > 3
    expected = CAPITAL * (1.0 + daily["net"]).cumprod()
    simulated = simulate(opens, dividends, rf_daily, predictions).reindex(daily.index)
    np.testing.assert_allclose(simulated.to_numpy(), expected.to_numpy(), rtol=1e-10)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_portfolio.py tests/test_backtest_validation.py -q -p no:cacheprovider`
Expected: FAIL. A assinatura atual exige `returns`.

- [ ] **Step 3: Implementar**

Restaure `git show b428777:engine/portfolio.py > engine/portfolio.py` e rode de novo. Só `test_no_realized_day_gives_an_empty_backtest` deve falhar, porque a janela vazia era a pendência 6 da rodada 2. Depois, em `backtest`, logo após o bloco `gaps`, acrescente:

```python
        if window.empty:
            return BacktestResult(daily=pd.DataFrame(columns=DAILY_COLUMNS), positions=pd.DataFrame(columns=POSITION_COLUMNS))
```

Acrescente também, ao lado de `POSITION_COLUMNS`:

```python
DAILY_COLUMNS = ["decision_date", "gross", "net", "bench", "rf", "n_positions", "invested", "turnover", "cost", "n_missing"]
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_portfolio.py tests/test_backtest_validation.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Mutação**

No `_turnover`, troque `drifted.shift(1)` por `weights.shift(1)` (sem deriva). O simulador tem que falhar. Desfaça a troca: PASS.

---

### Task 4: Decis com o retorno do dia seguinte

**Files:** `engine/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces: `deciles(predictions)` com colunas `prob, label, fwd_ret`, com médias primeiro dentro de cada data e depois entre datas. `decile_spread` não muda.

- [ ] **Step 1: Testes**

Em `tests/test_metrics.py`:
- em `ranked_predictions`, troque o `return` por `return frame.assign(label=(frame["fwd_ret"] > 0).astype(float))` e a docstring por `"""Two dates; prob rises with i; returns rise with i on the first date and fall on the second."""` (igual à atual);
- troque `test_deciles_average_within_each_date_then_across_dates` por:

```python
def test_deciles_average_within_each_date_then_across_dates():
    table = deciles(ranked_predictions())
    assert list(table.index) == list(range(1, 11))
    assert list(table.columns) == ["prob", "label", "fwd_ret"]
    assert table["prob"].tolist() == pytest.approx([(i + 0.5) / 10 for i in range(10)])
    assert table["label"].tolist() == pytest.approx([0.0] + [0.5] * 9)
    assert table["fwd_ret"].tolist() == pytest.approx([0.0] * 10)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_metrics.py -q -p no:cacheprovider`
Expected: FAIL (colunas `relative` em vez de `fwd_ret`).

- [ ] **Step 3: Implementar**

Em `engine/metrics.py`:
- em `_ranked`, apague a linha `relative=known["fwd_ret"] - by_date["fwd_ret"].transform("median"),` e troque a docstring por `"""Rows with a known outcome, with the probability rank and decile (1 = lowest) that day."""`;
- em `deciles`, troque `[["prob", "label", "relative"]]` por `[["prob", "label", "fwd_ret"]]` e a docstring por `"""Per probability decile (1 = lowest that day): mean prob, share that went up and mean next-day return — averaged within each date first, then across dates."""`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_metrics.py -q -p no:cacheprovider`
Expected: PASS.

---

### Task 5: `Project`, `RunKey` sem horizonte, CLI estável

**Files:** `engine/project.py`, `main.py`, `tests/synthetic.py`, `tests/test_project.py`, `tests/test_main.py`

**Interfaces:**
- Consumes: `build_dataset(..., fwd_ret, ...)` (Task 1), `WalkForward(..., retrain_every=...)` (Task 2), `Portfolio.backtest(predictions, market)` (Task 3).
- Produces:
  - `RunKey(window, retrain_every=1, model="lightgbm")`, com `.name` = `"lightgbm-w21-k1"` e `.label` = `"janela 21 · retreino a cada 1"`;
  - `FeatureSet(dataset, market, coverage, quality)` e `FEATURES_FORMAT = 3`;
  - `main.main(argv=None)`; `main.report(project, key)`.

- [ ] **Step 1: Testes**

Em `tests/synthetic.py`:
- `RUNS = (RunKey(40), RunKey(40, 3))`;
- no `synthetic_config`, apague `"horizon_days": 1, `.

Em `tests/test_project.py`:
- troque os testes parametrizados de lookahead e de membros: o corpo é o mesmo, só muda o parâmetro (`RUNS` novos);
- troque `test_decisions_start_at_backtest_start_and_follow_the_horizon` e `test_labels_are_relative_to_the_median_of_each_date` por:

```python
@pytest.mark.parametrize("key", RUNS)
def test_decisions_start_at_backtest_start_and_happen_every_session(synthetic_project, key):
    predictions = synthetic_project.load_run(key).predictions
    calendar = synthetic_project.features(with_dataset=False).calendar
    positions = calendar.get_indexer(sorted(predictions["date"].unique()))
    assert calendar[positions[0]] == pd.Timestamp(synthetic_project.config.backtest_start)
    assert set(np.diff(positions)) == {1}


def test_labels_say_whether_the_stock_went_up(synthetic_project):
    known = synthetic_project.load_run(RUNS[0]).predictions.dropna(subset=["label"])
    assert (known["label"] == (known["fwd_ret"] > 0)).all()
    assert 0.3 < known["label"].mean() < 0.7
```

Em `test_features_are_saved_with_every_column`:
- troque `assert list(features.dataset.columns) == FEATURES` por `assert list(features.dataset.columns) == FEATURES + ["fwd_ret", "label"]`;
- apague as duas linhas de `features.returns`.

Em `test_runs_are_saved_per_key_and_listed`, troque:
- `"lightgbm-h5-w40-k2"` por `"lightgbm-w40-k3"`;
- `RunKey(1, 21)` por `RunKey(21)`;
- a última linha por `assert RUNS[1].label == "janela 40 · retreino a cada 3"`.

Troque também:
- `RunKey(1, 100)` por `RunKey(100)`;
- `RunKey(1, 5)` por `RunKey(5)`;
- `"lightgbm-h1-w40-k1"` por `"lightgbm-w40-k1"`;
- `"lightgbm-h5-w21-k1"` por `"lightgbm-w21-k1"`.

Substitua `tests/test_main.py` por:

```python
import pytest
from synthetic import RUNS

import main


def test_report_prints_the_run_portfolio_benchmark_and_model(synthetic_project, capsys):
    main.report(synthetic_project, RUNS[1])
    out = capsys.readouterr().out
    assert RUNS[1].label in out
    assert "Carteira (líquida)" in out and "S&P 500 TR" in out and "sharpe" in out and "auc" in out


@pytest.mark.parametrize("option", ["--window", "--retrain"])
def test_run_options_must_be_positive(option, capsys):
    with pytest.raises(SystemExit):
        main.main(["run", option, "0"])
    assert "≥ 1" in capsys.readouterr().err


def test_expected_errors_print_a_message_instead_of_a_traceback(synthetic_project, monkeypatch):
    monkeypatch.setattr(main, "CONFIG", synthetic_project.config)
    with pytest.raises(SystemExit, match="erro: .*máximo"):
        main.main(["run", "--window", "500"])
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_project.py tests/test_main.py -q -p no:cacheprovider`
Expected: FAIL (erro de coleta: `RunKey(40)` ainda exige `window` como segundo campo).

- [ ] **Step 3: Implementar**

Em `engine/project.py`:
- troque o import `from engine.dataset import build_dataset, horizon_target` por `from engine.dataset import build_dataset`;
- `FEATURES_FORMAT = 3`;
- troque `RunKey` por:

```python
@dataclass(frozen=True, order=True)
class RunKey:
    """A run: train on the last `window` sessions of rows, refit every `retrain_every` sessions."""
    window: int
    retrain_every: int = 1
    model: str = "lightgbm"

    @property
    def name(self) -> str:
        return f"{self.model}-w{self.window}-k{self.retrain_every}"

    @property
    def label(self) -> str:
        return f"janela {self.window} · retreino a cada {self.retrain_every}"
```

- apague o campo `returns` de `FeatureSet`, a gravação de `returns.parquet` em `build_features` e a leitura dele em `features()`;
- em `run`, troque as linhas do `dataset = ...join(horizon_target(...))` e da construção do `WalkForward` por:

```python
        started = time.perf_counter()
        walk_forward = WalkForward(key.window, c.lgbm_params, retrain_every=key.retrain_every,
                                   min_child_share=c.min_child_share, min_child_floor=c.min_child_floor,
                                   n_jobs=c.n_jobs, model_class=MODELS[key.model])
        result = walk_forward.run(features.dataset, FEATURES, features.calendar, start=c.backtest_start, progress=progress)
```

- em `_make_features`, troque as duas linhas `returns = forward_open_return(...)` / `dataset = build_dataset(...)` por:

```python
        dataset = build_dataset(ticker_features, date_features, membership, panel.close,
                                forward_open_return(panel.open, panel.dividends), c.min_history_days)
```

- troque `return FeatureSet(dataset, market, returns, coverage, quality)` por `return FeatureSet(dataset, market, coverage, quality)`.

Em `main.py`, com `Edit` pontual:
- docstring da linha 1: `"""`python main.py download` baixa constituintes e preços; `features` calcula as features; `run --window X --retrain K` roda o walk-forward."""`;
- `report` usa `market = project.features(with_dataset=False).market` e `Portfolio(...).backtest(run.predictions, market).daily`, e o cabeçalho troca "giro médio por troca" por `f"giro médio diário {daily['turnover'].mean():.0%}"`;
- acrescente antes de `def main()`:

```python
def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"precisa ser um inteiro ≥ 1 (recebido: {text})")
    return value
```

- `def main(argv: list[str] | None = None) -> None:`;
- no subparser `run`, apague o `--horizon` e use `type=positive` em `--window` e `--retrain`, com a ajuda `"retreina a cada k pregões"`;
- `args = parser.parse_args(argv)`;
- envolva o corpo depois de `project = Project(CONFIG)` em `try:` … `except (ValueError, RuntimeError, OSError) as error: raise SystemExit(f"erro: {error}")`;
- no ramo `run`, use `key = RunKey(args.window, args.retrain)`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests -q -p no:cacheprovider --ignore=tests/test_app.py`
Expected: PASS (o `test_app` fica para a Task 6).

- [ ] **Step 5: Mutação (lookahead)**

Em `_fit_block`, troque `group[0] - 2` por `group[0] - 1` e rode `-k future`: os dois casos devem falhar. Desfaça a troca: PASS.

---

### Task 6: App

**Files:** `app/dashboard.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `RunKey(window, retrain_every)`, `Portfolio.backtest(predictions, market)`, `metrics.deciles` (colunas `prob, label, fwd_ret`).
- Produces:
  - barra lateral com `number_input` 0 = X e 1 = k;
  - abas `["Visão geral", "Previsto vs realizado", "Carteira por dia", "Recortes de período", "Runs", "Modelo"]`;
  - tabela de decis com `Prob. média`, `Subiu`, `Retorno médio`;
  - tabela do dia com `Subiu?`;
  - tabela de runs com `Giro médio diário`.

- [ ] **Step 1: Testes**

Em `tests/test_app.py`:
- `TABS = ["Visão geral", "Previsto vs realizado", "Carteira por dia", "Recortes de período", "Runs", "Modelo"]`;
- em `open_app`, troque a tupla por `(key.window, key.retrain_every)`;
- em `backtest`, troque `features = project.features(with_dataset=False)` e o `return` por `market = project.features(with_dataset=False).market` e `return Portfolio(0.55).backtest(predictions, market)`;
- `test_run_not_made_yet_offers_the_button` passa a usar `RunKey(21)`;
- em `test_signal_tab_shows_the_decile_table`, troque `"Bateu a mediana"` por `"Subiu"`;
- troque `test_decision_tab_shows_what_was_bought_and_what_happened` e `test_runs_tab_compares_every_run` e `test_run_without_any_known_outcome_still_renders` por:

```python
def test_day_tab_shows_what_was_bought_and_what_happened(synthetic_project, monkeypatch):
    key = RUNS[1]
    result = backtest(synthetic_project, key)
    days = list(result.daily.index)
    at = open_app(synthetic_project, monkeypatch, key)
    at.selectbox[0].set_value(days.index(result.positions["holding_date"].iloc[-1])).run()
    held = next(frame.value for frame in at.dataframe if "Subiu?" in frame.value.columns)
    assert {"Prob. prevista", "Retorno realizado", "Contribuição"} <= set(held.columns)
    assert any("Soma das contribuições" in info.value for info in at.info)


def test_runs_tab_compares_every_run(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    table = next(frame.value for frame in at.dataframe if "Giro médio diário" in frame.value.columns)
    assert list(table.index) == [key.label for key in RUNS]


def test_run_without_any_realized_day_shows_a_warning(tmp_path, monkeypatch):
    raw = make_market()
    dates = sorted(raw["prices"]["date"].unique())
    project = Project(synthetic_config(tmp_path, backtest_start=str(dates[-2].date())))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    project.run(RunKey(40))
    at = open_app(project, monkeypatch, RunKey(40))
    assert not at.exception
    assert any("Nenhum dia de carteira" in warning.value for warning in at.warning)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_app.py -q -p no:cacheprovider`
Expected: FAIL (o app ainda lê h e chama `backtest` com `returns`).

- [ ] **Step 3: Implementar**

Partindo do `app/dashboard.py` atual:
- **docstring:** "Escolha na barra lateral a janela de treino X e o retreino k: …".
- **Caches:** `max_entries=2` em `load_features`, `16` em `load_run`, `32` em `evaluate`, `256` em `evaluate_since`.
- **`backtest`:** usa `load_features(...).market` e `Portfolio(...).backtest(predictions, market)`.
- **Barra lateral:**
  - sai o `number_input` de h;
  - o de X continua;
  - o de k vira `"Retreinar a cada k pregões (1 = todo dia)"`;
  - `key = RunKey(window, retrain, model)`;
  - o limiar vira `"Limiar: probabilidade mínima de subir"`;
  - texto da carteira: "Todo dia, compra na abertura seguinte as ações cuja probabilidade prevista de subir passa do limiar, com peso proporcional à probabilidade, e vende na abertura do dia seguinte; se nenhuma passar, fica em caixa rendendo a T-bill (^IRX). Escolher limiar, X ou k olhando este resultado é otimizar dentro do próprio backtest (data snooping)."
- **Título:** `f"{key.label} · {len(daily)} dias de carteira · …"` (sem contagem de decisões).
- **Visão geral:**
  - legenda do modelo: `f"{model} retreinado a cada {retrain} pregão(ões) com os exemplos dos últimos {window} pregões; após cada fechamento, estima a probabilidade de cada ação do índice subir da abertura seguinte até a outra · em média …"`;
  - legenda da curva: "Decisão após o fechamento de t, compra na abertura de t+1 e rebalanceamento na abertura seguinte (retornos abertura → abertura, com dividendos). Arraste ou use a roda do mouse para aproximar."
- **Previsto vs realizado:**
  - textos com "subir/subiu" no lugar de "bater a mediana" e "dia" no lugar de "decisão";
  - barras em `fwd_ret`, com título "Retorno médio no dia seguinte" e cor por `alt.datum.fwd_ret > 0`;
  - calibração com y = `label` ("Fração que subiu");
  - tabela renomeando `{"prob": "Prob. média", "label": "Subiu", "fwd_ret": "Retorno médio"}`, com formatos `{:.3f}`, `{:.1%}`, `{:+.3%}`;
  - spread "Em cada dia, retorno médio das ações com as 10% maiores probabilidades menos o das 10% menores, somado ao longo do tempo. Sobe quando a ordenação acerta; não depende do limiar e não inclui custos."
- **Carteira por dia** (substitui "Carteira por decisão"):
  - `selectbox` "Dia de carteira" sobre `daily.index`, com rótulo `f"{data} · carteira {gross:+.2%} · S&P {bench:+.2%}"`;
  - 6 métricas: bruto, líquido, S&P, ações, `"Giro"` = `turnover:.0%`, `"Custo"` = `cost:.3%`;
  - legenda "Sinal calculado após o fechamento de {decisão}; compra na abertura de {dia} e rebalanceamento na abertura do pregão seguinte.";
  - tabela das posições do `holding_date`: `Ticker, Prob. prevista, Peso, Retorno realizado (fwd_ret), Subiu? (fwd_ret > 0 → "subiu"/"não subiu", NaN → "—"), Contribuição`;
  - `info` com a soma das contribuições e `warning` se `n_missing`;
  - dispersão das previsões do dia: x = prob; y = `fwd_ret` ("Retorno realizado no dia seguinte"); cor = `label` mapeado `{1.0: "subiu", 0.0: "não subiu"}`; linha do limiar;
  - "O resultado deste dia ainda não é conhecido." quando não há `fwd_ret`;
  - "Quase entraram" com `Ticker, Prob. prevista, Retorno realizado`.
- **Recortes:** igual.
- **Runs:** a coluna `"Giro médio por troca"` vira `"Giro médio diário": f"{other_daily['turnover'].mean():.0%}"`.
- **Modelo:**
  - IC com média móvel fixa de 63 dias ("IC diário (média móvel de 63 dias)", `min_periods=20`);
  - o resto igual.

- [ ] **Step 4: Rodar e ver passar (e a suíte inteira)**

Run: `venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: PASS.

---

### Task 7: Documentação, simplificação

**Files:** `README.md`, `.claude/CLAUDE.md`; code-simplifier nos arquivos tocados

- [ ] **Step 1: README**

Reescreva o primeiro parágrafo e a seção "How it works" para o dia seguinte:
- linha do tempo `close t → open t+1 (buy) → open t+2 (sell)`;
- rótulo "the stock's return from the open of t+1 to the open of t+2, dividends included, is above zero" e a probabilidade P(up);
- nota de que, num dia típico, 52% das ações sobem, mas essa fração varia ±22 pontos com o mercado; por isso o rótulo mistura acertar o mercado e escolher ações, e a aba "Previsto vs realizado" mostra qual dos dois o modelo faz;
- uma linha de que a rodada 3 (rótulo relativo e horizonte h) está no commit 9512cfd;
- treino nas linhas t−X−1 … t−2, com retreino a cada k pregões;
- features iguais;
- carteira diária e custos;
- avaliação com "subiu" no lugar de "bateu a mediana";
- tabela de parâmetros só com X, k, limiar e custo;
- em "Data and guarantees": preços sem ajuste de dividendos (`auto_adjust=False`, dividendo somado no dia ex) e ajustados por split (senão um split 4:1 pareceria −75%); cobertura de 86% (2018) a 99% (2026), com os que faltam listados no app;
- "How to run" com `run --window 21 --retrain 1`.

- [ ] **Step 2: CLAUDE.md**

- Status: acrescente a linha da rodada 4 (spec 2026-10-01).
- Comando do venv: `run --window X --retrain K`.
- Invariantes de lookahead: decisão a cada pregão; compra na abertura de t+1 e venda na de t+2; `label = fwd_ret > 0`; linhas de treino t−X−1 … t−2; refit a cada k.
- Layout:
  - dataset (features + `fwd_ret`/`label`);
  - walk_forward (decisão diária, refit a cada k, janela com classe única);
  - portfolio (diário);
  - project (`RunKey(window, retrain_every)`, `runs/lightgbm-w21-k1`, `FEATURES_FORMAT`).

- [ ] **Step 3: code-simplifier**

Rode o agente `code-simplifier:code-simplifier` em `engine/walk_forward.py`, `engine/portfolio.py`, `engine/metrics.py`, `engine/project.py`, `main.py` (notas intactas) e `app/dashboard.py`, sem mudar comportamento. Depois, a suíte.

Run: `venv/bin/python -m pytest -q -p no:cacheprovider`
Expected: PASS.

---

### Task 8: Dados reais e regressão

- [ ] **Step 1:** `venv/bin/python main.py features`, depois `run --window 21 --retrain 1` e `run --window 21 --retrain 5`, em segundo plano e com log.
- [ ] **Step 2: Regressão.** O run (21, 1) tem que dar 803.587 previsões, bruto +51,1%, líquido −45,6%, AUC 0,4898 e IC 0,0014, igual à rodada 2 após a correção PARA/BBT.
- [ ] **Step 3: Smoke do app nos dados reais.** Sem exceção, seis abas, os dois runs listados e a tabela de decis presente.
