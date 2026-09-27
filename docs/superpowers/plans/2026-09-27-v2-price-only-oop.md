# Rodada 2 — só preços, classes simples, janela variável — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tirar todo dado macro, baixar constituintes e preços mais recentes, reorganizar o código em classes simples e permitir escolher a janela de treino (CLI e app) com walk-forward paralelo, mantendo a convenção de tempo e os testes de lookahead.

**Architecture:** Cada unidade com estado vira uma classe:
- `Universe`, `PriceData`, `FeatureBuilder`, `WalkForward` e `Portfolio`;
- `Project` é a fachada usada pelo CLI e pelo app (download, features em cache, um run por janela).

As métricas continuam funções. As features são calculadas uma vez e salvas em `data/out/features/`. Cada janela vira `data/out/runs/wNNN/`. O walk-forward paraleliza as datas de decisão com `joblib`.

**Tech Stack:** Python 3.14 (venv), pandas 3, numpy 2, pyarrow, scikit-learn, LightGBM, joblib, yfinance, Streamlit + Altair, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-v2-price-only-oop-design.md` (o que ele não muda continua valendo de `docs/superpowers/specs/2026-09-26-ml-backtest-redesign-design.md`).

## Global Constraints

- **Nunca rodar `git commit` nem `git push`** (bloqueado); o usuário faz os commits. Este plano não tem passos de commit.
- Comandos pelo venv: `venv/bin/python -m pytest ...`, `venv/bin/python main.py ...`.
- A convenção de tempo não muda:
  - decisão após o fechamento de *t*, só com dados ≤ *t*;
  - execução da abertura de *t+1* à abertura de *t+2*;
  - `fwd_ret = (Open[t+2] + Div[t+2]) / Open[t+1] − 1`, com `label = fwd_ret > 0`;
  - treino em *s* ∈ [*t* − X − 1, *t* − 2];
  - saída sempre de 1 dia.
- Universo em *t* = membros do snapshot mais recente ≤ *t*. Aliases só os 21 da lista do spec (renomeações puras e conhecidas). Membro sem preço é descartado e reportado.
- yfinance com `auto_adjust=False, actions=True`; o `Adj Close` nunca é gravado.
- Sem dado macro (Fed Funds, GPR). O `^IRX` só serve de taxa do caixa e do Sharpe. O `^SP500TR` é o benchmark e a fonte das features de mercado.
- Paralelização só no `WalkForward`: processos `joblib` e LightGBM com `n_jobs=1`. O resultado tem que ser idêntico bit a bit ao sequencial.
- Defaults:
  - `price_start="2018-01-02"`, `backtest_start="2020-01-02"`;
  - `train_window_days=21`, `threshold=0.55`, `cost_bps=5.0`;
  - `min_history_days=63`, `peer_count=10`, `peer_lookback_days=63`;
  - `min_child_share=0.004`, `min_child_floor=20`;
  - `risk_free_max_staleness_days=5`, `n_jobs=-1`.
- Métricas:
  - carteira vs S&P: `total_return`, `volatility`, `sharpe`, `max_drawdown`, `hit_ratio`;
  - modelo: `auc`, `accuracy`, `ic`.
- Código o mais simples possível. Sem comentários, a não ser quando o porquê não for óbvio. Identificadores em inglês, UI do app em português, README e CLAUDE.md em inglês.
- As notas do usuário no fim do `main.py` (o bloco `"""` que começa em `Reorganizar em classes`) são preservadas.

## Review Focus

1. **O download dos constituintes volta HTML ou erro** (rate limit, URL mudou) → o CSV antigo é mantido e o erro é claro. *(Task 2: `test_download_rejects_non_table_and_keeps_old_file`)*
2. **O Yahoo não devolve `^SP500TR` ou `^IRX`** → os preços antigos são mantidos e o erro é claro. *(Task 8: `test_download_keeps_old_prices_when_an_index_is_missing`)*
3. **Janela maior que o histórico antes de `backtest_start`** → o erro informa o máximo, no CLI e no app. *(Task 5: `test_window_too_long_for_the_start_date_is_refused`; Task 8: `test_window_too_long_is_refused`)*
4. **App aberto enquanto um run é refeito** → mostra o resultado novo, não o cache velho. *(Task 10: `test_dashboard_shows_new_results_after_a_rerun`)*
5. **Download ou recálculo das features com runs antigos no disco** → os runs antigos são apagados, para nunca misturar dados. *(Task 8: `test_rebuilding_features_deletes_old_runs`, `test_download_saves_inputs_and_clears_derived_results`)*

---

## File Structure

```
config/config.py        Config, TICKER_ALIASES (21), LGBM_PARAMS, CONSTITUENTS_URL, CONFIG
engine/universe.py      class Universe
engine/prices.py        class PriceData, PricePanel, clean_prices, daily_total_return, forward_open_return
engine/features.py      class FeatureBuilder + listas de features
engine/dataset.py       build_dataset (sem mudança)
engine/walk_forward.py  class WalkForward, WalkForwardResult
engine/portfolio.py     class Portfolio, BacktestResult
engine/metrics.py       performance_metrics, model_metrics, drawdown, daily_ic, monthly_auc
engine/project.py       class Project, FeatureSet, RunResults
main.py                 CLI download | features | run --window N
app/dashboard.py        Streamlit
tests/synthetic.py      mercado sintético, perturb_after, yahoo_response, synthetic_config, SMALL_LGBM
tests/conftest.py       fixtures synthetic_panel e synthetic_project
```

**Saem:**

- `engine/data/` (universe, prices, macro);
- `engine/model.py`, `engine/pipeline.py`, `engine/results.py`;
- `tests/test_macro.py`, `tests/test_pipeline.py`, `tests/test_results.py`;
- `data/in/ff_futures_daily.csv`, `data/in/data_gpr_daily_recent.xls`, `data/in/sp500_data.csv` e `data/out/` (o formato antigo).

---

### Task 1: Config v2 e remoção do macro

**Files:**
- Modify: `config/config.py`, `engine/features.py:1-20`, `tests/test_config.py`, `tests/test_features.py`, `tests/synthetic.py`, `requirements.txt`, `pyproject.toml`
- Delete: `engine/data/macro.py`, `engine/pipeline.py`, `engine/results.py`, `tests/test_macro.py`, `tests/test_pipeline.py`, `tests/test_results.py`, `tests/test_main.py`, `tests/test_app.py`, `data/in/ff_futures_daily.csv`, `data/in/data_gpr_daily_recent.xls`, `data/in/sp500_data.csv`, `data/out/`

**Interfaces:**
- Produces: `Config` com os campos da seção Global Constraints e mais `data_in`, `data_out`, `constituents_url`, `constituents_file`, `prices_file`, `benchmark_ticker`, `risk_free_ticker` e `lgbm_params`; também `TICKER_ALIASES`, `LGBM_PARAMS` (sem `min_child_samples`, com `n_jobs: 1`), `CONSTITUENTS_URL` e `CONFIG`. As listas de features ficam sem macro: `DATE_FEATURES == MARKET_FEATURES`.
- Note: `main.py` e `app/dashboard.py` ficam quebrados até as Tasks 9 e 10 (não há teste para eles até lá).

- [ ] **Step 1: Escrever os testes do config**

`tests/test_config.py`:

```python
import dataclasses

import pandas as pd
import pytest

from config.config import CONFIG, LGBM_PARAMS, TICKER_ALIASES


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.threshold = 0.6


def test_aliases_point_to_final_tickers():
    assert not set(TICKER_ALIASES.values()) & set(TICKER_ALIASES)


def test_aliases_are_only_well_known_pure_renames():
    assert TICKER_ALIASES["FB"] == "META" and TICKER_ALIASES["ANTM"] == "ELV"
    assert len(TICKER_ALIASES) == 21
    assert not {"DISCA", "MYL", "CBS", "VIAC", "PARA", "FI", "MMC", "BK", "SATS"} & set(TICKER_ALIASES)


def test_backtest_period_and_model_defaults():
    assert pd.Timestamp(CONFIG.price_start) < pd.Timestamp(CONFIG.backtest_start)
    assert CONFIG.benchmark_ticker == "^SP500TR" and CONFIG.risk_free_ticker == "^IRX"
    assert "min_child_samples" not in LGBM_PARAMS and LGBM_PARAMS["n_jobs"] == 1
```

Em `tests/test_features.py`, trocar o corpo de `test_feature_lists` por:

```python
def test_feature_lists():
    assert len(FEATURES) == 30 and len(set(FEATURES)) == 30
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_config.py tests/test_features.py -q`
Expected: FAIL — `test_aliases_are_only_well_known_pure_renames`, `test_backtest_period_and_model_defaults` (`AttributeError: 'Config' object has no attribute 'backtest_start'`) e `test_feature_lists` (36 ≠ 30).

- [ ] **Step 3: Implementar**

`config/config.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path

# Pure renames only (same company, new name/ticker), publicly known, and a same-day 1:1 swap in the constituents file.
TICKER_ALIASES: dict[str, str] = {
    "FB": "META", "ANTM": "ELV", "ABC": "COR", "FLT": "CPAY", "PKI": "RVTY", "RE": "EG",
    "BLL": "BALL", "WLTW": "WTW", "LB": "BBWI", "SYMC": "GEN", "NLOK": "GEN", "HCP": "DOC",
    "PEAK": "DOC", "JEC": "J", "TMK": "GL", "BHGE": "BKR", "CTL": "LUMN", "HRS": "LHX",
    "UTX": "RTX", "DWDP": "DD", "ARNC": "HWM",
}

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)

LGBM_PARAMS: dict = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "importance_type": "gain",
    "random_state": 42,
    "deterministic": True,
    "force_col_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}


@dataclass(frozen=True)
class Config:
    data_in: Path = Path("data/in")
    data_out: Path = Path("data/out")
    constituents_url: str = CONSTITUENTS_URL
    constituents_file: str = "sp500_historical_constituents.csv"
    prices_file: str = "prices.parquet"
    price_start: str = "2018-01-02"
    backtest_start: str = "2020-01-02"
    benchmark_ticker: str = "^SP500TR"
    risk_free_ticker: str = "^IRX"
    train_window_days: int = 21
    threshold: float = 0.55
    cost_bps: float = 5.0
    min_history_days: int = 63
    peer_count: int = 10
    peer_lookback_days: int = 63
    min_child_share: float = 0.004
    min_child_floor: int = 20
    risk_free_max_staleness_days: int = 5
    n_jobs: int = -1
    lgbm_params: dict = field(default_factory=lambda: dict(LGBM_PARAMS))


CONFIG = Config()
```

Em `engine/features.py`:
- apagar a linha `from engine.data.macro import FF_FEATURES, GPR_FEATURES`;
- apagar a linha `MACRO_FEATURES = FF_FEATURES + GPR_FEATURES`;
- trocar `DATE_FEATURES = MARKET_FEATURES + MACRO_FEATURES` por `DATE_FEATURES = MARKET_FEATURES`.

Em `tests/synthetic.py`:
- `make_market` deixa de criar `ff_bars` e `gpr`: apagar os dois blocos e devolver `{"prices": pd.concat(frames, ignore_index=True), "snapshots": snapshots}`;
- `perturb_after` deixa de mexer neles: apagar os blocos `ff_bars` e `gpr` e devolver `{"prices": pd.concat([prices, newcomer], ignore_index=True), "snapshots": snapshots}`.

`requirements.txt`: trocar a linha `xlrd>=2.0` por `joblib>=1.4`. `pyproject.toml`: trocar `"xlrd>=2.0",` por `"joblib>=1.4",`.

Apagar o macro, a orquestração antiga, os testes que dependiam dela e os dados sem uso:

```bash
rm engine/data/macro.py engine/pipeline.py engine/results.py tests/test_macro.py tests/test_pipeline.py tests/test_results.py tests/test_main.py tests/test_app.py
rm data/in/ff_futures_daily.csv data/in/data_gpr_daily_recent.xls data/in/sp500_data.csv
rm -rf data/out
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 55 passed (config 4, universe 7, prices 11, features 7, dataset 3, walk_forward 6, portfolio 8, metrics 9).

---

### Task 2: `Universe`

**Files:**
- Create: `engine/universe.py`
- Modify: `tests/test_universe.py` (reescrito), `tests/conftest.py`
- Delete: `engine/data/universe.py`

**Interfaces:**
- Produces:
  - `to_yahoo(ticker) -> str`;
  - `Universe(snapshots)`, com `snapshots` nas colunas `date` e `tickers` (`list[str]`);
  - `Universe.from_frame(raw, aliases)` e `Universe.from_csv(path, aliases)`;
  - `Universe.download(url, path, fetch=None)`;
  - `.members_on(date) -> list[str]`, `.tickers_since(start) -> list[str]`, `.membership(dates, tickers) -> DataFrame[bool]`, `.sizes(dates) -> ndarray`.

- [ ] **Step 1: Escrever os testes**

`tests/test_universe.py`:

```python
import pandas as pd
import pytest

from engine.universe import Universe, to_yahoo

RAW = pd.DataFrame({
    "date": ["2020-01-02", "2020-02-03", "2020-03-02"],
    "tickers": ["AAA,BRK.B,OLD", "AAA,BRK.B,NEW", "BRK.B,NEW,ZZZ"],
})
ALIASES = {"OLD": "NEW"}


@pytest.fixture
def universe():
    return Universe.from_frame(RAW, ALIASES)


def test_to_yahoo_replaces_dots():
    assert to_yahoo("BRK.B") == "BRK-B"
    assert to_yahoo(" BF.B ") == "BF-B"


def test_from_frame_applies_yahoo_format_and_aliases(universe):
    assert universe.snapshots["tickers"].iloc[0] == ["AAA", "BRK-B", "NEW"]
    assert universe.snapshots["tickers"].iloc[1] == ["AAA", "BRK-B", "NEW"]


def test_from_frame_drops_duplicates_created_by_aliases():
    raw = pd.DataFrame({"date": ["2020-01-02"], "tickers": ["OLD,NEW,AAA"]})
    assert Universe.from_frame(raw, ALIASES).snapshots["tickers"].iloc[0] == ["NEW", "AAA"]


def test_members_on_uses_last_snapshot_not_after_date(universe):
    assert universe.members_on("2020-01-01") == []
    assert universe.members_on("2020-01-02") == ["AAA", "BRK-B", "NEW"]
    assert universe.members_on("2020-02-28") == ["AAA", "BRK-B", "NEW"]
    assert universe.members_on("2020-03-02") == ["BRK-B", "NEW", "ZZZ"]


def test_tickers_since_includes_snapshot_in_force_at_start(universe):
    assert universe.tickers_since("2020-02-15") == ["AAA", "BRK-B", "NEW", "ZZZ"]
    assert universe.tickers_since("2020-03-10") == ["BRK-B", "NEW", "ZZZ"]


def test_membership_is_point_in_time(universe):
    dates = pd.DatetimeIndex(pd.to_datetime(["2019-12-31", "2020-01-02", "2020-02-28", "2020-03-02", "2020-03-03"]))
    matrix = universe.membership(dates, ["AAA", "NEW", "ZZZ", "NOPE"])
    assert not matrix.loc["2019-12-31"].any()
    assert matrix.loc["2020-02-28"].tolist() == [True, True, False, False]
    assert matrix.loc["2020-03-02"].tolist() == [False, True, True, False]
    assert matrix.loc["2020-03-03", "ZZZ"]


def test_sizes_count_members_per_date(universe):
    dates = pd.DatetimeIndex(pd.to_datetime(["2019-12-31", "2020-01-02", "2020-03-02"]))
    assert universe.sizes(dates).tolist() == [0, 3, 3]


def test_from_csv_reads_the_snapshots(tmp_path):
    path = tmp_path / "constituents.csv"
    RAW.to_csv(path, index=False)
    loaded = Universe.from_csv(path, ALIASES)
    assert loaded.snapshots["date"].iloc[0] == pd.Timestamp("2020-01-02")
    assert loaded.snapshots["tickers"].iloc[2] == ["BRK-B", "NEW", "ZZZ"]


def test_download_saves_the_snapshots_table(tmp_path):
    path = tmp_path / "in" / "constituents.csv"
    Universe.download("url", path, fetch=lambda url: RAW.to_csv(index=False).encode())
    assert Universe.from_csv(path, ALIASES).members_on("2020-03-02") == ["BRK-B", "NEW", "ZZZ"]


def test_download_rejects_non_table_and_keeps_old_file(tmp_path):
    path = tmp_path / "constituents.csv"
    path.write_text("date,tickers\n2020-01-02,AAA\n")
    with pytest.raises(ValueError):
        Universe.download("url", path, fetch=lambda url: b"<html>rate limited</html>\n<body></body>\n")
    assert path.read_text().startswith("date,tickers")
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_universe.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.universe'`.

- [ ] **Step 3: Implementar**

`engine/universe.py`:

```python
"""S&P 500 membership as of each date: the last snapshot dated on or before it."""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


def to_yahoo(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


class Universe:
    """Snapshots `date` (sorted) → `tickers` (Yahoo format): the index composition from that date on."""

    def __init__(self, snapshots: pd.DataFrame):
        self.snapshots = snapshots

    @classmethod
    def from_frame(cls, raw: pd.DataFrame, aliases: dict[str, str]) -> "Universe":
        snapshots = raw.assign(date=pd.to_datetime(raw["date"])).sort_values("date").reset_index(drop=True)

        def normalize(field: str) -> list[str]:
            yahoo = (to_yahoo(t) for t in field.split(",") if t.strip())
            return list(dict.fromkeys(aliases.get(t, t) for t in yahoo))

        return cls(pd.DataFrame({"date": snapshots["date"], "tickers": snapshots["tickers"].map(normalize)}))

    @classmethod
    def from_csv(cls, path: Path, aliases: dict[str, str]) -> "Universe":
        return cls.from_frame(pd.read_csv(path), aliases)

    @staticmethod
    def download(url: str, path: Path, fetch=None) -> None:
        """Replaces the local CSV only when what came back really is the snapshots table."""
        content = (fetch or _fetch)(url)
        if list(pd.read_csv(io.BytesIO(content), nrows=1).columns) != ["date", "tickers"]:
            raise ValueError(f"o arquivo baixado de {url} não é a tabela de constituintes (date, tickers)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def members_on(self, date) -> list[str]:
        pos = self.snapshots["date"].searchsorted(pd.Timestamp(date), side="right") - 1
        return [] if pos < 0 else list(self.snapshots["tickers"].iloc[pos])

    def tickers_since(self, start) -> list[str]:
        first = max(self.snapshots["date"].searchsorted(pd.Timestamp(start), side="right") - 1, 0)
        return sorted(set().union(*self.snapshots["tickers"].iloc[first:]))

    def membership(self, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
        column = {ticker: j for j, ticker in enumerate(tickers)}
        table = np.zeros((len(self.snapshots), len(tickers)), dtype=bool)
        for i, members in enumerate(self.snapshots["tickers"]):
            table[i, [column[t] for t in members if t in column]] = True
        pos = self._positions(dates)
        values = table[np.clip(pos, 0, None)]
        values[pos < 0] = False
        return pd.DataFrame(values, index=dates, columns=tickers)

    def sizes(self, dates: pd.DatetimeIndex) -> np.ndarray:
        pos = self._positions(dates)
        counts = self.snapshots["tickers"].map(len).to_numpy()
        return np.where(pos >= 0, counts[np.clip(pos, 0, None)], 0)

    def _positions(self, dates) -> np.ndarray:
        return self.snapshots["date"].searchsorted(dates, side="right") - 1
```

`tests/conftest.py`:
- trocar `from engine.data import universe` por `from engine.universe import Universe`;
- trocar `membership = universe.membership_matrix(raw["snapshots"], calendar, tickers)` por `membership = Universe(raw["snapshots"]).membership(calendar, tickers)`.

Apagar o módulo antigo: `rm engine/data/universe.py`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 58 passed (universe 10).

---

### Task 3: `PriceData`

**Files:**
- Create: `engine/prices.py`
- Modify: `tests/test_prices.py` (reescrito), `tests/conftest.py`, `tests/test_features.py:6`, `engine/features.py:5`
- Delete: `engine/data/` (pasta inteira)

**Interfaces:**
- Produces:
  - `PRICE_FIELDS`, `PANEL_FIELDS` e `PricePanel` (dataclass com os campos open, high, low, close, volume e dividends);
  - `PriceData(prices)`, com `PriceData.download(tickers, start, end=None, batch_size=100, downloader=None) -> tuple[PriceData, list[str]]` e `PriceData.load(path)`;
  - `.save(path)`, `.tickers -> set[str]`, `.calendar(benchmark, start=None) -> DatetimeIndex`, `.panel(calendar, tickers) -> PricePanel`, `.series(ticker, field, calendar) -> Series`;
  - as funções `clean_prices`, `daily_total_return(close, dividends)` e `forward_open_return(open_, dividends=None)`.

- [ ] **Step 1: Escrever os testes**

`tests/test_prices.py`:

```python
import numpy as np
import pandas as pd
import pytest

from engine.prices import PRICE_FIELDS, PriceData, clean_prices, daily_total_return, forward_open_return

DATES = ["2024-01-02", "2024-01-03", "2024-01-04"]


def yahoo_frame(data: dict[str, dict[str, list]]) -> pd.DataFrame:
    """Same shape as yf.download(group_by='ticker'): columns MultiIndex (Ticker, Price)."""
    frames = {t: pd.DataFrame(fields, index=pd.DatetimeIndex(pd.to_datetime(DATES), name="Date")) for t, fields in data.items()}
    raw = pd.concat(frames, axis=1)
    raw.columns = raw.columns.set_names(["Ticker", "Price"])
    return raw


def bars(close=(10.0, 11.0, 12.0)):
    close = list(close)
    return {
        "Open": close, "High": close, "Low": close, "Close": close,
        "Adj Close": [c * 0.9 for c in close], "Volume": [1000.0] * 3,
        "Dividends": [0.0] * 3, "Stock Splits": [0.0] * 3,
    }


def test_download_drops_adj_close_and_renames_fields():
    def fake(tickers, **kwargs):
        assert kwargs["auto_adjust"] is False and kwargs["actions"] is True
        return yahoo_frame({t: bars() for t in tickers})

    data, failed = PriceData.download(["AAA", "BBB"], "2024-01-01", downloader=fake)
    assert list(data.prices.columns) == ["date", "ticker", *PRICE_FIELDS]
    assert failed == []
    assert len(data.prices) == 6
    assert data.tickers == {"AAA", "BBB"}


def test_download_reports_tickers_without_data_and_survives_empty_batches():
    def fake(tickers, **kwargs):
        return pd.DataFrame() if "GONE" in tickers else yahoo_frame({t: bars() for t in tickers})

    data, failed = PriceData.download(["AAA", "GONE"], "2024-01-01", batch_size=1, downloader=fake)
    assert failed == ["GONE"]
    assert data.tickers == {"AAA"}


def test_download_reports_all_nan_tickers_as_failed():
    def fake(tickers, **kwargs):
        return yahoo_frame({"AAA": bars(), "NAN": {k: [np.nan] * 3 for k in bars()}})

    data, failed = PriceData.download(["AAA", "NAN"], "2024-01-01", downloader=fake)
    assert failed == ["NAN"]
    assert data.tickers == {"AAA"}


def test_download_fails_loudly_when_nothing_comes_back():
    with pytest.raises(RuntimeError):
        PriceData.download(["AAA"], "2024-01-01", downloader=lambda tickers, **kwargs: pd.DataFrame())


def test_clean_prices_fixes_bad_values_and_duplicates():
    raw = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"],
        "ticker": ["AAA"] * 4,
        "open": [10.0, 10.0, 0.0, 11.0], "high": 11.0, "low": 9.0,
        "close": [10.0, 10.5, 10.2, np.nan], "volume": 100.0,
        "dividends": [np.nan, np.nan, 0.5, 0.0], "splits": [np.nan, 0.0, 0.0, 0.0],
    })
    out = clean_prices(raw)
    assert len(out) == 2
    assert out.loc[0, "close"] == 10.5
    assert np.isnan(out.loc[1, "open"])
    assert out["dividends"].tolist() == [0.0, 0.5]
    assert out["splits"].tolist() == [0.0, 0.0]


def test_calendar_comes_from_benchmark():
    data = PriceData(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-05"]), "ticker": ["^B", "^B", "AAA"], "close": 1.0,
    }))
    assert list(data.calendar("^B")) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert list(data.calendar("^B", start="2024-01-03")) == [pd.Timestamp("2024-01-03")]


def test_panel_aligns_to_calendar():
    data = PriceData(pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
        "ticker": ["AAA", "AAA", "BBB"], "open": 1.0, "high": 1.0, "low": 1.0,
        "close": [10.0, 11.0, 20.0], "volume": 5.0, "dividends": [0.0, 0.3, 0.0], "splits": 0.0,
    }))
    calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), name="date")
    panel = data.panel(calendar, ["AAA", "BBB", "CCC"])
    assert panel.close.shape == (3, 3)
    assert np.isnan(panel.close.loc["2024-01-03", "BBB"])
    assert panel.dividends.loc["2024-01-03", "AAA"] == 0.3
    assert panel.dividends.isna().sum().sum() == 0


def test_series_aligns_one_ticker_to_the_calendar():
    data = PriceData(pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-04"]), "ticker": "^B", "close": [1.0, 2.0]}))
    calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    np.testing.assert_array_equal(data.series("^B", "close", calendar).to_numpy(), [1.0, np.nan, 2.0])


def test_daily_total_return_includes_dividends_and_bridges_gaps():
    close = pd.DataFrame({"A": [100.0, 102.0, 101.0], "B": [100.0, np.nan, 110.0]})
    dividends = pd.DataFrame({"A": [0.0, 0.0, 1.0], "B": [0.0, 0.0, 0.0]})
    r = daily_total_return(close, dividends)
    assert np.isnan(r.loc[0, "A"])
    assert r.loc[1, "A"] == pytest.approx(0.02)
    assert r.loc[2, "A"] == pytest.approx((101.0 + 1.0) / 102.0 - 1.0)
    assert np.isnan(r.loc[1, "B"])
    assert r.loc[2, "B"] == pytest.approx(0.10)


def test_forward_open_return_is_next_open_to_following_open():
    open_ = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]})
    dividends = pd.DataFrame({"A": [0.0, 0.0, 0.5, 0.0]})
    fwd = forward_open_return(open_, dividends)
    assert fwd.loc[0, "A"] == pytest.approx((12.0 + 0.5) / 11.0 - 1.0)
    assert fwd.loc[1, "A"] == pytest.approx(13.0 / 12.0 - 1.0)
    assert fwd.loc[2:, "A"].isna().all()


def test_forward_open_return_is_nan_when_entry_open_missing():
    assert np.isnan(forward_open_return(pd.Series([10.0, np.nan, 12.0])).iloc[0])


def test_save_and_load_roundtrip(tmp_path):
    data = PriceData(clean_prices(pd.DataFrame({
        "date": ["2024-01-02"], "ticker": ["AAA"], "open": 1.0, "high": 1.0, "low": 1.0,
        "close": 1.0, "volume": 1.0, "dividends": 0.0, "splits": 0.0,
    })))
    data.save(tmp_path / "sub" / "prices.parquet")
    pd.testing.assert_frame_equal(PriceData.load(tmp_path / "sub" / "prices.parquet").prices, data.prices)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_prices.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.prices'`.

- [ ] **Step 3: Implementar**

`engine/prices.py`:

```python
"""Daily bars from Yahoo adjusted for splits only; dividends come separately and are added back into returns."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_FIELDS = ["open", "high", "low", "close", "volume", "dividends", "splits"]
PANEL_FIELDS = ["open", "high", "low", "close", "volume", "dividends"]
_YAHOO_FIELDS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Volume": "volume", "Dividends": "dividends", "Stock Splits": "splits",
}


@dataclass(frozen=True)
class PricePanel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    dividends: pd.DataFrame


class PriceData:
    """Long table (date, ticker, open, high, low, close, volume, dividends, splits); 'Adj Close' never enters it."""

    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    @classmethod
    def download(cls, tickers: list[str], start: str, end: str | None = None, batch_size: int = 100,
                 downloader=None) -> tuple["PriceData", list[str]]:
        """'Adj Close' is dropped here: it is back-adjusted with dividends paid after each date."""
        if downloader is None:
            import yfinance as yf
            downloader = yf.download
        frames = []
        for i in range(0, len(tickers), batch_size):
            raw = downloader(
                list(tickers[i:i + batch_size]), start=start, end=end, interval="1d",
                auto_adjust=False, actions=True, group_by="ticker", multi_level_index=True,
                progress=False, threads=True,
            )
            if not raw.empty:
                frames.append(_to_long(raw))
        if not frames:
            raise RuntimeError("yfinance não devolveu dados para nenhum ticker")
        prices = clean_prices(pd.concat(frames, ignore_index=True))
        return cls(prices), sorted(set(tickers) - set(prices["ticker"]))

    @classmethod
    def load(cls, path: Path) -> "PriceData":
        return cls(pd.read_parquet(path))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prices.to_parquet(path, index=False)

    @property
    def tickers(self) -> set[str]:
        return set(self.prices["ticker"])

    def calendar(self, benchmark: str, start=None) -> pd.DatetimeIndex:
        dates = pd.DatetimeIndex(self.prices.loc[self.prices["ticker"] == benchmark, "date"].unique()).sort_values()
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        return dates.rename("date")

    def panel(self, calendar: pd.DatetimeIndex, tickers: list[str]) -> PricePanel:
        subset = self.prices[self.prices["ticker"].isin(tickers)]
        tables = {
            field: subset.pivot(index="date", columns="ticker", values=field).reindex(index=calendar, columns=tickers)
            for field in PANEL_FIELDS
        }
        tables["dividends"] = tables["dividends"].fillna(0.0)
        return PricePanel(**tables)

    def series(self, ticker: str, field: str, calendar: pd.DatetimeIndex) -> pd.Series:
        return self.prices[self.prices["ticker"] == ticker].set_index("date")[field].reindex(calendar)


def _to_long(raw: pd.DataFrame) -> pd.DataFrame:
    long = raw.stack(level="Ticker").rename(columns=_YAHOO_FIELDS)
    long.index = long.index.set_names(["date", "ticker"])
    return long.reindex(columns=PRICE_FIELDS).reset_index()


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.assign(date=pd.to_datetime(prices["date"]))
    out = out.dropna(subset=["close"]).drop_duplicates(["date", "ticker"], keep="last")
    out.loc[out["open"] <= 0, "open"] = np.nan
    out[["dividends", "splits"]] = out[["dividends", "splits"]].fillna(0.0)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def daily_total_return(close: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close with the dividend going ex that day; after a missing day, it spans back to the last close."""
    return (close + dividends) / close.ffill().shift(1) - 1.0


def forward_open_return(open_, dividends=None):
    """Row t: buy at the open of t+1, sell at the open of t+2, keeping the dividend that goes ex on t+2."""
    received = 0.0 if dividends is None else dividends.shift(-2)
    return (open_.shift(-2) + received) / open_.shift(-1) - 1.0
```

Em `engine/features.py`, trocar `from engine.data.prices import PricePanel, daily_total_return` por `from engine.prices import PricePanel, daily_total_return`.

Em `tests/test_features.py`, trocar `from engine.data.prices import PANEL_FIELDS, PricePanel` por `from engine.prices import PANEL_FIELDS, PricePanel`.

`tests/conftest.py` (inteiro):

```python
import pytest
from synthetic import BENCH, RF, make_market

from engine.prices import PriceData, clean_prices
from engine.universe import Universe


@pytest.fixture(scope="session")
def synthetic_panel():
    raw = make_market()
    prices = PriceData(clean_prices(raw["prices"]))
    calendar = prices.calendar(BENCH)
    tickers = sorted(prices.tickers - {BENCH, RF})
    membership = Universe(raw["snapshots"]).membership(calendar, tickers)
    return prices.panel(calendar, tickers), prices.series(BENCH, "close", calendar), membership
```

Apagar a pasta antiga: `rm -rf engine/data`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 59 passed (prices 12).

---

### Task 4: `FeatureBuilder` e pares por correlação

**Files:**
- Modify: `engine/features.py` (reescrito), `tests/test_features.py` (reescrito)

**Interfaces:**
- Consumes: `PricePanel`, `daily_total_return` (Task 3).
- Produces:
  - `STOCK_FEATURES` (19), `XS_SOURCES`, `XS_FEATURES` (5), `PEER_FEATURES = ["peer_ret_1d", "peer_ret_5d", "peer_ret_21d", "peer_gap_5d"]` e `MARKET_FEATURES` (6);
  - `TICKER_FEATURES = STOCK_FEATURES + XS_FEATURES + PEER_FEATURES`, `DATE_FEATURES = MARKET_FEATURES` e `FEATURES = TICKER_FEATURES + DATE_FEATURES` (34);
  - `FeatureBuilder(peer_count=10, peer_lookback=63)`, com os métodos:
    - `.build(panel, market_close, membership) -> tuple[dict[str, DataFrame], DataFrame]`;
    - `.stock(panel, market_close)`, `.cross_section(stock, membership)`, `.peers(stock, membership)` e `.market(market_close, stock, membership)`.

- [ ] **Step 1: Escrever os testes**

`tests/test_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from engine.features import FEATURES, PEER_FEATURES, TICKER_FEATURES, FeatureBuilder
from engine.prices import PANEL_FIELDS, PricePanel

DATES = pd.bdate_range("2020-01-01", periods=300, name="date")
BUILDER = FeatureBuilder()


def make_panel(close: pd.DataFrame, dividends: pd.DataFrame | None = None) -> PricePanel:
    return PricePanel(
        open=close, high=close, low=close, close=close,
        volume=pd.DataFrame(1000.0, index=close.index, columns=close.columns),
        dividends=pd.DataFrame(0.0, index=close.index, columns=close.columns) if dividends is None else dividends,
    )


def test_feature_lists():
    assert len(FEATURES) == 34 and len(set(FEATURES)) == 34
    assert TICKER_FEATURES[-4:] == PEER_FEATURES


def test_returns_momentum_and_trend_on_constant_growth():
    close = pd.DataFrame({"A": 100.0 * 1.01 ** np.arange(300)}, index=DATES)
    market = pd.Series(1000.0 * 1.005 ** np.arange(300), index=DATES)
    last = {name: frame["A"].iloc[-1] for name, frame in BUILDER.stock(make_panel(close), market).items()}
    assert last["ret_1d"] == pytest.approx(0.01)
    assert last["ret_5d"] == pytest.approx(1.01 ** 5 - 1)
    assert last["ret_63d"] == pytest.approx(1.01 ** 63 - 1)
    assert last["mom_12_1"] == pytest.approx(1.01 ** 231 - 1)
    assert last["vol_21d"] == pytest.approx(0.0, abs=1e-12)
    assert last["rsi_14"] == pytest.approx(100.0)
    assert last["dist_high_252"] == pytest.approx(0.0)
    assert last["intraday_1d"] == pytest.approx(0.0)


def test_dividend_counts_as_return_and_yield():
    close = pd.DataFrame({"A": np.full(300, 100.0)}, index=DATES)
    dividends = pd.DataFrame({"A": np.zeros(300)}, index=DATES)
    dividends.iloc[250, 0] = 1.0
    market = pd.Series(np.linspace(1000.0, 1100.0, 300), index=DATES)
    f = BUILDER.stock(make_panel(close, dividends), market)
    assert f["ret_1d"]["A"].iloc[250] == pytest.approx(0.01)
    assert f["gap_1d"]["A"].iloc[250] == pytest.approx(0.01)
    assert f["div_yield_252"]["A"].iloc[260] == pytest.approx(0.01)
    assert f["ret_21d"]["A"].iloc[260] == pytest.approx(0.01)


def test_beta_recovers_leverage_on_market():
    m = np.random.default_rng(1).normal(0.0, 0.01, 300)
    market = pd.Series(1000.0 * np.cumprod(1 + m), index=DATES)
    close = pd.DataFrame({"A": 50.0 * np.cumprod(1 + 2.0 * m)}, index=DATES)
    assert BUILDER.stock(make_panel(close), market)["beta_63d"]["A"].iloc[-1] == pytest.approx(2.0, rel=1e-9)


def test_cross_section_ranks_only_members():
    dates = DATES[:2]
    ret = pd.DataFrame({"A": [0.01, 0.01], "B": [0.02, 0.02], "C": [0.03, 0.03]}, index=dates)
    stock = {name: ret for name in ("ret_1d", "ret_5d", "ret_21d", "mom_12_1", "vol_21d")}
    membership = pd.DataFrame({"A": [True, True], "B": [True, False], "C": [True, True]}, index=dates)
    xs = BUILDER.cross_section(stock, membership)["xs_ret_1d"]
    assert xs.iloc[0].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert np.isnan(xs.iloc[1]["B"]) and xs.iloc[1][["A", "C"]].tolist() == pytest.approx([0.5, 1.0])


def test_breadth_counts_members_above_their_ma50():
    dates = DATES[:1]
    stock = {"dist_ma50": pd.DataFrame({"A": [0.1], "B": [-0.1], "C": [0.2], "D": [0.3]}, index=dates)}
    membership = pd.DataFrame({"A": [True], "B": [True], "C": [True], "D": [False]}, index=dates)
    assert BUILDER.market(pd.Series([1000.0], index=dates), stock, membership)["breadth_ma50"].iloc[0] == pytest.approx(2 / 3)


def test_peers_are_the_most_correlated_members():
    dates = DATES[:12]
    rng = np.random.default_rng(3)
    a, c = rng.normal(0.0, 0.01, 12), rng.normal(0.0, 0.01, 12)
    ret = pd.DataFrame({"A": a, "B": 2 * a, "C": c, "D": a}, index=dates)
    stock = {"ret_1d": ret, "ret_5d": 5 * ret, "ret_21d": 21 * ret}
    membership = pd.DataFrame(True, index=dates, columns=ret.columns).assign(D=False)
    peers = FeatureBuilder(peer_count=1, peer_lookback=5).peers(stock, membership)
    last = peers["peer_ret_1d"].iloc[-1]
    assert last["A"] == pytest.approx(ret["B"].iloc[-1])
    assert last["B"] == pytest.approx(ret["A"].iloc[-1])
    assert np.isnan(last["D"])
    assert peers["peer_gap_5d"]["A"].iloc[-1] == pytest.approx(5 * a[-1] - 5 * 2 * a[-1])
    assert peers["peer_ret_1d"].iloc[:4].isna().all().all()


def test_all_features_are_causal(synthetic_panel):
    panel, market_close, membership = synthetic_panel
    cutoff = panel.close.index[200]
    truncated = PricePanel(**{field: getattr(panel, field).loc[:cutoff] for field in PANEL_FIELDS})
    builder = FeatureBuilder(peer_count=3, peer_lookback=21)
    full_ticker, full_date = builder.build(panel, market_close, membership)
    part_ticker, part_date = builder.build(truncated, market_close.loc[:cutoff], membership.loc[:cutoff])
    assert list(full_ticker) == TICKER_FEATURES
    for name in TICKER_FEATURES:
        pd.testing.assert_frame_equal(full_ticker[name].loc[:cutoff], part_ticker[name], check_exact=True)
    pd.testing.assert_frame_equal(full_date.loc[:cutoff], part_date, check_exact=True)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_features.py -q`
Expected: FAIL — `ImportError: cannot import name 'PEER_FEATURES'`.

- [ ] **Step 3: Implementar**

`engine/features.py`:

```python
"""Point-in-time features: every value at date t uses only data dated on or before t."""
import warnings

import numpy as np
import pandas as pd

from engine.prices import PricePanel, daily_total_return

STOCK_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "mom_12_1", "gap_1d", "intraday_1d",
    "vol_21d", "vol_63d", "vol_ratio", "range_21d", "dist_ma50", "dist_ma200",
    "dist_high_252", "rsi_14", "volume_ratio", "log_dollar_volume", "div_yield_252", "beta_63d",
]
XS_SOURCES = ["ret_1d", "ret_5d", "ret_21d", "mom_12_1", "vol_21d"]
XS_FEATURES = [f"xs_{name}" for name in XS_SOURCES]
PEER_FEATURES = ["peer_ret_1d", "peer_ret_5d", "peer_ret_21d", "peer_gap_5d"]
MARKET_FEATURES = ["mkt_ret_1d", "mkt_ret_5d", "mkt_ret_21d", "mkt_vol_21d", "mkt_dist_ma200", "breadth_ma50"]
TICKER_FEATURES = STOCK_FEATURES + XS_FEATURES + PEER_FEATURES
DATE_FEATURES = MARKET_FEATURES
FEATURES = TICKER_FEATURES + DATE_FEATURES


def _rolling(frame, window: int, stat: str):
    return getattr(frame.rolling(window, min_periods=int(np.ceil(0.8 * window))), stat)()


class FeatureBuilder:
    def __init__(self, peer_count: int = 10, peer_lookback: int = 63):
        self.peer_count = peer_count
        self.peer_lookback = peer_lookback

    def build(self, panel: PricePanel, market_close: pd.Series,
              membership: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        stock = self.stock(panel, market_close)
        ticker = {**stock, **self.cross_section(stock, membership), **self.peers(stock, membership)}
        return ticker, self.market(market_close, stock, membership)

    @staticmethod
    def stock(panel: PricePanel, market_close: pd.Series) -> dict[str, pd.DataFrame]:
        close, open_, dividends = panel.close, panel.open, panel.dividends
        prev_close = close.ffill().shift(1)
        r = daily_total_return(close, dividends)
        tri = (1.0 + r.fillna(0.0)).cumprod().where(close.ffill().notna())
        m = market_close / market_close.shift(1) - 1.0
        m_wide = pd.DataFrame(np.repeat(m.to_numpy()[:, None], r.shape[1], axis=1), index=r.index, columns=r.columns).where(r.notna())

        f = {"ret_1d": r}
        for k in (5, 21, 63):
            f[f"ret_{k}d"] = tri / tri.shift(k) - 1.0
        f["mom_12_1"] = tri.shift(21) / tri.shift(252) - 1.0
        f["gap_1d"] = (open_ + dividends) / prev_close - 1.0
        f["intraday_1d"] = close / open_ - 1.0
        f["vol_21d"] = _rolling(r, 21, "std")
        f["vol_63d"] = _rolling(r, 63, "std")
        f["vol_ratio"] = f["vol_21d"] / f["vol_63d"]
        f["range_21d"] = _rolling((panel.high - panel.low) / close, 21, "mean")
        f["dist_ma50"] = tri / _rolling(tri, 50, "mean") - 1.0
        f["dist_ma200"] = tri / _rolling(tri, 200, "mean") - 1.0
        f["dist_high_252"] = close / _rolling(close, 252, "max") - 1.0
        gain = _rolling(r.clip(lower=0.0), 14, "mean")
        loss = _rolling((-r).clip(lower=0.0), 14, "mean")
        f["rsi_14"] = 100.0 * gain / (gain + loss)
        f["volume_ratio"] = panel.volume / _rolling(panel.volume, 21, "mean")
        with np.errstate(divide="ignore"):
            f["log_dollar_volume"] = np.log(_rolling(close * panel.volume, 21, "mean"))
        f["div_yield_252"] = dividends.rolling(252, min_periods=1).sum() / close
        mean_r, mean_m = _rolling(r, 63, "mean"), _rolling(m_wide, 63, "mean")
        covariance = _rolling(r * m_wide, 63, "mean") - mean_r * mean_m
        variance = _rolling(m_wide * m_wide, 63, "mean") - mean_m * mean_m
        f["beta_63d"] = covariance / variance
        return {name: f[name].replace([np.inf, -np.inf], np.nan) for name in STOCK_FEATURES}

    @staticmethod
    def cross_section(stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {f"xs_{name}": stock[name].where(membership).rank(axis=1, pct=True) for name in XS_SOURCES}

    def peers(self, stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """For each member at t: mean return of the peer_count members whose last peer_lookback daily returns correlate most with its own."""
        returns, member = stock["ret_1d"].to_numpy(), membership.to_numpy()
        horizons = {name: stock[name].to_numpy() for name in ("ret_1d", "ret_5d", "ret_21d")}
        out = {name: np.full(returns.shape, np.nan) for name in horizons}
        lookback, count = self.peer_lookback, self.peer_count
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(lookback - 1, len(returns)):
                window = returns[i - lookback + 1:i + 1]
                cols = np.flatnonzero(member[i] & (np.isfinite(window).sum(axis=0) >= 0.8 * lookback))
                if len(cols) <= count:
                    continue
                z = window[:, cols]
                z = np.nan_to_num((z - np.nanmean(z, axis=0)) / np.nanstd(z, axis=0))
                corr = z.T @ z / lookback
                np.fill_diagonal(corr, -np.inf)
                peers = np.argpartition(-corr, count - 1, axis=1)[:, :count]
                for name, values in horizons.items():
                    out[name][i, cols] = np.nanmean(values[i, cols][peers], axis=1)
        frames = {f"peer_{name}": pd.DataFrame(values, index=membership.index, columns=membership.columns)
                  for name, values in out.items()}
        frames["peer_gap_5d"] = stock["ret_5d"] - frames["peer_ret_5d"]
        return {name: frames[name] for name in PEER_FEATURES}

    @staticmethod
    def market(market_close: pd.Series, stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> pd.DataFrame:
        m = market_close / market_close.shift(1) - 1.0
        dist = stock["dist_ma50"]
        above = (dist > 0).astype(float).where(dist.notna() & membership)
        return pd.DataFrame({
            "mkt_ret_1d": m,
            "mkt_ret_5d": market_close / market_close.shift(5) - 1.0,
            "mkt_ret_21d": market_close / market_close.shift(21) - 1.0,
            "mkt_vol_21d": _rolling(m, 21, "std"),
            "mkt_dist_ma200": market_close / _rolling(market_close, 200, "mean") - 1.0,
            "breadth_ma50": above.mean(axis=1),
        }, index=market_close.index)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 60 passed (features 8). Se o teste de causalidade falhar, **há lookahead**: investigar com superpowers:systematic-debugging, sem afrouxar o teste.

- [ ] **Step 5: Tempo dos pares em escala real**

```bash
venv/bin/python - <<'EOF'
import time, numpy as np, pandas as pd
from engine.features import FeatureBuilder
rng = np.random.default_rng(0)
dates = pd.bdate_range("2018-01-02", periods=2200); cols = [f"T{i}" for i in range(620)]
ret = pd.DataFrame(rng.normal(0, 0.02, (2200, 620)), index=dates, columns=cols)
member = pd.DataFrame(rng.uniform(size=(2200, 620)) < 0.8, index=dates, columns=cols)
t = time.perf_counter(); FeatureBuilder().peers({"ret_1d": ret, "ret_5d": ret, "ret_21d": ret}, member); print(f"{time.perf_counter() - t:.1f}s")
EOF
```

Expected: menos de ~60 s. Se passar disso, registrar no ledger. Não otimizar sem necessidade.

---

### Task 5: `WalkForward` paralelo

**Files:**
- Modify: `engine/walk_forward.py` (reescrito), `tests/test_walk_forward.py` (reescrito)
- Delete: `engine/model.py`

**Interfaces:**
- Consumes: dataset `MultiIndex(date, ticker)` com as colunas de features, `fwd_ret` e `label`.
- Produces:
  - `WalkForwardResult(predictions, importance)`, com `predictions` nas colunas `date, ticker, prob, fwd_ret, label` e `importance` com índice `date` e uma coluna por feature;
  - `WalkForward(train_window, params, min_child_share=0.004, min_child_floor=20, n_jobs=1, model_class=LGBMClassifier)`;
  - `.decisions(row_pos, calendar, start=None) -> ndarray`, que levanta `ValueError` com "máximo" quando a janela não cabe antes de `start`;
  - `.run(dataset, features, calendar, start=None, progress=None) -> WalkForwardResult`, em que `progress(fração)` é chamado ao fim de cada bloco.

- [ ] **Step 1: Escrever os testes**

`tests/test_walk_forward.py`:

```python
import numpy as np
import pandas as pd
import pytest

from config.config import LGBM_PARAMS
from engine.walk_forward import WalkForward

DATES = pd.bdate_range("2021-01-01", periods=15, name="date")


def spy_dataset() -> pd.DataFrame:
    rows = [
        {"date": day, "ticker": ticker, "day": float(k), "noise": 0.0, "fwd_ret": 0.01, "label": float(k % 2)}
        for k, day in enumerate(DATES) if k >= 2
        for ticker in ("A", "B")
    ]
    return pd.DataFrame(rows).set_index(["date", "ticker"])


class Spy:
    """Records which calendar days (feature 'day') each fit and predict sees, and the params it got."""
    log: list = []

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y):
        Spy.log.append(("fit", sorted({int(v) for v in X[:, 0]}), self.params["min_child_samples"]))
        self.feature_importances_ = np.array([3.0, 1.0])
        return self

    def predict_proba(self, X):
        Spy.log.append(("predict", sorted({int(v) for v in X[:, 0]}), None))
        p = np.full(len(X), 0.7)
        return np.column_stack([1 - p, p])


def run_spy(dataset, window=4, start=None, **options):
    Spy.log = []
    return WalkForward(window, {}, n_jobs=1, model_class=Spy, **options).run(dataset, ["day", "noise"], DATES, start=start)


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
    fits = [days for kind, days, _ in Spy.log if kind == "fit"]
    predicts = [days for kind, days, _ in Spy.log if kind == "predict"]
    decisions = list(range(2 + 4 + 1, len(DATES)))
    assert predicts == [[k] for k in decisions]
    assert fits == [list(range(k - 4 - 1, k - 1)) for k in decisions]


def test_rows_with_unknown_label_are_not_used_for_training():
    dataset = spy_dataset()
    dataset.loc[dataset.index.get_level_values("date") == DATES[5], "label"] = np.nan
    run_spy(dataset)
    fits = [days for kind, days, _ in Spy.log if kind == "fit"]
    assert fits[0] == [2, 3, 4]
    assert all(5 not in days for days in fits)


def test_min_child_samples_grows_with_the_training_rows():
    run_spy(spy_dataset(), min_child_share=0.5, min_child_floor=2)
    sizes = {size for kind, _, size in Spy.log if kind == "fit"}
    assert sizes == {4}
    run_spy(spy_dataset())
    assert {size for kind, _, size in Spy.log if kind == "fit"} == {20}


def test_outputs_predictions_and_importance():
    result = run_spy(spy_dataset())
    predictions = result.predictions
    assert list(predictions.columns) == ["date", "ticker", "prob", "fwd_ret", "label"]
    assert predictions["date"].min() == DATES[7]
    assert len(predictions) == 2 * (len(DATES) - 7)
    assert (predictions["prob"] == 0.7).all()
    assert list(result.importance.columns) == ["day", "noise"]
    assert result.importance.iloc[0].tolist() == [0.75, 0.25]
    assert list(result.importance.index) == list(DATES[7:])


def test_start_sets_the_first_decision():
    result = run_spy(spy_dataset(), start=DATES[9])
    assert result.predictions["date"].min() == DATES[9]


def test_window_too_long_for_the_start_date_is_refused():
    with pytest.raises(ValueError, match="máximo é 5"):
        run_spy(spy_dataset(), window=6, start=DATES[8])


def test_progress_reaches_one():
    seen = []
    Spy.log = []
    WalkForward(4, {}, n_jobs=1, model_class=Spy).run(spy_dataset(), ["day", "noise"], DATES, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == pytest.approx(1.0)


def test_parallel_run_is_identical_to_sequential():
    dataset, dates = random_dataset()
    params = {**LGBM_PARAMS, "n_estimators": 20}
    sequential = WalkForward(10, params, n_jobs=1).run(dataset, ["a", "b", "c"], dates)
    parallel = WalkForward(10, params, n_jobs=2).run(dataset, ["a", "b", "c"], dates)
    pd.testing.assert_frame_equal(sequential.predictions, parallel.predictions, check_exact=True)
    pd.testing.assert_frame_equal(sequential.importance, parallel.importance, check_exact=True)
    assert ((sequential.predictions["prob"] > 0) & (sequential.predictions["prob"] < 1)).all()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_walk_forward.py -q`
Expected: FAIL — `ImportError: cannot import name 'WalkForward'`.

- [ ] **Step 3: Implementar**

`engine/walk_forward.py`:

```python
"""Daily retraining on a rolling window, predicting only the next decision date; decision dates run in parallel."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from lightgbm import LGBMClassifier


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    importance: pd.DataFrame


class WalkForward:
    """Row s's label needs the open of s+2, so the fit for decision t uses exactly the rows dated t-X-1 .. t-2."""

    def __init__(self, train_window: int, params: dict, min_child_share: float = 0.004, min_child_floor: int = 20,
                 n_jobs: int = 1, model_class=LGBMClassifier):
        self.train_window = train_window
        self.params = params
        self.min_child_share = min_child_share
        self.min_child_floor = min_child_floor
        self.n_jobs = n_jobs
        self.model_class = model_class

    def decisions(self, row_pos: np.ndarray, calendar: pd.DatetimeIndex, start=None) -> np.ndarray:
        first = row_pos[0] + self.train_window + 1
        begin = first if start is None else calendar.searchsorted(pd.Timestamp(start))
        if begin < first:
            raise ValueError(f"a janela de {self.train_window} pregões precisa de {self.train_window + 1} pregões de dados "
                             f"antes de {pd.Timestamp(start).date()}; o máximo é {begin - row_pos[0] - 1}")
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
        blocks = np.array_split(decisions, min(len(decisions), 4 * effective_n_jobs(self.n_jobs)))
        jobs = (delayed(_fit_block)(X, y, row_pos, block, self.train_window, self.params, self.min_child_share,
                                    self.min_child_floor, self.model_class) for block in blocks)
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
                                  index=pd.DatetimeIndex(calendar[decisions], name="date"), columns=features)
        return WalkForwardResult(predictions, importance)


def _fit_block(X, y, row_pos, decisions, window, params, min_child_share, min_child_floor, model_class):
    probs, gains = [], []
    for k in decisions:
        lo = np.searchsorted(row_pos, k - window - 1, side="left")
        hi = np.searchsorted(row_pos, k - 2, side="right")
        known = ~np.isnan(y[lo:hi])
        model = model_class(**params, min_child_samples=max(min_child_floor, round(min_child_share * known.sum())))
        model.fit(X[lo:hi][known], y[lo:hi][known].astype(int))
        first, last = np.searchsorted(row_pos, k, side="left"), np.searchsorted(row_pos, k, side="right")
        probs.append(model.predict_proba(X[first:last])[:, 1])
        gain = np.asarray(model.feature_importances_, dtype=float)
        gains.append(gain / gain.sum() if gain.sum() > 0 else gain)
    return np.concatenate(probs), np.vstack(gains)
```

Apagar a fábrica antiga: `rm engine/model.py`.

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 62 passed (walk_forward 8).

---

### Task 6: `Portfolio`

**Files:**
- Modify: `engine/portfolio.py` (reescrito), `tests/test_portfolio.py`

**Interfaces:**
- Consumes: `predictions` (`date, ticker, prob, fwd_ret, label`) e `market` (índice `decision_date`; colunas `holding_date`, `bench_fwd_ret`, `rf_daily`).
- Produces:
  - `Portfolio(threshold, cost_bps=0.0)`, com `.select(predictions) -> DataFrame` e `.backtest(predictions, market) -> BacktestResult`;
  - `BacktestResult(daily, positions)`, com as mesmas colunas de hoje.

- [ ] **Step 1: Adaptar os testes**

Em `tests/test_portfolio.py`:
- trocar `from engine.portfolio import run_backtest` por `from engine.portfolio import Portfolio`;
- trocar cada chamada `run_backtest(P, M, TH)` por `Portfolio(TH).backtest(P, M)` e cada `run_backtest(P, M, TH, cost_bps=C)` por `Portfolio(TH, cost_bps=C).backtest(P, M)`. Exemplos:
  - `run_backtest(predictions, market, 0.55).positions` → `Portfolio(0.55).backtest(predictions, market).positions`;
  - `run_backtest(predictions, market, 0.55, cost_bps=10.0).daily` → `Portfolio(0.55, cost_bps=10.0).backtest(predictions, market).daily`.

Acrescentar:

```python
def test_select_keeps_only_probabilities_above_the_threshold(predictions):
    chosen = Portfolio(0.6).select(predictions)
    assert set(zip(chosen["date"], chosen["ticker"])) == {(D[0], "B"), (D[2], "A"), (D[4], "A")}
    assert (chosen["weight"] == 1.0).all()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_portfolio.py -q`
Expected: FAIL — `ImportError: cannot import name 'Portfolio'`.

- [ ] **Step 3: Implementar**

`engine/portfolio.py`:

```python
"""Long-only daily portfolio built from the probabilities predicted after each close."""
from dataclasses import dataclass

import pandas as pd

POSITION_COLUMNS = ["decision_date", "holding_date", "ticker", "prob", "weight", "fwd_ret", "contribution"]


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    positions: pd.DataFrame


class Portfolio:
    """Buys at the next open the stocks with prob > threshold, weighted by prob; otherwise cash at the risk-free rate."""

    def __init__(self, threshold: float, cost_bps: float = 0.0):
        self.threshold = threshold
        self.cost_bps = cost_bps

    def select(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Only `prob` drives the choice."""
        chosen = predictions.loc[predictions["prob"] > self.threshold, ["date", "ticker", "prob", "fwd_ret"]].copy()
        chosen["weight"] = chosen["prob"] / chosen.groupby("date")["prob"].transform("sum")
        return chosen

    def backtest(self, predictions: pd.DataFrame, market: pd.DataFrame) -> BacktestResult:
        """A held stock without a realized return (delisting, halt) counts as 0: dropping it would use future information."""
        window = market.loc[predictions["date"].min():predictions["date"].max()]
        realized = window["bench_fwd_ret"].notna().to_numpy()
        window = window.iloc[: realized.nonzero()[0].max() + 1] if realized.any() else window.iloc[:0]
        gaps = window.index[window["bench_fwd_ret"].isna()]
        if len(gaps):
            raise ValueError(f"benchmark sem retorno realizado dentro do período ({', '.join(str(d.date()) for d in gaps)}); "
                             "baixe os preços de novo")
        dates = window.index
        chosen = self.select(predictions[predictions["date"].isin(dates)])
        chosen["missing"] = chosen["fwd_ret"].isna()
        chosen["contribution"] = chosen["weight"] * chosen["fwd_ret"].fillna(0.0)
        by_date = chosen.groupby("date")
        n_positions = by_date.size().reindex(dates, fill_value=0)
        invested = n_positions > 0
        stock_return = by_date["contribution"].sum().reindex(dates, fill_value=0.0)
        gross = stock_return.where(invested, window["rf_daily"])
        turnover = self._turnover(chosen, dates, stock_return)
        cost = turnover * self.cost_bps / 10_000.0
        daily = pd.DataFrame({
            "decision_date": dates.to_numpy(),
            "gross": gross.to_numpy(),
            "net": ((1.0 + gross) * (1.0 - cost) - 1.0).to_numpy(),
            "bench": window["bench_fwd_ret"].to_numpy(),
            "rf": window["rf_daily"].to_numpy(),
            "n_positions": n_positions.to_numpy(),
            "invested": invested.to_numpy(),
            "turnover": turnover.to_numpy(),
            "cost": cost.to_numpy(),
            "n_missing": by_date["missing"].sum().reindex(dates, fill_value=0).astype(int).to_numpy(),
        }, index=pd.DatetimeIndex(window["holding_date"], name="holding_date"))
        positions = chosen.rename(columns={"date": "decision_date"})
        positions["holding_date"] = positions["decision_date"].map(window["holding_date"])
        return BacktestResult(daily=daily, positions=positions[POSITION_COLUMNS].reset_index(drop=True))

    @staticmethod
    def _turnover(chosen: pd.DataFrame, dates: pd.DatetimeIndex, stock_return: pd.Series) -> pd.Series:
        """Buys + sells at each open (Σ|Δw|): target weights vs yesterday's weights after they drifted with prices."""
        weights = chosen.pivot(index="date", columns="ticker", values="weight").reindex(dates).fillna(0.0)
        growth = 1.0 + chosen.pivot(index="date", columns="ticker", values="fwd_ret").reindex(dates).fillna(0.0)
        drifted = (weights * growth).div(1.0 + stock_return, axis=0)
        return (weights - drifted.shift(1).fillna(0.0)).abs().sum(axis=1)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 63 passed (portfolio 9).

---

### Task 7: Métricas enxutas

**Files:**
- Modify: `engine/metrics.py` (reescrito), `tests/test_metrics.py` (reescrito)

**Interfaces:**
- Produces:
  - `TRADING_DAYS = 252` e `drawdown(returns) -> Series`;
  - `performance_metrics(returns, rf, invested=None) -> dict`, com as chaves `total_return, volatility, sharpe, max_drawdown, hit_ratio`;
  - `daily_ic(predictions) -> Series`;
  - `model_metrics(predictions) -> dict`, com as chaves `auc, accuracy, ic`;
  - `monthly_auc(predictions) -> Series`.

- [ ] **Step 1: Escrever os testes**

`tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd
import pytest

from engine.metrics import drawdown, model_metrics, monthly_auc, performance_metrics


def series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)), dtype=float)


def test_drawdown_counts_initial_capital():
    assert drawdown(series([0.1, -0.2, 0.05])).min() == pytest.approx(0.88 / 1.1 - 1)
    assert drawdown(series([-0.1, 0.05])).tolist() == pytest.approx([-0.1, 0.9 * 1.05 - 1])
    assert drawdown(series([0.01, 0.02])).min() == 0.0


def test_performance_metrics():
    r = series([0.02, 0.0, 0.02, 0.0])
    m = performance_metrics(r, series([0.0] * 4))
    spread = np.std([0.02, 0.0, 0.02, 0.0], ddof=1)
    assert list(m) == ["total_return", "volatility", "sharpe", "max_drawdown", "hit_ratio"]
    assert m["total_return"] == pytest.approx(1.02 ** 2 - 1)
    assert m["volatility"] == pytest.approx(spread * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(0.01 / spread * np.sqrt(252))
    assert m["max_drawdown"] == 0.0
    assert m["hit_ratio"] == pytest.approx(0.5)


def test_sharpe_uses_excess_over_the_risk_free_rate():
    r = series([0.03, -0.01, 0.02, -0.02])
    rf = series([0.001] * 4)
    excess = r - rf
    assert performance_metrics(r, rf)["sharpe"] == pytest.approx(excess.mean() / excess.std(ddof=1) * np.sqrt(252))


def test_hit_ratio_ignores_cash_days_when_invested_mask_given():
    r = series([0.01, 0.0001, -0.01, 0.02])
    invested = series([1, 0, 1, 1]).astype(bool)
    assert performance_metrics(r, series([0.0001] * 4), invested=invested)["hit_ratio"] == pytest.approx(2 / 3)


def test_empty_returns_do_not_crash():
    m = performance_metrics(series([]), series([]))
    assert m["total_return"] == 0.0 and np.isnan(m["sharpe"]) and np.isnan(m["hit_ratio"])


def test_model_metrics_on_perfect_ranking():
    predictions = pd.DataFrame({
        "date": np.repeat(pd.bdate_range("2022-01-03", periods=3), 4),
        "prob": np.tile([0.2, 0.4, 0.6, 0.8], 3),
        "fwd_ret": np.tile([-0.02, -0.01, 0.01, 0.02], 3),
    })
    predictions["label"] = (predictions["fwd_ret"] > 0).astype(float)
    assert model_metrics(predictions) == pytest.approx({"auc": 1.0, "accuracy": 1.0, "ic": 1.0})


def test_monthly_auc():
    predictions = pd.DataFrame({
        "date": np.repeat(pd.to_datetime(["2022-01-10", "2022-02-10"]), 4),
        "prob": [0.2, 0.4, 0.6, 0.8] * 2,
        "label": [0, 0, 1, 1, 1, 1, 0, 0],
    })
    assert monthly_auc(predictions).tolist() == pytest.approx([1.0, 0.0])
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_metrics.py -q`
Expected: FAIL — `ImportError: cannot import name 'monthly_auc'`.

- [ ] **Step 3: Implementar**

`engine/metrics.py`:

```python
"""Performance and model-quality metrics for daily data (252 trading days per year)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRADING_DAYS = 252


def drawdown(returns: pd.Series) -> pd.Series:
    """Distance from the running peak; the initial capital (1.0) is the first peak."""
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax().clip(lower=1.0) - 1.0


def performance_metrics(returns: pd.Series, rf: pd.Series, invested: pd.Series | None = None) -> dict[str, float]:
    excess = returns - rf
    spread = excess.std(ddof=1)
    active = returns[invested.astype(bool)] if invested is not None else returns
    return {
        "total_return": float((1.0 + returns).prod() - 1.0),
        "volatility": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": float(excess.mean() / spread * np.sqrt(TRADING_DAYS)) if spread > 0 else float("nan"),
        "max_drawdown": float(drawdown(returns).min()),
        "hit_ratio": float((active > 0).mean()) if len(active) else float("nan"),
    }


def daily_ic(predictions: pd.DataFrame) -> pd.Series:
    """Spearman correlation between predicted probability and realized return, per decision date."""
    realized = predictions.dropna(subset=["fwd_ret"])
    return realized.groupby("date")[["prob", "fwd_ret"]].apply(lambda day: day["prob"].corr(day["fwd_ret"], method="spearman"))


def model_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    known = predictions.dropna(subset=["label"])
    label, prob = known["label"].astype(int), known["prob"]
    return {
        "auc": float(roc_auc_score(label, prob)),
        "accuracy": float(((prob > 0.5).astype(int) == label).mean()),
        "ic": float(daily_ic(known).mean()),
    }


def monthly_auc(predictions: pd.DataFrame) -> pd.Series:
    known = predictions.dropna(subset=["label"])

    def auc(period: pd.DataFrame) -> float:
        return float(roc_auc_score(period["label"], period["prob"])) if period["label"].nunique() == 2 else float("nan")

    return known.groupby(pd.Grouper(key="date", freq="ME"))[["label", "prob"]].apply(auc)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 61 passed (metrics 7).

---

### Task 8: `Project` e teste de lookahead ponta a ponta

**Files:**
- Create: `engine/project.py`, `tests/test_project.py`
- Modify: `tests/synthetic.py` (acrescentar helpers), `tests/conftest.py` (fixture `synthetic_project`)

**Interfaces:**
- Consumes: `Universe`, `PriceData`, `FeatureBuilder`, `build_dataset`, `WalkForward`, `forward_open_return`, `daily_total_return`, `FEATURES`, `TICKER_ALIASES`, `Config`.
- Produces:
  - `FeatureSet(dataset | None, market, coverage, quality)`, com a propriedade `.calendar`;
  - `RunResults(window, predictions, importance, info)`;
  - `Project(config)`, com:
    - `.features_dir` e `.runs_dir`;
    - `.download(price_downloader=None, fetch=None) -> list[str]`;
    - `.build_features(universe=None, prices=None) -> FeatureSet` e `.features(with_dataset=True) -> FeatureSet`;
    - `.run(window, progress=None) -> RunResults`, `.save_run(results)`, `.load_run(window)`, `.runs() -> list[int]` e `.stamp(window=None) -> int`;
  - em `tests/synthetic.py`: `SMALL_LGBM`, `synthetic_config(root, **overrides)` e `yahoo_response(tickers, dates, skip=())`.

- [ ] **Step 1: Helpers e fixture de teste**

Acrescentar `from config.config import Config` aos imports do topo de `tests/synthetic.py` e, ao fim do arquivo:

```python
SMALL_LGBM = {
    "n_estimators": 10, "learning_rate": 0.1, "num_leaves": 4, "subsample": 0.8, "subsample_freq": 1,
    "colsample_bytree": 0.8, "importance_type": "gain", "random_state": 0, "deterministic": True,
    "force_col_wise": True, "n_jobs": 1, "verbose": -1,
}


def synthetic_config(root, **overrides) -> Config:
    """Decisions start at business day 210 of the synthetic market; features are complete from day ~160."""
    start = pd.bdate_range("2019-01-01", periods=330)[210]
    settings = {
        "data_in": root / "in", "data_out": root / "out", "price_start": "2019-01-01",
        "backtest_start": str(start.date()), "train_window_days": 40, "peer_count": 3,
        "peer_lookback_days": 21, "n_jobs": 1, "lgbm_params": SMALL_LGBM,
    }
    return Config(**{**settings, **overrides})


def yahoo_response(tickers, dates, skip=()) -> pd.DataFrame:
    """Same shape as yf.download(group_by='ticker') for the tickers not in `skip`."""
    bar = {"Open": 10.0, "High": 10.0, "Low": 10.0, "Close": 10.0, "Adj Close": 9.0,
           "Volume": 100.0, "Dividends": 0.0, "Stock Splits": 0.0}
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    frames = {t: pd.DataFrame(bar, index=index) for t in tickers if t not in skip}
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, axis=1)
    raw.columns = raw.columns.set_names(["Ticker", "Price"])
    return raw
```

Acrescentar ao `tests/conftest.py`:

```python
from synthetic import synthetic_config

from engine.project import Project


@pytest.fixture(scope="session")
def synthetic_project(tmp_path_factory):
    raw = make_market()
    project = Project(synthetic_config(tmp_path_factory.mktemp("project")))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    project.run(40)
    return project
```

(juntar os imports com os que já existem no topo do arquivo).

- [ ] **Step 2: Escrever os testes**

`tests/test_project.py`:

```python
import numpy as np
import pandas as pd
import pytest
from synthetic import BENCH, RF, make_market, perturb_after, synthetic_config, yahoo_response

from engine.features import DATE_FEATURES, FEATURES
from engine.prices import PriceData, clean_prices
from engine.project import Project
from engine.universe import Universe

CONSTITUENTS = b'date,tickers\n2020-01-02,"AAA,BRK.B,GONE"\n'


def build(raw: dict, root, **overrides) -> Project:
    project = Project(synthetic_config(root, **overrides))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    return project


def test_predictions_do_not_depend_on_future_data(synthetic_project, tmp_path):
    base = synthetic_project.load_run(40).predictions
    decision_dates = base["date"].drop_duplicates().sort_values()
    cutoff = decision_dates.iloc[len(decision_dates) // 2]
    perturbed = build(perturb_after(make_market(), cutoff), tmp_path, n_jobs=2).run(40).predictions
    columns = ["date", "ticker", "prob"]
    before = base.loc[base["date"] <= cutoff, columns].reset_index(drop=True)
    after = perturbed.loc[perturbed["date"] <= cutoff, columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    later = base[base["date"] > cutoff].merge(perturbed[perturbed["date"] > cutoff], on=["date", "ticker"], suffixes=("_base", "_pert"))
    assert len(later) > 0 and not np.allclose(later["prob_base"], later["prob_pert"])


def test_predictions_only_for_members_on_each_date(synthetic_project):
    universe = Universe(make_market()["snapshots"])
    for date, group in synthetic_project.load_run(40).predictions.groupby("date"):
        assert set(group["ticker"]) <= set(universe.members_on(date))


def test_decisions_start_at_backtest_start(synthetic_project):
    predictions = synthetic_project.load_run(40).predictions
    assert predictions["date"].min() == pd.Timestamp(synthetic_project.config.backtest_start)


def test_market_frame_aligns_benchmark_and_risk_free(synthetic_project):
    market = synthetic_project.features(with_dataset=False).market
    prices = make_market()["prices"]
    bench_open = prices[prices["ticker"] == BENCH].set_index("date")["open"]
    irx = prices[prices["ticker"] == RF].set_index("date")["close"]
    calendar, t = market.index, market.index[100]
    assert market.loc[t, "holding_date"] == calendar[101]
    assert market.loc[t, "bench_fwd_ret"] == pytest.approx(bench_open[calendar[102]] / bench_open[calendar[101]] - 1)
    assert market.loc[t, "rf_daily"] == pytest.approx(irx[t] / 100 / 252)
    assert pd.isna(market["holding_date"].iloc[-1]) and pd.isna(market["bench_fwd_ret"].iloc[-2])


def test_features_are_saved_with_every_column(synthetic_project):
    features = synthetic_project.features()
    assert list(features.dataset.columns) == FEATURES + ["fwd_ret", "label"]
    assert features.dataset[DATE_FEATURES].notna().all().all()
    assert synthetic_project.features(with_dataset=False).dataset is None


def test_quality_reports_members_without_usable_history(tmp_path):
    raw = make_market()
    dates = sorted(raw["prices"]["date"].unique())
    stub = raw["prices"][(raw["prices"]["ticker"] == "T05") & (raw["prices"]["date"] >= dates[-5])].assign(ticker="STUB")
    raw["prices"] = pd.concat([raw["prices"], stub], ignore_index=True)
    raw["snapshots"] = raw["snapshots"].assign(tickers=raw["snapshots"]["tickers"].map(lambda members: members + ["GONE", "STUB"]))
    quality = build(raw, tmp_path).features(with_dataset=False).quality
    assert quality["missing_tickers"] == ["GONE", "STUB"]
    detail = {row["ticker"]: row for row in quality["missing_detail"]}
    assert detail["GONE"] == {"ticker": "GONE", "member_days": len(dates), "days_with_price": 0}
    assert detail["STUB"] == {"ticker": "STUB", "member_days": len(dates), "days_with_price": 5}
    assert quality["date_ranges"]["constituents"] == [str(dates[0].date()), str(raw["snapshots"]["date"].max().date())]
    assert set(quality["coverage_by_year"]) == {"2019", "2020"}
    assert all(0 < share <= 1 for share in quality["coverage_by_year"].values())


def test_runs_are_saved_per_window_and_listed(synthetic_project):
    assert synthetic_project.runs() == [40]
    run = synthetic_project.load_run(40)
    assert run.info["window"] == 40
    assert list(run.importance.columns) == FEATURES
    assert synthetic_project.stamp(40) > 0 and synthetic_project.stamp(21) == 0


def test_rebuilding_features_deletes_old_runs(tmp_path):
    project = build(make_market(), tmp_path)
    project.run(40)
    assert project.runs() == [40]
    raw = make_market()
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    assert project.runs() == []


def test_window_too_long_is_refused(synthetic_project):
    with pytest.raises(ValueError, match="máximo"):
        synthetic_project.run(100)
    assert synthetic_project.runs() == [40]


def test_download_saves_inputs_and_clears_derived_results(tmp_path):
    project = Project(synthetic_config(tmp_path))
    (project.runs_dir / "w021").mkdir(parents=True)
    (project.runs_dir / "w021" / "run_info.json").write_text("{}")
    failed = project.download(
        price_downloader=lambda tickers, **kwargs: yahoo_response(tickers, ["2024-01-02", "2024-01-03"], skip={"GONE"}),
        fetch=lambda url: CONSTITUENTS,
    )
    assert failed == ["GONE"]
    assert (tmp_path / "in" / "prices.parquet").exists()
    assert (tmp_path / "in" / "sp500_historical_constituents.csv").read_bytes() == CONSTITUENTS
    assert project.runs() == []


def test_download_keeps_old_prices_when_an_index_is_missing(tmp_path):
    project = Project(synthetic_config(tmp_path))
    with pytest.raises(RuntimeError, match=r"\^SP500TR"):
        project.download(
            price_downloader=lambda tickers, **kwargs: yahoo_response(tickers, ["2024-01-02"], skip={"^SP500TR"}),
            fetch=lambda url: CONSTITUENTS,
        )
    assert not (tmp_path / "in" / "prices.parquet").exists()
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_project.py -q`
Expected: FAIL (erro de coleta) — `ModuleNotFoundError: No module named 'engine.project'` vindo do `conftest.py`.

- [ ] **Step 4: Implementar**

`engine/project.py`:

```python
"""What the CLI and the app do: download inputs, build features once, run one walk-forward per window, load results."""
import dataclasses
import json
import platform
import shutil
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from config.config import TICKER_ALIASES, Config
from engine.dataset import build_dataset
from engine.features import FEATURES, FeatureBuilder
from engine.prices import PriceData, daily_total_return, forward_open_return
from engine.universe import Universe
from engine.walk_forward import WalkForward

PACKAGES = ("pandas", "numpy", "lightgbm", "scikit-learn", "joblib", "yfinance", "streamlit")


@dataclass
class FeatureSet:
    dataset: pd.DataFrame | None
    market: pd.DataFrame
    coverage: pd.DataFrame
    quality: dict

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.market.index)


@dataclass
class RunResults:
    window: int
    predictions: pd.DataFrame
    importance: pd.DataFrame
    info: dict


class Project:
    def __init__(self, config: Config):
        self.config = config
        self.features_dir = config.data_out / "features"
        self.runs_dir = config.data_out / "runs"

    def download(self, price_downloader=None, fetch=None) -> list[str]:
        """Latest constituents and prices; features and runs are deleted because they no longer match the data."""
        c = self.config
        constituents = c.data_in / c.constituents_file
        Universe.download(c.constituents_url, constituents, fetch)
        universe = Universe.from_csv(constituents, TICKER_ALIASES)
        index_tickers = [c.benchmark_ticker, c.risk_free_ticker]
        prices, failed = PriceData.download(universe.tickers_since(c.price_start) + index_tickers, c.price_start,
                                            downloader=price_downloader)
        lost = [t for t in index_tickers if t in failed]
        if lost:
            raise RuntimeError(f"o Yahoo não devolveu {', '.join(lost)}; os preços antigos foram mantidos")
        prices.save(c.data_in / c.prices_file)
        self._clear(self.features_dir, self.runs_dir)
        return failed

    def build_features(self, universe: Universe | None = None, prices: PriceData | None = None) -> FeatureSet:
        """Features do not depend on the training window: built once, shared by every run; old runs are deleted."""
        if universe is None or prices is None:
            universe, prices = self._load_inputs()
        features = self._make_features(universe, prices)
        self._clear(self.features_dir, self.runs_dir)
        self.features_dir.mkdir(parents=True)
        features.dataset.to_parquet(self.features_dir / "dataset.parquet")
        features.market.to_parquet(self.features_dir / "market.parquet")
        features.coverage.to_parquet(self.features_dir / "coverage.parquet")
        (self.features_dir / "quality.json").write_text(json.dumps(features.quality, indent=2, default=str))
        return features

    def features(self, with_dataset: bool = True) -> FeatureSet:
        if not (self.features_dir / "quality.json").exists():
            return self.build_features()
        return FeatureSet(
            dataset=pd.read_parquet(self.features_dir / "dataset.parquet") if with_dataset else None,
            market=pd.read_parquet(self.features_dir / "market.parquet"),
            coverage=pd.read_parquet(self.features_dir / "coverage.parquet"),
            quality=json.loads((self.features_dir / "quality.json").read_text()),
        )

    def run(self, window: int, progress=None) -> RunResults:
        c = self.config
        features = self.features()
        started = time.perf_counter()
        result = WalkForward(window, c.lgbm_params, c.min_child_share, c.min_child_floor, c.n_jobs).run(
            features.dataset, FEATURES, features.calendar, start=c.backtest_start, progress=progress)
        results = RunResults(window, result.predictions, result.importance, info={
            "window": window,
            "features": FEATURES,
            "runtime_minutes": round((time.perf_counter() - started) / 60.0, 2),
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "versions": {package: version(package) for package in PACKAGES},
            "config": dataclasses.asdict(c),
        })
        self.save_run(results)
        return results

    def save_run(self, results: RunResults) -> None:
        folder = self._run_dir(results.window)
        folder.mkdir(parents=True, exist_ok=True)
        results.predictions.to_parquet(folder / "predictions.parquet")
        results.importance.to_parquet(folder / "importance.parquet")
        (folder / "run_info.json").write_text(json.dumps(results.info, indent=2, default=str))

    def load_run(self, window: int) -> RunResults:
        folder = self._run_dir(window)
        return RunResults(window, pd.read_parquet(folder / "predictions.parquet"),
                          pd.read_parquet(folder / "importance.parquet"), json.loads((folder / "run_info.json").read_text()))

    def runs(self) -> list[int]:
        return sorted(int(path.parent.name[1:]) for path in self.runs_dir.glob("w*/run_info.json"))

    def stamp(self, window: int | None = None) -> int:
        """mtime of the file written last (quality.json / run_info.json); 0 when absent. Keys the app caches."""
        path = self.features_dir / "quality.json" if window is None else self._run_dir(window) / "run_info.json"
        return path.stat().st_mtime_ns if path.exists() else 0

    def _run_dir(self, window: int) -> Path:
        return self.runs_dir / f"w{window:03d}"

    def _load_inputs(self) -> tuple[Universe, PriceData]:
        c = self.config
        return Universe.from_csv(c.data_in / c.constituents_file, TICKER_ALIASES), PriceData.load(c.data_in / c.prices_file)

    @staticmethod
    def _clear(*folders: Path) -> None:
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)

    def _make_features(self, universe: Universe, prices: PriceData) -> FeatureSet:
        c = self.config
        calendar = prices.calendar(c.benchmark_ticker, start=c.price_start)
        wanted = universe.tickers_since(c.price_start)
        tickers = sorted(set(wanted) & (prices.tickers - {c.benchmark_ticker, c.risk_free_ticker}))
        panel = prices.panel(calendar, tickers)
        membership = universe.membership(calendar, tickers)
        ticker_features, date_features = FeatureBuilder(c.peer_count, c.peer_lookback_days).build(
            panel, prices.series(c.benchmark_ticker, "close", calendar), membership)
        dataset = build_dataset(ticker_features, date_features, membership, panel.close,
                                forward_open_return(panel.open, panel.dividends), c.min_history_days)
        market = pd.DataFrame({
            "holding_date": pd.Series(calendar, index=calendar).shift(-1),
            "bench_fwd_ret": forward_open_return(prices.series(c.benchmark_ticker, "open", calendar)),
            "rf_daily": prices.series(c.risk_free_ticker, "close", calendar)
                              .ffill(limit=c.risk_free_max_staleness_days) / 100.0 / 252.0,
        }, index=calendar).rename_axis("decision_date")
        coverage = pd.DataFrame({
            "n_members": universe.sizes(calendar),
            "n_with_prices": (membership & panel.close.notna()).sum(axis=1).to_numpy(),
            "n_eligible": dataset.groupby(level="date").size().reindex(calendar, fill_value=0).to_numpy(),
        }, index=calendar)
        quality = _quality(universe, calendar, wanted, tickers, panel, membership, dataset, coverage)
        return FeatureSet(dataset, market, coverage, quality)


def _quality(universe, calendar, wanted, tickers, panel, membership, dataset, coverage) -> dict:
    returns = daily_total_return(panel.close, panel.dividends).where(membership)
    extreme = returns.where(returns.abs() > 0.5).stack().dropna()
    top = extreme.loc[extreme.abs().sort_values(ascending=False).index[:20]]
    missing = _members_without_history(universe, calendar, wanted, panel.close)
    in_dataset = coverage[coverage["n_eligible"] > 0]
    share = in_dataset["n_eligible"] / in_dataset["n_members"]
    return {
        "coverage_by_year": {str(year): round(float(value), 4) for year, value in share.groupby(share.index.year).mean().items()},
        "missing_tickers": [row["ticker"] for row in missing],
        "missing_detail": missing,
        "n_tickers_with_prices": len(tickers),
        "n_extreme_returns": len(extreme),
        "extreme_returns": [{"date": str(d.date()), "ticker": t, "ret": round(float(v), 4)} for (d, t), v in top.items()],
        "date_ranges": {
            "prices": _date_range(calendar),
            "constituents": _date_range(pd.DatetimeIndex(universe.snapshots["date"])),
            "dataset": _date_range(dataset.index.get_level_values("date")),
        },
    }


def _members_without_history(universe, calendar, wanted, close) -> list[dict]:
    """Members priced on under half of their member days: delisted, or a ticker that now names another company."""
    member = universe.membership(calendar, wanted)
    priced = close.notna().reindex(columns=wanted, fill_value=False)
    member_days, days_with_price = member.sum(), (member & priced).sum()
    poor = member_days[(member_days > 0) & (days_with_price < 0.5 * member_days)].index
    return [{"ticker": t, "member_days": int(member_days[t]), "days_with_price": int(days_with_price[t])} for t in sorted(poor)]


def _date_range(dates: pd.DatetimeIndex) -> list[str]:
    return [str(dates.min().date()), str(dates.max().date())]
```

- [ ] **Step 5: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 72 passed (project 11). Se `test_predictions_do_not_depend_on_future_data` falhar, **há lookahead**: investigar com superpowers:systematic-debugging, sem afrouxar o teste.

---

### Task 9: CLI e runs reais

**Files:**
- Modify: `main.py` (reescrito, preservando as notas do usuário no fim)
- Create: `tests/test_main.py`

**Interfaces:**
- Consumes: `Project`, `Portfolio`, `metrics`, `CONFIG`.
- Produces:
  - `report(project, window) -> None`, que imprime a tabela da carteira bruta, da líquida e do S&P e as métricas do modelo;
  - `main()` com os subcomandos `download`, `features` e `run [--window N]`.

- [ ] **Step 1: Escrever o teste**

`tests/test_main.py`:

```python
from main import report


def test_report_prints_portfolio_benchmark_and_model(synthetic_project, capsys):
    report(synthetic_project, 40)
    out = capsys.readouterr().out
    assert "Carteira (líquida)" in out and "S&P 500 TR" in out and "sharpe" in out and "auc" in out
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_main.py -q`
Expected: FAIL — `ImportError: cannot import name 'report'`, porque o `main.py` antigo quebra ao importar `engine.pipeline`.

- [ ] **Step 3: Implementar preservando as notas do usuário**

```bash
venv/bin/python - <<'EOF'
from pathlib import Path
text = Path("main.py").read_text()
notes = text[text.index('\n"""\nReorganizar'):]
code = '''"""`python main.py download` baixa constituintes e preços; `features` calcula as features; `run --window N` roda o walk-forward."""
import argparse

import pandas as pd

from config.config import CONFIG
from engine import metrics
from engine.portfolio import Portfolio
from engine.project import Project


def report(project: Project, window: int) -> None:
    config = project.config
    market = project.features(with_dataset=False).market
    run = project.load_run(window)
    daily = Portfolio(config.threshold, config.cost_bps).backtest(run.predictions, market).daily
    table = pd.DataFrame({
        "Carteira (bruta)": metrics.performance_metrics(daily["gross"], daily["rf"], daily["invested"]),
        "Carteira (líquida)": metrics.performance_metrics(daily["net"], daily["rf"], daily["invested"]),
        "S&P 500 TR": metrics.performance_metrics(daily["bench"], daily["rf"]),
    })
    print(f"\\nJanela {window} pregões · {daily.index.min().date()} → {daily.index.max().date()} · "
          f"limiar {config.threshold} · custo {config.cost_bps} bps")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    realized = run.predictions[run.predictions["date"].isin(daily["decision_date"])]
    print("Modelo:", {name: round(value, 4) for name, value in metrics.model_metrics(realized).items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML do S&P 500")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="baixa os constituintes (GitHub) e os preços (yfinance) mais recentes")
    commands.add_parser("features", help="calcula as features uma vez (servem para todas as janelas)")
    run_command = commands.add_parser("run", help="walk-forward para uma janela de treino e relatório")
    run_command.add_argument("--window", type=int, default=CONFIG.train_window_days,
                             help="pregões de treino (5 = 1 semana, 21 = 1 mês)")
    args = parser.parse_args()
    project = Project(CONFIG)
    if args.command == "download":
        failed = project.download()
        print(f"Constituintes e preços atualizados. Sem dados no Yahoo ({len(failed)}): {', '.join(failed)}")
    elif args.command == "features":
        dataset = project.build_features().dataset
        dates = dataset.index.get_level_values("date")
        print(f"Features: {len(dataset):,} linhas, {dates.min().date()} → {dates.max().date()}")
    else:
        run = project.run(args.window, progress=lambda done: print(f"\\rtreinando {done:.0%}", end="", flush=True))
        print(f"\\n{len(run.predictions):,} previsões em {run.info['runtime_minutes']} min")
        report(project, args.window)


if __name__ == "__main__":
    main()
'''
Path("main.py").write_text(code + notes)
print(Path("main.py").read_text()[-400:])
EOF
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 73 passed. O fim do `main.py` impresso no passo anterior mostra as notas do usuário intactas.

- [ ] **Step 5: Download real**

Run (timeout 10 min): `venv/bin/python main.py download`
Expected:
- `data/in/sp500_historical_constituents.csv` com o último snapshot em 2026-08-18 ou depois;
- `prices.parquet` desde 2018-01-02 até o último pregão;
- uma lista de tickers sem dado (deslistados, e agora também os antigos sem alias, como MMC, BK, FI, PARA...).

Se aparecerem tickers grandes atuais na lista (AAPL, MSFT...), é rate limit: esperar alguns minutos e rodar de novo.

- [ ] **Step 6: Features reais e três janelas**

```bash
time venv/bin/python main.py features
time venv/bin/python main.py run --window 21
time venv/bin/python main.py run --window 5
time venv/bin/python main.py run --window 63
```

Expected:
- as features em ~1–3 min;
- os runs em ~1–5 min cada, com o relatório da carteira bruta, da líquida, do S&P 500 TR e do modelo impresso ao final.
- Anotar os tempos.
- **Sinais de alerta** (parar e investigar lookahead com superpowers:systematic-debugging): AUC > 0,60, Sharpe > 3 ou retorno absurdo.

---

### Task 10: App

**Files:**
- Modify: `app/dashboard.py` (reescrito)
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: `Project`, `RunResults`, `Portfolio`, as funções de `engine.metrics`, `CONFIG`.
- Produces: o app, que lê `BACKTEST_DATA_IN` e `BACKTEST_DATA_OUT` (variáveis de ambiente; os defaults são `data/in` e `data/out` na raiz do projeto) e tem 5 abas.

- [ ] **Step 1: Escrever os testes**

`tests/test_app.py`:

```python
from pathlib import Path

from streamlit.testing.v1 import AppTest
from synthetic import make_market, synthetic_config

from engine.portfolio import Portfolio
from engine.prices import PriceData, clean_prices
from engine.project import Project, RunResults
from engine.universe import Universe

APP = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"
TABS = ["Visão geral", "Carteira por dia", "Recortes de período", "Janelas", "Modelo"]


def open_app(project: Project, monkeypatch, window: int = 40) -> AppTest:
    monkeypatch.setenv("BACKTEST_DATA_IN", str(project.config.data_in))
    monkeypatch.setenv("BACKTEST_DATA_OUT", str(project.config.data_out))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.number_input[0].set_value(window).run()
    return at


def test_dashboard_renders_a_saved_run(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    assert not at.exception
    assert [tab.label for tab in at.tabs] == TABS


def test_dashboard_survives_threshold_nobody_passes(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch)
    at.sidebar.slider[0].set_value(0.8).run()
    assert not at.exception


def test_window_not_run_yet_offers_the_button(synthetic_project, monkeypatch):
    at = open_app(synthetic_project, monkeypatch, window=21)
    assert not at.exception
    assert "Rodar janela de 21 pregões" in [button.label for button in at.sidebar.button]
    assert any("ainda não foi rodada" in info.value for info in at.info)


def test_missing_inputs_show_how_to_download(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_DATA_IN", str(tmp_path / "in"))
    monkeypatch.setenv("BACKTEST_DATA_OUT", str(tmp_path / "out"))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert at.error


def test_dashboard_shows_new_results_after_a_rerun(tmp_path, monkeypatch):
    raw = make_market()
    project = Project(synthetic_config(tmp_path))
    project.build_features(Universe(raw["snapshots"]), PriceData(clean_prices(raw["prices"])))
    run = project.run(40)
    market = project.features(with_dataset=False).market
    first = open_app(project, monkeypatch)
    dates = sorted(run.predictions["date"].unique())
    cut = run.predictions[run.predictions["date"] <= dates[59]]
    project.save_run(RunResults(40, cut, run.importance.loc[:dates[59]], run.info))
    second = open_app(project, monkeypatch)
    n_full = len(Portfolio(0.55).backtest(run.predictions, market).daily)
    n_cut = len(Portfolio(0.55).backtest(cut, market).daily)
    assert n_full != n_cut
    assert any(f"{n_full} dias de carteira" in caption.value for caption in first.caption)
    assert any(f"{n_cut} dias de carteira" in caption.value for caption in second.caption)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_app.py -q`
Expected: FAIL — o app antigo quebra ao importar `engine.results`.

- [ ] **Step 3: Implementar**

`app/dashboard.py`:

```python
"""Resultados do backtest. Rode com: streamlit run app/dashboard.py

Escolha a janela de treino na barra lateral: se ela ainda não foi rodada, o botão treina (em paralelo) e salva;
as janelas já rodadas abrem na hora e são comparadas na aba "Janelas".
"""
import dataclasses
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config.config import CONFIG
from engine import metrics
from engine.portfolio import Portfolio
from engine.project import Project

PROJECT = Project(dataclasses.replace(
    CONFIG,
    data_in=Path(os.environ.get("BACKTEST_DATA_IN", ROOT / CONFIG.data_in)),
    data_out=Path(os.environ.get("BACKTEST_DATA_OUT", ROOT / CONFIG.data_out)),
))
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

st.set_page_config(page_title="Backtest ML — S&P 500", layout="wide")


@st.cache_resource(show_spinner="Carregando as features…")
def load_features(data_out: str, stamp: int):
    return PROJECT.features(with_dataset=False)


@st.cache_resource(show_spinner=False)
def load_run(data_out: str, window: int, stamp: int):
    return PROJECT.load_run(window)


@st.cache_data(show_spinner="Recalculando a carteira…", max_entries=64)
def backtest(data_out: str, window: int, run_stamp: int, features_stamp: int, threshold: float, cost_bps: float):
    market = load_features(data_out, features_stamp).market
    return Portfolio(threshold, cost_bps).backtest(load_run(data_out, window, run_stamp).predictions, market)


@st.cache_data(show_spinner=False)
def evaluate(data_out: str, window: int, run_stamp: int, features_stamp: int):
    predictions = load_run(data_out, window, run_stamp).predictions
    realized = load_features(data_out, features_stamp).market["bench_fwd_ret"].dropna().index
    evaluated = predictions[predictions["date"].isin(realized)]
    return metrics.model_metrics(evaluated), metrics.monthly_auc(evaluated), metrics.daily_ic(evaluated)


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


def metrics_table(columns: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame({
        name: {label: fmt(values[key], pattern) for key, (label, pattern) in METRIC_FORMATS.items()}
        for name, values in columns.items()
    })


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
    st.header("Janela de treino")
    window = int(st.number_input("Pregões de treino (5 = 1 semana, 21 = 1 mês, 63 = 3 meses)",
                                 min_value=1, max_value=1000, value=CONFIG.train_window_days, step=1))
    if window not in PROJECT.runs() and st.button(f"Rodar janela de {window} pregões", type="primary"):
        bar = st.progress(0.0, text="Treinando…")
        try:
            PROJECT.run(window, progress=lambda done: bar.progress(done, text=f"Treinando… {done:.0%}"))
        except ValueError as error:
            st.error(str(error))
        else:
            st.rerun()
    st.caption(f"Janelas já rodadas: {', '.join(map(str, PROJECT.runs())) or 'nenhuma'}")
    st.header("Carteira")
    threshold = st.slider("Limiar de probabilidade", 0.40, 0.80, float(CONFIG.threshold), 0.005, format="%.3f")
    cost_bps = st.number_input("Custo de transação (bps por lado, em cada compra ou venda)", 0.0, 50.0, float(CONFIG.cost_bps), 0.5)
    st.caption(
        "Compra as ações com probabilidade prevista acima do limiar, com peso proporcional à probabilidade; "
        "se nenhuma passar, fica em caixa rendendo a T-bill (^IRX). Escolher o limiar ou a janela olhando este resultado "
        "é otimizar dentro do próprio backtest (data snooping)."
    )

if window not in PROJECT.runs():
    st.info(f"A janela de {window} pregões ainda não foi rodada: use o botão na barra lateral.")
    st.stop()

run_stamp = PROJECT.stamp(window)
run = load_run(OUT, window, run_stamp)
bt = backtest(OUT, window, run_stamp, features_stamp, threshold, cost_bps)
daily, positions = bt.daily, bt.positions
if daily.empty:
    st.warning("Nenhum dia de carteira com retorno realizado nesta janela.")
    st.stop()
summary, month_auc, ic = evaluate(OUT, window, run_stamp, features_stamp)
window_coverage = features.coverage.loc[daily["decision_date"].min():daily["decision_date"].max()]

st.title("Backtest ML — S&P 500")
st.caption(f"Janela de {window} pregões · {len(daily)} dias de carteira · {daily.index.min().date()} → "
           f"{daily.index.max().date()} · gerado em {run.info.get('generated_at', '?')}")
tab_overview, tab_day, tab_periods, tab_windows, tab_model = st.tabs(
    ["Visão geral", "Carteira por dia", "Recortes de período", "Janelas", "Modelo"])

with tab_overview:
    for col, (key, (label, pattern)) in zip(st.columns(3), MODEL_FORMATS.items()):
        col.metric(label, fmt(summary[key], pattern))
    st.caption(
        f"LightGBM treinado em cada pregão com os últimos {window} pregões e prevendo só o dia seguinte · em média "
        f"{window_coverage['n_eligible'].mean():.0f} ações elegíveis por dia, de "
        f"{window_coverage['n_members'].mean():.0f} membros do índice."
    )
    st.subheader("Carteira vs S&P 500 Total Return")
    st.dataframe(metrics_table(compare(daily)))
    line_chart(curves(daily, growth), "Capital (1,0 no início)")
    st.caption("Decisão após o fechamento de t, compra na abertura de t+1 e rebalanceamento na abertura seguinte "
               "(retornos abertura → abertura, com dividendos). Arraste ou use a roda do mouse para aproximar.")
    st.markdown("**Drawdown**")
    line_chart(curves(daily, metrics.drawdown), "Drawdown", height=220, area=True)
    st.markdown("**Nº de ações na carteira**")
    line_chart(daily[["n_positions"]].rename(columns={"n_positions": "Ações"}), "Ações", height=220, y_format=".0f")

with tab_day:
    holding_dates = list(daily.index)
    choice = st.selectbox(
        "Dia de carteira",
        options=range(len(holding_dates)),
        index=len(holding_dates) - 1,
        format_func=lambda i: (f"{holding_dates[i].date()}  ·  carteira {daily['gross'].iloc[i]:+.2%}  ·  "
                               f"S&P {daily['bench'].iloc[i]:+.2%}"),
    )
    day = daily.iloc[choice]
    holding_date, decision_date = holding_dates[choice], pd.Timestamp(day["decision_date"])
    cols = st.columns(5)
    cols[0].metric("Retorno bruto", f"{day['gross']:+.2%}")
    cols[1].metric("Retorno líquido", f"{day['net']:+.2%}")
    cols[2].metric(BENCH, f"{day['bench']:+.2%}")
    cols[3].metric("Ações", int(day["n_positions"]))
    cols[4].metric("Decisão", str(decision_date.date()))
    st.caption(f"Sinal calculado após o fechamento de {decision_date.date()}; compra na abertura de "
               f"{holding_date.date()} e rebalanceamento na abertura do pregão seguinte.")
    held = positions[positions["holding_date"] == holding_date].sort_values("weight", ascending=False)
    if held.empty:
        st.info(f"Nenhuma ação passou do limiar: carteira em caixa rendendo {day['rf']:.4%} no dia.")
    else:
        table = held[["ticker", "prob", "weight", "fwd_ret", "contribution"]].rename(columns={
            "ticker": "Ticker", "prob": "Prob. prevista", "weight": "Peso",
            "fwd_ret": "Retorno realizado", "contribution": "Contribuição",
        })
        st.dataframe(
            table.style.format({"Prob. prevista": "{:.3f}", "Peso": "{:.2%}", "Retorno realizado": "{:+.2%}",
                                "Contribuição": "{:+.3%}"}, na_rep="—")
            .map(shade, subset=["Retorno realizado", "Contribuição"]),
            hide_index=True, height=min(38 * (len(table) + 1), 520),
        )
        st.info(f"Soma das contribuições = {held['contribution'].sum():+.4%} = retorno bruto do dia ({day['gross']:+.4%}).")
        if day["n_missing"]:
            st.warning(f"{int(day['n_missing'])} ação(ões) sem retorno realizado (deslistagem/halt) contada(s) como 0%.")
    day_predictions = run.predictions[run.predictions["date"] == decision_date]
    st.markdown(f"**Probabilidades previstas em {decision_date.date()}** ({len(day_predictions)} ações do índice)")
    histogram = alt.Chart(day_predictions).mark_bar(opacity=0.8).encode(
        x=alt.X("prob:Q", bin=alt.Bin(maxbins=40), title="Probabilidade prevista de subir"),
        y=alt.Y("count():Q", title="Ações"),
    )
    rule = alt.Chart(pd.DataFrame({"limiar": [threshold]})).mark_rule(color="red", strokeDash=[4, 4]).encode(x="limiar:Q")
    st.altair_chart((histogram + rule).properties(width=CHART_WIDTH, height=220), width="content")
    near = day_predictions[day_predictions["prob"] <= threshold].nlargest(10, "prob")
    if len(near):
        st.markdown("**Quase entraram** (maiores probabilidades abaixo do limiar)")
        st.dataframe(
            near[["ticker", "prob", "fwd_ret"]]
            .rename(columns={"ticker": "Ticker", "prob": "Prob. prevista", "fwd_ret": "Retorno realizado"})
            .style.format({"Prob. prevista": "{:.3f}", "Retorno realizado": "{:+.2%}"}, na_rep="—"),
            hide_index=True,
        )

with tab_periods:
    st.caption("As mesmas métricas, recalculadas só no trecho final de cada período.")
    periods = {"1 mês": 1, "3 meses": 3, "6 meses": 6, "12 meses": 12, "Tudo": None}
    last = daily.index.max()

    def cut(months):
        return daily if months is None else daily[daily.index > last - pd.DateOffset(months=months)]

    blocks = {}
    for label, months in periods.items():
        part = cut(months)
        if len(part) >= 2:
            blocks.update({f"{label} · {name}": values for name, values in compare(part).items()})
    st.dataframe(metrics_table(blocks))
    selected = st.radio("Período do gráfico", list(periods), horizontal=True, index=len(periods) - 1)
    part = cut(periods[selected])
    if len(part) >= 2:
        line_chart(curves(part, growth), "Capital (1,0 no início do período)")

with tab_windows:
    st.caption("Todas as janelas já rodadas, no mesmo período, com o limiar e o custo da barra lateral (carteira líquida).")
    rows = {}
    for other in PROJECT.runs():
        other_stamp = PROJECT.stamp(other)
        other_daily = backtest(OUT, other, other_stamp, features_stamp, threshold, cost_bps).daily
        performance = metrics.performance_metrics(other_daily["net"], other_daily["rf"], other_daily["invested"])
        model = evaluate(OUT, other, other_stamp, features_stamp)[0]
        rows[f"{other} pregões"] = {
            **{label: fmt(performance[key], pattern) for key, (label, pattern) in METRIC_FORMATS.items()},
            **{label: fmt(model[key], pattern) for key, (label, pattern) in MODEL_FORMATS.items()},
        }
    st.dataframe(pd.DataFrame(rows).T)

with tab_model:
    st.markdown("**AUC por mês**")
    line_chart(month_auc.to_frame("AUC"), "AUC", height=220)
    st.markdown("**IC diário (média móvel de 63 dias)**")
    line_chart(ic.rolling(63, min_periods=20).mean().to_frame("IC"), "IC", height=220)
    st.markdown("**Importância das features** (média do ganho normalizado nos retreinos)")
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
        st.write(f"Retornos diários com |r| > 50% entre membros (erro de dado ou spin-off): {quality['n_extreme_returns']}")
        if quality["extreme_returns"]:
            st.dataframe(pd.DataFrame(quality["extreme_returns"]), hide_index=True)
        st.write(f"Posições sem retorno realizado (contadas como 0%): {int(daily['n_missing'].sum())}")
        st.json(quality["date_ranges"])
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest -q`
Expected: 78 passed.

- [ ] **Step 5: Subir o app de verdade (headless) sobre os runs reais**

```bash
venv/bin/streamlit run app/dashboard.py --server.headless true --server.port 8599 > /tmp/claude-streamlit.log 2>&1 &
sleep 10; curl -s http://localhost:8599/_stcore/health; echo; grep -iE "error|traceback" /tmp/claude-streamlit.log; kill %1
```

Expected: `ok` e nenhum erro. A checagem visual fica com o usuário.

---

### Task 11: Documentação, limpeza, simplificação e verificação final

**Files:**
- Modify: `README.md`, `.claude/CLAUDE.md`
- Delete: `.superpowers/sdd/2026-09-26-ml-backtest-redesign/`

- [ ] **Step 1: README curto**

Substituir `README.md` por:

````markdown
# Stock Behavior Prediction

Every trading day, after the close, a LightGBM classifier estimates the probability that each stock in the S&P 500 **on that day** goes up the next day. A long-only portfolio buys, at the next open, the stocks whose probability is above a threshold, weighted by that probability; if none qualifies, it holds cash at the 13-week T-bill rate.

## Objective and scope

- **Objective:** test whether daily direction predictions from price history alone beat the S&P 500, quickly and for several training windows.
- **Training window:** the model is retrained every day on the last *X* trading days (5 = one week, 21 = one month, 63 = three months…) and always predicts only the next day. All windows are evaluated over the same period (from 2020-01-02).
- **Data:** Yahoo Finance prices (split-adjusted, dividends added separately) and point-in-time S&P 500 constituents ([fja05680/sp500](https://github.com/fja05680/sp500)). Price history only, no macro data.
- **No lookahead:** a decision after the close of *t* uses only data dated on or before *t*; the model trains only on labels already known then. The tests enforce it.
- **Out of scope:** live trading, hyperparameter tuning, and delisted companies (Yahoo has no data for them; the app lists them).

## Benchmark

S&P 500 Total Return (`^SP500TR`), open to open, the same holding period as the portfolio.

## How to run

```bash
python3 -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python main.py download          # latest constituents and prices
venv/bin/python main.py features          # features, once per download
venv/bin/python main.py run --window 21   # walk-forward with a 21-day training window
venv/bin/streamlit run app/dashboard.py   # results; pick another window in the sidebar and click "Rodar"
venv/bin/python -m pytest
```

Defaults (window, threshold, transaction cost, backtest start) live in `config/config.py`.
````

- [ ] **Step 2: CLAUDE.md**

Em `.claude/CLAUDE.md`, substituir a primeira linha de descrição, a seção `## Status`, as linhas de `## Working conventions` depois da primeira (sobre commits) e a seção `## Engine layout` por:

```markdown
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

## Engine layout

- `engine/universe.py` — `Universe`: point-in-time constituents, download.
- `engine/prices.py` — `PriceData`: yfinance download (split-adjusted, dividends separate), calendar, panel, returns.
- `engine/features.py` — `FeatureBuilder`: stock, cross-section, correlation peers and market features (34, causal).
- `engine/dataset.py` — the (date, ticker) training table.
- `engine/walk_forward.py` — `WalkForward`: daily retraining on the last X days, decision dates in parallel (joblib).
- `engine/portfolio.py`, `engine/metrics.py` — `Portfolio` (threshold, probability weights, cash at ^IRX, costs) and metrics.
- `engine/project.py` — `Project`: download, cached features (`data/out/features`), one run per window (`data/out/runs/wNNN`).
```

(a seção `## Plugins available` continua).

- [ ] **Step 3: Apagar o workspace antigo**

```bash
rm -rf .superpowers/sdd/2026-09-26-ml-backtest-redesign
```

- [ ] **Step 4: Passada do code-simplifier**

Despachar o agente `code-simplifier:code-simplifier` sobre `engine/*.py`, `main.py`, `app/dashboard.py` e `config/config.py`, com estas restrições:
- não mudar comportamento;
- todos os números bit a bit iguais;
- não tocar na convenção de tempo nem na aritmética da janela;
- manter as assinaturas públicas;
- sem comentários novos, sem paralelização nova e sem commit;
- não tocar em `tests/`, `data/` ou nas notas do usuário no fim do `main.py`;
- manter `venv/bin/python -m pytest -q` verde.

Depois, verificar o diff dos arquivos tocados e rodar `venv/bin/python main.py run --window 5`. As previsões têm que ser idênticas às de antes da simplificação: guardar uma cópia de `data/out/runs/w005/predictions.parquet` antes e comparar com `pd.testing.assert_frame_equal(check_exact=True)`.

- [ ] **Step 5: Verificação final e revisão independente**

Com superpowers:verification-before-completion:

```bash
venv/bin/python -m pytest -q
venv/bin/python -m pytest tests/test_project.py::test_predictions_do_not_depend_on_future_data tests/test_features.py tests/test_walk_forward.py -v
git status --short
```

Depois, superpowers:requesting-code-review (revisor novo, modelo mais capaz) sobre o diff completo contra o spec da rodada 2. O foco é lookahead, integridade dos dados sem aliases duvidosos, paralelo = sequencial e os 5 itens de Review Focus.
