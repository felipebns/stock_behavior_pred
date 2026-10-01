# ML Backtest Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Substituir o projeto antigo por um pipeline que, todo pregão, prevê com LightGBM (janela rolante de X pregões) a probabilidade de cada ação do S&P 500 point-in-time subir no próximo dia de carteira, monta uma carteira long-only ponderada pela probabilidade acima de um limiar e mostra o backtest num app Streamlit — sem lookahead, verificado por testes.

**Architecture:** Funções puras sobre DataFrames em `engine/` (dados → features → dataset → walk-forward → carteira → métricas), orquestradas por `engine/pipeline.py` e expostas por `main.py` (`download`, `run`). O walk-forward (caro) roda uma vez e grava as previsões em `data/out/`; a carteira é recalculada a partir delas pela mesma função no `main.py` e no app.

**Tech Stack:** Python 3.14 (venv local), pandas 3, numpy 2, pyarrow, scikit-learn, LightGBM, yfinance, xlrd, Streamlit + Altair, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-ml-backtest-redesign-design.md`

## Global Constraints

- **Nunca rodar `git commit` nem `git push`** (bloqueado em `.claude/settings.json`); os commits são do usuário. Este plano não tem passos de commit. *(Regra revogada em 2026-10-01: commits são liberados dentro do repositório; `git push` continua bloqueado — ver `.claude/CLAUDE.md`.)*
- Comandos sempre pelo venv do projeto: `venv/bin/python -m pytest ...`, `venv/bin/python main.py ...`.
- Nenhum código de `data/reference/` é importado ou copiado; ele será apagado.
- Preços: `yfinance.download(..., auto_adjust=False, actions=True)`; o `Adj Close` é descartado dentro de `download_prices` e nunca é usado.
- Convenção de tempo: decisão após o fechamento de *t*; features só com dados de data ≤ *t* (GPR ≤ *t* − 7 dias corridos); compra na abertura de *t+1* e rebalanceia na abertura de *t+2*; `fwd_ret(i, t) = (Open[t+2] + Div[t+2]) / Open[t+1] − 1`; alvo `label = fwd_ret > 0`; treino da decisão *t* usa exatamente as linhas *s* ∈ [*t* − X − 1, *t* − 2].
- Universo em *t*: snapshot de constituintes mais recente com data ≤ *t*, tickers no formato Yahoo, `TICKER_ALIASES` aplicados.
- Benchmark `^SP500TR` (abertura→abertura); taxa livre de risco `^IRX`: `rf_daily = IRX / 100 / 252`, as-of ≤ *t*.
- Carteira: `prob > limiar`, pesos `prob_i / Σ prob`; ninguém acima → 100% caixa rendendo `rf_daily`. Custos só na Task 14.
- Sem paralelização própria (joblib/multiprocessing); só as threads internas do LightGBM.
- Defaults: `train_window_days=252`, `threshold=0.55`, `cost_bps=5`, `gpr_lag_days=7`, `macro_max_staleness_days=5`, `min_history_days=63`, `price_start="2018-06-01"`, `ff_horizon_months=12`, `ff_max_gap_days=45`, `ff_change_days=21`.
- Código em inglês, sem comentários (só quando o porquê não é óbvio); interface do app em português; README e CLAUDE.md em inglês (idioma atual deles).

## Review Focus

1. yfinance devolve um lote vazio ou tickers só com NaN (rate limit, deslistadas) → o download continua e esses tickers saem na lista de falhas. *(Task 3: `test_download_reports_tickers_without_data_and_survives_empty_batches`, `test_download_reports_all_nan_tickers_as_failed`)*
2. Um pregão dentro do backtest sem nenhuma previsão → vira dia de caixa, não some da série. *(Task 10: `test_daily_returns_cash_and_missing`)*
3. Limiar acima de todas as probabilidades → backtest 100% caixa; métricas e app não quebram com carteira vazia. *(Task 10: `test_threshold_never_met_is_all_cash`; Task 13: `test_dashboard_survives_threshold_nobody_passes`)*
4. Futuros acabam antes dos preços (07/2026 vs 09/2026) → dataset para no fim do macro + tolerância; últimas decisões sem retorno realizado ficam fora do backtest. *(Task 9: `test_dataset_stops_when_macro_data_goes_stale`; Task 10: fixture com último dia sem benchmark)*
5. Linhas (data, ticker) duplicadas vindas do Yahoo → deduplicadas antes do pivot (o pivot quebraria). *(Task 3: `test_clean_prices_fixes_bad_values_and_duplicates`)*

---

## File Structure

```
config/__init__.py          vazio
config/config.py            Config (dataclass congelada), TICKER_ALIASES, LGBM_PARAMS, CONFIG
engine/__init__.py          vazio
engine/data/__init__.py     vazio
engine/data/universe.py     composição point-in-time + aliases → matriz de membros
engine/data/prices.py       download yfinance, limpeza, calendário, painel largo, retornos
engine/data/macro.py        Fed Funds futures, GPR, alinhamento as-of, taxa livre de risco
engine/features.py          36 features causais + listas de nomes
engine/dataset.py           tabela longa (date, ticker) com features, fwd_ret, label
engine/model.py             fábrica do LGBMClassifier
engine/walk_forward.py      retreino diário e previsão
engine/results.py           RunResults + gravar/ler data/out
engine/pipeline.py          MarketInputs, load_inputs, build_feature_set, run_pipeline
engine/portfolio.py         seleção, pesos, retornos diários (e custos na Task 14)
engine/metrics.py           métricas de desempenho e do modelo
main.py                     CLI: download | run
app/dashboard.py            Streamlit
tests/synthetic.py          mercado sintético (preços, constituintes, futuros, GPR)
tests/test_*.py             um por módulo + test_pipeline.py (lookahead ponta a ponta) + test_app.py
```

---

### Task 1: Limpeza, venv e config

**Files:**
- Delete: `engine/algorithms/`, `engine/backtesting/`, `engine/log/`, `engine/pipeline/`, `engine/plotting/`, `engine/reproducibility/`, `engine/stock/`, `engine/strategies/`, `engine/validation/`, `output/`, `tests/`, `main.py`
- Modify: `requirements.txt`, `pyproject.toml`, `.gitignore`, `config/config.py`, `engine/__init__.py`
- Create: `config/__init__.py`, `engine/data/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `config.config.Config` (campos da seção Global Constraints + `data_in`, `data_out`, nomes de arquivos, `benchmark_ticker`, `risk_free_ticker`, `progress_every`, `lgbm_params`), `CONFIG = Config()`, `TICKER_ALIASES: dict[str, str]`, `LGBM_PARAMS: dict`.

- [ ] **Step 1: Apagar o código antigo**

```bash
git status --short
rm -rf engine/algorithms engine/backtesting engine/log engine/pipeline engine/plotting engine/reproducibility engine/stock engine/strategies engine/validation output tests main.py
```

Expected: `git status --short` antes mostra só `?? .claude/`, `?? data/`, `?? docs/` (nada modificado que possa se perder). Depois do `rm`, `ls engine` mostra só `__init__.py`.

- [ ] **Step 2: Dependências**

`requirements.txt` (substitui o freeze antigo):

```
pandas>=3.0
numpy>=2.0
pyarrow>=20
scikit-learn>=1.7
lightgbm>=4.6
yfinance>=1.2
xlrd>=2.0
streamlit>=1.60
altair>=5.5
pytest>=8.0
```

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "quantfund_engine"
version = "0.2.0"
description = "Daily S&P 500 direction model (LightGBM, rolling window) and long-only backtest"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "pandas>=3.0",
    "numpy>=2.0",
    "pyarrow>=20",
    "scikit-learn>=1.7",
    "lightgbm>=4.6",
    "yfinance>=1.2",
    "xlrd>=2.0",
    "streamlit>=1.60",
    "altair>=5.5",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
where = ["."]
include = ["engine*", "config*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
pythonpath = ["."]
```

`.gitignore`:

```
venv
.env
__pycache__
*.pdf
*.log
.pytest_cache/
data/out/
data/in/prices.parquet
```

- [ ] **Step 3: Criar o venv e instalar**

```bash
python3 -m venv venv
venv/bin/pip install -q -r requirements.txt
venv/bin/python -c "import pandas, lightgbm, streamlit, yfinance, xlrd; print(pandas.__version__, lightgbm.__version__, streamlit.__version__)"
```

Expected: imprime as versões (pandas 3.x, lightgbm 4.x, streamlit 1.6x), sem erro.

- [ ] **Step 4: Escrever o teste do config**

`tests/test_config.py`:

```python
import dataclasses

import pytest

from config.config import CONFIG, TICKER_ALIASES


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIG.threshold = 0.6


def test_aliases_point_to_final_tickers():
    assert not set(TICKER_ALIASES.values()) & set(TICKER_ALIASES)


def test_lookahead_sensitive_defaults():
    assert CONFIG.gpr_lag_days == 7
    assert CONFIG.benchmark_ticker == "^SP500TR"
    assert CONFIG.risk_free_ticker == "^IRX"
```

- [ ] **Step 5: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `ImportError: cannot import name 'TICKER_ALIASES'`.

- [ ] **Step 6: Implementar o config**

`config/__init__.py` e `engine/data/__init__.py`: arquivos vazios. `engine/__init__.py`: esvaziar.

`config/config.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path

# Old ticker in the constituents file -> ticker Yahoo uses today for the same security.
# Only 1:1 renames visible in the constituents file itself (same-date swap, successor has history).
TICKER_ALIASES: dict[str, str] = {
    "HRS": "LHX", "TMK": "GL", "BHGE": "BKR", "JEC": "J", "CTL": "LUMN",
    "MYL": "VTRS", "WLTW": "WTW", "DISCA": "WBD", "BLL": "BALL", "ANTM": "ELV",
    "SYMC": "GEN", "NLOK": "GEN", "PKI": "RVTY", "RE": "EG", "ABC": "COR",
    "HCP": "DOC", "PEAK": "DOC", "FLT": "CPAY", "CBS": "PSKY", "VIAC": "PSKY",
    "PARA": "PSKY", "FI": "FISV", "MMC": "MRSH", "BK": "BNY", "UTX": "RTX",
    "ARNC": "HWM", "DWDP": "DD",
}

LGBM_PARAMS: dict = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 500,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "importance_type": "gain",
    "random_state": 42,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}


@dataclass(frozen=True)
class Config:
    data_in: Path = Path("data/in")
    data_out: Path = Path("data/out")
    constituents_file: str = "sp500_historical_constituents.csv"
    ff_futures_file: str = "ff_futures_daily.csv"
    gpr_file: str = "data_gpr_daily_recent.xls"
    prices_file: str = "prices.parquet"
    price_start: str = "2018-06-01"
    benchmark_ticker: str = "^SP500TR"
    risk_free_ticker: str = "^IRX"
    train_window_days: int = 252
    threshold: float = 0.55
    cost_bps: float = 5.0
    gpr_lag_days: int = 7
    macro_max_staleness_days: int = 5
    min_history_days: int = 63
    ff_horizon_months: int = 12
    ff_max_gap_days: int = 45
    ff_change_days: int = 21
    progress_every: int = 50
    lgbm_params: dict = field(default_factory=lambda: dict(LGBM_PARAMS))


CONFIG = Config()
```

- [ ] **Step 7: Rodar e ver passar**

Run: `venv/bin/python -m pytest -v`
Expected: 3 passed.

---

### Task 2: Universo point-in-time

**Files:**
- Create: `engine/data/universe.py`, `tests/test_universe.py`

**Interfaces:**
- Produces:
  - `to_yahoo(ticker: str) -> str`
  - `parse_constituents(raw: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame` — colunas `date` (datetime) e `tickers` (`list[str]`), ordenado por data.
  - `load_constituents(path, aliases) -> pd.DataFrame` (mesmo formato)
  - `members_on(snapshots, date) -> list[str]`
  - `tickers_since(snapshots, start) -> list[str]` (ordenado)
  - `membership_matrix(snapshots, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame` (bool, `dates × tickers`)

- [ ] **Step 1: Escrever os testes**

`tests/test_universe.py`:

```python
import pandas as pd
import pytest

from engine.data.universe import (
    load_constituents,
    members_on,
    membership_matrix,
    parse_constituents,
    tickers_since,
    to_yahoo,
)

RAW = pd.DataFrame({
    "date": ["2020-01-02", "2020-02-03", "2020-03-02"],
    "tickers": ["AAA,BRK.B,OLD", "AAA,BRK.B,NEW", "BRK.B,NEW,ZZZ"],
})
ALIASES = {"OLD": "NEW"}


@pytest.fixture
def snapshots():
    return parse_constituents(RAW, ALIASES)


def test_to_yahoo_replaces_dots():
    assert to_yahoo("BRK.B") == "BRK-B"
    assert to_yahoo(" BF.B ") == "BF-B"


def test_parse_applies_yahoo_format_and_aliases(snapshots):
    assert snapshots["tickers"].iloc[0] == ["AAA", "BRK-B", "NEW"]
    assert snapshots["tickers"].iloc[1] == ["AAA", "BRK-B", "NEW"]


def test_parse_drops_duplicates_created_by_aliases():
    raw = pd.DataFrame({"date": ["2020-01-02"], "tickers": ["OLD,NEW,AAA"]})
    assert parse_constituents(raw, ALIASES)["tickers"].iloc[0] == ["NEW", "AAA"]


def test_members_on_uses_last_snapshot_not_after_date(snapshots):
    assert members_on(snapshots, "2020-01-01") == []
    assert members_on(snapshots, "2020-01-02") == ["AAA", "BRK-B", "NEW"]
    assert members_on(snapshots, "2020-02-28") == ["AAA", "BRK-B", "NEW"]
    assert members_on(snapshots, "2020-03-02") == ["BRK-B", "NEW", "ZZZ"]


def test_tickers_since_includes_snapshot_in_force_at_start(snapshots):
    assert tickers_since(snapshots, "2020-02-15") == ["AAA", "BRK-B", "NEW", "ZZZ"]
    assert tickers_since(snapshots, "2020-03-10") == ["BRK-B", "NEW", "ZZZ"]


def test_membership_matrix_is_point_in_time(snapshots):
    dates = pd.DatetimeIndex(pd.to_datetime(["2019-12-31", "2020-01-02", "2020-02-28", "2020-03-02", "2020-03-03"]))
    matrix = membership_matrix(snapshots, dates, ["AAA", "NEW", "ZZZ", "NOPE"])
    assert not matrix.loc["2019-12-31"].any()
    assert matrix.loc["2020-02-28"].tolist() == [True, True, False, False]
    assert matrix.loc["2020-03-02"].tolist() == [False, True, True, False]
    assert matrix.loc["2020-03-03", "ZZZ"]


def test_load_constituents_reads_csv(tmp_path):
    path = tmp_path / "constituents.csv"
    RAW.to_csv(path, index=False)
    loaded = load_constituents(path, ALIASES)
    assert loaded["date"].iloc[0] == pd.Timestamp("2020-01-02")
    assert loaded["tickers"].iloc[2] == ["BRK-B", "NEW", "ZZZ"]
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_universe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.data.universe'`.

- [ ] **Step 3: Implementar**

`engine/data/universe.py`:

```python
"""S&P 500 membership as of each date (last snapshot dated on or before it)."""
import numpy as np
import pandas as pd


def to_yahoo(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def parse_constituents(raw: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    snapshots = raw.assign(date=pd.to_datetime(raw["date"])).sort_values("date").reset_index(drop=True)

    def normalize(field: str) -> list[str]:
        mapped = (aliases.get(to_yahoo(t), to_yahoo(t)) for t in field.split(",") if t.strip())
        return list(dict.fromkeys(mapped))

    return pd.DataFrame({"date": snapshots["date"], "tickers": snapshots["tickers"].map(normalize)})


def load_constituents(path, aliases: dict[str, str]) -> pd.DataFrame:
    return parse_constituents(pd.read_csv(path), aliases)


def members_on(snapshots: pd.DataFrame, date) -> list[str]:
    pos = snapshots["date"].searchsorted(pd.Timestamp(date), side="right") - 1
    return [] if pos < 0 else list(snapshots["tickers"].iloc[pos])


def tickers_since(snapshots: pd.DataFrame, start) -> list[str]:
    first = max(snapshots["date"].searchsorted(pd.Timestamp(start), side="right") - 1, 0)
    return sorted(set().union(*snapshots["tickers"].iloc[first:]))


def membership_matrix(snapshots: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    column = {ticker: j for j, ticker in enumerate(tickers)}
    table = np.zeros((len(snapshots), len(tickers)), dtype=bool)
    for i, members in enumerate(snapshots["tickers"]):
        table[i, [column[t] for t in members if t in column]] = True
    pos = snapshots["date"].searchsorted(dates, side="right") - 1
    values = table[np.clip(pos, 0, None)]
    values[pos < 0] = False
    return pd.DataFrame(values, index=dates, columns=tickers)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_universe.py -v`
Expected: 7 passed.

---

### Task 3: Preços (download, limpeza, painel, retornos)

**Files:**
- Create: `engine/data/prices.py`, `tests/test_prices.py`

**Interfaces:**
- Produces:
  - `PRICE_FIELDS = ["open", "high", "low", "close", "volume", "dividends", "splits"]`, `PANEL_FIELDS = ["open", "high", "low", "close", "volume", "dividends"]`
  - `download_prices(tickers: list[str], start: str, end: str | None = None, batch_size: int = 100, downloader=None) -> tuple[pd.DataFrame, list[str]]` — formato longo `date, ticker, *PRICE_FIELDS` + tickers sem dado.
  - `clean_prices(prices) -> pd.DataFrame`, `save_prices(prices, path)`, `load_prices(path) -> pd.DataFrame`
  - `trading_calendar(prices, benchmark: str, start=None) -> pd.DatetimeIndex` (nome `date`)
  - `PricePanel` (dataclass congelada com `open, high, low, close, volume, dividends`, cada um `calendar × tickers`)
  - `build_panel(prices, calendar, tickers) -> PricePanel`
  - `daily_total_return(close, dividends) -> pd.DataFrame`
  - `forward_open_return(open_, dividends=None)` — DataFrame ou Series; linha *t* = `(open[t+2] + div[t+2]) / open[t+1] − 1`

- [ ] **Step 1: Escrever os testes**

`tests/test_prices.py`:

```python
import numpy as np
import pandas as pd
import pytest

from engine.data import prices as px

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

    prices, failed = px.download_prices(["AAA", "BBB"], "2024-01-01", downloader=fake)
    assert list(prices.columns) == ["date", "ticker", *px.PRICE_FIELDS]
    assert failed == []
    assert len(prices) == 6
    assert prices.loc[prices["ticker"] == "AAA", "close"].tolist() == [10.0, 11.0, 12.0]


def test_download_reports_tickers_without_data_and_survives_empty_batches():
    def fake(tickers, **kwargs):
        return pd.DataFrame() if "GONE" in tickers else yahoo_frame({t: bars() for t in tickers})

    prices, failed = px.download_prices(["AAA", "GONE"], "2024-01-01", batch_size=1, downloader=fake)
    assert failed == ["GONE"]
    assert set(prices["ticker"]) == {"AAA"}


def test_download_reports_all_nan_tickers_as_failed():
    def fake(tickers, **kwargs):
        return yahoo_frame({"AAA": bars(), "NAN": {k: [np.nan] * 3 for k in bars()}})

    prices, failed = px.download_prices(["AAA", "NAN"], "2024-01-01", downloader=fake)
    assert failed == ["NAN"]
    assert set(prices["ticker"]) == {"AAA"}


def test_download_fails_loudly_when_nothing_comes_back():
    with pytest.raises(RuntimeError):
        px.download_prices(["AAA"], "2024-01-01", downloader=lambda tickers, **kwargs: pd.DataFrame())


def test_clean_prices_fixes_bad_values_and_duplicates():
    raw = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"],
        "ticker": ["AAA"] * 4,
        "open": [10.0, 10.0, 0.0, 11.0], "high": 11.0, "low": 9.0,
        "close": [10.0, 10.5, 10.2, np.nan], "volume": 100.0,
        "dividends": [np.nan, np.nan, 0.5, 0.0], "splits": [np.nan, 0.0, 0.0, 0.0],
    })
    out = px.clean_prices(raw)
    assert len(out) == 2
    assert out.loc[0, "close"] == 10.5
    assert np.isnan(out.loc[1, "open"])
    assert out["dividends"].tolist() == [0.0, 0.5]
    assert out["splits"].tolist() == [0.0, 0.0]


def test_trading_calendar_comes_from_benchmark():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-05"]),
        "ticker": ["^B", "^B", "AAA"], "close": 1.0,
    })
    assert list(px.trading_calendar(prices, "^B")) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert list(px.trading_calendar(prices, "^B", start="2024-01-03")) == [pd.Timestamp("2024-01-03")]


def test_build_panel_aligns_to_calendar():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
        "ticker": ["AAA", "AAA", "BBB"], "open": 1.0, "high": 1.0, "low": 1.0,
        "close": [10.0, 11.0, 20.0], "volume": 5.0, "dividends": [0.0, 0.3, 0.0], "splits": 0.0,
    })
    calendar = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), name="date")
    panel = px.build_panel(prices, calendar, ["AAA", "BBB", "CCC"])
    assert panel.close.shape == (3, 3)
    assert np.isnan(panel.close.loc["2024-01-03", "BBB"])
    assert panel.dividends.loc["2024-01-03", "AAA"] == 0.3
    assert panel.dividends.isna().sum().sum() == 0


def test_daily_total_return_includes_dividends_and_bridges_gaps():
    close = pd.DataFrame({"A": [100.0, 102.0, 101.0], "B": [100.0, np.nan, 110.0]})
    dividends = pd.DataFrame({"A": [0.0, 0.0, 1.0], "B": [0.0, 0.0, 0.0]})
    r = px.daily_total_return(close, dividends)
    assert np.isnan(r.loc[0, "A"])
    assert r.loc[1, "A"] == pytest.approx(0.02)
    assert r.loc[2, "A"] == pytest.approx((101.0 + 1.0) / 102.0 - 1.0)
    assert np.isnan(r.loc[1, "B"])
    assert r.loc[2, "B"] == pytest.approx(0.10)


def test_forward_open_return_is_next_open_to_following_open():
    open_ = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0]})
    dividends = pd.DataFrame({"A": [0.0, 0.0, 0.5, 0.0]})
    fwd = px.forward_open_return(open_, dividends)
    assert fwd.loc[0, "A"] == pytest.approx((12.0 + 0.5) / 11.0 - 1.0)
    assert fwd.loc[1, "A"] == pytest.approx(13.0 / 12.0 - 1.0)
    assert fwd.loc[2:, "A"].isna().all()


def test_forward_open_return_is_nan_when_entry_open_missing():
    assert np.isnan(px.forward_open_return(pd.Series([10.0, np.nan, 12.0])).iloc[0])


def test_save_and_load_roundtrip(tmp_path):
    prices = px.clean_prices(pd.DataFrame({
        "date": ["2024-01-02"], "ticker": ["AAA"], "open": 1.0, "high": 1.0, "low": 1.0,
        "close": 1.0, "volume": 1.0, "dividends": 0.0, "splits": 0.0,
    }))
    px.save_prices(prices, tmp_path / "sub" / "prices.parquet")
    pd.testing.assert_frame_equal(px.load_prices(tmp_path / "sub" / "prices.parquet"), prices)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_prices.py -v`
Expected: FAIL — `ImportError: cannot import name 'prices' from 'engine.data'`.

- [ ] **Step 3: Implementar**

`engine/data/prices.py`:

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


def download_prices(tickers: list[str], start: str, end: str | None = None,
                    batch_size: int = 100, downloader=None) -> tuple[pd.DataFrame, list[str]]:
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
    return prices, sorted(set(tickers) - set(prices["ticker"]))


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


def save_prices(prices: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path, index=False)


def load_prices(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def trading_calendar(prices: pd.DataFrame, benchmark: str, start=None) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(prices.loc[prices["ticker"] == benchmark, "date"].unique()).sort_values()
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    return dates.rename("date")


@dataclass(frozen=True)
class PricePanel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    dividends: pd.DataFrame


def build_panel(prices: pd.DataFrame, calendar: pd.DatetimeIndex, tickers: list[str]) -> PricePanel:
    subset = prices[prices["ticker"].isin(tickers)]
    tables = {
        field: subset.pivot(index="date", columns="ticker", values=field).reindex(index=calendar, columns=tickers)
        for field in PANEL_FIELDS
    }
    tables["dividends"] = tables["dividends"].fillna(0.0)
    return PricePanel(**tables)


def daily_total_return(close: pd.DataFrame, dividends: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close with the dividend going ex that day; after a missing day, it spans back to the last close."""
    return (close + dividends) / close.ffill().shift(1) - 1.0


def forward_open_return(open_, dividends=None):
    """Row t: buy at the open of t+1, sell at the open of t+2, keeping the dividend that goes ex on t+2."""
    received = 0.0 if dividends is None else dividends.shift(-2)
    return (open_.shift(-2) + received) / open_.shift(-1) - 1.0
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_prices.py -v`
Expected: 11 passed.

---

### Task 4: `main.py download` e download real

**Files:**
- Create: `main.py`

**Interfaces:**
- Consumes: `universe.load_constituents`, `universe.tickers_since`, `px.download_prices`, `px.save_prices`, `CONFIG`, `TICKER_ALIASES`.
- Produces: `data/in/prices.parquet` (formato longo de `download_prices`, inclui `^SP500TR` e `^IRX`); função `download(config: Config) -> None`; `main()` com subcomando `download`.

- [ ] **Step 1: Implementar o CLI com `download`**

`main.py`:

```python
"""`python main.py download` baixa os preços; `python main.py run` roda features, walk-forward e backtest."""
import argparse

from config.config import CONFIG, TICKER_ALIASES, Config
from engine.data import prices as px
from engine.data import universe


def download(config: Config) -> None:
    snapshots = universe.load_constituents(config.data_in / config.constituents_file, TICKER_ALIASES)
    tickers = universe.tickers_since(snapshots, config.price_start) + [config.benchmark_ticker, config.risk_free_ticker]
    prices, failed = px.download_prices(tickers, start=config.price_start)
    path = config.data_in / config.prices_file
    px.save_prices(prices, path)
    print(f"{prices['ticker'].nunique()} tickers salvos em {path} "
          f"({prices['date'].min().date()} → {prices['date'].max().date()})")
    print(f"Sem dados no Yahoo ({len(failed)}): {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML do S&P 500")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="baixa preços sem ajuste de dividendos (yfinance)")
    args = parser.parse_args()
    if args.command == "download":
        download(CONFIG)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodar o download real**

Run (timeout 10 min): `venv/bin/python main.py download`
Expected: por volta de 600 tickers salvos, de 2018-06-01 até o último pregão, e uma lista de uns 70–100 tickers sem dado, todos de empresas deslistadas (AGN, CELG, TWTR, ATVI...). **Se aparecerem tickers atuais na lista de falhas** (AAPL, MSFT etc.), é rate limit do Yahoo: espere alguns minutos e rode de novo.

- [ ] **Step 3: Checar a qualidade do que foi baixado**

```bash
venv/bin/python - <<'EOF'
import pandas as pd
from config.config import CONFIG, TICKER_ALIASES
from engine.data import prices as px, universe
p = px.load_prices(CONFIG.data_in / CONFIG.prices_file)
print(p[p.ticker.isin(["^SP500TR", "^IRX"])].groupby("ticker")["date"].agg(["min", "max", "count"]))
print(p.loc[(p.ticker == "AAPL") & (p.dividends > 0) & p.date.between("2020-05-01", "2020-09-30"), ["date", "close", "dividends"]])
print("colunas:", list(p.columns))
snap = universe.load_constituents(CONFIG.data_in / CONFIG.constituents_file, TICKER_ALIASES)
have = set(p.ticker)
for d in ["2019-12-02", "2021-01-04", "2023-01-03", "2025-01-02", "2026-06-30"]:
    members = universe.members_on(snap, d)
    print(d, len(members), "sem dado:", sum(t not in have for t in members))
EOF
```

Expected:
- `^SP500TR` e `^IRX` vão de 2018-06-01 até ~2026-09-25.
- Os dividendos do AAPL em 2020 aparecem como ~0,205 (já na base pós-split) e o close fica em torno de 110–113.
- Não existe coluna de adj close.
- O número de membros sem dado cai ao longo do tempo (dezenas em 2019 → 0 em 2026) e fica abaixo dos 68 de 2019 medidos antes dos aliases.
- Se algo divergir, parar e investigar com superpowers:systematic-debugging.

---

### Task 5: Macro (Fed Funds futures, GPR, taxa livre de risco)

**Files:**
- Create: `engine/data/macro.py`, `tests/test_macro.py`

**Interfaces:**
- Produces:
  - `FF_FEATURES = ["ff_rate_12m", "ff_rate_12m_chg", "ff_slope"]`, `GPR_FEATURES = ["gpr_level", "gpr_trend", "gpr_threat_act"]`
  - `parse_ff_futures(raw) -> pd.DataFrame` (`date, symbol, expiration, close`), `load_ff_futures(path)`
  - `ff_rates(bars, horizon_months=12, max_gap_days=45, change_days=21) -> pd.DataFrame` — índice = data dos futuros; colunas `ff_rate_12m, ff_rate_front, ff_rate_12m_chg, ff_slope`
  - `parse_gpr(raw) -> pd.DataFrame` (índice `date`, colunas `gpr, act, threat`), `load_gpr(path)`
  - `gpr_features(gpr) -> pd.DataFrame` (índice diário corrido, colunas `GPR_FEATURES`)
  - `align_asof(frame, dates, lag_days=0, max_staleness_days=None) -> pd.DataFrame` (índice = `dates`)
  - `risk_free_daily(prices, ticker, dates, max_staleness_days) -> pd.Series` (nome `rf_daily`)

- [ ] **Step 1: Escrever os testes**

`tests/test_macro.py`:

```python
import numpy as np
import pandas as pd
import pytest

from engine.data import macro


def ff_bar(date, expiration, close):
    return {"date": pd.Timestamp(date), "symbol": "ZQ", "expiration": pd.Timestamp(expiration), "close": close}


def gpr_frame(columns: dict, periods: int = 60) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=periods, freq="D", name="date")
    return pd.DataFrame({k: np.broadcast_to(np.asarray(v, dtype=float), periods).copy() for k, v in columns.items()}, index=index)


def test_load_ff_futures_normalizes_columns(tmp_path):
    raw = pd.DataFrame({"Date": ["2020-01-02"], "symbol": ["ZQF0"], "expiration": ["2020-01-31"],
                        "Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [98.45], "Volume": [10]})
    raw.to_csv(tmp_path / "ff.csv", index=False)
    bars = macro.load_ff_futures(tmp_path / "ff.csv")
    assert list(bars.columns) == ["date", "symbol", "expiration", "close"]
    assert bars["date"].iloc[0] == pd.Timestamp("2020-01-02")


def test_ff_rates_picks_12m_contract_front_month_and_drops_sundays():
    bars = pd.DataFrame([
        ff_bar("2020-01-02", "2020-01-31", 98.45),
        ff_bar("2020-01-02", "2020-12-31", 98.60),
        ff_bar("2020-01-02", "2021-01-29", 98.62),
        ff_bar("2020-01-05", "2020-01-31", 90.00),
        ff_bar("2020-01-05", "2020-12-31", 90.00),
    ])
    rates = macro.ff_rates(bars, change_days=1)
    assert list(rates.index) == [pd.Timestamp("2020-01-02")]
    row = rates.loc["2020-01-02"]
    assert row["ff_rate_12m"] == pytest.approx(0.0140)
    assert row["ff_rate_front"] == pytest.approx(0.0155)
    assert row["ff_slope"] == pytest.approx(0.0140 - 0.0155)


def test_ff_rates_drops_dates_without_contract_near_horizon():
    bars = pd.DataFrame([ff_bar("2020-01-02", "2020-01-31", 98.45), ff_bar("2020-01-02", "2020-06-30", 98.50)])
    assert macro.ff_rates(bars).empty


def test_ff_rates_are_causal():
    days = pd.bdate_range("2020-01-01", periods=60)
    bars = pd.DataFrame([
        ff_bar(day, day + pd.offsets.MonthEnd(m), 99.0 - 0.01 * k - 0.05 * m)
        for k, day in enumerate(days) for m in range(15)
    ])
    cutoff = days[40]
    full = macro.ff_rates(bars, change_days=5)
    part = macro.ff_rates(bars[bars["date"] <= cutoff], change_days=5)
    pd.testing.assert_frame_equal(full.loc[:cutoff], part, check_exact=True)


def test_ff_rate_change_is_a_difference_not_pct_change():
    bars = pd.DataFrame([
        ff_bar(day, expiration, close)
        for day, close in [("2020-01-02", 99.90), ("2020-01-03", 99.80)]
        for expiration in ["2020-01-31", "2020-12-31"]
    ])
    rates = macro.ff_rates(bars, change_days=1)
    assert rates["ff_rate_12m_chg"].iloc[1] == pytest.approx(0.001)


def test_parse_gpr_selects_and_renames():
    raw = pd.DataFrame({"DAY": [20200101], "N10D": [1], "GPRD": [100.0], "GPRD_ACT": [50.0],
                        "GPRD_THREAT": [80.0], "date": [pd.Timestamp("2020-01-01")], "event": [None]})
    out = macro.parse_gpr(raw)
    assert list(out.columns) == ["gpr", "act", "threat"]
    assert out.index[0] == pd.Timestamp("2020-01-01")


def test_gpr_features_on_constant_series():
    last = macro.gpr_features(gpr_frame({"gpr": 100.0, "act": 50.0, "threat": 100.0})).iloc[-1]
    assert last["gpr_level"] == pytest.approx(np.log(100.0))
    assert last["gpr_trend"] == pytest.approx(0.0)
    assert last["gpr_threat_act"] == pytest.approx(np.log(2.0))


def test_gpr_features_are_causal():
    rng = np.random.default_rng(0)
    gpr = gpr_frame({"gpr": rng.uniform(50, 200, 60), "act": rng.uniform(20, 100, 60), "threat": rng.uniform(20, 100, 60)})
    cutoff = gpr.index[40]
    pd.testing.assert_frame_equal(macro.gpr_features(gpr).loc[:cutoff], macro.gpr_features(gpr.loc[:cutoff]), check_exact=True)


def test_align_asof_respects_lag_and_staleness():
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-10"]))
    dates = pd.DatetimeIndex(pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-05", "2020-01-09", "2020-01-10"]))
    np.testing.assert_array_equal(macro.align_asof(frame, dates)["x"], [np.nan, 2.0, 2.0, 2.0, 3.0])
    np.testing.assert_array_equal(macro.align_asof(frame, dates, lag_days=1)["x"], [np.nan, 1.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(macro.align_asof(frame, dates, max_staleness_days=2)["x"], [np.nan, 2.0, np.nan, np.nan, 3.0])


def test_align_asof_never_reads_rows_after_t_minus_lag():
    frame = pd.DataFrame({"x": np.arange(30.0)}, index=pd.date_range("2020-01-01", periods=30, freq="D"))
    dates = pd.date_range("2020-01-05", periods=20, freq="D")
    base = macro.align_asof(frame, dates, lag_days=7)["x"].to_numpy()
    for i, t in enumerate(dates):
        changed = frame.copy()
        changed.loc[changed.index > t - pd.Timedelta(days=7), "x"] = -999.0
        np.testing.assert_array_equal(macro.align_asof(changed, dates[i:i + 1], lag_days=7)["x"].to_numpy(), base[i:i + 1])


def test_risk_free_daily_converts_percent_yield():
    prices = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "ticker": "^IRX", "close": [5.04, 5.04]})
    dates = pd.DatetimeIndex(pd.to_datetime(["2020-01-03", "2020-01-06"]))
    rf = macro.risk_free_daily(prices, "^IRX", dates, max_staleness_days=5)
    assert rf.tolist() == pytest.approx([0.0504 / 252, 0.0504 / 252])
    assert rf.name == "rf_daily"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_macro.py -v`
Expected: FAIL — `ImportError: cannot import name 'macro'`.

- [ ] **Step 3: Implementar**

`engine/data/macro.py`:

```python
"""Macro inputs aligned to trading days without lookahead: fed funds futures, GPR and the risk-free rate."""
import numpy as np
import pandas as pd

FF_FEATURES = ["ff_rate_12m", "ff_rate_12m_chg", "ff_slope"]
GPR_FEATURES = ["gpr_level", "gpr_trend", "gpr_threat_act"]


def parse_ff_futures(raw: pd.DataFrame) -> pd.DataFrame:
    bars = raw.rename(columns=str.lower)
    bars = bars.assign(date=pd.to_datetime(bars["date"]), expiration=pd.to_datetime(bars["expiration"]))
    return bars[["date", "symbol", "expiration", "close"]]


def load_ff_futures(path) -> pd.DataFrame:
    return parse_ff_futures(pd.read_csv(path))


def ff_rates(bars: pd.DataFrame, horizon_months: int = 12, max_gap_days: int = 45, change_days: int = 21) -> pd.DataFrame:
    """Daily bars are UTC days, so each one closes (~19-20h ET) before the next US open."""
    df = bars[bars["date"].dt.dayofweek != 6].dropna(subset=["close"])  # Sunday bars are partial Globex sessions
    df = df.assign(rate=(100.0 - df["close"]) / 100.0)
    targets = {day: day + pd.DateOffset(months=horizon_months) for day in df["date"].unique()}
    df = df.assign(gap=(df["expiration"] - df["date"].map(targets)).abs())
    nearest = df.sort_values(["date", "gap"]).groupby("date").first()
    rate_12m = nearest.loc[nearest["gap"] <= pd.Timedelta(days=max_gap_days), "rate"]
    live = df[df["expiration"] >= df["date"]]
    rate_front = live.sort_values(["date", "expiration"]).groupby("date")["rate"].first()
    out = pd.DataFrame({"ff_rate_12m": rate_12m, "ff_rate_front": rate_front}).dropna().sort_index()
    out["ff_rate_12m_chg"] = out["ff_rate_12m"].diff(change_days)
    out["ff_slope"] = out["ff_rate_12m"] - out["ff_rate_front"]
    return out


def parse_gpr(raw: pd.DataFrame) -> pd.DataFrame:
    gpr = raw[["date", "GPRD", "GPRD_ACT", "GPRD_THREAT"]].dropna(subset=["date"])
    gpr = gpr.rename(columns={"GPRD": "gpr", "GPRD_ACT": "act", "GPRD_THREAT": "threat"})
    return gpr.assign(date=pd.to_datetime(gpr["date"])).set_index("date").sort_index()


def load_gpr(path) -> pd.DataFrame:
    return parse_gpr(pd.read_excel(path))


def gpr_features(gpr: pd.DataFrame) -> pd.DataFrame:
    """Trailing calendar-day means ending at each GPR date; the publication lag is applied by the caller."""
    daily = gpr.asfreq("D")
    mean_7 = daily.rolling(7, min_periods=6).mean()
    mean_30 = daily["gpr"].rolling(30, min_periods=24).mean()
    with np.errstate(divide="ignore"):
        out = pd.DataFrame({
            "gpr_level": np.log(mean_7["gpr"]),
            "gpr_trend": np.log(mean_7["gpr"]) - np.log(mean_30),
            "gpr_threat_act": np.log(mean_7["threat"]) - np.log(mean_7["act"]),
        })
    return out.replace([np.inf, -np.inf], np.nan)


def align_asof(frame: pd.DataFrame, dates, lag_days: int = 0, max_staleness_days: int | None = None) -> pd.DataFrame:
    """Row t = last row of `frame` dated on or before t - lag_days; NaN if none or if it is older than the limit."""
    frame = frame.sort_index()
    dates = pd.DatetimeIndex(dates)
    lookup = dates - pd.Timedelta(days=lag_days)
    pos = frame.index.searchsorted(lookup, side="right") - 1
    safe = np.clip(pos, 0, None)
    values = frame.to_numpy(dtype=float)[safe]
    invalid = pos < 0
    if max_staleness_days is not None:
        invalid |= np.asarray((lookup - frame.index[safe]) > pd.Timedelta(days=max_staleness_days))
    values[invalid] = np.nan
    return pd.DataFrame(values, index=dates, columns=frame.columns)


def risk_free_daily(prices: pd.DataFrame, ticker: str, dates, max_staleness_days: int) -> pd.Series:
    """^IRX closes in % per year; one trading day of carry is rate / 100 / 252."""
    yields = prices.loc[prices["ticker"] == ticker].set_index("date")["close"].sort_index().to_frame("rate")
    return (align_asof(yields, dates, 0, max_staleness_days)["rate"] / 100.0 / 252.0).rename("rf_daily")
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_macro.py -v`
Expected: 11 passed.

- [ ] **Step 5: Conferir nos dados reais**

```bash
venv/bin/python - <<'EOF'
from config.config import CONFIG
from engine.data import macro
ff = macro.ff_rates(macro.load_ff_futures(CONFIG.data_in / CONFIG.ff_futures_file))
print(ff.describe().T[["count", "min", "max"]]); print(ff.index.min().date(), ff.index.max().date())
g = macro.gpr_features(macro.load_gpr(CONFIG.data_in / CONFIG.gpr_file))
print(g.dropna().describe().T[["count", "min", "max"]]); print(g.index.max().date())
EOF
```

Expected:
- `ff_rate_12m` fica entre ~0 e ~0,055, com datas de 2019-11 a 2026-07-31.
- `ff_rate_12m_chg` fica em poucas centésimas.
- O GPR termina em 2026-09-21, sem infinitos.

---

### Task 6: Features causais

**Files:**
- Create: `engine/features.py`, `tests/synthetic.py`, `tests/conftest.py`, `tests/test_features.py`

**Interfaces:**
- Consumes: `PricePanel`, `PANEL_FIELDS`, `daily_total_return`, `build_panel`, `trading_calendar` (Task 3); `FF_FEATURES`, `GPR_FEATURES` (Task 5); `membership_matrix` (Task 2).
- Produces:
  - `STOCK_FEATURES` (19 nomes), `XS_SOURCES`, `XS_FEATURES` (5), `MARKET_FEATURES` (6), `MACRO_FEATURES = FF_FEATURES + GPR_FEATURES`, `TICKER_FEATURES = STOCK_FEATURES + XS_FEATURES`, `DATE_FEATURES = MARKET_FEATURES + MACRO_FEATURES`, `FEATURES = TICKER_FEATURES + DATE_FEATURES` (36)
  - `stock_features(panel: PricePanel, market_close: pd.Series) -> dict[str, pd.DataFrame]`
  - `cross_sectional_features(stock: dict, membership: pd.DataFrame) -> dict[str, pd.DataFrame]`
  - `market_features(market_close: pd.Series, stock: dict, membership: pd.DataFrame) -> pd.DataFrame`
  - `tests/synthetic.py`: `BENCH`, `RF`, `make_market(n_days=330, n_tickers=12, seed=7) -> dict` com chaves `prices, snapshots, ff_bars, gpr`; `perturb_after(raw: dict, cutoff, seed=99) -> dict`
  - fixture `synthetic_panel` → `(panel, market_close, membership)`

- [ ] **Step 1: Criar o mercado sintético e a fixture**

`tests/synthetic.py`:

```python
"""Synthetic market used by the tests: prices, constituents, fed funds futures and GPR."""
import numpy as np
import pandas as pd

BENCH = "^SP500TR"
RF = "^IRX"


def _bars(rng, dates, ticker, log_returns, dividend_every=None):
    n = len(dates)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    dividends = np.zeros(n)
    if dividend_every:
        dividends[dividend_every::dividend_every] = 0.5
    return pd.DataFrame({
        "date": dates, "ticker": ticker, "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.integers(100_000, 1_000_000, n).astype(float), "dividends": dividends, "splits": 0.0,
    })


def make_market(n_days: int = 330, n_tickers: int = 12, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    market = rng.normal(0.0004, 0.01, n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    frames = [
        _bars(rng, dates, t, (0.5 + i / n_tickers) * market + rng.normal(0.0, 0.015, n_days), 63 if i % 3 == 0 else None)
        for i, t in enumerate(tickers)
    ]
    frames.append(_bars(rng, dates, BENCH, market))
    frames.append(pd.DataFrame({
        "date": dates, "ticker": RF, "open": np.nan, "high": np.nan, "low": np.nan,
        "close": 2.0 + 0.5 * np.sin(np.arange(n_days) / 40.0), "volume": 0.0, "dividends": 0.0, "splits": 0.0,
    }))
    snapshots = pd.DataFrame({
        "date": [dates[0], dates[150], dates[260]],
        "tickers": [tickers[:10], tickers[1:11], tickers[2:12]],
    })
    ff_bars = pd.DataFrame([
        {"date": day, "symbol": f"ZQ{m}", "expiration": day + pd.offsets.MonthEnd(m),
         "close": 100.0 - 100.0 * (0.02 + 0.001 * m + 0.003 * np.sin(k / 50.0))}
        for k, day in enumerate(dates) for m in range(15)
    ])
    days = pd.date_range(dates[0] - pd.Timedelta(days=60), dates[-1] + pd.Timedelta(days=30), freq="D", name="date")
    gpr = pd.DataFrame({
        "gpr": rng.uniform(50.0, 200.0, len(days)),
        "act": rng.uniform(20.0, 150.0, len(days)),
        "threat": rng.uniform(20.0, 150.0, len(days)),
    }, index=days)
    return {"prices": pd.concat(frames, ignore_index=True), "snapshots": snapshots, "ff_bars": ff_bars, "gpr": gpr}


def perturb_after(raw: dict, cutoff: pd.Timestamp, seed: int = 99) -> dict:
    """Same market up to `cutoff`; everything dated after it is scrambled and a new ticker joins the index."""
    rng = np.random.default_rng(seed)
    prices = raw["prices"].copy()
    future = prices["date"] > cutoff
    for column in ("open", "high", "low", "close", "volume"):
        prices.loc[future, column] = prices.loc[future, column] * rng.uniform(0.5, 1.5, int(future.sum()))
    prices.loc[future, "dividends"] = rng.uniform(0.0, 2.0, int(future.sum()))
    newcomer = raw["prices"].loc[raw["prices"]["ticker"] == "T00"].assign(ticker="NEW")
    snapshots = pd.concat([
        raw["snapshots"],
        pd.DataFrame({"date": [cutoff + pd.Timedelta(days=1)], "tickers": [["T02", "T03", "NEW"]]}),
    ], ignore_index=True)
    ff_bars = raw["ff_bars"].copy()
    later = ff_bars["date"] > cutoff
    ff_bars.loc[later, "close"] = rng.uniform(94.0, 99.0, int(later.sum()))
    gpr = raw["gpr"].copy()
    gpr.loc[gpr.index > cutoff] = rng.uniform(1.0, 500.0, size=(int((gpr.index > cutoff).sum()), 3))
    return {"prices": pd.concat([prices, newcomer], ignore_index=True), "snapshots": snapshots, "ff_bars": ff_bars, "gpr": gpr}
```

`tests/conftest.py`:

```python
import pytest
from synthetic import BENCH, RF, make_market

from engine.data import prices as px
from engine.data import universe


@pytest.fixture(scope="session")
def synthetic_panel():
    raw = make_market()
    prices = raw["prices"]
    calendar = px.trading_calendar(prices, BENCH)
    tickers = sorted(set(prices["ticker"]) - {BENCH, RF})
    panel = px.build_panel(prices, calendar, tickers)
    membership = universe.membership_matrix(raw["snapshots"], calendar, tickers)
    market_close = prices[prices["ticker"] == BENCH].set_index("date")["close"].reindex(calendar)
    return panel, market_close, membership
```

- [ ] **Step 2: Escrever os testes das features**

`tests/test_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from engine.data.prices import PANEL_FIELDS, PricePanel
from engine.features import (
    FEATURES,
    STOCK_FEATURES,
    XS_FEATURES,
    cross_sectional_features,
    market_features,
    stock_features,
)

DATES = pd.bdate_range("2020-01-01", periods=300, name="date")


def make_panel(close: pd.DataFrame, dividends: pd.DataFrame | None = None) -> PricePanel:
    return PricePanel(
        open=close, high=close, low=close, close=close,
        volume=pd.DataFrame(1000.0, index=close.index, columns=close.columns),
        dividends=pd.DataFrame(0.0, index=close.index, columns=close.columns) if dividends is None else dividends,
    )


def test_feature_lists():
    assert len(FEATURES) == 36 and len(set(FEATURES)) == 36


def test_returns_momentum_and_trend_on_constant_growth():
    close = pd.DataFrame({"A": 100.0 * 1.01 ** np.arange(300)}, index=DATES)
    market = pd.Series(1000.0 * 1.005 ** np.arange(300), index=DATES)
    last = {name: frame["A"].iloc[-1] for name, frame in stock_features(make_panel(close), market).items()}
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
    f = stock_features(make_panel(close, dividends), market)
    assert f["ret_1d"]["A"].iloc[250] == pytest.approx(0.01)
    assert f["gap_1d"]["A"].iloc[250] == pytest.approx(0.01)
    assert f["div_yield_252"]["A"].iloc[260] == pytest.approx(0.01)
    assert f["ret_21d"]["A"].iloc[260] == pytest.approx(0.01)


def test_beta_recovers_leverage_on_market():
    m = np.random.default_rng(1).normal(0.0, 0.01, 300)
    market = pd.Series(1000.0 * np.cumprod(1 + m), index=DATES)
    close = pd.DataFrame({"A": 50.0 * np.cumprod(1 + 2.0 * m)}, index=DATES)
    assert stock_features(make_panel(close), market)["beta_63d"]["A"].iloc[-1] == pytest.approx(2.0, rel=1e-9)


def test_cross_section_ranks_only_members():
    dates = DATES[:2]
    ret = pd.DataFrame({"A": [0.01, 0.01], "B": [0.02, 0.02], "C": [0.03, 0.03]}, index=dates)
    stock = {name: ret for name in ("ret_1d", "ret_5d", "ret_21d", "mom_12_1", "vol_21d")}
    membership = pd.DataFrame({"A": [True, True], "B": [True, False], "C": [True, True]}, index=dates)
    xs = cross_sectional_features(stock, membership)["xs_ret_1d"]
    assert xs.iloc[0].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert np.isnan(xs.iloc[1]["B"]) and xs.iloc[1][["A", "C"]].tolist() == pytest.approx([0.5, 1.0])


def test_breadth_counts_members_above_their_ma50():
    dates = DATES[:1]
    stock = {"dist_ma50": pd.DataFrame({"A": [0.1], "B": [-0.1], "C": [0.2], "D": [0.3]}, index=dates)}
    membership = pd.DataFrame({"A": [True], "B": [True], "C": [True], "D": [False]}, index=dates)
    market = market_features(pd.Series([1000.0], index=dates), stock, membership)
    assert market["breadth_ma50"].iloc[0] == pytest.approx(2 / 3)


def test_stock_cross_section_and_market_features_are_causal(synthetic_panel):
    panel, market_close, membership = synthetic_panel
    cutoff = panel.close.index[200]
    truncated = PricePanel(**{field: getattr(panel, field).loc[:cutoff] for field in PANEL_FIELDS})
    full = stock_features(panel, market_close)
    part = stock_features(truncated, market_close.loc[:cutoff])
    for name in STOCK_FEATURES:
        pd.testing.assert_frame_equal(full[name].loc[:cutoff], part[name], check_exact=True)
    full_xs = cross_sectional_features(full, membership)
    part_xs = cross_sectional_features(part, membership.loc[:cutoff])
    for name in XS_FEATURES:
        pd.testing.assert_frame_equal(full_xs[name].loc[:cutoff], part_xs[name], check_exact=True)
    pd.testing.assert_frame_equal(
        market_features(market_close, full, membership).loc[:cutoff],
        market_features(market_close.loc[:cutoff], part, membership.loc[:cutoff]),
        check_exact=True,
    )
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.features'`.

- [ ] **Step 4: Implementar**

`engine/features.py`:

```python
"""Point-in-time features: every value at date t uses only data dated on or before t."""
import numpy as np
import pandas as pd

from engine.data.macro import FF_FEATURES, GPR_FEATURES
from engine.data.prices import PricePanel, daily_total_return

STOCK_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "mom_12_1", "gap_1d", "intraday_1d",
    "vol_21d", "vol_63d", "vol_ratio", "range_21d", "dist_ma50", "dist_ma200",
    "dist_high_252", "rsi_14", "volume_ratio", "log_dollar_volume", "div_yield_252", "beta_63d",
]
XS_SOURCES = ["ret_1d", "ret_5d", "ret_21d", "mom_12_1", "vol_21d"]
XS_FEATURES = [f"xs_{name}" for name in XS_SOURCES]
MARKET_FEATURES = ["mkt_ret_1d", "mkt_ret_5d", "mkt_ret_21d", "mkt_vol_21d", "mkt_dist_ma200", "breadth_ma50"]
MACRO_FEATURES = FF_FEATURES + GPR_FEATURES
TICKER_FEATURES = STOCK_FEATURES + XS_FEATURES
DATE_FEATURES = MARKET_FEATURES + MACRO_FEATURES
FEATURES = TICKER_FEATURES + DATE_FEATURES


def _rolling(frame, window: int, stat: str):
    return getattr(frame.rolling(window, min_periods=int(np.ceil(0.8 * window))), stat)()


def stock_features(panel: PricePanel, market_close: pd.Series) -> dict[str, pd.DataFrame]:
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


def cross_sectional_features(stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {f"xs_{name}": stock[name].where(membership).rank(axis=1, pct=True) for name in XS_SOURCES}


def market_features(market_close: pd.Series, stock: dict[str, pd.DataFrame], membership: pd.DataFrame) -> pd.DataFrame:
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

- [ ] **Step 5: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_features.py -v`
Expected: 7 passed. Se o teste de causalidade falhar só por ruído de ponto flutuante (diferença < 1e-12), trocar para `check_exact=False, rtol=1e-12`. Se a diferença for maior que isso, há vazamento: investigar com superpowers:systematic-debugging.

---

### Task 7: Dataset

**Files:**
- Create: `engine/dataset.py`, `tests/test_dataset.py`

**Interfaces:**
- Consumes: `FEATURES`, `TICKER_FEATURES`, `DATE_FEATURES` (Task 6).
- Produces: `build_dataset(ticker_features: dict[str, pd.DataFrame], date_features: pd.DataFrame, membership: pd.DataFrame, close: pd.DataFrame, fwd_ret: pd.DataFrame, min_history_days: int) -> pd.DataFrame`. O índice é `MultiIndex(date, ticker)`, ordenado por data; as colunas são `FEATURES + ["fwd_ret", "label"]`; as features ficam em float32.

- [ ] **Step 1: Escrever os testes**

`tests/test_dataset.py`:

```python
import numpy as np
import pandas as pd
import pytest

from engine.dataset import build_dataset
from engine.features import DATE_FEATURES, FEATURES, TICKER_FEATURES

DATES = pd.bdate_range("2021-01-01", periods=4, name="date")
TICKERS = ["A", "B", "C"]


def frame(values) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES, columns=TICKERS, dtype=float)


@pytest.fixture
def parts():
    ticker_features = {name: frame(np.arange(12).reshape(4, 3) + i) for i, name in enumerate(TICKER_FEATURES)}
    date_features = pd.DataFrame(1.0, index=DATES, columns=DATE_FEATURES)
    date_features.iloc[0, 0] = np.nan
    membership = frame([[1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 1, 1]]).astype(bool)
    close = frame([[1, 1, np.nan], [1, np.nan, 1], [1, 1, 1], [1, 1, 1]])
    fwd = frame([[0.1, 0.1, 0.1], [0.02, -0.01, 0.0], [0.0, np.nan, 0.03], [np.nan, np.nan, np.nan]])
    return ticker_features, date_features, membership, close, fwd


def test_rows_are_members_with_close_history_and_date_features(parts):
    dataset = build_dataset(*parts, min_history_days=2)
    assert list(dataset.index) == [
        (DATES[1], "A"), (DATES[2], "A"), (DATES[2], "B"), (DATES[2], "C"), (DATES[3], "B"), (DATES[3], "C"),
    ]
    assert dataset.index.names == ["date", "ticker"]


def test_columns_and_values(parts):
    dataset = build_dataset(*parts, min_history_days=2)
    assert list(dataset.columns) == FEATURES + ["fwd_ret", "label"]
    assert dataset.loc[(DATES[2], "B"), TICKER_FEATURES[0]] == 7.0
    assert dataset.loc[(DATES[2], "B"), TICKER_FEATURES[3]] == 10.0
    assert dataset[FEATURES].dtypes.eq(np.float32).all()


def test_label_is_positive_forward_return(parts):
    dataset = build_dataset(*parts, min_history_days=2)
    np.testing.assert_array_equal(dataset["fwd_ret"], [0.02, 0.0, np.nan, 0.03, np.nan, np.nan])
    np.testing.assert_array_equal(dataset["label"], [1.0, 0.0, np.nan, 1.0, np.nan, np.nan])
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.dataset'`.

- [ ] **Step 3: Implementar**

`engine/dataset.py`:

```python
"""Long table indexed by (date, ticker): features known after the close of `date`, realized forward return and label."""
import numpy as np
import pandas as pd

from engine.features import DATE_FEATURES, FEATURES, TICKER_FEATURES


def build_dataset(ticker_features: dict[str, pd.DataFrame], date_features: pd.DataFrame, membership: pd.DataFrame,
                  close: pd.DataFrame, fwd_ret: pd.DataFrame, min_history_days: int) -> pd.DataFrame:
    """A row per stock that, on that date, is in the index, traded, has enough history and all date features."""
    has_close = close.notna().to_numpy()
    eligible = (
        membership.to_numpy()
        & has_close
        & (np.cumsum(has_close, axis=0) >= min_history_days)
        & date_features[DATE_FEATURES].notna().all(axis=1).to_numpy()[:, None]
    )
    rows, cols = np.nonzero(eligible)
    index = pd.MultiIndex.from_arrays([close.index[rows], close.columns[cols]], names=["date", "ticker"])
    data = {name: ticker_features[name].to_numpy(dtype=np.float32)[rows, cols] for name in TICKER_FEATURES}
    date_values = date_features[DATE_FEATURES].to_numpy(dtype=np.float32)[rows]
    data.update({name: date_values[:, j] for j, name in enumerate(DATE_FEATURES)})
    dataset = pd.DataFrame(data, index=index)[FEATURES]
    realized = fwd_ret.to_numpy(dtype=float)[rows, cols]
    dataset["fwd_ret"] = realized
    dataset["label"] = np.where(np.isnan(realized), np.nan, (realized > 0).astype(float))
    return dataset
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_dataset.py -v`
Expected: 3 passed.

---

### Task 8: Modelo e walk-forward

**Files:**
- Create: `engine/model.py`, `engine/walk_forward.py`, `tests/test_walk_forward.py`

**Interfaces:**
- Consumes: dataset no formato da Task 7; `LGBM_PARAMS` (Task 1).
- Produces:
  - `make_model(params: dict) -> LGBMClassifier`
  - `WalkForwardResult(predictions, importance)`:
    - `predictions`: colunas `date, ticker, prob, fwd_ret, label`;
    - `importance`: índice `date` (datas de decisão), uma coluna por feature, com a fração do gain.
  - `walk_forward(dataset, features: list[str], calendar: pd.DatetimeIndex, train_window: int, model_factory: Callable, start=None, end=None, progress_every: int = 0, log=print) -> WalkForwardResult`

- [ ] **Step 1: Escrever os testes**

`tests/test_walk_forward.py`:

```python
import numpy as np
import pandas as pd
import pytest

from config.config import LGBM_PARAMS
from engine.model import make_model
from engine.walk_forward import walk_forward

DATES = pd.bdate_range("2021-01-01", periods=15, name="date")


def spy_dataset() -> pd.DataFrame:
    rows = [
        {"date": day, "ticker": ticker, "day": float(k), "noise": 0.0, "fwd_ret": 0.01, "label": float(k % 2)}
        for k, day in enumerate(DATES) if k >= 2
        for ticker in ("A", "B")
    ]
    return pd.DataFrame(rows).set_index(["date", "ticker"])


class Spy:
    """Records which calendar days (feature 'day') each fit and predict sees."""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y):
        self.log.append(("fit", sorted({int(v) for v in X[:, 0]})))
        self.feature_importances_ = np.array([3.0, 1.0])
        return self

    def predict_proba(self, X):
        self.log.append(("predict", sorted({int(v) for v in X[:, 0]})))
        p = np.full(len(X), 0.7)
        return np.column_stack([1 - p, p])


def run_spy(dataset, window=4, **kwargs):
    log = []
    result = walk_forward(dataset, ["day", "noise"], DATES, window, lambda: Spy(log), **kwargs)
    return result, log


def test_trains_on_exactly_the_labels_known_at_decision_time():
    _, log = run_spy(spy_dataset())
    fits = [days for kind, days in log if kind == "fit"]
    predicts = [days for kind, days in log if kind == "predict"]
    decisions = list(range(2 + 4 + 1, len(DATES)))
    assert predicts == [[k] for k in decisions]
    assert fits == [list(range(k - 4 - 1, k - 1)) for k in decisions]


def test_rows_with_unknown_label_are_not_used_for_training():
    dataset = spy_dataset()
    dataset.loc[dataset.index.get_level_values("date") == DATES[5], "label"] = np.nan
    _, log = run_spy(dataset)
    fits = [days for kind, days in log if kind == "fit"]
    assert fits[0] == [2, 3, 4]
    assert all(5 not in days for days in fits)


def test_outputs_predictions_and_importance():
    result, _ = run_spy(spy_dataset())
    predictions = result.predictions
    assert list(predictions.columns) == ["date", "ticker", "prob", "fwd_ret", "label"]
    assert predictions["date"].min() == DATES[7]
    assert len(predictions) == 2 * (len(DATES) - 7)
    assert (predictions["prob"] == 0.7).all()
    assert list(result.importance.columns) == ["day", "noise"]
    assert result.importance.iloc[0].tolist() == [0.75, 0.25]
    assert list(result.importance.index) == list(DATES[7:])


def test_start_and_end_limit_decisions():
    result, _ = run_spy(spy_dataset(), start=DATES[9], end=DATES[11])
    assert sorted(result.predictions["date"].unique()) == list(DATES[9:12])


def test_no_decision_dates_raises():
    with pytest.raises(ValueError):
        run_spy(spy_dataset(), window=50)


def test_lightgbm_model_is_deterministic_and_gives_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 3))
    y = (X[:, 0] + rng.normal(0.0, 1.0, 3000) > 0).astype(int)
    first = make_model(LGBM_PARAMS).fit(X, y).predict_proba(X)[:, 1]
    second = make_model(LGBM_PARAMS).fit(X, y).predict_proba(X)[:, 1]
    np.testing.assert_array_equal(first, second)
    assert ((first > 0) & (first < 1)).all()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_walk_forward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.model'`.

- [ ] **Step 3: Implementar**

`engine/model.py`:

```python
"""The classifier behind P(stock goes up); swap it here."""
from lightgbm import LGBMClassifier


def make_model(params: dict) -> LGBMClassifier:
    return LGBMClassifier(**params)
```

`engine/walk_forward.py`:

```python
"""Daily retraining on a rolling window, predicting only the next decision date."""
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    importance: pd.DataFrame


def walk_forward(dataset: pd.DataFrame, features: list[str], calendar: pd.DatetimeIndex, train_window: int,
                 model_factory: Callable, start=None, end=None, progress_every: int = 0, log=print) -> WalkForwardResult:
    """Row s's label needs the open of s+2, so the fit for decision t uses exactly the rows dated t-X-1 .. t-2."""
    row_pos = calendar.get_indexer(dataset.index.get_level_values("date"))
    X = dataset[features].to_numpy(dtype=np.float32)
    y = dataset["label"].to_numpy(dtype=float)
    decisions = np.unique(row_pos)
    decisions = decisions[decisions - train_window - 1 >= row_pos[0]]
    if start is not None:
        decisions = decisions[calendar[decisions] >= pd.Timestamp(start)]
    if end is not None:
        decisions = decisions[calendar[decisions] <= pd.Timestamp(end)]
    if len(decisions) == 0:
        raise ValueError("Nenhuma data de decisão com janela de treino completa no intervalo pedido.")

    predictions, importance = [], []
    started = time.perf_counter()
    for n, k in enumerate(decisions, start=1):
        lo = np.searchsorted(row_pos, k - train_window - 1, side="left")
        hi = np.searchsorted(row_pos, k - 2, side="right")
        known = ~np.isnan(y[lo:hi])
        model = model_factory()
        model.fit(X[lo:hi][known], y[lo:hi][known].astype(int))

        first, last = np.searchsorted(row_pos, k, side="left"), np.searchsorted(row_pos, k, side="right")
        block = dataset.iloc[first:last]
        predictions.append(pd.DataFrame({
            "date": block.index.get_level_values("date"),
            "ticker": block.index.get_level_values("ticker"),
            "prob": model.predict_proba(X[first:last])[:, 1],
            "fwd_ret": block["fwd_ret"].to_numpy(),
            "label": block["label"].to_numpy(),
        }))
        gain = np.asarray(model.feature_importances_, dtype=float)
        importance.append(gain / gain.sum() if gain.sum() > 0 else gain)

        if progress_every and n % progress_every == 0:
            elapsed = time.perf_counter() - started
            remaining = elapsed / n * (len(decisions) - n)
            log(f"walk-forward {n}/{len(decisions)} ({calendar[k].date()}) · {elapsed / 60:.1f} min · faltam ~{remaining / 60:.1f} min")

    return WalkForwardResult(
        predictions=pd.concat(predictions, ignore_index=True),
        importance=pd.DataFrame(importance, index=pd.DatetimeIndex(calendar[decisions], name="date"), columns=features),
    )
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_walk_forward.py -v`
Expected: 6 passed.

---

### Task 9: Resultados, pipeline e teste de lookahead ponta a ponta

**Files:**
- Create: `engine/results.py`, `engine/pipeline.py`, `tests/test_results.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: tudo das Tasks 2–8.
- Produces:
  - `RunResults(predictions, importance, market, coverage, run_info: dict)`
    - `market`: índice `decision_date`, colunas `holding_date, bench_fwd_ret, rf_daily`.
    - `coverage`: índice = calendário, colunas `n_members, n_with_prices, n_eligible`.
  - `save_results(results, out_dir: Path)`, `load_results(out_dir: Path) -> RunResults`
  - `MarketInputs(prices, snapshots, ff_bars, gpr)` (dataclass congelada), `load_inputs(config) -> MarketInputs`
  - `FeatureSet(dataset, calendar, market, coverage, data_quality)`, `build_feature_set(inputs, config) -> FeatureSet`
  - `run_pipeline(inputs, config, start=None, end=None, log=print) -> RunResults`. O `run_info` sai com as chaves `features` e `data_quality`.
  - `data_quality` tem as chaves `missing_tickers, n_tickers_with_prices, n_extreme_returns, extreme_returns, date_ranges`.

- [ ] **Step 1: Escrever o teste de resultados**

`tests/test_results.py`:

```python
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
```

- [ ] **Step 2: Escrever os testes do pipeline**

`tests/test_pipeline.py`:

```python
import numpy as np
import pandas as pd
import pytest
from synthetic import BENCH, RF, make_market, perturb_after

from config.config import Config
from engine.data import universe
from engine.features import DATE_FEATURES, FEATURES
from engine.pipeline import MarketInputs, build_feature_set, run_pipeline

SMALL_LGBM = {
    "n_estimators": 10, "learning_rate": 0.1, "num_leaves": 4, "min_child_samples": 10,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "importance_type": "gain",
    "random_state": 0, "deterministic": True, "force_col_wise": True, "n_jobs": 1, "verbose": -1,
}


def quiet(*_):
    pass


@pytest.fixture(scope="module")
def config():
    return Config(price_start="2019-01-01", train_window_days=40, progress_every=0, lgbm_params=SMALL_LGBM)


@pytest.fixture(scope="module")
def inputs():
    return MarketInputs(**make_market())


@pytest.fixture(scope="module")
def baseline(inputs, config):
    return run_pipeline(inputs, config, log=quiet)


def test_predictions_do_not_depend_on_future_data(config, baseline):
    decision_dates = baseline.predictions["date"].drop_duplicates().sort_values()
    cutoff = decision_dates.iloc[len(decision_dates) // 2]
    perturbed = run_pipeline(MarketInputs(**perturb_after(make_market(), cutoff)), config, log=quiet)
    columns = ["date", "ticker", "prob"]
    before = baseline.predictions.loc[baseline.predictions["date"] <= cutoff, columns].reset_index(drop=True)
    after = perturbed.predictions.loc[perturbed.predictions["date"] <= cutoff, columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    later = baseline.predictions[baseline.predictions["date"] > cutoff].merge(
        perturbed.predictions[perturbed.predictions["date"] > cutoff], on=["date", "ticker"], suffixes=("_base", "_pert"))
    assert len(later) > 0 and not np.allclose(later["prob_base"], later["prob_pert"])


def test_predictions_only_for_point_in_time_members(inputs, baseline):
    for date, group in baseline.predictions.groupby("date"):
        assert set(group["ticker"]) <= set(universe.members_on(inputs.snapshots, date))


def test_market_frame_aligns_benchmark_and_risk_free(inputs, config):
    features = build_feature_set(inputs, config)
    market, calendar = features.market, features.calendar
    bench_open = inputs.prices[inputs.prices["ticker"] == BENCH].set_index("date")["open"]
    irx = inputs.prices[inputs.prices["ticker"] == RF].set_index("date")["close"]
    t = calendar[100]
    assert market.loc[t, "holding_date"] == calendar[101]
    assert market.loc[t, "bench_fwd_ret"] == pytest.approx(bench_open[calendar[102]] / bench_open[calendar[101]] - 1)
    assert market.loc[t, "rf_daily"] == pytest.approx(irx[t] / 100 / 252)
    assert pd.isna(market["holding_date"].iloc[-1]) and pd.isna(market["bench_fwd_ret"].iloc[-2])


def test_dataset_has_all_features_and_known_date_features(inputs, config):
    dataset = build_feature_set(inputs, config).dataset
    assert list(dataset.columns) == FEATURES + ["fwd_ret", "label"]
    assert dataset[DATE_FEATURES].notna().all().all()


def test_dataset_stops_when_macro_data_goes_stale(config):
    raw = make_market()
    last_ff = pd.Timestamp(np.sort(raw["ff_bars"]["date"].unique())[280])
    raw["ff_bars"] = raw["ff_bars"][raw["ff_bars"]["date"] <= last_ff]
    dataset = build_feature_set(MarketInputs(**raw), config).dataset
    assert dataset.index.get_level_values("date").max() <= last_ff + pd.Timedelta(days=config.macro_max_staleness_days)


def test_run_info_reports_quality_and_coverage(baseline):
    assert list(baseline.coverage.columns) == ["n_members", "n_with_prices", "n_eligible"]
    quality = baseline.run_info["data_quality"]
    assert {"missing_tickers", "n_tickers_with_prices", "n_extreme_returns", "extreme_returns", "date_ranges"} <= set(quality)
    assert baseline.run_info["features"] == FEATURES
    assert (baseline.coverage["n_eligible"] <= baseline.coverage["n_with_prices"]).all()
```

- [ ] **Step 3: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_results.py tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.results'`.

- [ ] **Step 4: Implementar os resultados**

`engine/results.py`:

```python
"""What `python main.py run` writes to data/out, so the app never retrains."""
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TABLES = ("predictions", "importance", "market", "coverage")


@dataclass
class RunResults:
    predictions: pd.DataFrame
    importance: pd.DataFrame
    market: pd.DataFrame
    coverage: pd.DataFrame
    run_info: dict


def save_results(results: RunResults, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        getattr(results, name).to_parquet(out_dir / f"{name}.parquet")
    (out_dir / "run_info.json").write_text(json.dumps(results.run_info, indent=2, default=str))


def load_results(out_dir: Path) -> RunResults:
    tables = {name: pd.read_parquet(out_dir / f"{name}.parquet") for name in TABLES}
    return RunResults(**tables, run_info=json.loads((out_dir / "run_info.json").read_text()))
```

- [ ] **Step 5: Implementar o pipeline**

`engine/pipeline.py`:

```python
"""Data → features → dataset → walk-forward, over in-memory inputs."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.config import TICKER_ALIASES, Config
from engine.data import macro
from engine.data import prices as px
from engine.data import universe
from engine.dataset import build_dataset
from engine.features import FEATURES, cross_sectional_features, market_features, stock_features
from engine.model import make_model
from engine.results import RunResults
from engine.walk_forward import walk_forward


@dataclass(frozen=True)
class MarketInputs:
    prices: pd.DataFrame
    snapshots: pd.DataFrame
    ff_bars: pd.DataFrame
    gpr: pd.DataFrame


@dataclass
class FeatureSet:
    dataset: pd.DataFrame
    calendar: pd.DatetimeIndex
    market: pd.DataFrame
    coverage: pd.DataFrame
    data_quality: dict


def load_inputs(config: Config) -> MarketInputs:
    return MarketInputs(
        prices=px.load_prices(config.data_in / config.prices_file),
        snapshots=universe.load_constituents(config.data_in / config.constituents_file, TICKER_ALIASES),
        ff_bars=macro.load_ff_futures(config.data_in / config.ff_futures_file),
        gpr=macro.load_gpr(config.data_in / config.gpr_file),
    )


def build_feature_set(inputs: MarketInputs, config: Config) -> FeatureSet:
    calendar = px.trading_calendar(inputs.prices, config.benchmark_ticker, start=config.price_start)
    wanted = universe.tickers_since(inputs.snapshots, config.price_start)
    available = set(inputs.prices["ticker"]) - {config.benchmark_ticker, config.risk_free_ticker}
    tickers = sorted(set(wanted) & available)
    panel = px.build_panel(inputs.prices, calendar, tickers)
    membership = universe.membership_matrix(inputs.snapshots, calendar, tickers)
    bench = inputs.prices[inputs.prices["ticker"] == config.benchmark_ticker].set_index("date").sort_index()
    market_close = bench["close"].reindex(calendar)

    stock = stock_features(panel, market_close)
    ff = macro.ff_rates(inputs.ff_bars, config.ff_horizon_months, config.ff_max_gap_days, config.ff_change_days)
    date_features = pd.concat([
        market_features(market_close, stock, membership),
        macro.align_asof(ff[macro.FF_FEATURES], calendar, 0, config.macro_max_staleness_days),
        macro.align_asof(macro.gpr_features(inputs.gpr), calendar, config.gpr_lag_days, config.macro_max_staleness_days),
    ], axis=1)
    dataset = build_dataset(
        {**stock, **cross_sectional_features(stock, membership)}, date_features, membership,
        panel.close, px.forward_open_return(panel.open, panel.dividends), config.min_history_days,
    )
    market = pd.DataFrame({
        "holding_date": pd.Series(calendar, index=calendar).shift(-1),
        "bench_fwd_ret": px.forward_open_return(bench["open"].reindex(calendar)),
        "rf_daily": macro.risk_free_daily(inputs.prices, config.risk_free_ticker, calendar, config.macro_max_staleness_days),
    }, index=calendar).rename_axis("decision_date")
    return FeatureSet(
        dataset=dataset,
        calendar=calendar,
        market=market,
        coverage=_coverage(inputs.snapshots, calendar, membership, panel.close, dataset),
        data_quality=_data_quality(inputs, calendar, wanted, tickers, panel, membership, dataset, ff),
    )


def _coverage(snapshots, calendar, membership, close, dataset) -> pd.DataFrame:
    sizes = snapshots["tickers"].map(len).to_numpy()
    pos = snapshots["date"].searchsorted(calendar, side="right") - 1
    return pd.DataFrame({
        "n_members": np.where(pos >= 0, sizes[np.clip(pos, 0, None)], 0),
        "n_with_prices": (membership & close.notna()).sum(axis=1).to_numpy(),
        "n_eligible": dataset.groupby(level="date").size().reindex(calendar, fill_value=0).to_numpy(),
    }, index=calendar)


def _data_quality(inputs, calendar, wanted, tickers, panel, membership, dataset, ff) -> dict:
    returns = px.daily_total_return(panel.close, panel.dividends).where(membership)
    extreme = returns.where(returns.abs() > 0.5).stack().dropna()
    top = extreme.loc[extreme.abs().sort_values(ascending=False).index[:20]]
    dataset_dates = dataset.index.get_level_values("date")
    return {
        "missing_tickers": sorted(set(wanted) - set(tickers)),
        "n_tickers_with_prices": len(tickers),
        "n_extreme_returns": int(len(extreme)),
        "extreme_returns": [{"date": str(d.date()), "ticker": t, "ret": round(float(v), 4)} for (d, t), v in top.items()],
        "date_ranges": {
            "prices": [str(calendar.min().date()), str(calendar.max().date())],
            "ff_futures": [str(ff.index.min().date()), str(ff.index.max().date())],
            "gpr": [str(inputs.gpr.index.min().date()), str(inputs.gpr.index.max().date())],
            "dataset": [str(dataset_dates.min().date()), str(dataset_dates.max().date())],
        },
    }


def run_pipeline(inputs: MarketInputs, config: Config, start=None, end=None, log=print) -> RunResults:
    features = build_feature_set(inputs, config)
    dates = features.dataset.index.get_level_values("date")
    log(f"dataset: {len(features.dataset):,} linhas, {dates.min().date()} → {dates.max().date()}")
    result = walk_forward(
        features.dataset, FEATURES, features.calendar, config.train_window_days,
        lambda: make_model(config.lgbm_params), start=start, end=end,
        progress_every=config.progress_every, log=log,
    )
    return RunResults(
        predictions=result.predictions,
        importance=result.importance,
        market=features.market,
        coverage=features.coverage,
        run_info={"features": FEATURES, "data_quality": features.data_quality},
    )
```

- [ ] **Step 6: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_results.py tests/test_pipeline.py -v`
Expected: 7 passed. Se `test_predictions_do_not_depend_on_future_data` falhar, **há lookahead**: não afrouxar o teste, investigar com superpowers:systematic-debugging até achar qual entrada vazou.

- [ ] **Step 7: Suíte inteira**

Run: `venv/bin/python -m pytest -q`
Expected: 55 passed.

---

### Task 10: Carteira (sem custos)

**Files:**
- Create: `engine/portfolio.py`, `tests/test_portfolio.py`

**Interfaces:**
- Consumes: `predictions` e `market` no formato de `RunResults`.
- Produces:
  - `BacktestResult(daily, positions)`:
    - `daily`: índice `holding_date`, colunas `decision_date, gross, bench, rf, n_positions, invested, n_missing`;
    - `positions`: colunas `decision_date, holding_date, ticker, prob, weight, fwd_ret, contribution`.
  - `select_positions(predictions, threshold) -> pd.DataFrame`
  - `run_backtest(predictions, market, threshold) -> BacktestResult`

- [ ] **Step 1: Escrever os testes**

`tests/test_portfolio.py`:

```python
import numpy as np
import pandas as pd
import pytest

from engine.portfolio import run_backtest

D = pd.bdate_range("2022-01-03", periods=5, name="decision_date")


def preds(rows) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["date", "ticker", "prob", "fwd_ret"])
    return frame.assign(label=(frame["fwd_ret"] > 0).astype(float).where(frame["fwd_ret"].notna()))


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
        (D[4], "A", 0.90, 0.100),
    ])


def test_selection_uses_threshold_and_probability_weights(predictions, market):
    positions = run_backtest(predictions, market, 0.55).positions
    day0 = positions[positions["decision_date"] == D[0]].set_index("ticker")
    assert set(day0.index) == {"A", "B"}
    assert day0.loc["A", "weight"] == pytest.approx(0.6 / 1.3)
    assert day0.loc["B", "weight"] == pytest.approx(0.7 / 1.3)


def test_daily_returns_cash_and_missing(predictions, market):
    daily = run_backtest(predictions, market, 0.55).daily
    assert list(daily.index) == list(D[1:5])
    assert daily["decision_date"].tolist() == list(D[:4])
    assert daily["gross"].iloc[0] == pytest.approx(0.6 / 1.3 * 0.01 + 0.7 / 1.3 * -0.02)
    assert daily["gross"].iloc[1] == pytest.approx(0.0002)
    assert daily["gross"].iloc[2] == pytest.approx(0.0)
    assert daily["gross"].iloc[3] == pytest.approx(0.0002)
    assert daily["n_missing"].tolist() == [0, 0, 1, 0]
    assert daily["invested"].tolist() == [True, False, True, False]
    assert daily["bench"].tolist() == pytest.approx([0.01, -0.01, 0.002, 0.003])


def test_selection_never_looks_at_realized_returns(predictions, market):
    shuffled = predictions.assign(fwd_ret=predictions["fwd_ret"].sample(frac=1.0, random_state=3).to_numpy())
    columns = ["decision_date", "ticker", "weight"]
    pd.testing.assert_frame_equal(
        run_backtest(predictions, market, 0.55).positions[columns],
        run_backtest(shuffled, market, 0.55).positions[columns],
    )


def test_threshold_never_met_is_all_cash(predictions, market):
    backtest = run_backtest(predictions, market, 0.99)
    assert backtest.positions.empty
    assert not backtest.daily["invested"].any()
    assert backtest.daily["gross"].tolist() == pytest.approx([0.0002] * 4)


def test_positions_have_holding_dates_and_contributions(predictions, market):
    positions = run_backtest(predictions, market, 0.55).positions
    assert list(positions.columns) == ["decision_date", "holding_date", "ticker", "prob", "weight", "fwd_ret", "contribution"]
    assert (positions["holding_date"] == positions["decision_date"].map(market["holding_date"])).all()
    day0 = positions[positions["decision_date"] == D[0]]
    assert day0["contribution"].sum() == pytest.approx(0.6 / 1.3 * 0.01 + 0.7 / 1.3 * -0.02)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_portfolio.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.portfolio'`.

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


def select_positions(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Stocks above the threshold, weighted by predicted probability; only `prob` drives the choice."""
    chosen = predictions.loc[predictions["prob"] > threshold, ["date", "ticker", "prob", "fwd_ret"]].copy()
    chosen["weight"] = chosen["prob"] / chosen.groupby("date")["prob"].transform("sum")
    return chosen


def run_backtest(predictions: pd.DataFrame, market: pd.DataFrame, threshold: float) -> BacktestResult:
    """A held stock without a realized return (delisting, halt) counts as 0: dropping it would use future information."""
    window = market.loc[predictions["date"].min():predictions["date"].max()]
    window = window[window["bench_fwd_ret"].notna()]
    dates = window.index
    chosen = select_positions(predictions[predictions["date"].isin(dates)], threshold)
    chosen["missing"] = chosen["fwd_ret"].isna()
    chosen["contribution"] = chosen["weight"] * chosen["fwd_ret"].fillna(0.0)
    by_date = chosen.groupby("date")
    n_positions = by_date.size().reindex(dates, fill_value=0)
    invested = n_positions > 0
    stock_return = by_date["contribution"].sum().reindex(dates, fill_value=0.0)
    daily = pd.DataFrame({
        "decision_date": dates.to_numpy(),
        "gross": stock_return.where(invested, window["rf_daily"]).to_numpy(),
        "bench": window["bench_fwd_ret"].to_numpy(),
        "rf": window["rf_daily"].to_numpy(),
        "n_positions": n_positions.to_numpy(),
        "invested": invested.to_numpy(),
        "n_missing": by_date["missing"].sum().reindex(dates, fill_value=0).astype(int).to_numpy(),
    }, index=pd.DatetimeIndex(window["holding_date"], name="holding_date"))
    positions = chosen.rename(columns={"date": "decision_date"})
    positions["holding_date"] = positions["decision_date"].map(window["holding_date"])
    return BacktestResult(daily=daily, positions=positions[POSITION_COLUMNS].reset_index(drop=True))
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_portfolio.py -v`
Expected: 5 passed.

---

### Task 11: Métricas

**Files:**
- Create: `engine/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `TRADING_DAYS = 252`
  - `drawdown(returns) -> pd.Series`, `max_drawdown(returns) -> float`
  - `performance_metrics(returns, rf, benchmark=None, invested=None) -> dict[str, float]`, com as chaves `total_return, cagr, volatility, sharpe, sortino, max_drawdown, calmar, hit_ratio`; com benchmark, soma `beta, alpha, tracking_error, information_ratio`
  - `daily_ic(predictions) -> pd.Series`, `model_metrics(predictions) -> dict` (chaves `auc, accuracy, brier, log_loss, base_rate, ic_mean, ic_tstat, n_predictions`)
  - `rolling_auc(predictions, freq="ME") -> pd.Series`, `calibration_table(predictions, bins=10) -> pd.DataFrame` (colunas `prob_mean, up_rate, n`)
  - `position_hit_ratio(positions) -> float`, `annual_returns(returns) -> pd.Series`

- [ ] **Step 1: Escrever os testes**

`tests/test_metrics.py`:

```python
import numpy as np
import pandas as pd
import pytest

from engine.metrics import (
    annual_returns,
    calibration_table,
    drawdown,
    max_drawdown,
    model_metrics,
    performance_metrics,
    position_hit_ratio,
    rolling_auc,
)


def series(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)), dtype=float)


def test_max_drawdown_counts_initial_capital():
    assert max_drawdown(series([0.1, -0.2, 0.05])) == pytest.approx(0.88 / 1.1 - 1)
    assert max_drawdown(series([-0.1, 0.05])) == pytest.approx(-0.1)
    assert max_drawdown(series([0.01, 0.02])) == 0.0
    assert drawdown(series([-0.1, 0.05])).tolist() == pytest.approx([-0.1, 0.9 * 1.05 - 1])


def test_return_volatility_sharpe_sortino():
    r = series([0.02, 0.0, 0.02, 0.0])
    m = performance_metrics(r, series([0.0] * 4))
    spread = np.std([0.02, 0.0, 0.02, 0.0], ddof=1)
    assert m["total_return"] == pytest.approx(1.02 ** 2 - 1)
    assert m["cagr"] == pytest.approx((1.02 ** 2) ** (252 / 4) - 1)
    assert m["volatility"] == pytest.approx(spread * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(0.01 / spread * np.sqrt(252))
    assert np.isnan(m["sortino"])
    assert m["hit_ratio"] == pytest.approx(0.5)
    assert np.isnan(m["calmar"])


def test_sortino_uses_downside_deviation():
    r = series([0.03, -0.01, 0.02, -0.02])
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    assert performance_metrics(r, series([0.0] * 4))["sortino"] == pytest.approx(r.mean() / downside * np.sqrt(252))


def test_hit_ratio_ignores_cash_days_when_invested_mask_given():
    r = series([0.01, 0.0001, -0.01, 0.02])
    invested = series([1, 0, 1, 1]).astype(bool)
    assert performance_metrics(r, series([0.0001] * 4), invested=invested)["hit_ratio"] == pytest.approx(2 / 3)


def test_beta_alpha_tracking_error():
    b = series(np.random.default_rng(0).normal(0.0005, 0.01, 300))
    m = performance_metrics(2 * b, series(np.zeros(300)), benchmark=b)
    assert m["beta"] == pytest.approx(2.0)
    assert m["alpha"] == pytest.approx(0.0, abs=1e-12)
    assert m["tracking_error"] == pytest.approx(b.std(ddof=1) * np.sqrt(252))
    assert m["information_ratio"] == pytest.approx(b.mean() * 252 / (b.std(ddof=1) * np.sqrt(252)))


def test_model_metrics_on_perfect_ranking():
    predictions = pd.DataFrame({
        "date": np.repeat(pd.bdate_range("2022-01-03", periods=3), 4),
        "prob": np.tile([0.2, 0.4, 0.6, 0.8], 3),
        "fwd_ret": np.tile([-0.02, -0.01, 0.01, 0.02], 3),
    })
    predictions["label"] = (predictions["fwd_ret"] > 0).astype(float)
    m = model_metrics(predictions)
    assert m["auc"] == pytest.approx(1.0)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["base_rate"] == pytest.approx(0.5)
    assert m["ic_mean"] == pytest.approx(1.0)
    assert m["n_predictions"] == 12


def test_calibration_table_has_bins():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 0.7, 1000)
    table = calibration_table(pd.DataFrame({"prob": p, "label": (rng.uniform(size=1000) < p).astype(float)}))
    assert len(table) == 10
    assert table["n"].sum() == 1000
    assert table["prob_mean"].is_monotonic_increasing


def test_annual_returns_and_position_hit_ratio():
    r = pd.Series([0.1, 0.1, -0.5], index=pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-03"]))
    yearly = annual_returns(r)
    assert yearly.loc[2021] == pytest.approx(0.21)
    assert yearly.loc[2022] == pytest.approx(-0.5)
    assert position_hit_ratio(pd.DataFrame({"fwd_ret": [0.01, -0.01, np.nan, 0.02]})) == pytest.approx(2 / 3)


def test_rolling_auc_is_monthly():
    predictions = pd.DataFrame({
        "date": np.repeat(pd.to_datetime(["2022-01-10", "2022-02-10"]), 4),
        "prob": [0.2, 0.4, 0.6, 0.8] * 2,
        "label": [0, 0, 1, 1, 1, 1, 0, 0],
    })
    assert rolling_auc(predictions).tolist() == pytest.approx([1.0, 0.0])
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engine.metrics'`.

- [ ] **Step 3: Implementar**

`engine/metrics.py`:

```python
"""Performance and model-quality metrics for daily data (252 trading days per year)."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

TRADING_DAYS = 252


def drawdown(returns: pd.Series) -> pd.Series:
    """Distance from the running peak; the initial capital (1.0) is the first peak."""
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax().clip(lower=1.0) - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def _annualized(mean: float, spread: float) -> float:
    return float(mean / spread * np.sqrt(TRADING_DAYS)) if spread > 0 else float("nan")


def performance_metrics(returns: pd.Series, rf: pd.Series, benchmark: pd.Series | None = None,
                        invested: pd.Series | None = None) -> dict[str, float]:
    total = float((1.0 + returns).prod() - 1.0)
    cagr = float((1.0 + total) ** (TRADING_DAYS / len(returns)) - 1.0)
    excess = returns - rf
    worst = max_drawdown(returns)
    active_days = returns[invested.astype(bool)] if invested is not None else returns
    out = {
        "total_return": total,
        "cagr": cagr,
        "volatility": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": _annualized(excess.mean(), excess.std(ddof=1)),
        "sortino": _annualized(excess.mean(), float(np.sqrt((excess.clip(upper=0.0) ** 2).mean()))),
        "max_drawdown": worst,
        "calmar": cagr / abs(worst) if worst < 0 else float("nan"),
        "hit_ratio": float((active_days > 0).mean()) if len(active_days) else float("nan"),
    }
    if benchmark is not None:
        bench_excess = benchmark - rf
        beta = float(np.cov(excess, bench_excess, ddof=1)[0, 1] / bench_excess.var(ddof=1))
        active = returns - benchmark
        tracking = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS))
        out.update({
            "beta": beta,
            "alpha": float((excess.mean() - beta * bench_excess.mean()) * TRADING_DAYS),
            "tracking_error": tracking,
            "information_ratio": float(active.mean() * TRADING_DAYS / tracking) if tracking > 0 else float("nan"),
        })
    return out


def daily_ic(predictions: pd.DataFrame) -> pd.Series:
    """Spearman correlation between predicted probability and realized return, per decision date."""
    realized = predictions.dropna(subset=["fwd_ret"])
    return realized.groupby("date")[["prob", "fwd_ret"]].apply(lambda day: day["prob"].corr(day["fwd_ret"], method="spearman"))


def model_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    known = predictions.dropna(subset=["label"])
    label, prob = known["label"].astype(int), known["prob"]
    ic = daily_ic(known).dropna()
    ic_spread = ic.std(ddof=1)
    return {
        "auc": float(roc_auc_score(label, prob)),
        "accuracy": float(((prob > 0.5).astype(int) == label).mean()),
        "brier": float(brier_score_loss(label, prob)),
        "log_loss": float(log_loss(label, prob)),
        "base_rate": float(label.mean()),
        "ic_mean": float(ic.mean()),
        "ic_tstat": float(ic.mean() / ic_spread * np.sqrt(len(ic))) if ic_spread > 0 else float("nan"),
        "n_predictions": int(len(known)),
    }


def rolling_auc(predictions: pd.DataFrame, freq: str = "ME") -> pd.Series:
    known = predictions.dropna(subset=["label"])

    def auc(period: pd.DataFrame) -> float:
        return float(roc_auc_score(period["label"], period["prob"])) if period["label"].nunique() == 2 else float("nan")

    return known.groupby(pd.Grouper(key="date", freq=freq))[["label", "prob"]].apply(auc)


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    known = predictions.dropna(subset=["label"])
    bucket = pd.qcut(known["prob"], q=bins, duplicates="drop")
    table = known.groupby(bucket, observed=True).agg(prob_mean=("prob", "mean"), up_rate=("label", "mean"), n=("label", "size"))
    return table.reset_index(drop=True)


def position_hit_ratio(positions: pd.DataFrame) -> float:
    realized = positions["fwd_ret"].dropna()
    return float((realized > 0).mean()) if len(realized) else float("nan")


def annual_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).groupby(returns.index.year).prod() - 1.0
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_metrics.py -v`
Expected: 9 passed.

---

### Task 12: `main.py run` e o run real

**Files:**
- Modify: `main.py`

**Interfaces:**
- Consumes: `load_inputs`, `run_pipeline`, `save_results`, `run_backtest`, `metrics`.
- Produces: `run(config, start, end) -> None`, `report(results, config) -> None`; subcomando `run [--start YYYY-MM-DD] [--end YYYY-MM-DD]`; `data/out/{predictions,importance,market,coverage}.parquet` + `run_info.json` (com `config`, `runtime_minutes`, `generated_at`, `python`, `versions`, `features`, `data_quality`).

- [ ] **Step 1: Implementar `run` e `report`**

Substituir o `main.py` inteiro por:

```python
"""`python main.py download` baixa os preços; `python main.py run` roda features, walk-forward e backtest."""
import argparse
import dataclasses
import platform
import time
from importlib.metadata import version

import pandas as pd

from config.config import CONFIG, TICKER_ALIASES, Config
from engine import metrics
from engine.data import prices as px
from engine.data import universe
from engine.pipeline import load_inputs, run_pipeline
from engine.portfolio import run_backtest
from engine.results import RunResults, save_results

PACKAGES = ("pandas", "numpy", "lightgbm", "scikit-learn", "yfinance", "streamlit")


def download(config: Config) -> None:
    snapshots = universe.load_constituents(config.data_in / config.constituents_file, TICKER_ALIASES)
    tickers = universe.tickers_since(snapshots, config.price_start) + [config.benchmark_ticker, config.risk_free_ticker]
    prices, failed = px.download_prices(tickers, start=config.price_start)
    path = config.data_in / config.prices_file
    px.save_prices(prices, path)
    print(f"{prices['ticker'].nunique()} tickers salvos em {path} "
          f"({prices['date'].min().date()} → {prices['date'].max().date()})")
    print(f"Sem dados no Yahoo ({len(failed)}): {', '.join(failed)}")


def run(config: Config, start: str | None, end: str | None) -> None:
    started = time.perf_counter()
    results = run_pipeline(load_inputs(config), config, start=start, end=end)
    results.run_info.update({
        "config": dataclasses.asdict(config),
        "runtime_minutes": round((time.perf_counter() - started) / 60.0, 2),
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "versions": {package: version(package) for package in PACKAGES},
    })
    save_results(results, config.data_out)
    print(f"Resultados em {config.data_out}/ ({results.run_info['runtime_minutes']} min)")
    report(results, config)


def report(results: RunResults, config: Config) -> None:
    backtest = run_backtest(results.predictions, results.market, config.threshold)
    daily = backtest.daily
    invested = daily["invested"]
    table = pd.DataFrame({
        "Carteira": metrics.performance_metrics(daily["gross"], daily["rf"], daily["bench"], invested),
        "S&P 500 TR": metrics.performance_metrics(daily["bench"], daily["rf"]),
    })
    print(f"\nBacktest {daily.index.min().date()} → {daily.index.max().date()} · {len(daily)} dias · limiar {config.threshold}")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"Dias investido: {invested.mean():.1%} · posições por dia investido: "
          f"{daily.loc[invested, 'n_positions'].mean():.1f} · hit ratio das posições: "
          f"{metrics.position_hit_ratio(backtest.positions):.1%}")
    print("Modelo:", {name: round(value, 4) for name, value in metrics.model_metrics(results.predictions).items()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest ML do S&P 500")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="baixa preços sem ajuste de dividendos (yfinance)")
    run_command = commands.add_parser("run", help="features, walk-forward e backtest; grava data/out")
    run_command.add_argument("--start", help="primeira data de decisão (YYYY-MM-DD)")
    run_command.add_argument("--end", help="última data de decisão (YYYY-MM-DD)")
    args = parser.parse_args()
    if args.command == "download":
        download(CONFIG)
    else:
        run(CONFIG, args.start, args.end)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Rodada curta (fumaça e tempo)**

Run (timeout 10 min): `venv/bin/python main.py run --start 2026-05-01`
Expected:
- Uma linha `dataset: ... linhas, 2019-12-.. → 2026-07/08-..`, a tabela de métricas, a linha do modelo, e os 5 arquivos em `data/out/`.
- Anotar o `runtime_minutes` e estimar o run completo: (tempo / nº de decisões) × ~1.400.
- **Se a estimativa passar de 60 min, parar e reportar ao usuário antes de adicionar qualquer paralelização.**

- [ ] **Step 3: Run completo (em background)**

Run (Bash com `run_in_background: true`): `venv/bin/python main.py run > data/out/run.log 2>&1`
Expected: o processo termina e `data/out/run.log` traz as linhas de progresso a cada 50 decisões, a tabela final e o modelo.

- [ ] **Step 4: Revisar sinais de alerta antes de seguir**

```bash
venv/bin/python - <<'EOF'
import json, pandas as pd
from config.config import CONFIG
from engine.results import load_results
r = load_results(CONFIG.data_out)
q = r.run_info["data_quality"]
print("datas:", q["date_ranges"]); print("sem dado:", len(q["missing_tickers"]), "| |r|>50%:", q["n_extreme_returns"])
print(pd.DataFrame(q["extreme_returns"]).head(10))
cov = r.coverage.loc[r.predictions["date"].min():]
print((cov["n_eligible"] / cov["n_members"]).describe())
print(r.predictions["prob"].describe())
print("ações acima de 0,55 por dia:", r.predictions.assign(a=r.predictions.prob > 0.55).groupby("date")["a"].sum().describe())
print(r.importance.mean().sort_values(ascending=False).head(10))
EOF
tail -25 data/out/run.log
```

Faixas esperadas (qualquer coisa fora delas é um alerta):
- **AUC do modelo:** entre ~0,48 e ~0,56.
- **Cobertura elegível/membros:** mediana ≥ ~0,85.
- **Probabilidades:** centradas perto da taxa base (~0,5).
- **Suspeita de lookahead (parar e investigar com superpowers:systematic-debugging antes de seguir):** AUC > 0,60, Sharpe > 3 ou CAGR > 60%.
- **Retornos extremos:** se algum deles cair num dia em que a ação estava na carteira, anotar para o usuário (possível spin-off ou erro de dado).

---

### Task 13: App Streamlit

**Files:**
- Create: `app/dashboard.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: `load_results`, `run_backtest(predictions, market, threshold)`, as funções de `engine.metrics`, `CONFIG`, `FEATURES`, `RunResults`, `save_results`.
- Produces: `app/dashboard.py`, que lê o diretório `BACKTEST_RESULTS_DIR` (variável de ambiente; default `data/out`).

- [ ] **Step 1: Escrever o smoke test**

`tests/test_app.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from engine.features import FEATURES
from engine.results import RunResults, save_results

APP = Path(__file__).resolve().parent.parent / "app" / "dashboard.py"


def fake_results(n_days: int = 90, tickers=("AAA", "BBB", "CCC", "DDD", "EEE")) -> RunResults:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=n_days, name="date")
    predictions = pd.DataFrame([(d, t) for d in dates[:-1] for t in tickers], columns=["date", "ticker"])
    predictions["prob"] = rng.uniform(0.45, 0.65, len(predictions))
    predictions["fwd_ret"] = rng.normal(0.0005, 0.02, len(predictions))
    predictions["label"] = (predictions["fwd_ret"] > 0).astype(float)
    market = pd.DataFrame({
        "holding_date": pd.Series(dates, index=dates).shift(-1),
        "bench_fwd_ret": rng.normal(0.0004, 0.01, n_days),
        "rf_daily": 0.0002,
    }, index=pd.DatetimeIndex(dates, name="decision_date"))
    market.iloc[-2:, market.columns.get_loc("bench_fwd_ret")] = np.nan
    return RunResults(
        predictions=predictions,
        importance=pd.DataFrame(rng.dirichlet(np.ones(len(FEATURES)), n_days - 1), index=dates[:-1], columns=FEATURES),
        market=market,
        coverage=pd.DataFrame({"n_members": 5, "n_with_prices": 5, "n_eligible": 5}, index=dates),
        run_info={
            "config": {"train_window_days": 252}, "generated_at": "2026-09-26T12:00:00", "features": FEATURES,
            "data_quality": {"missing_tickers": ["GONE"], "n_extreme_returns": 0, "extreme_returns": [], "date_ranges": {}},
        },
    )


@pytest.fixture
def app(tmp_path, monkeypatch):
    save_results(fake_results(), tmp_path)
    monkeypatch.setenv("BACKTEST_RESULTS_DIR", str(tmp_path))
    return AppTest.from_file(str(APP), default_timeout=120)


def test_dashboard_renders_all_tabs(app):
    app.run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Visão geral", "Carteira por dia", "Recortes de período", "Modelo"]


def test_dashboard_survives_threshold_nobody_passes(app):
    app.run()
    app.sidebar.slider[0].set_value(0.8).run()
    assert not app.exception


def test_dashboard_without_results_shows_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKTEST_RESULTS_DIR", str(tmp_path / "empty"))
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert at.error
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_app.py -v`
Expected: FAIL — o AppTest não encontra `app/dashboard.py`.

- [ ] **Step 3: Implementar o app**

`app/dashboard.py`:

```python
"""Resultados do backtest. Rode com: streamlit run app/dashboard.py

Lê o que `python main.py run` gravou em data/out e nunca retreina o modelo: só recalcula a
carteira, que é barata, para o limiar escolhido na barra lateral.
"""
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
from engine.portfolio import run_backtest
from engine.results import load_results

RESULTS_DIR = Path(os.environ.get("BACKTEST_RESULTS_DIR", ROOT / CONFIG.data_out))
BENCH = "S&P 500 TR"
LEVELS = {"gross": "Carteira"}
CHART_WIDTH = 980
METRIC_FORMATS = {
    "total_return": ("Retorno acumulado", "{:+.2%}"),
    "cagr": ("Retorno anualizado (CAGR)", "{:+.2%}"),
    "volatility": ("Volatilidade anualizada", "{:.2%}"),
    "sharpe": ("Sharpe", "{:.2f}"),
    "sortino": ("Sortino", "{:.2f}"),
    "max_drawdown": ("Máximo drawdown", "{:.2%}"),
    "calmar": ("Calmar", "{:.2f}"),
    "hit_ratio": ("Hit ratio (dias positivos)", "{:.1%}"),
    "beta": ("Beta vs S&P 500", "{:.2f}"),
    "alpha": ("Alfa anualizado", "{:+.2%}"),
    "tracking_error": ("Tracking error", "{:.2%}"),
    "information_ratio": ("Information ratio", "{:.2f}"),
}

st.set_page_config(page_title="Backtest ML — S&P 500", layout="wide")


@st.cache_data(show_spinner=False)
def load(results_dir: str):
    return load_results(Path(results_dir))


@st.cache_data(show_spinner="Recalculando a carteira…")
def backtest(results_dir: str, threshold: float):
    results = load(results_dir)
    return run_backtest(results.predictions, results.market, threshold)


@st.cache_data(show_spinner=False)
def model_evaluation(results_dir: str):
    results = load(results_dir)
    realized = results.market.index[results.market["bench_fwd_ret"].notna()]
    evaluated = results.predictions[results.predictions["date"].isin(realized)]
    return (metrics.model_metrics(evaluated), metrics.rolling_auc(evaluated),
            metrics.daily_ic(evaluated), metrics.calibration_table(evaluated))


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
        name: {label: fmt.format(values[key]) if pd.notna(values.get(key, np.nan)) else "—"
               for key, (label, fmt) in METRIC_FORMATS.items()}
        for name, values in columns.items()
    })


def compare(part: pd.DataFrame) -> dict[str, dict]:
    columns = {label: metrics.performance_metrics(part[level], part["rf"], part["bench"], part["invested"])
               for level, label in LEVELS.items()}
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
    results = load(str(RESULTS_DIR))
except FileNotFoundError:
    st.error(f"Nenhum resultado em `{RESULTS_DIR}`. Rode `python main.py run` primeiro.")
    st.stop()

with st.sidebar:
    st.header("Carteira")
    threshold = st.slider("Limiar de probabilidade", 0.40, 0.80, float(CONFIG.threshold), 0.005, format="%.3f")
    st.caption(
        "Compra as ações com probabilidade prevista acima do limiar, com peso proporcional à probabilidade; "
        "se nenhuma passar, fica em caixa rendendo a T-bill (^IRX). Escolher o limiar olhando este resultado "
        f"é otimizar dentro do próprio backtest (data snooping). O default ({CONFIG.threshold}) foi fixado a priori."
    )

bt = backtest(str(RESULTS_DIR), threshold)
daily, positions = bt.daily, bt.positions
if daily.empty:
    st.warning("Nenhum dia de carteira com retorno realizado nos resultados.")
    st.stop()

st.title("Backtest ML — S&P 500")
st.caption(f"{len(daily)} dias de carteira · {daily.index.min().date()} → {daily.index.max().date()} · "
           f"gerado em {results.run_info.get('generated_at', '?')}")
window_coverage = results.coverage.loc[daily["decision_date"].min():daily["decision_date"].max()]
tab_overview, tab_day, tab_periods, tab_model = st.tabs(["Visão geral", "Carteira por dia", "Recortes de período", "Modelo"])

with tab_overview:
    invested = daily["invested"]
    cols = st.columns(5)
    cols[0].metric("Janela de treino", f"{results.run_info.get('config', {}).get('train_window_days', '?')} pregões")
    cols[1].metric("Limiar", f"{threshold:.3f}")
    cols[2].metric("Dias investido", f"{invested.mean():.1%}")
    cols[3].metric("Ações por dia investido", f"{daily.loc[invested, 'n_positions'].mean():.1f}" if invested.any() else "—")
    cols[4].metric("Hit ratio das posições", f"{metrics.position_hit_ratio(positions):.1%}" if len(positions) else "—")
    st.caption(
        f"Modelo LightGBM retreinado todo pregão · em média {window_coverage['n_eligible'].mean():.0f} ações elegíveis "
        f"por dia, de {window_coverage['n_members'].mean():.0f} membros do índice "
        f"({(window_coverage['n_eligible'] / window_coverage['n_members']).mean():.0%} de cobertura)."
    )

    st.subheader("Carteira vs S&P 500 Total Return")
    st.dataframe(metrics_table(compare(daily)))
    line_chart(curves(daily, growth), "Capital (1,0 no início)")
    st.caption("Decisão após o fechamento de t, compra na abertura de t+1 e rebalanceamento na abertura seguinte "
               "(retornos abertura → abertura, com dividendos). Arraste ou use a roda do mouse para aproximar.")
    st.markdown("**Drawdown**")
    line_chart(curves(daily, metrics.drawdown), "Drawdown", height=220, area=True)
    st.markdown("**Volatilidade anualizada rolante (63 dias)**")
    line_chart(curves(daily, lambda r: r.rolling(63, min_periods=20).std() * np.sqrt(252)), "Volatilidade", height=220)
    st.markdown("**Exposição: nº de ações na carteira**")
    line_chart(daily[["n_positions"]].rename(columns={"n_positions": "Ações"}), "Ações", height=220, y_format=".0f")
    st.markdown("**Retorno por ano**")
    st.dataframe(curves(daily, metrics.annual_returns).style.format("{:+.2%}"))

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
    cols = st.columns(4)
    cols[0].metric("Retorno da carteira", f"{day['gross']:+.2%}")
    cols[1].metric(BENCH, f"{day['bench']:+.2%}")
    cols[2].metric("Ações", int(day["n_positions"]))
    cols[3].metric("Decisão", str(decision_date.date()))
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
        st.info(f"Reconciliação: soma das contribuições = {held['contribution'].sum():+.4%}, "
                f"igual ao retorno bruto do dia ({day['gross']:+.4%}).")
        if day["n_missing"]:
            st.warning(f"{int(day['n_missing'])} ação(ões) sem retorno realizado (deslistagem/halt) contada(s) como 0%.")

    day_predictions = results.predictions[results.predictions["date"] == decision_date]
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

with tab_model:
    summary, monthly_auc, ic, calibration = model_evaluation(str(RESULTS_DIR))
    cols = st.columns(5)
    cols[0].metric("AUC", f"{summary['auc']:.3f}")
    cols[1].metric("Acurácia (corte 0,5)", f"{summary['accuracy']:.1%}")
    cols[2].metric("Brier", f"{summary['brier']:.4f}")
    cols[3].metric("IC médio", f"{summary['ic_mean']:+.4f}", help=f"t-stat {summary['ic_tstat']:.2f}")
    cols[4].metric("Taxa base (subiu)", f"{summary['base_rate']:.1%}")
    st.caption(f"{summary['n_predictions']:,} previsões out-of-sample. IC = correlação de Spearman entre a "
               "probabilidade prevista e o retorno realizado, por dia.")
    st.markdown("**AUC por mês**")
    line_chart(monthly_auc.to_frame("AUC"), "AUC", height=220)
    st.markdown("**IC diário (média móvel de 63 dias)**")
    line_chart(ic.rolling(63, min_periods=20).mean().to_frame("IC"), "IC", height=220)

    st.markdown("**Calibração**: probabilidade prevista (decis) vs frequência com que a ação de fato subiu")
    low, high = float(calibration["prob_mean"].min()), float(calibration["prob_mean"].max())
    points = alt.Chart(calibration).mark_line(point=True).encode(
        x=alt.X("prob_mean:Q", title="Probabilidade prevista", scale=alt.Scale(zero=False)),
        y=alt.Y("up_rate:Q", title="Frequência realizada de alta", scale=alt.Scale(zero=False)),
        tooltip=["prob_mean:Q", "up_rate:Q", "n:Q"],
    )
    diagonal = alt.Chart(pd.DataFrame({"x": [low, high], "y": [low, high]})).mark_line(
        color="gray", strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    st.altair_chart((points + diagonal).properties(width=600, height=360), width="content")

    st.markdown("**Ações acima do limiar por dia**")
    in_window = results.predictions[results.predictions["date"].isin(daily["decision_date"])]
    above = in_window.assign(above=in_window["prob"] > threshold).groupby("date")["above"].sum()
    line_chart(above.to_frame("Acima do limiar"), "Ações", height=220, y_format=".0f")

    st.markdown("**Importância das features** (média do ganho normalizado nos retreinos)")
    importance = results.importance.mean().sort_values(ascending=False).rename_axis("feature").reset_index(name="importância")
    st.altair_chart(
        alt.Chart(importance).mark_bar().encode(
            x=alt.X("importância:Q", title="Participação no ganho"), y=alt.Y("feature:N", sort="-x", title=None),
        ).properties(width=CHART_WIDTH, height=18 * len(importance)),
        width="content",
    )

    st.markdown("**Cobertura do universo**: membros do índice, com preço no Yahoo e elegíveis ao modelo")
    line_chart(window_coverage.rename(columns={
        "n_members": "Membros do índice", "n_with_prices": "Com preço", "n_eligible": "Elegíveis"}), "Ações", height=240, y_format=".0f")
    quality = results.run_info.get("data_quality", {})
    with st.expander("Qualidade dos dados"):
        missing = quality.get("missing_tickers", [])
        st.write(f"Tickers do índice sem dado no Yahoo (deslistados): {len(missing)}")
        st.write(", ".join(missing) or "—")
        st.write(f"Retornos diários com |r| > 50% entre membros (erro de dado ou spin-off): {quality.get('n_extreme_returns', 0)}")
        if quality.get("extreme_returns"):
            st.dataframe(pd.DataFrame(quality["extreme_returns"]), hide_index=True)
        st.write(f"Posições sem retorno realizado (contadas como 0%): {int(daily['n_missing'].sum())}")
        st.json(quality.get("date_ranges", {}))
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_app.py -v`
Expected: 3 passed. Se o Streamlit avisar sobre um parâmetro deprecado, trocar pela forma nova que o aviso indicar.

- [ ] **Step 5: Subir o app de verdade (headless) sobre os resultados reais**

```bash
venv/bin/streamlit run app/dashboard.py --server.headless true --server.port 8599 > /tmp/claude-streamlit.log 2>&1 &
sleep 8; curl -s http://localhost:8599/_stcore/health; echo; grep -iE "error|traceback" /tmp/claude-streamlit.log; kill %1
```

Expected: `ok` e nenhum erro no log. Avisar o usuário que a checagem visual no navegador é com ele (`streamlit run app/dashboard.py`).

---

### Task 14: Custos de transação

**Files:**
- Modify: `engine/portfolio.py`, `tests/test_portfolio.py`, `app/dashboard.py`, `main.py`

**Interfaces:**
- Produces: `run_backtest(predictions, market, threshold, cost_bps: float = 0.0)`; `daily` ganha `net, turnover, cost`.

- [ ] **Step 1: Escrever os testes de custo**

Acrescentar ao fim de `tests/test_portfolio.py`:

```python
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
    daily = run_backtest(predictions, market, 0.55, cost_bps=10.0).daily
    assert daily["turnover"].tolist() == pytest.approx([1.0, 0.1, 2.0, 1.0, 1.0])
    assert daily["cost"].tolist() == pytest.approx([0.001, 0.0001, 0.002, 0.001, 0.001])
    assert daily["net"].iloc[0] == pytest.approx((1 + daily["gross"].iloc[0]) * (1 - 0.001) - 1)
    assert daily["net"].iloc[3] == pytest.approx((1 + 0.0001) * (1 - 0.001) - 1)


def test_zero_cost_means_net_equals_gross(predictions, market):
    daily = run_backtest(predictions, market, 0.55).daily
    assert daily["net"].tolist() == pytest.approx(daily["gross"].tolist())
    assert (daily["cost"] == 0).all()
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `venv/bin/python -m pytest tests/test_portfolio.py -v`
Expected: FAIL nos dois testes novos — `TypeError: run_backtest() got an unexpected keyword argument 'cost_bps'` e `KeyError: 'net'`.

- [ ] **Step 3: Implementar os custos**

Em `engine/portfolio.py`, substituir `run_backtest` e acrescentar `_turnover`:

```python
def run_backtest(predictions: pd.DataFrame, market: pd.DataFrame, threshold: float, cost_bps: float = 0.0) -> BacktestResult:
    """A held stock without a realized return (delisting, halt) counts as 0: dropping it would use future information."""
    window = market.loc[predictions["date"].min():predictions["date"].max()]
    window = window[window["bench_fwd_ret"].notna()]
    dates = window.index
    chosen = select_positions(predictions[predictions["date"].isin(dates)], threshold)
    chosen["missing"] = chosen["fwd_ret"].isna()
    chosen["contribution"] = chosen["weight"] * chosen["fwd_ret"].fillna(0.0)
    by_date = chosen.groupby("date")
    n_positions = by_date.size().reindex(dates, fill_value=0)
    invested = n_positions > 0
    stock_return = by_date["contribution"].sum().reindex(dates, fill_value=0.0)
    gross = stock_return.where(invested, window["rf_daily"])
    turnover = _turnover(chosen, dates, stock_return)
    cost = turnover * cost_bps / 10_000.0
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


def _turnover(chosen: pd.DataFrame, dates: pd.DatetimeIndex, stock_return: pd.Series) -> pd.Series:
    """One-way turnover at each open: target weights vs yesterday's weights after they drifted with prices."""
    weights = chosen.pivot(index="date", columns="ticker", values="weight").reindex(dates).fillna(0.0)
    growth = 1.0 + chosen.pivot(index="date", columns="ticker", values="fwd_ret").reindex(dates).fillna(0.0)
    drifted = (weights * growth).div(1.0 + stock_return, axis=0)
    return (weights - drifted.shift(1).fillna(0.0)).abs().sum(axis=1)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `venv/bin/python -m pytest tests/test_portfolio.py -v`
Expected: 7 passed.

- [ ] **Step 5: Custos no app**

Em `app/dashboard.py`:

1. `LEVELS = {"gross": "Carteira"}` → `LEVELS = {"gross": "Carteira (bruta)", "net": "Carteira (líquida)"}`
2. Trocar a função `backtest` por:

```python
@st.cache_data(show_spinner="Recalculando a carteira…")
def backtest(results_dir: str, threshold: float, cost_bps: float):
    results = load(results_dir)
    return run_backtest(results.predictions, results.market, threshold, cost_bps)
```

3. Na sidebar, logo depois do `threshold = st.slider(...)`:

```python
    cost_bps = st.number_input("Custo de transação (bps, one-way)", 0.0, 50.0, float(CONFIG.cost_bps), 0.5)
```

4. `bt = backtest(str(RESULTS_DIR), threshold)` → `bt = backtest(str(RESULTS_DIR), threshold, cost_bps)`
5. Na aba "Visão geral", logo depois do bloco `cols = st.columns(5)` (as 5 métricas):

```python
    cost_cols = st.columns(3)
    cost_cols[0].metric("Turnover médio por dia", f"{daily['turnover'].mean():.1%}")
    cost_cols[1].metric("Custo total (soma)", f"{daily['cost'].sum():.2%}")
    cost_cols[2].metric("Custo médio por ano", f"{daily['cost'].mean() * 252:.2%}")
```

6. Na aba "Carteira por dia", trocar o bloco `cols = st.columns(4)` e as 4 métricas por:

```python
    cols = st.columns(5)
    cols[0].metric("Retorno bruto", f"{day['gross']:+.2%}")
    cols[1].metric("Retorno líquido", f"{day['net']:+.2%}", help=f"custo do dia {day['cost']:.3%} (turnover {day['turnover']:.1%})")
    cols[2].metric(BENCH, f"{day['bench']:+.2%}")
    cols[3].metric("Ações", int(day["n_positions"]))
    cols[4].metric("Decisão", str(decision_date.date()))
```

- [ ] **Step 6: Custos no `main.py`**

Em `report`, trocar as linhas do `backtest = run_backtest(...)` até o `print(table...)` por:

```python
    backtest = run_backtest(results.predictions, results.market, config.threshold, config.cost_bps)
    daily = backtest.daily
    invested = daily["invested"]
    table = pd.DataFrame({
        "Carteira (bruta)": metrics.performance_metrics(daily["gross"], daily["rf"], daily["bench"], invested),
        "Carteira (líquida)": metrics.performance_metrics(daily["net"], daily["rf"], daily["bench"], invested),
        "S&P 500 TR": metrics.performance_metrics(daily["bench"], daily["rf"]),
    })
    print(f"\nBacktest {daily.index.min().date()} → {daily.index.max().date()} · {len(daily)} dias · "
          f"limiar {config.threshold} · custo {config.cost_bps} bps")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"Turnover médio: {daily['turnover'].mean():.1%}/dia · custo total: {daily['cost'].sum():.2%}")
```

(as duas linhas finais de `print`, com a exposição e o modelo, continuam iguais).

- [ ] **Step 7: Suíte inteira e relatório com custos**

Run: `venv/bin/python -m pytest -q && venv/bin/python -c "from config.config import CONFIG; from engine.results import load_results; from main import report; report(load_results(CONFIG.data_out), CONFIG)"`
Expected: todos os testes passando e a tabela com as colunas bruta, líquida e S&P 500 TR, com o custo total > 0.

---

### Task 15: Documentação, simplificação e verificação final

**Files:**
- Modify: `README.md`, `.claude/CLAUDE.md`

- [ ] **Step 1: README**

Substituir `README.md` por:

````markdown
# Stock Behavior Prediction — S&P 500 daily direction backtest

Every trading day, after the close, a LightGBM classifier retrained on the last *X* trading days estimates, for each stock that is in the S&P 500 **on that day**, the probability that it goes up over the next portfolio day. The long-only portfolio holds the stocks whose probability is above a threshold, weighted by that probability; if none qualifies it stays in cash at the 13-week T-bill rate. Results are compared with the S&P 500 Total Return index in a Streamlit app.

## Quick start

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python main.py download        # prices without dividend adjustment (yfinance) → data/in/prices.parquet
venv/bin/python main.py run             # features, walk-forward and backtest → data/out/
venv/bin/streamlit run app/dashboard.py # results app (never retrains)
venv/bin/python -m pytest               # includes the lookahead tests
```

`main.py run --start YYYY-MM-DD --end YYYY-MM-DD` limits the decision dates (handy for quick runs).

## Time convention (no lookahead)

| Step | When | Uses |
|---|---|---|
| Decision | after the close of *t* | prices/volumes/dividends dated ≤ *t*, index membership ≤ *t*, fed funds futures ≤ *t*, GPR ≤ *t* − 7 days |
| Execution | open of *t+1*, rebalanced at the open of *t+2* | — |
| Realized return / target | `(Open[t+2] + Div[t+2]) / Open[t+1] − 1`, label = return > 0 | — |
| Training for decision *t* | rows dated *t* − X − 1 … *t* − 2 | only labels already known at the close of *t* |

`tests/test_pipeline.py::test_predictions_do_not_depend_on_future_data` runs the whole pipeline twice — once with every input dated after a cutoff scrambled — and requires identical predictions up to the cutoff.

## Data (`data/in/`)

- `sp500_historical_constituents.csv` — point-in-time membership snapshots; renamed tickers are mapped in `config.TICKER_ALIASES`.
- `prices.parquet` — Yahoo daily bars with `auto_adjust=False, actions=True` (split-adjusted, not dividend-adjusted); `Adj Close` is never stored because it embeds future dividends. Includes `^SP500TR` (benchmark) and `^IRX` (risk-free).
- `ff_futures_daily.csv` — 30-day fed funds futures: implied rate ~12 months ahead, its 21-day change and the slope vs the current month.
- `data_gpr_daily_recent.xls` — Caldara & Iacoviello daily geopolitical risk index, used with a 7-day publication lag.

## Model and portfolio

- 36 causal features (`engine/features.py`): stock returns/momentum/volatility/trend/volume/dividend yield/beta, cross-sectional ranks among index members, market and breadth, fed funds and GPR.
- LightGBM with fixed conservative parameters (`config/config.py`), retrained every day (sequential loop).
- Weights `prob_i / Σ prob` over stocks with `prob > threshold` (default 0.55, fixed a priori); transaction costs in bps of one-way turnover measured against drifted weights (default 5 bps).
- Metrics: cumulative return, CAGR, volatility, Sharpe/Sortino (excess over ^IRX), max drawdown, Calmar, hit ratios, beta/alpha/information ratio vs the benchmark; model AUC, IC and calibration.

## Known limitations

- Delisted companies are not on Yahoo, so they drop out of the universe (survivorship bias); the app plots daily coverage.
- Yahoo does not adjust closes for spin-offs; rare fake drops are counted in `run_info.json` but not corrected.
- The backtest starts when the fed funds futures file allows (Dec 2020 decisions with X = 252) and ends where it ends.

## Layout

```
config/config.py     all parameters, ticker aliases, LightGBM params
main.py              CLI: download | run
app/dashboard.py     Streamlit app
engine/data/         universe (membership), prices (yfinance, returns), macro (futures, GPR, risk-free)
engine/features.py   causal features      engine/dataset.py   (date, ticker) table + target
engine/walk_forward.py  daily retraining  engine/pipeline.py  orchestration
engine/portfolio.py  selection, weights, costs   engine/metrics.py   metrics
engine/results.py    data/out persistence
tests/               unit tests, synthetic market, end-to-end lookahead test, app smoke test
```
````

- [ ] **Step 2: CLAUDE.md**

Em `.claude/CLAUDE.md`, substituir as seções `## Status`, `## Working conventions` (manter a primeira linha, sobre commits) e `## Engine layout` por:

```markdown
## Status

Rebuilt on 2026-09-26 (spec: `docs/superpowers/specs/2026-09-26-ml-backtest-redesign-design.md`). Daily LightGBM direction model over the point-in-time S&P 500, long-only probability-weighted portfolio, Streamlit app.

## Working conventions

- **Never run `git commit` or `git push`.** This is blocked at the permission level (see `.claude/settings.json`, `permissions.deny`). Commits and pushes are entirely the user's responsibility — prepare and stage changes if asked, but leave committing to them.
- Use the project venv: `venv/bin/python -m pytest`, `venv/bin/python main.py download|run`, `venv/bin/streamlit run app/dashboard.py`.
- All parameters live in `config/config.py` (`Config`, `TICKER_ALIASES`, `LGBM_PARAMS`).
- Lookahead invariants (keep `tests/test_pipeline.py` and `tests/test_features.py` green): decision after the close of t with data dated ≤ t (GPR ≤ t−7); execution open t+1 → open t+2; training rows t−X−1 … t−2; never use Yahoo's `Adj Close`.
- No custom parallelization unless a full run gets too slow; `data/reference/` is not to be imported.

## Engine layout

- `engine/data/universe.py` — constituent snapshots → membership matrix (aliases applied).
- `engine/data/prices.py` — yfinance download (split-adjusted, dividends separate), calendar, wide panel, total and forward open-to-open returns.
- `engine/data/macro.py` — fed funds futures rates, GPR features, as-of alignment with lag/staleness, ^IRX risk-free.
- `engine/features.py`, `engine/dataset.py` — 36 causal features and the (date, ticker) training table.
- `engine/model.py`, `engine/walk_forward.py` — LightGBM factory and daily rolling retraining.
- `engine/pipeline.py`, `engine/results.py` — orchestration and `data/out` persistence.
- `engine/portfolio.py`, `engine/metrics.py` — portfolio (threshold, probability weights, cash at rf, costs) and metrics.
```

(a seção `## Plugins available` continua igual).

- [ ] **Step 3: Passada do code-simplifier**

Despachar o agente `code-simplifier:code-simplifier` com o prompt:

> Simplifique para clareza e consistência, **sem mudar comportamento**: `engine/**/*.py`, `main.py`, `app/dashboard.py`, `config/config.py`. Restrições:
> - `venv/bin/python -m pytest -q` tem que continuar 100% verde;
> - não mude a convenção de tempo (decisão após o fechamento de t, execução abertura t+1 → abertura t+2, treino t−X−1…t−2, GPR com lag de 7 dias);
> - não mude assinaturas públicas usadas pelos testes;
> - não adicione comentários nem docstrings longas;
> - não introduza paralelização;
> - não rode `git commit`.
>
> Relate o que mudou, arquivo por arquivo.

Depois, revisar o diff (`git diff --stat` e `git diff` dos arquivos tocados) e rodar `venv/bin/python -m pytest -q`.
Expected: testes verdes; mudanças só de forma.

- [ ] **Step 4: Verificação final**

Usar superpowers:verification-before-completion e rodar:

```bash
venv/bin/python -m pytest -q
venv/bin/python -m pytest tests/test_pipeline.py::test_predictions_do_not_depend_on_future_data tests/test_features.py -v
venv/bin/python -c "from config.config import CONFIG; from engine.results import load_results; from main import report; report(load_results(CONFIG.data_out), CONFIG)"
git status --short
```

Expected:
- Tudo verde e o relatório impresso.
- `git status` mostra só as mudanças esperadas: o código antigo apagado, os arquivos novos e `docs/`. Nada em `data/in` além do `prices.parquet`, que é ignorado.

- [ ] **Step 5: Revisão final independente**

Usar superpowers:requesting-code-review para um revisor novo checar o diff completo (`git diff HEAD` + arquivos novos) contra o spec. Foco:
- lookahead;
- coerência entre alvo, execução e contabilidade;
- os 5 itens de Review Focus.

Corrigir o que for confirmado (superpowers:receiving-code-review) e rodar a suíte de novo.
