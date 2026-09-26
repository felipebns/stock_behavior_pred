# Redesign do backtest de ML — Design

- **Data:** 2026-09-26
- **Status:** design aprovado em conversa; este arquivo aguarda revisão.
- **Escopo:** substitui quase todo o código atual (`engine/`, `main.py`, `config/`, `tests/`, `output/`).

## 1. Objetivo e critérios de sucesso

Pipeline de pesquisa:

1. Todo pregão *t*, estimar para cada ação que está no S&P 500 **naquele dia** a probabilidade de ela **subir** no próximo dia de carteira.
2. Usar ML clássico treinado numa janela rolante de *X* pregões.
3. Montar uma carteira long-only só com as ações de probabilidade acima de um limiar.
4. Comparar a carteira com o S&P 500 Total Return num app.

**Sucesso significa:**

- `python main.py download` e `python main.py run` rodam de ponta a ponta nos dados reais.
- `streamlit run app/dashboard.py` mostra desempenho, métricas e o que foi comprado em cada dia.
- A suíte `pytest` está verde, incluindo os testes de lookahead da seção 9.
- Nenhum código de `data/reference/` é importado ou copiado. A pasta serviu só de referência e será apagada.

**Não-objetivos:**

- Otimizar hiperparâmetros ou limiar.
- Deep learning.
- Execução ao vivo.
- Paralelização própria (só entra se o run real ficar lento demais).
- Dados de empresas deslistadas.

## 2. Convenção de tempo (regra que governa todo o resto)

**Decisão em *t*.** Ocorre depois do fechamento do pregão *t*. Só pode usar:

| Fonte | Informação permitida na decisão *t* |
|---|---|
| Preços, volume, dividendos, splits das ações e índices | data ≤ *t* |
| Composição do S&P 500 | snapshot mais recente com data ≤ *t* |
| Fed Funds futures (barra diária por dia UTC, fecha ~19–20h ET) | data ≤ *t* |
| GPR diário | data ≤ *t* − 7 dias corridos (defasagem de publicação) |
| Taxa livre de risco (^IRX) | data ≤ *t* |

**Execução.** Compra na abertura de *t+1* e rebalanceia na abertura de *t+2*. Aqui *t+1* e *t+2* são os próximos pregões do calendário.

**Retorno realizado da linha (i, t):**

```
fwd_ret(i, t) = (Open[i, t+2] + Div[i, t+2]) / Open[i, t+1] − 1
```

`Div[t+2]` é o dividendo com data-ex em *t+2*. Ele pertence a quem segurou a ação no fechamento de *t+1*.

**Alvo.** `label(i, t) = 1 se fwd_ret(i, t) > 0, senão 0`. Se `fwd_ret` for NaN, o rótulo também é NaN.

**Disponibilidade do rótulo.** O rótulo da linha *s* só é conhecido na abertura de *s+2*. Por isso, na decisão *t*, o treino usa **exatamente** as linhas com *s* ∈ [*t* − *X* − 1, *t* − 2]: são *X* pregões, e a linha *t* − 1 fica de fora.

## 3. Dados

### 3.1 Composição do índice

- **Fonte:** `data/in/sp500_historical_constituents.csv`, com colunas `date` e `tickers` (lista separada por vírgula).
- **Membros em *t*:** snapshot mais recente com `date ≤ t`. Antes do primeiro snapshot, o universo é vazio.
- **Formato dos tickers:** internamente tudo segue o formato do Yahoo (`.` → `-`, ou seja, `BRK.B` → `BRK-B`).
- **Aliases de renomeação** (`config.TICKER_ALIASES`): cada ticker antigo do snapshot é trocado pelo ticker atual. Só entram renomeações comprovadas no próprio arquivo, isto é, troca 1:1 na mesma data com um sucessor que tem histórico no Yahoo:

  ```
  HRS→LHX  TMK→GL   BHGE→BKR  JEC→J     CTL→LUMN  MYL→VTRS  WLTW→WTW
  DISCA→WBD BLL→BALL ANTM→ELV SYMC→GEN NLOK→GEN PKI→RVTY RE→EG
  ABC→COR  HCP→DOC  PEAK→DOC  FLT→CPAY CBS→PSKY VIAC→PSKY PARA→PSKY
  FI→FISV  MMC→MRSH BK→BNY    UTX→RTX  ARNC→HWM DWDP→DD
  ```

  Depois do alias, duplicatas no mesmo snapshot são removidas.
- **Saída:** matriz booleana `membership` (pregões × tickers).

### 3.2 Preços das ações (download)

- **Comando:** `python main.py download`.
- **Chamada:** `yfinance.download(..., auto_adjust=False, actions=True)` em lotes de 100 tickers.
- **Tickers baixados:** a união dos membros (já com alias) de todos os snapshots desde o snapshot vigente em `price_start = 2018-06-01`, mais `^SP500TR` e `^IRX`. O início em 2018-06 aquece as janelas de 252 pregões antes do começo do dataset.
- **Campos guardados:** `open, high, low, close, volume, dividends, splits`. O **`Adj Close` é descartado dentro da função de download**, porque ele embute dividendos futuros e nunca pode ser usado.
- **Convenção do Yahoo (verificada com AAPL em 2020):**
  - `Close`, `Open` etc. vêm ajustados por **split**, mas não por dividendos.
  - `Dividends` também vem ajustado por split, então fica na mesma base do preço.
  - Isso não vaza informação porque nenhuma feature usa nível absoluto de preço (só razões). Assim, retornos totais e features ficam corretos sem precisar desfazer os splits.
- **Limpeza mínima:**
  - `open ≤ 0` vira NaN.
  - `dividends` e `splits` NaN viram 0.
  - Linhas totalmente vazias são descartadas.
  - Nada é interpolado.
- **Armazenamento:** `data/in/prices.parquet`, em formato longo (`date, ticker, open, high, low, close, volume, dividends, splits`). O download registra os tickers que falharam.
- **Leitura:** as colunas são pivotadas em matrizes largas (pregões × tickers).
- **Calendário:** os pregões são as datas em que `^SP500TR` tem fechamento.

### 3.3 Benchmark e taxa livre de risco

- **Benchmark:** `^SP500TR`, S&P 500 Total Return. O retorno do dia de carteira tem a mesma convenção da carteira: `Open[t+2] / Open[t+1] − 1`.
- **Taxa livre de risco:** `^IRX`, T-bill de 13 semanas em % a.a.
  - `rf_daily(t) = IRX[t] / 100 / 252`, com as-of ≤ *t* e tolerância de 5 dias corridos.
  - Rende o caixa quando nenhuma ação passa o limiar.
  - Também é a base de Sharpe, Sortino e alfa.

### 3.4 Fed Funds futures

- **Fonte:** `data/in/ff_futures_daily.csv`, com colunas `Date, symbol, expiration, Open, High, Low, Close, Volume`, de 2019-11-01 a 2026-07-31.
- **Barras de domingo** (sessão parcial da Globex) são descartadas.
- **Taxa implícita de um contrato:** `(100 − Close) / 100`.
- **`ff_rate_12m`:** em cada data, o contrato cuja expiração é a mais próxima de data + 12 meses. Se nenhum contrato ficar a até 45 dias dessa data-alvo, o dia é descartado.
- **`ff_rate_front`:** o contrato de menor expiração ≥ data, ou seja, o do mês corrente.
- **Features:**
  - `ff_rate_12m`.
  - `ff_rate_12m_chg`: diferença em 21 pregões da série de futuros. Troca o `pct_change` da referência, que explode com taxa perto de zero (2020–21).
  - `ff_slope = ff_rate_12m − ff_rate_front`.
- **Junção com o calendário:** as-of ≤ *t* com tolerância de 5 dias corridos. Fora da tolerância vira NaN, e a linha sai do dataset.

### 3.5 GPR diário

- **Fonte:** `data/in/data_gpr_daily_recent.xls`, com colunas `date, GPRD, GPRD_ACT, GPRD_THREAT` em dias corridos de 1985 a 2026-09-21. As médias do arquivo são ignoradas e recalculadas.
- **Médias próprias**, calculadas sobre dias corridos que terminam na data *d*: `gpr_7d`, `gpr_30d`, `act_7d`, `threat_7d`.
- **Features na decisão *t*:** valores com *d* = última data do GPR ≤ *t* − 7 dias.
  - `gpr_level = log(gpr_7d)`
  - `gpr_trend = log(gpr_7d) − log(gpr_30d)`
  - `gpr_threat_act = log(threat_7d) − log(act_7d)`

### 3.6 Período

O fator limitante são os Fed Funds futures (a partir de 11/2019, com 21 pregões de aquecimento):

- O dataset começa por volta de 2019-12 e termina na última data com futuros válidos (2026-07-31, mais a tolerância).
- Com *X* = 252, as decisões vão de ~12/2020 até o fim do dataset.

### 3.7 Limitações conhecidas (documentadas no README e visíveis no app)

- **Viés de sobrevivência residual.** O Yahoo não tem empresas deslistadas. Constituintes sem dado ficam fora do universo daquele dia. A cobertura diária (membros com dado / membros) é salva e plotada.
- **Spin-offs.** O Yahoo não os ajusta no `Close`, o que gera quedas "falsas" raras. Não são corrigidas.
- **Retornos diários com |r| > 50%.** São contados no `run_info` como possíveis erros de dado, mas não são alterados.
- **Ticker reciclado.** Um ticker só entra no universo nas datas em que é membro e tem dado, o que já exclui casos como o `NFX` atual.

## 4. Features (36, todas causais)

**Notação:**

- `C, O, H, L, V, D` são matrizes pregões × tickers.
- `Cf = C.ffill()`.
- `r = (C + D) / Cf.shift(1) − 1` é o retorno total diário.
- `TRI = (1 + r.fillna(0)).cumprod()`, mascarado com NaN antes do primeiro fechamento válido.
- `M` é o fechamento do `^SP500TR` e `m = M / M.shift(1) − 1`.
- Janelas rolantes exigem pelo menos 80% de observações válidas.

**Ação:**

| # | Feature | Definição |
|---|---|---|
| 1 | `ret_1d` | `r` |
| 2–4 | `ret_5d`, `ret_21d`, `ret_63d` | `TRI / TRI.shift(k) − 1` |
| 5 | `mom_12_1` | `TRI.shift(21) / TRI.shift(252) − 1` |
| 6 | `gap_1d` | `(O + D) / Cf.shift(1) − 1` |
| 7 | `intraday_1d` | `C / O − 1` |
| 8–9 | `vol_21d`, `vol_63d` | desvio-padrão rolante de `r` |
| 10 | `vol_ratio` | `vol_21d / vol_63d` |
| 11 | `range_21d` | média rolante 21 de `(H − L) / C` |
| 12–13 | `dist_ma50`, `dist_ma200` | `TRI / média rolante(TRI) − 1` |
| 14 | `dist_high_252` | `C / máximo rolante 252(C) − 1` |
| 15 | `rsi_14` | `100 · ganho_médio / (ganho_médio + perda_média)` com média rolante 14 de `r` |
| 16 | `volume_ratio` | `V / média rolante 21(V)` |
| 17 | `log_dollar_volume` | `log(média rolante 21(C · V))` |
| 18 | `div_yield_252` | `soma rolante 252(D) / C` |
| 19 | `beta_63d` | `cov_63(r, m) / var_63(m)` |

**Cross-section** (rank percentual no dia, só entre os membros de *t*):

| # | Feature |
|---|---|
| 20–24 | `xs_ret_1d`, `xs_ret_5d`, `xs_ret_21d`, `xs_mom_12_1`, `xs_vol_21d` |

**Mercado:**

| # | Feature | Definição |
|---|---|---|
| 25–27 | `mkt_ret_1d`, `mkt_ret_5d`, `mkt_ret_21d` | retornos de `M` |
| 28 | `mkt_vol_21d` | desvio-padrão rolante 21 de `m` |
| 29 | `mkt_dist_ma200` | `M / média rolante 200(M) − 1` |
| 30 | `breadth_ma50` | fração dos membros de *t* com `dist_ma50 > 0` |

**Macro:**

| # | Feature |
|---|---|
| 31–33 | `ff_rate_12m`, `ff_rate_12m_chg`, `ff_slope` (seção 3.4) |
| 34–36 | `gpr_level`, `gpr_trend`, `gpr_threat_act` (seção 3.5) |

Nenhuma feature usa nível absoluto de preço, datas futuras, normalização global ou `bfill`.

## 5. Dataset

- **Formato:** tabela longa indexada por `(date, ticker)` e ordenada por data. Colunas: as 36 features, `fwd_ret` e `label`.
- **Critério de inclusão da linha (i, t):**
  - *i* é membro em *t*;
  - `C[i, t]` é válido;
  - a ação tem pelo menos `min_history_days` (63) fechamentos válidos até *t*;
  - todas as features de mercado e macro são válidas.
- **NaN nas demais features de ação:** é permitido (o LightGBM trata).
- **`fwd_ret` e `label` NaN:** a linha pode existir. Não entra no treino, mas é prevista.

## 6. Modelo e walk-forward

- **Modelo:** `lightgbm.LGBMClassifier` com parâmetros fixos, sem tuning.

  ```
  n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=500,
  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
  random_state=42, deterministic=True, force_col_wise=True, verbose=-1
  ```

- **Por que LightGBM:**
  - Capta interações entre as features da ação e o regime macro (num modelo linear, a macro só desloca o intercepto).
  - Trata NaN sozinho e não precisa de escala.
  - É rápido o bastante para cerca de 1.400 retreinos.
  - Dá probabilidades utilizáveis com limiar absoluto.
  - Alternativas descartadas: SVM (O(n²) e sem probabilidade nativa) e Random Forest (lento e com probabilidades comprimidas).
- **Loop sequencial.** Para cada pregão de decisão *t*:
  1. Treina um modelo novo nas linhas com rótulo e *s* ∈ [*t* − *X* − 1, *t* − 2].
  2. Prevê `prob = P(label = 1)` para todas as linhas de *t*.
  3. Guarda a importância (gain normalizado) desse treino.
- **Primeira decisão:** o primeiro pregão com *X* dias completos de treino.
- **Paralelização:** nenhuma no nosso código; o LightGBM usa as threads dele. Se o run real passar de ~1 h, reavaliamos.
- **Log de progresso:** a cada 50 dias, com ETA.

## 7. Carteira e backtest

A carteira é uma função pura sobre as previsões. `main.py` e o app chamam **a mesma** função.

### 7.1 Seleção e pesos

- **Seleção:** `S(t) = {i : prob(i, t) > limiar}`.
- **Pesos proporcionais à confiança prevista:** `w_i = prob_i / Σ_{j∈S} prob_j`. A carteira fica 100% investida entre as selecionadas.
- **Nenhuma selecionada:** 100% em caixa rendendo `rf_daily(t)`.
- **Nota:** com probabilidades entre ~0,55 e 0,65, os pesos variam cerca de ±10% em torno do peso igual. Se quiser inclinação mais forte, a alternativa é pesar por `prob − limiar` (é uma troca de uma linha).

### 7.2 Retornos (indexados pelo dia de carteira *h* = *t+1*)

```
gross(h) = Σ_i w_i · R_i            (R_i = fwd_ret(i, t); NaN → 0 e é contado)
gross(h) = rf_daily(t)              (se S(t) vazio)
bench(h) = Open_SP500TR[t+2] / Open_SP500TR[t+1] − 1
```

Uma ação selecionada sem retorno realizado (deslistagem ou halt) conta como retorno 0. O total de ocorrências aparece no app. Excluí-la usaria informação futura.

### 7.3 Custos (implementados por último)

```
w̃_i        = w_i(t−1) · (1 + R_i(t−1)) / (1 + gross(h−1))    (pesos que derivaram durante o dia de carteira anterior; 0 se ele foi caixa)
turnover(h) = Σ_i |w_i(t) − w̃_i|                             (compras + vendas, só ações; o custo em bps é por lado)
cost(h)     = turnover(h) · cost_bps / 10_000
net(h)      = (1 + gross(h)) · (1 − cost(h)) − 1
```

Aqui `h−1` é o dia de carteira anterior, cuja decisão foi em `t−1`.

- Default de 5 bps, ajustável no app.
- A primeira montagem conta como turnover.
- Não há liquidação final.

### 7.4 Saídas da função

- **`daily`**, indexado por *h*: `decision_date, gross, net, bench, rf, n_positions, invested, turnover, cost, n_missing`.
- **`positions`**: `decision_date, holding_date, ticker, prob, weight, fwd_ret, contribution`.

## 8. Métricas

**Da carteira** (bruta e líquida) **e do benchmark:**

- retorno acumulado;
- CAGR (base 252);
- volatilidade anualizada;
- Sharpe e Sortino sobre o excesso a `rf`;
- máximo drawdown (equity começando em 1);
- Calmar;
- hit ratio de dias (dias com retorno > 0; para a carteira, só dias investidos);
- beta e alfa anualizado (CAPM em excessos);
- tracking error e information ratio vs benchmark.

**Operacionais:**

- hit ratio das posições (% das posições com `fwd_ret > 0`);
- % de dias investido;
- nº médio de posições;
- turnover médio;
- custo total.

**Do modelo** (out-of-sample, sobre todas as previsões com rótulo):

- AUC, acurácia com corte em 0,5, Brier e log-loss;
- taxa base;
- IC diário (Spearman entre `prob` e `fwd_ret` por data), com média e t-stat;
- curva de calibração (10 quantis: probabilidade média vs frequência realizada).

## 9. Verificação de lookahead e testes

Todos os testes usam `pytest` e um gerador de mercado sintético (`tests/conftest.py`). Os módulos são funções puras sobre DataFrames, então dá para testar sem rede.

1. **Causalidade das features.** Features calculadas com todos os dados e com os dados truncados em *T* têm que ser idênticas para toda data ≤ *T*. Vale para ação, cross-section, mercado, futuros e GPR.
2. **Defasagem do GPR.** Mudar valores do GPR em (*t* − 7, *t*] não pode alterar as features em *t*.
3. **Alvo.** Com preços e dividendos sintéticos conhecidos, `fwd_ret` bate com a conta manual, inclusive com dividendo em *t+2* e com a abertura de *t+1* faltando.
4. **Janela de treino.** Um modelo "espião" registra as datas recebidas. Para cada *t*, o treino tem que ser exatamente [*t* − *X* − 1, *t* − 2] e a previsão só pode conter linhas de *t*.
5. **Ponta a ponta.**
   - Roda o pipeline inteiro (features → dataset → walk-forward) duas vezes; na segunda, todos os dados com data > *T* são embaralhados (preços, dividendos, futuros, GPR, taxa).
   - As previsões com data de decisão ≤ *T* têm que ser idênticas bit a bit.
6. **Universo.** Uma ação que entra no índice depois de *t* nunca é prevista em *t*. Os aliases e a regra de snapshot ≤ *t* são testados, incluindo a data exata do snapshot e datas antes do primeiro snapshot.
7. **Carteira.** A seleção depende só de `prob`: embaralhar `fwd_ret` não muda nem a seleção nem os pesos. Também há casos numéricos conhecidos para pesos, caixa, retorno ausente, turnover e custo.
8. **Métricas.** Séries com resposta conhecida (drawdown, CAGR, Sharpe, hit ratio, beta).
9. **Dados.** Seleção de contrato dos futuros (12m, tolerância de 45 dias, domingo descartado, front month), leitura do GPR e o download descartando `Adj Close` (com o yfinance mockado).
10. **App.** Smoke test com `streamlit.testing.v1.AppTest` sobre saídas sintéticas.

Além dos testes, `main.py run` grava em `run_info.json` um relatório de qualidade dos dados: cobertura por ano, tickers sem dado, contagem de |r| > 50% e datas de início e fim de cada fonte.

## 10. App (`app/dashboard.py`, Streamlit + Altair)

O app é escrito do zero, inspirado só no layout do `dashboard.py` de referência. Ele lê `data/out/`, nunca retreina e usa as funções de `engine.portfolio` e `engine.metrics`. A interface é em português.

**Sidebar:**

- limiar (default do config);
- custo em bps (default do config);
- aviso: "escolher o limiar olhando este resultado é otimizar no próprio backtest (data snooping)".

**Abas:**

1. **Visão geral**
   - parâmetros da rodada (janela *X*, modelo, período, universo médio, cobertura);
   - KPIs da carteira vs S&P 500 TR;
   - curva de capital (bruta, líquida, benchmark);
   - drawdown;
   - volatilidade rolante de 63d;
   - exposição (nº de posições e % investido);
   - retornos por ano.
2. **Carteira por dia**
   - seletor da data de decisão;
   - período de carteira;
   - retorno bruto e líquido vs benchmark;
   - tabela das ações compradas (ticker, prob, peso, retorno realizado, contribuição), com cor verde ou vermelha;
   - reconciliação (Σ contribuições = retorno bruto);
   - histograma das probabilidades do dia com a linha do limiar;
   - as 10 maiores probabilidades que ficaram abaixo do limiar.
3. **Recortes de período**
   - métricas nos últimos 1, 3, 6 e 12 meses e no período todo;
   - curva vs benchmark em cada recorte.
4. **Modelo**
   - AUC mensal e IC rolante;
   - calibração;
   - nº de ações acima do limiar por dia;
   - importância média das features;
   - cobertura do universo ao longo do tempo;
   - contagem de retornos ausentes.

## 11. Estrutura, saídas e remoções

```
config/config.py        Config (dataclass congelada) + TICKER_ALIASES
main.py                 CLI: download | run
app/dashboard.py        Streamlit
engine/data/universe.py composição point-in-time + aliases → membership
engine/data/prices.py   download (yfinance), leitura, calendário, retornos, fwd_ret
engine/data/macro.py    Fed Funds futures, GPR, taxa livre de risco
engine/features.py      features causais (seção 4)
engine/dataset.py       montagem do dataset (seção 5)
engine/model.py         fábrica do LightGBM
engine/walk_forward.py  treino rolante e previsão (seção 6)
engine/portfolio.py     seleção, pesos, retornos e custos (seção 7)
engine/metrics.py       métricas (seção 8)
engine/results.py       gravar e ler data/out
tests/                  um arquivo por módulo + test_lookahead_e2e.py + test_app.py
```

**Saídas em `data/out/`:**

- `predictions.parquet` (`date, ticker, prob, fwd_ret, label`);
- `market.parquet` (`decision_date, holding_date, bench_fwd_ret, rf_daily`);
- `importance.parquet`;
- `coverage.parquet`;
- `run_info.json` (config, features, faixas de datas, tempo de execução, versões, qualidade dos dados).

**Remoções:**

- `engine/{algorithms,backtesting,log,pipeline,plotting,reproducibility,stock,strategies,validation}`;
- `output/`;
- os testes antigos;
- as dependências torch, matplotlib, seaborn, black e flake8.

**Atualizações:**

- `requirements.txt` e `pyproject.toml`: pandas, numpy, pyarrow, scikit-learn, lightgbm, yfinance, xlrd, streamlit, altair, com pytest como dev; Python ≥ 3.11.
- `README.md`.
- `.gitignore`: `data/out/` e `data/in/prices.parquet`.
- `.claude/CLAUDE.md`: layout novo.
- venv local em `venv/`.

**Fica intocado:** `data/in/*` e `data/reference/`. Quem decide apagar é você. O `sp500_data.csv` antigo, ajustado por dividendos, deixa de ser usado.

## 12. Configuração (defaults)

| Parâmetro | Default |
|---|---|
| `train_window_days` (*X*) | 252 |
| `threshold` | 0,55 (escolhido a priori, sem olhar resultado) |
| `cost_bps` | 5 |
| `gpr_lag_days` | 7 |
| `macro_max_staleness_days` | 5 |
| `min_history_days` | 63 |
| `price_start` | 2018-06-01 |
| `benchmark_ticker` / `risk_free_ticker` | `^SP500TR` / `^IRX` |
| `ff_horizon_months` / `ff_max_gap_days` / `ff_change_days` | 12 / 45 / 21 |
| `lgbm_params` | seção 6 |

## 13. Ordem de implementação

1. Limpeza do código antigo, esqueleto, venv, dependências e config.
2. `universe.py`.
3. `prices.py`, seguido do download real e de uma checagem de qualidade.
4. `macro.py`.
5. `features.py`, com os testes de causalidade.
6. `dataset.py`.
7. `model.py` e `walk_forward.py`, com o teste do espião e o teste ponta a ponta.
8. `portfolio.py` (sem custos) e `metrics.py`.
9. `results.py` e `main.py run`, seguido do run real.
10. App e o smoke test dele.
11. Custos (carteira, app e testes).
12. README, CLAUDE.md, passada do code-simplifier e verificação final.
