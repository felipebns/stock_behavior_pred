# Rodada 2 — só preços, classes simples, janela variável — Design

- **Data:** 2026-09-27
- **Status:** design aprovado em conversa; este arquivo aguarda revisão.
- **Substitui:** as partes correspondentes de `2026-09-26-ml-backtest-redesign-design.md`. O que não é citado aqui continua valendo, em especial a convenção de tempo, que não muda.

## 1. Objetivo

Com a mesma convenção de tempo e a mesma carteira:

- tirar todo dado macro;
- usar os constituintes e preços mais recentes;
- reorganizar o código em classes simples;
- permitir escolher a janela de treino no app e rodar de novo em poucos minutos.

O objetivo é conseguir testar várias janelas rapidamente, com o código o mais simples possível e sempre íntegro.

**Sucesso significa:**

- `python main.py download`, `python main.py features` e `python main.py run --window N` funcionam.
- O app permite escolher a janela, rodar e comparar as janelas já rodadas.
- Os testes estão verdes, incluindo o de lookahead ponta a ponta e o de "paralelo = sequencial".

## 2. Convenção de tempo (inalterada)

- **Decisão em *t*:** depois do fechamento, só com preços, dividendos e constituintes datados ≤ *t*.
- **Execução:** compra na abertura de *t+1* e rebalanceia na abertura de *t+2*.
- **Alvo:** `(Open[t+2] + Div[t+2]) / Open[t+1] − 1 > 0`.
- **Treino da decisão *t*:** linhas *s* ∈ [*t* − X − 1, *t* − 2].
- **Saída:** sempre 1 dia.

## 3. Dados

- **Sai tudo de macro:**
  - `engine/data/macro.py` (Fed Funds futures e GPR), as features `ff_*` e `gpr_*`, os parâmetros de config, os testes e a dependência `xlrd`;
  - os arquivos `data/in/ff_futures_daily.csv`, `data/in/data_gpr_daily_recent.xls` e o antigo `data/in/sp500_data.csv` (ajustado por dividendos e sem uso). Eles continuam recuperáveis pelo git.
- **Continua:** o `^IRX`, como taxa do caixa e base do Sharpe (não é feature), e o `^SP500TR`, como benchmark e fonte das features de mercado.
- **Constituintes:** `download` baixa o CSV mais recente do repositório fja05680:

  ```
  https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv
  ```

  Ele vai até 2026-08-18 e é salvo em `data/in/sp500_historical_constituents.csv`.
- **Preços:** `download` baixa pelo yfinance (`auto_adjust=False, actions=True`, sem `Adj Close`) a partir de `price_start = 2018-01-02`, até o último pregão disponível.

## 4. Universo point-in-time

- Em cada *t* só entram os membros do snapshot vigente (o mais recente com data ≤ *t*). Isso vale para as features cross-section, os pares, o breadth, as linhas de treino e as previsões.
- O ticker da época é convertido só para o formato do Yahoo (`.` → `-`).
- **Aliases:** só renomeações puras (mesma empresa, só mudou nome ou ticker) que são publicamente conhecidas e aparecem no arquivo como troca 1:1 na mesma data:

  ```
  FB→META  ANTM→ELV  ABC→COR  FLT→CPAY  PKI→RVTY  RE→EG  BLL→BALL  WLTW→WTW
  LB→BBWI  SYMC→GEN  NLOK→GEN  HCP→DOC  PEAK→DOC  JEC→J  TMK→GL  BHGE→BKR
  CTL→LUMN  HRS→LHX  UTX→RTX  DWDP→DD  ARNC→HWM
  ```

  Saem da lista anterior:
  - as fusões e holdings novas: DISCA→WBD, MYL→VTRS, CBS/VIAC/PARA→PSKY;
  - as trocas que eu só inferi pelos dados: FI→FISV, MMC→MRSH, BK→BNY, SATS→ECHO.
- **Descarte:** um membro sem preço no Yahoo durante a membresia é descartado. A lista (dias como membro vs dias com preço) sai no relatório de qualidade e no app.
- O viés de sobrevivência que resta, de empresas deslistadas que o Yahoo não tem, é medido e mostrado, não escondido.

## 5. Features (sem macro)

- **Continuam** as 19 de ação, os 5 ranks cross-section e as 6 de mercado (índice e breadth), sem mudança.
- **Novas: pares por correlação.** Em cada *t*:
  1. Pega os retornos totais diários dos últimos `peer_lookback_days = 63` pregões.
  2. Entre os membros de *t* com pelo menos 80% de retornos válidos nessa janela, calcula a correlação: centraliza e padroniza cada coluna ignorando NaN, preenche NaN com 0 e faz `Zᵀ Z / L`.
  3. Para cada membro, escolhe os `peer_count = 10` membros de maior correlação (sem ele mesmo).
  4. Calcula as features:
     - `peer_ret_1d`, `peer_ret_5d`, `peer_ret_21d`: média de `ret_1d`, `ret_5d` e `ret_21d` dos pares em *t*;
     - `peer_gap_5d = ret_5d − peer_ret_5d`.
- **Por que não setor:** o Yahoo só dá o setor de hoje, é estático (o GICS mudou em 2018 e 2023) e não existe para as deslistadas. Os pares usam só dados ≤ *t* e acompanham as mudanças.
- **Normalização:** não há normalização global nem PCA. As árvores não precisam, e estatística calculada no período todo seria lookahead. Os ranks cross-section, do mesmo dia, continuam.
- Total: 34 features (19 + 5 + 4 + 6).

## 6. Modelo, janela e paralelização

- **Um modelo por dia, treinado com todas as ações** (não um por ação): com janela de 1 semana, uma ação teria 5 exemplos.
- **Janela:** `train_window_days` é escolhida no CLI ou no app (5 = 1 semana, 21 = 1 mês, 63, 252...). O default é 21.
- **Mesmo período para todas as janelas:** as decisões vão de `backtest_start = 2020-01-02` até o último dia com dados. A janela X exige X + 1 pregões do dataset antes de `backtest_start` (a primeira decisão treina em [*t* − X − 1, *t* − 2]). Se faltar histórico, a janela é recusada com mensagem clara que informa o máximo possível (~340 pregões com os dados desde 2018).
- **Parâmetros:** os mesmos do LightGBM de antes, exceto `min_child_samples = max(20, round(0.004 × linhas_de_treino))`. Com janela de 252 isso fica em ~480 (próximo dos 500 de antes); com janela de 5, em ~20. Nada é otimizado.
- **Paralelização:**
  - as datas de decisão são divididas em blocos e rodam em processos `joblib` (`n_jobs` do config, default = todos os núcleos);
  - cada treino usa `n_jobs=1` no LightGBM;
  - o resultado é idêntico bit a bit ao sequencial, e há teste para isso;
  - a barra de progresso avança por bloco.
- **Tempo medido para o backtest inteiro (8 processos):** ~1,3 min (5d), ~2 min (21d), ~4 min (63d), ~13 min (252d).

## 7. Carteira (inalterada)

- Seleção por `prob > limiar` (default 0,55), com pesos `prob / Σ prob`.
- Ninguém acima do limiar → caixa rendendo `^IRX / 100 / 252`.
- Custos em bps por lado sobre compras + vendas, contra os pesos já derivados.
- Benchmark: `^SP500TR`, abertura → abertura.
- Uma lacuna do benchmark no meio do período continua gerando erro.

## 8. Métricas (enxutas)

- **Carteira (bruta e líquida) vs S&P 500 TR:**
  - retorno acumulado;
  - volatilidade anualizada;
  - Sharpe (excesso sobre o `^IRX`);
  - máximo drawdown (capital inicial 1,0);
  - hit ratio (dias com retorno > 0; para a carteira, só dias investidos).
- **Modelo:**
  - AUC;
  - acurácia (corte 0,5);
  - IC (média da correlação de Spearman diária entre prob e retorno).
- **Sai todo o resto:** CAGR, beta, alfa, Sortino, Calmar, tracking error, information ratio, Brier, log-loss, taxa base, calibração, hit ratio das posições e retorno por ano.

## 9. Estrutura (classes simples)

```
config/config.py          Config (dataclass congelada), TICKER_ALIASES, LGBM_PARAMS
engine/universe.py        class Universe      — download(url, path), from_csv, members_on, tickers_since, membership, sizes
engine/prices.py          class PriceData     — download, load/save, calendar, panel, series
                          PricePanel (dataclass); funções daily_total_return, forward_open_return
engine/features.py        class FeatureBuilder — build(panel, market_close, membership) → (features por ticker, features por data)
                          constantes STOCK_/XS_/PEER_/MARKET_FEATURES, TICKER_FEATURES, DATE_FEATURES, FEATURES
engine/dataset.py         build_dataset(...) (função pura, como hoje)
engine/walk_forward.py    class WalkForward   — run(dataset, calendar, start, progress) → predições + importância
engine/portfolio.py       class Portfolio     — backtest(predictions, market) → BacktestResult
engine/metrics.py         funções: performance_metrics, model_metrics, drawdown, daily_ic, monthly_auc
engine/project.py         class Project       — fachada usada pelo CLI e pelo app (download, build_features, features, run, runs, load_run)
main.py                   CLI: download | features | run --window N
app/dashboard.py          Streamlit
```

Saem:

- `engine/data/` (o `universe` e o `prices` sobem um nível; o `macro` é apagado);
- `engine/pipeline.py` e `engine/results.py` (absorvidos pelo `Project`);
- `--start/--end` e a pasta `partial`.

As métricas continuam funções, porque não têm estado.

**Arquivos em `data/out/`:**

- `features/`: `dataset.parquet`, `market.parquet`, `coverage.parquet`, `quality.json`. É gerado por `features` e reaproveitado por todas as janelas.
- `runs/w{janela:03d}/`: `predictions.parquet`, `importance.parquet`, `run_info.json`.
- Recalcular as features apaga `runs/`, porque os runs antigos ficariam inconsistentes com os dados novos.

## 10. App

- **Barra lateral:**
  - janela de treino (número de pregões) com um botão "Rodar", que treina com barra de progresso se a janela ainda não existe e carrega se já existe;
  - limiar;
  - custo;
  - o aviso de data snooping.
- **Abas:**
  - **Visão geral:** tabela de métricas (carteira bruta, carteira líquida, S&P 500 TR), KPIs do modelo (AUC, acurácia, IC), curva de capital, drawdown e nº de ações por dia.
  - **Carteira por dia:** como hoje (o que foi comprado, prob, peso, retorno, contribuição, reconciliação, histograma, "quase entraram").
  - **Recortes de período:** as métricas da seção 8 em 1, 3, 6 e 12 meses e no período todo.
  - **Janelas:** uma linha por janela já rodada, com as métricas da carteira líquida e do modelo no limiar e custo atuais.
  - **Modelo:** AUC mensal, IC rolante, importância das features, cobertura do universo e qualidade dos dados.
- Os caches são chaveados pelo `run_info.json` de cada run e pelo `quality.json` das features.

## 11. Testes (adaptados e novos)

- **Adaptados:** o teste ponta a ponta de lookahead (perturba preços, dividendos, `^IRX` e composição depois de *T*), o espião da janela, a causalidade das features, o universo point-in-time, a carteira, os custos, as métricas e o smoke do app.
- **Novos:**
  - causalidade e valor conhecido dos pares (duas ações idênticas são par uma da outra; não-membro não vira par);
  - regra do `min_child_samples`;
  - paralelo (n_jobs=2) idêntico bit a bit ao sequencial;
  - janela grande demais é recusada;
  - `Project`: cache de features, um run por janela, listagem das janelas e `build_features` limpando `runs/`;
  - download dos constituintes com fetch falso (sem rede).

## 12. Documentação

- **README curto:** o que é, objetivo, escopo, benchmark, como rodar e como mudar a janela (CLI e app).
- **`.claude/CLAUDE.md`:** layout novo.
- **Workspace antigo:** `.superpowers/sdd/2026-09-26-ml-backtest-redesign/` é apagado ao final (o plano dele foi superado).
- **Notas no `main.py`:** as suas notas no fim do arquivo são preservadas.
