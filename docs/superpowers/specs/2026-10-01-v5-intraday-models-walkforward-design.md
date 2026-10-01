# Rodada 5 — intraday, universo completo no passado, vários modelos (único ou por ação), varredura e walk-forward de parâmetros

Data: 2026-10-01. Decisões do dono nesta rodada: operar intraday; universo "completo no passado"; walk-forward de parâmetros com Sharpe líquido de 1 ano, reescolhido a cada trimestre; tipo de modelo escolhido no app e aplicado igualmente a todas as ações, com várias opções e parâmetros prontos para tuning. Commits passam a ser feitos pelos agentes (push não).

## 1. Operação intraday

- Decisão após o fechamento de *t*; compra na abertura de *t*+1; vende no fechamento de *t*+1.
- `fwd_ret = Close[t+1] / Open[t+1] − 1`; `label = fwd_ret > 0`. Sem dividendo: quem compra na abertura do dia ex já paga o preço sem ele.
- O rótulo de *s* é conhecido no fechamento de *s*+1, então o treino da decisão *t* usa as linhas **[*t*−X, *t*−1]**.
- Benchmark: `^SP500TR` da abertura ao fechamento do mesmo dia.
- Custo: cada dia investido compra tudo na abertura e vende tudo no fechamento; com taxa *c* = bps/10⁴ por lado, `líquido = (1 + bruto)(1 − c)² − 1`; giro = 2 nos dias investidos. Dias em caixa rendem `rf_daily` e não pagam custo.
- Fato medido que acompanha esta escolha: em 2020–2026 o S&P 500 TR rendeu +36,6% só nos intervalos abertura→fechamento e +92,2% só nos fechamento→abertura; a 5 bps por lado, operar todo dia custa ~22% ao ano.

## 2. Universo

- Em *t*, uma ação é elegível se é membro do índice em *t*, tem preço em **todos os últimos 252 pregões** (incluindo *t*) e as features de data existem. Substitui a regra "≥ 63 pregões com preço" (`complete_history_days = 252` no lugar de `min_history_days`).
- As checagens de dado errado continuam (tickers reatribuídos descartados; liquidez suspeita e retornos > 50% reportados).
- `FEATURES_FORMAT = 4`.

## 3. Modelos e escopo

- `engine/models.py`: `MODELS = {nome: (rótulo, fábrica(params, n_linhas))}`:
  - `lightgbm` (LightGBM; `min_child_samples = max(piso, round(fração × linhas))`, piso e fração dentro dos parâmetros);
  - `logistic` (imputação pela mediana + padronização + logística L2);
  - `random_forest` e `extra_trees` (imputação + 100 árvores, profundidade 6, ≥ 20 exemplos por folha);
  - `naive_bayes` (imputação + Gaussian NB).
  - Imputação e padronização ficam dentro do pipeline, ajustadas só nas linhas de treino.
- Parâmetros padrão em `config.MODEL_PARAMS` (um dicionário por modelo), copiados para `Config.model_params`.
- Importância: ganho/impureza das árvores, |coeficiente| da logística, zero no Naive Bayes; normalizada.
- **Escopo:**
  - `pooled` (um modelo para todas): como hoje.
  - `per_stock` (um modelo por ação): no ajuste da decisão *t*, cada ação presente nas decisões do grupo treina só com as próprias linhas em [*t*−X, *t*−1]. Com menos de `per_stock_min_rows = 63` exemplos conhecidos, ou uma classe só, prevê a fração de altas da janela.
  - Importância do ajuste = média das importâncias das ações.
- Defaults do app: escopo `pooled` com X = 21 e k = 1; escopo `per_stock` com X = 252 e k = 21.
- `RunKey(model, scope, window, retrain_every)`, pasta `runs/{model}-{scope}-w{X}-k{k}`.
- O fingerprint de um run guarda `backtest_start`, `per_stock_min_rows` e os parâmetros **do próprio modelo**: mudar os parâmetros da logística não esconde os runs do LightGBM.

## 4. Varredura

`python main.py sweep --models … --scopes … --windows … --retrain …`:
- roda todas as combinações que ainda não existem, em sequência (cada uma já paraleliza por dentro);
- imprime uma linha por combinação (tempo, AUC, IC) ou o erro (por exemplo, janela grande demais), sem parar a varredura.

## 5. Walk-forward de parâmetros (`engine/selection.py`)

- **Candidatos:** cada par (run, limiar), sendo o limiar de uma grade (padrão 0,500 a 0,700, passo 0,025). A curva diária de cada par vem do `Portfolio` com o custo atual.
- **Escolha:**
  - a cada `selection_step_days = 63` decisões, a partir da primeira com `selection_lookback_days = 252` dias anteriores, escolhe o candidato com maior Sharpe líquido (excesso sobre `rf`) nesses 252 dias;
  - na decisão da linha *i*, usa só as linhas < *i*, já realizadas no fechamento de *t*;
  - Sharpe indefinido (desvio zero) conta como 0; empate fica com o primeiro na ordem;
  - o candidato escolhido é usado até a próxima escolha.
- **Resultado:**
  - curva diária costurada (com o nome do candidato de cada dia);
  - tabela de escolhas (início, candidato, Sharpe do período anterior, dias);
  - comparação com o S&P intraday nos mesmos dias.
- Como o intraday zera a carteira todo dia, trocar de candidato não gera custo extra além do que cada dia já paga.
- Nada é retreinado: a camada só escolhe entre previsões que já são fora da amostra.

## 6. App

- **Barra lateral:** modelo (lista), escopo (rádio), X e k (padrões por escopo), botão "Rodar"; limiar e custo como hoje.
- **Abas:**
  - Visão geral;
  - Previsto vs realizado;
  - Carteira por dia;
  - Recortes de período;
  - Runs;
  - **Walk-forward** (nova): seleção dos runs candidatos, grade de limiares, período de olhar para trás e passo; curva e métricas contra o S&P intraday; tabela das escolhas; frequência de cada run e limiar;
  - Modelo.
- Textos explicam a operação intraday.

## 7. Testes

- `forward_intraday_return`.
- Regra de universo completo.
- Modelos: todos ajustam e preveem com NaN nas features; regra do `min_child_samples`; importância.
- `WalkForward`:
  - linhas [*t*−X, *t*−1];
  - escopo por ação: cada ação treina só com as próprias linhas, mínimo de linhas e classe única;
  - alinhamento de cada probabilidade com a sua linha;
  - paralelo = sequencial nos dois escopos.
- `Portfolio` intraday e simulador independente intraday.
- Seleção:
  - escolhe o melhor do passado;
  - não usa o futuro (perturbar dias ≥ início não muda a escolha);
  - períodos e tamanhos;
  - candidatos desalinhados dão erro.
- Lookahead ponta a ponta com um run único e um run por ação, n_jobs = 2, e mutação no recorte de treino.
- CLI `sweep`; app com a aba nova.

## 8. Depois da implementação

- Refazer as features reais.
- Rodar uma varredura inicial nos dois escopos e com vários modelos.
- Rodar o walk-forward de parâmetros nos dados reais e reportar a curva, as escolhas e o quanto elas variaram.
