# Rodada 3 — rótulo relativo, horizonte h, retreino a cada k, diagnóstico previsto × realizado

Data: 2026-09-29. Parte do estado da rodada 2 (`2026-09-27-v2-price-only-oop-design.md`), que continua valendo onde este documento não diz o contrário.

## 1. Objetivo

Projeto pessoal: achar sinal e iterar rápido. Diagnóstico da rodada 2 nos dados reais:

- o rótulo "a ação sobe" é dominado pelo mercado (fração de ações em alta por dia: média 51,8%, desvio 22,5 pp);
- o modelo aprendeu a prever o mercado: a probabilidade média do dia vai de 0,16 a 0,83 (p5–p95); features de mercado têm 47% (janela 21) e 70% (janela 63) da importância; IC ≈ 0;
- por isso a carteira oscila de 28 a 414 ações de um dia para o outro, gira 1,2–1,4× por dia e o custo come 15–18% ao ano. A conta de custo está certa; o problema é o giro.

## 2. Decisões do dono

- **Rótulo relativo:** 1 se o retorno da ação no horizonte fica acima da mediana das ações elegíveis do índice naquela data.
- **Mantém o grau de confiança:** compra as ações com P(bater a mediana) > limiar ajustável, peso ∝ probabilidade; se nenhuma passar, caixa no ^IRX.
- **Horizonte h variável, rebalanceia a cada h pregões:** decide, segura h pregões, troca.
- **Modelos:** só LightGBM nesta versão; um dicionário `MODELS` deixa outro entrar com poucas linhas (o seletor aparece no app quando houver mais de um).
- **Retreino a cada k decisões**, ajustável no app.
- Fora do escopo agora: outra fonte de dados (Binance só tem cripto; dados sem viés de sobrevivência para ações dos EUA são pagos ou via WRDS/CRSP), framework de backtesting pronto (validamos com um simulador independente no estilo do curso), tuning (próxima rodada, com período final reservado fora da amostra).

## 3. Modelagem

- *r(s)*: retorno de um pregão após a decisão em *s* — abertura de *s*+1 → abertura de *s*+2, com o dividendo que vai ex em *s*+2 (como hoje).
- *Rₕ(s)* = ∏ₖ₌₀^{h−1} (1 + *r*(*s*+k)) − 1: abertura de *s*+1 → abertura de *s*+1+h. NaN se algum dos h dias não tem retorno.
- **Rótulo:** 1 se *Rₕ(s)* > mediana de *Rₕ(s)* entre as linhas do dataset (membros elegíveis) na data *s*; NaN se *Rₕ(s)* é NaN.
- O rótulo de *s* só é conhecido na abertura de *s*+1+h. **Treino da decisão *t*: linhas *s* ∈ [*t*−h−X, *t*−h−1]** (X dias de linhas). Com h = 1 é a regra de hoje ([*t*−X−1, *t*−2]).
- **Decisões:** posições de calendário `backtest_start`, +h, +2h, … até a última data do dataset (só as que têm linhas).
- **Janela máxima:** a primeira decisão exige X + h pregões do dataset antes dela; a mensagem de erro informa o máximo.
- **Retreino a cada k decisões:** as decisões são agrupadas de k em k; cada grupo tem um único ajuste, feito na primeira decisão do grupo com as linhas acima, e esse modelo prevê todas as decisões do grupo (todas posteriores ao ajuste: sem lookahead).
- **Um run** = `RunKey(horizon, window, retrain_every, model="lightgbm")`, pasta `data/out/runs/lightgbm-h5-w21-k1/`.
- **Features não mudam.** O dataset salvo passa a ter só as 34 features; o alvo de cada h é calculado na hora do run (`horizon_target`) a partir do painel de retornos diários *r* salvo junto (`features/returns.parquet`, datas × tickers, inclusive fora do índice).
- `FEATURES_FORMAT = 2` entra nos settings das features: features salvas no formato antigo são refeitas automaticamente.
- Métricas do modelo: AUC e acurácia contra o rótulo relativo; IC = Spearman diário entre prob e *Rₕ*.

## 4. Carteira e custos

- Em cada decisão *t*: seleção como hoje (prob > limiar, peso ∝ prob; nenhuma → caixa).
- Compra na abertura de *t*+1 e **segura até a próxima decisão** (abertura de *t*+1+h); os pesos andam com os preços (compra e segura). Se uma decisão agendada não existir, a carteira anterior simplesmente segue.
- **Curva diária:** retorno de cada dia = Σ valor de cada posição no início do dia × *r* / Σ valor. Dias em caixa rendem `rf_daily`. Métricas continuam diárias e comparáveis entre horizontes; S&P 500 TR nos mesmos dias.
- Ação sem retorno num dia conta como 0 (valor congelado) e segue até a próxima troca.
- **Custo só no dia da troca:** giro = Σ|peso novo − peso anterior já deslocado pelos preços|; custo = giro × bps/10⁴; líquido = (1 + bruto)(1 − custo) − 1. Convenção: a taxa incide sobre o valor negociado para chegar nos pesos-alvo calculados sobre o patrimônio antes da taxa, e sai proporcionalmente de todas as posições.
- Janela do backtest: da primeira decisão até a última decisão + h − 1, cortada no último retorno realizado do benchmark (lacuna no meio continua sendo erro).
- `positions`: decision_date, holding_date (primeiro dia), ticker, prob, weight, **realized** (retorno do período como foi carregado, ausentes = 0), contribution = weight × realized. Σ contribuições = retorno bruto composto do período.
- `daily` ganha `rebalance` (dia de troca) e `decision_date` passa a ser a decisão em vigor.

**Validação (o material do curso):** `tests/test_backtest_validation.py` tem um simulador evento a evento no estilo das Aulas 4 e 5 — capital 1.000.000, caixa e quantidade de ações, marcação a mercado a cada abertura, dividendos reinvestidos na abertura em que vão ex, caixa rendendo `rf_daily`, taxa sobre o valor negociado, preço congelado após deslistagem. Ele escolhe os pesos a partir das probabilidades com código próprio. A curva do simulador tem de bater com `capital × ∏(1 + líquido)` do `Portfolio` (tolerância relativa 1e-10), para h = 1 e h = 5, com dias em caixa, dividendos e uma ação deslistada no meio de uma posição.

## 5. Diagnóstico (novas funções em `metrics.py`)

Em cada data de decisão, as ações com rótulo conhecido são divididas em 10 decis de probabilidade (1 = menor):

- `deciles(predictions)`: por decil, médias (primeiro no dia, depois entre dias) da probabilidade, da fração que bateu a mediana e do retorno relativo à mediana do dia. Mostra se há sinal (barras crescentes) e a calibração (prob média × fração que bateu).
- `decile_spread(predictions)`: por data, retorno médio do decil 10 menos o do decil 1 — o ganho da ordenação antes de custos.

## 6. App

- **Barra lateral:** horizonte h (1–63), janela X (1–1000), retreino k (1–100); botão "Rodar" se o run não existe (com progresso); lista dos runs existentes; limiar e custo como hoje. Seletor de modelo só se `len(MODELS) > 1`.
- **Abas:**
  - **Visão geral:** KPIs do modelo, tabela carteira bruta/líquida × S&P, curva, drawdown, nº de ações; textos explicam h, X, k e o rótulo relativo.
  - **Previsto vs realizado (nova):** barras do retorno relativo médio por decil; calibração (prob média do decil × fração que bateu a mediana, com a diagonal); soma acumulada do spread decil 10 − decil 1; texto de leitura.
  - **Carteira por decisão** (era "por dia"): escolhe uma data de decisão; retorno do período bruto, líquido e S&P, ações, giro, custo; tabela das ações compradas com prob, peso, retorno realizado, retorno relativo à mediana, "bateu a mediana?" e contribuição; dispersão de todas as ações do dia (prob × retorno relativo, cor = bateu) com a linha do limiar; "quase entraram".
  - **Recortes de período:** como hoje.
  - **Runs** (era "Janelas"): uma linha por run (h, X, k, tempo de execução, métricas da carteira líquida e do modelo no limiar e custo atuais).
  - **Modelo:** AUC mensal, IC em média móvel (max(5, 63 // h) decisões), importância, cobertura, qualidade.

## 7. CLI e config

- `python main.py run --horizon 5 --window 21 --retrain 1`.
- `Config`: `horizon_days = 5` (padrão semanal: menos giro), `train_window_days = 21`, `retrain_every = 1`.

## 8. Testes

- dataset: `horizon_target` (composição de h dias, mediana por data, NaN).
- walk_forward: espiões para h e k (linhas de treino, decisões de h em h, reaproveitamento do ajuste, janela máxima), alinhamento de cada probabilidade com a sua linha (data, ticker), paralelo = sequencial com h e k > 1.
- portfolio: casos à mão para h = 1 e h = 3 (deriva, giro só na troca, caixa, ausentes), simulador independente.
- metrics: decis e spread em casos à mão.
- project: `RunKey`, formato das features, runs por chave, lookahead ponta a ponta com (h=1, k=1) e (h=5, k=2) e n_jobs = 2.
- app e CLI com as novas entradas e abas.

## 9. Depois da implementação

Refazer as features reais e rodar (h, X, k) = (1, 21, 1), (5, 21, 1), (21, 63, 1) e (1, 21, 5); medir uma vez, fora do app, o efeito de tirar as features de pares (h = 5, X = 21) para responder "a correlação ajuda?".
