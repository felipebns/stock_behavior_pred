# Rodada 4 — volta ao "sobe ou desce" no dia seguinte, pipeline estável e rápido

Data: 2026-10-01. Pedido do dono: voltar ao rótulo antigo (a ação sobe ou não), manter o limiar de confiança ajustável no app, tirar o horizonte h por enquanto (prever só o próximo dia) e deixar o pipeline estável e o mais rápido possível. A rodada 3 (rótulo relativo, horizonte h) fica no histórico do git (commits 9512cfd e 191308b).

## 1. Modelagem (volta à rodada 2)

- Decisão após o fechamento de *t*, todo pregão.
- **Rótulo:** 1 se `fwd_ret = (Open[t+2] + Div[t+2]) / Open[t+1] − 1 > 0`; NaN se `fwd_ret` é NaN.
- **Treino da decisão *t*:** linhas *s* ∈ [*t*−X−1, *t*−2].
- O dataset salvo volta a ter `fwd_ret` e `label`; sai o painel `returns.parquet`; `FEATURES_FORMAT = 3`.
- Sai `horizon` de `Config`, `RunKey`, `WalkForward`, `Portfolio`, CLI e app; sai `horizon_target`.
- **Fica** o retreino a cada k pregões (`RunKey(window, retrain_every, model)`, pasta `runs/lightgbm-w21-k1/`), porque é a alavanca de desempenho que não depende de mudar o modelo.

## 2. Carteira (volta à rodada 2)

- Todo dia: compra na abertura de *t*+1 as ações com P(subir) > limiar, peso ∝ probabilidade, vende na abertura de *t*+2.
- Se nenhuma passar: caixa no ^IRX. Custo = giro × bps.
- `Portfolio.backtest(predictions, market)` usa o `fwd_ret` das previsões (equivale ao da rodada 3 com h = 1; o revisor da rodada 3 mediu diferença ≤ 4e-16).
- Janela sem nenhum retorno realizado devolve tabelas vazias, sem erro.
- O simulador evento a evento continua validando a carteira (h = 1).

## 3. Estabilidade

- Janela de treino com uma classe só (por exemplo X = 1 num dia em que tudo caiu): prevê a frequência dessa classe; hoje o LightGBM inverteria a probabilidade.
- Início sem espaço para nenhuma janela: mensagem clara em vez de "máximo" negativo.
- CLI: `--window` e `--retrain` precisam ser inteiros ≥ 1; erros esperados (dados ausentes, janela grande demais, falha de download) viram uma mensagem `erro: …` em vez de traceback.
- App: caches com `max_entries` (memória limitada em sessões longas).

## 4. Desempenho (medido em 2026-10-01, i7-10510U, 4 núcleos / 8 threads)

- Um ajuste do LightGBM com ~10 mil linhas leva ~0,5–0,57 s; o run é quase só treino.
- Paralelo: n_jobs = 1 → 82,5 s, 4 → 37,0 s, 8 → 30,0 s para 144 ajustes: o teto do hardware.
- Ajustes que preservam o resultado (`deterministic=False`, `force_row_wise`) não aceleram.
- Os que aceleram mudam o modelo (`max_bin=63` −24%, 100 árvores com taxa 0,1 −36%): ficam como opção do dono, não entram.
- O retreino a cada k divide o tempo por ~k.

## 5. App

- **Barra lateral:** janela X, retreino k, limiar de probabilidade de subir, custo.
- **Abas:**
  - Visão geral;
  - Previsto vs realizado (decis por probabilidade de subir: retorno médio, fração que subiu, calibração, spread 10% maior − 10% menor);
  - Carteira por dia (o que foi comprado, prob, peso, retorno, "subiu?", contribuição, dispersão prob × retorno de todas as ações do dia, quase entraram);
  - Recortes de período;
  - Runs;
  - Modelo.

## 6. Verificação

- Testes de cada módulo.
- Lookahead ponta a ponta com k = 1 e k = 3 e n_jobs = 2.
- Mutação no recorte de treino.
- Simulador independente.
- **Regressão com os dados reais:** o run (X = 21, k = 1) tem de reproduzir a rodada 2 depois da correção PARA/BBT: 803.587 previsões, bruto +51,1%, líquido −45,6%, AUC 0,4898, IC 0,0014.
