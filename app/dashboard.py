"""Resultados do backtest. Rode com: streamlit run app/dashboard.py

Escolha na barra lateral o modelo, o escopo (um modelo para todas as ações ou um por ação), a janela de treino X e o
retreino k: se esse run ainda não existe, o botão treina (em paralelo) e salva; os runs já feitos abrem na hora, são
comparados na aba "Runs" e servem de candidatos na aba "Walk-forward".
"""
import dataclasses
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
from engine.models import MODELS
from engine.portfolio import Portfolio
from engine.project import Project, RunKey
from engine.selection import walk_forward_selection
from engine.walk_forward import SCOPES

PROJECT = Project(dataclasses.replace(CONFIG, data_in=ROOT / CONFIG.data_in, data_out=ROOT / CONFIG.data_out))
OUT = str(PROJECT.config.data_out)
BENCH = "S&P 500 TR (intraday)"
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
WENT_UP = {1.0: "subiu", 0.0: "não subiu"}

st.set_page_config(page_title="Backtest ML — S&P 500", layout="wide")


@st.cache_resource(show_spinner="Carregando as features…", max_entries=2)
def load_features(data_out: str, stamp: int):
    return PROJECT.features(with_dataset=False)


@st.cache_resource(show_spinner=False, max_entries=16)
def load_run(data_out: str, key: RunKey, stamp: int):
    return PROJECT.load_run(key)


@st.cache_data(show_spinner="Recalculando a carteira…", max_entries=64)
def backtest(data_out: str, key: RunKey, run_stamp: int, features_stamp: int, threshold: float, cost_bps: float):
    market = load_features(data_out, features_stamp).market
    predictions = load_run(data_out, key, run_stamp).predictions
    return Portfolio(threshold, cost_bps).backtest(predictions, market)


def known(data_out: str, key: RunKey, run_stamp: int) -> pd.DataFrame:
    """Predictions whose next-day return is already known: the ones the model is evaluated on."""
    return load_run(data_out, key, run_stamp).predictions.dropna(subset=["label"])


@st.cache_data(show_spinner=False, max_entries=32)
def evaluate(data_out: str, key: RunKey, run_stamp: int) -> dict:
    evaluated = known(data_out, key, run_stamp)
    return {
        "summary": metrics.model_metrics(evaluated),
        "monthly_auc": metrics.monthly_auc(evaluated),
        "ic": metrics.daily_ic(evaluated),
        "deciles": metrics.deciles(evaluated),
        "spread": metrics.decile_spread(evaluated),
    }


@st.cache_data(show_spinner=False, max_entries=256)
def evaluate_since(data_out: str, key: RunKey, run_stamp: int, start: pd.Timestamp) -> dict:
    evaluated = known(data_out, key, run_stamp)
    return metrics.model_metrics(evaluated[evaluated["date"] >= start])


@st.cache_data(show_spinner="Escolhendo runs e limiares…", max_entries=16)
def select(data_out: str, keys: tuple, stamps: tuple, features_stamp: int, thresholds: tuple, cost_bps: float,
           lookback: int, step: int):
    """Every (run, threshold) pair is a candidate; returns the walk-forward selection and each candidate's (run, threshold)."""
    candidates, names = {}, {}
    for key, stamp in zip(keys, stamps):
        for threshold in thresholds:
            name = f"{key.name}@{threshold:.3f}"
            candidates[name] = backtest(data_out, key, stamp, features_stamp, threshold, cost_bps).daily
            names[name] = (key.label, threshold)
    return walk_forward_selection(candidates, lookback, step), names


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
    threshold = st.slider("Limiar: probabilidade mínima de subir", 0.40, 0.80, float(CONFIG.threshold), 0.005, format="%.3f")
    cost_bps = st.number_input("Custo de transação (bps por lado, em cada compra ou venda)", 0.0, 50.0, float(CONFIG.cost_bps), 0.5)
    st.caption(
        "Todo dia, compra na abertura as ações cuja probabilidade prevista de subir durante o dia passa do limiar, com "
        "peso proporcional à probabilidade, e vende no fechamento do mesmo dia; se nenhuma passar, fica em caixa rendendo "
        "a T-bill (^IRX). Escolher limiar, modelo, X ou k olhando este resultado é otimizar dentro do próprio backtest "
        "(data snooping): a aba Walk-forward faz essa escolha só com o passado."
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
st.caption(f"{key.label} · {len(daily)} dias de carteira · {daily.index.min().date()} → {daily.index.max().date()} · "
           f"gerado em {run.info.get('generated_at', '?')}")
tab_overview, tab_signal, tab_day, tab_periods, tab_runs, tab_walk, tab_model = st.tabs(
    ["Visão geral", "Previsto vs realizado", "Carteira por dia", "Recortes de período", "Runs", "Walk-forward", "Modelo"])

with tab_overview:
    for col, (label, text) in zip(st.columns(3), labeled(evaluation["summary"], MODEL_FORMATS).items()):
        col.metric(label, text)
    st.caption(
        f"{MODELS[model][0]} ({SCOPES[scope]}) retreinado a cada {retrain} pregão(ões) com os exemplos dos últimos "
        f"{window} pregões; após cada fechamento, estima a probabilidade de cada ação do índice subir da abertura ao "
        f"fechamento do dia seguinte · em média "
        f"{window_coverage['n_eligible'].mean():.0f} ações elegíveis por dia, de "
        f"{window_coverage['n_members'].mean():.0f} membros do índice."
    )
    st.subheader("Carteira vs S&P 500 Total Return (abertura → fechamento)")
    st.dataframe(metrics_table(compare(daily)))
    line_chart(curves(daily, growth), "Capital (1,0 no início)")
    st.caption("Decisão após o fechamento de t, compra na abertura de t+1 e venda no fechamento de t+1 (retorno "
               "abertura → fechamento); o custo incide na compra e na venda de cada dia investido. Arraste ou use a roda "
               "do mouse para aproximar.")
    st.markdown("**Drawdown**")
    line_chart(curves(daily, metrics.drawdown), "Drawdown", height=220, area=True)
    st.markdown("**Nº de ações na carteira**")
    line_chart(daily[["n_positions"]].rename(columns={"n_positions": "Ações"}), "Ações", height=220, y_format=".0f")

with tab_signal:
    st.caption(
        "Em cada dia, as ações com resultado já conhecido são divididas em 10 grupos pela probabilidade prevista de subir "
        "(decil 1 = menor, 10 = maior). Se o modelo tem sinal, o retorno médio cresce do decil 1 ao 10 e a fração que "
        "subiu acompanha a probabilidade prevista (pontos perto da diagonal). AUC: chance de uma ação que subiu ter "
        "recebido probabilidade maior que uma que não subiu (0,5 = sorteio). IC: correlação de postos entre "
        "probabilidade e retorno em cada dia."
    )
    table = evaluation["deciles"].rename_axis("decil").reset_index()
    if table.empty:
        st.info("Ainda não há dia com resultado conhecido neste run.")
    else:
        bars = alt.Chart(table).mark_bar().encode(
            x=alt.X("decil:O", title="Decil de probabilidade prevista"),
            y=alt.Y("fwd_ret:Q", title="Retorno médio da abertura ao fechamento do dia seguinte", axis=alt.Axis(format="%")),
            color=alt.condition(alt.datum.fwd_ret > 0, alt.value("#26a65b"), alt.value("#d64541")),
            tooltip=[alt.Tooltip("decil:O"), alt.Tooltip("fwd_ret:Q", format="+.3%")],
        )
        low = float(min(table["prob"].min(), table["label"].min())) - 0.02
        high = float(max(table["prob"].max(), table["label"].max())) + 0.02
        scale = alt.Scale(domain=[low, high])
        diagonal = alt.Chart(pd.DataFrame({"x": [low, high]})).mark_line(color="gray", strokeDash=[4, 4]).encode(
            x=alt.X("x:Q", scale=scale), y=alt.Y("x:Q", scale=scale))
        calibration = alt.Chart(table).mark_line(point=True).encode(
            x=alt.X("prob:Q", title="Probabilidade média prevista", scale=scale),
            y=alt.Y("label:Q", title="Fração que subiu", scale=scale),
            tooltip=[alt.Tooltip("decil:O"), alt.Tooltip("prob:Q", format=".3f"), alt.Tooltip("label:Q", format=".1%")],
        )
        left, right = st.columns(2)
        left.markdown("**Retorno por decil**")
        left.altair_chart(bars.properties(width=CHART_WIDTH // 2 - 20, height=260), width="content")
        right.markdown("**Calibração**")
        right.altair_chart((diagonal + calibration).properties(width=CHART_WIDTH // 2 - 20, height=260), width="content")
        st.dataframe(
            table.set_index("decil").rename(columns={"prob": "Prob. média", "label": "Subiu", "fwd_ret": "Retorno médio"})
            .style.format({"Prob. média": "{:.3f}", "Subiu": "{:.1%}", "Retorno médio": "{:+.3%}"}),
        )
        st.markdown("**Spread acumulado: 10% de maior probabilidade − 10% de menor**")
        line_chart(evaluation["spread"].cumsum().to_frame("Spread acumulado"), "Soma dos spreads", height=240)
        st.caption("Em cada dia, retorno médio das ações com as 10% maiores probabilidades menos o das 10% menores, "
                   "somado ao longo do tempo. Sobe quando a ordenação acerta; não depende do limiar e não inclui custos.")

with tab_day:
    holding_dates = list(daily.index)
    choice = st.selectbox(
        "Dia de carteira",
        key="day",
        options=range(len(holding_dates)),
        index=len(holding_dates) - 1,
        format_func=lambda i: (f"{holding_dates[i].date()}  ·  carteira {daily['gross'].iloc[i]:+.2%}  ·  "
                               f"S&P {daily['bench'].iloc[i]:+.2%}"),
    )
    day = daily.iloc[choice]
    holding_date, decision_date = holding_dates[choice], pd.Timestamp(day["decision_date"])
    cols = st.columns(6)
    cols[0].metric("Retorno bruto", f"{day['gross']:+.2%}")
    cols[1].metric("Retorno líquido", f"{day['net']:+.2%}")
    cols[2].metric(BENCH, f"{day['bench']:+.2%}")
    cols[3].metric("Ações", int(day["n_positions"]))
    cols[4].metric("Giro", f"{day['turnover']:.0%}")
    cols[5].metric("Custo", f"{day['cost']:.3%}")
    st.caption(f"Sinal calculado após o fechamento de {decision_date.date()}; compra na abertura de "
               f"{holding_date.date()} e venda no fechamento do mesmo dia.")
    held = positions[positions["holding_date"] == holding_date]
    if held.empty:
        st.info(f"Nenhuma ação passou do limiar: carteira em caixa rendendo {day['rf']:.4%} no dia.")
    else:
        table = pd.DataFrame({
            "Ticker": held["ticker"], "Prob. prevista": held["prob"], "Peso": held["weight"],
            "Retorno realizado": held["fwd_ret"],
            "Subiu?": (held["fwd_ret"] > 0).astype(float).where(held["fwd_ret"].notna()).map(WENT_UP),
            "Contribuição": held["contribution"],
        }).sort_values("Peso", ascending=False)
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
    outcome = day_predictions.dropna(subset=["fwd_ret"]).assign(went_up=lambda frame: frame["label"].map(WENT_UP))
    st.markdown(f"**Previsto × realizado em {decision_date.date()}** ({len(day_predictions)} ações do índice)")
    if outcome.empty:
        st.info("O resultado deste dia ainda não é conhecido.")
    else:
        dots = alt.Chart(outcome).mark_circle(size=45, opacity=0.7).encode(
            x=alt.X("prob:Q", title="Probabilidade prevista de subir", scale=alt.Scale(zero=False)),
            y=alt.Y("fwd_ret:Q", title="Retorno da abertura ao fechamento do dia seguinte", axis=alt.Axis(format="%")),
            color=alt.Color("went_up:N", title=None,
                            scale=alt.Scale(domain=list(WENT_UP.values()), range=["#26a65b", "#d64541"])),
            tooltip=["ticker", alt.Tooltip("prob:Q", format=".3f"), alt.Tooltip("fwd_ret:Q", format="+.2%")],
        )
        rule = alt.Chart(pd.DataFrame({"limiar": [threshold]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="limiar:Q")
        st.altair_chart((dots + rule).properties(width=CHART_WIDTH, height=320), width="content")
        st.caption("Cada ponto é uma ação do índice neste dia: à direita da linha tracejada, as que passaram do limiar e "
                   "foram compradas; acima de zero, as que subiram.")
    near = day_predictions[day_predictions["prob"] <= threshold].nlargest(10, "prob")
    if len(near):
        st.markdown("**Quase entraram** (maiores probabilidades abaixo do limiar)")
        st.dataframe(
            pd.DataFrame({"Ticker": near["ticker"], "Prob. prevista": near["prob"], "Retorno realizado": near["fwd_ret"]})
            .style.format({"Prob. prevista": "{:.3f}", "Retorno realizado": "{:+.2%}"}, na_rep="—"),
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
            "Giro médio diário": f"{other_daily['turnover'].mean():.0%}",
            "Tempo de treino (min)": fmt(load_run(OUT, other, other_stamp).info.get("runtime_minutes", float("nan")), "{:.1f}"),
        }
    st.dataframe(pd.DataFrame(rows).T)

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
            picked = selection.choices["candidate"]
            choices = selection.choices.assign(
                Run=picked.map(lambda name: names[name][0]),
                Limiar=picked.map(lambda name: names[name][1]),
            ).rename(columns={"start": "Início", "lookback_sharpe": "Sharpe no período anterior", "days": "Dias"})
            st.markdown("**Escolhas**")
            st.dataframe(choices[["Início", "Run", "Limiar", "Sharpe no período anterior", "Dias"]]
                         .style.format({"Limiar": "{:.3f}", "Sharpe no período anterior": "{:.2f}"}), hide_index=True)
            st.markdown("**Quanto tempo cada escolha ficou valendo**")
            st.dataframe(choices.groupby(["Run", "Limiar"])["Dias"].sum().sort_values(ascending=False).reset_index(),
                         hide_index=True)

with tab_model:
    st.markdown("**AUC por mês**")
    line_chart(evaluation["monthly_auc"].to_frame("AUC"), "AUC", height=220)
    st.markdown("**IC diário (média móvel de 63 dias)**")
    line_chart(evaluation["ic"].rolling(63, min_periods=20).mean().to_frame("IC"), "IC", height=220)
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
        st.write(f"Posições sem retorno realizado (contadas como 0%): {int(daily['n_missing'].sum())}")
        st.json(quality["date_ranges"])
