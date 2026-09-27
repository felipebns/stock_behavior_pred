"""Resultados do backtest. Rode com: streamlit run app/dashboard.py

Escolha a janela de treino na barra lateral: se ela ainda não foi rodada, o botão treina (em paralelo) e salva;
as janelas já rodadas abrem na hora e são comparadas na aba "Janelas".
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
from engine.portfolio import Portfolio
from engine.project import Project

PROJECT = Project(dataclasses.replace(CONFIG, data_in=ROOT / CONFIG.data_in, data_out=ROOT / CONFIG.data_out))
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


def realized(data_out: str, window: int, run_stamp: int, features_stamp: int) -> pd.DataFrame:
    """Predictions of the decisions whose returns are already known: the ones the model is evaluated on."""
    predictions = load_run(data_out, window, run_stamp).predictions
    dates = load_features(data_out, features_stamp).market["bench_fwd_ret"].dropna().index
    return predictions[predictions["date"].isin(dates)]


@st.cache_data(show_spinner=False)
def evaluate(data_out: str, window: int, run_stamp: int, features_stamp: int):
    evaluated = realized(data_out, window, run_stamp, features_stamp)
    return metrics.model_metrics(evaluated), metrics.monthly_auc(evaluated), metrics.daily_ic(evaluated)


@st.cache_data(show_spinner=False)
def evaluate_since(data_out: str, window: int, run_stamp: int, features_stamp: int, start: pd.Timestamp) -> dict:
    evaluated = realized(data_out, window, run_stamp, features_stamp)
    return metrics.model_metrics(evaluated[evaluated["date"] >= start])


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
    for col, (label, text) in zip(st.columns(3), labeled(summary, MODEL_FORMATS).items()):
        col.metric(label, text)
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

    blocks, model = {}, {}
    for label, months in periods.items():
        part = cut(months)
        if len(part) >= 2:
            blocks.update({f"{label} · {name}": values for name, values in compare(part).items()})
            model[label] = evaluate_since(OUT, window, run_stamp, features_stamp, part["decision_date"].min())
    st.dataframe(metrics_table(blocks))
    st.markdown("**Modelo** (não depende do limiar nem do custo)")
    st.dataframe(pd.DataFrame({label: labeled(values, MODEL_FORMATS) for label, values in model.items()}))
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
        rows[f"{other} pregões"] = {**labeled(performance, METRIC_FORMATS), **labeled(model, MODEL_FORMATS)}
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
        st.write("Membros com volume financeiro mediano abaixo de US$ 1 milhão/dia (possível ticker reatribuído a outra "
                 f"empresa no Yahoo): {', '.join(quality['suspect_tickers']) or '—'}")
        st.write(f"Retornos diários com |r| > 50% entre membros (erro de dado ou spin-off): {quality['n_extreme_returns']}")
        if quality["extreme_returns"]:
            st.dataframe(pd.DataFrame(quality["extreme_returns"]), hide_index=True)
        st.write(f"Posições sem retorno realizado (contadas como 0%): {int(daily['n_missing'].sum())}")
        st.json(quality["date_ranges"])
