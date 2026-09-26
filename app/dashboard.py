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
LEVELS = {"gross": "Carteira (bruta)", "net": "Carteira (líquida)"}
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


def results_stamp(results_dir: Path) -> int:
    """run_info.json is written last by `main.py run`: its mtime keys the caches to the current results."""
    info = results_dir / "run_info.json"
    return info.stat().st_mtime_ns if info.exists() else 0


@st.cache_data(show_spinner=False)
def load(results_dir: str, stamp: int):
    return load_results(Path(results_dir))


@st.cache_data(show_spinner="Recalculando a carteira…", max_entries=32)
def backtest(results_dir: str, stamp: int, threshold: float, cost_bps: float):
    results = load(results_dir, stamp)
    return run_backtest(results.predictions, results.market, threshold, cost_bps)


@st.cache_data(show_spinner=False)
def model_evaluation(results_dir: str, stamp: int):
    results = load(results_dir, stamp)
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
    stamp = results_stamp(RESULTS_DIR)
    results = load(str(RESULTS_DIR), stamp)
except FileNotFoundError:
    st.error(f"Nenhum resultado em `{RESULTS_DIR}`. Rode `python main.py run` primeiro.")
    st.stop()

with st.sidebar:
    st.header("Carteira")
    threshold = st.slider("Limiar de probabilidade", 0.40, 0.80, float(CONFIG.threshold), 0.005, format="%.3f")
    cost_bps = st.number_input("Custo de transação (bps por lado, em cada compra ou venda)", 0.0, 50.0, float(CONFIG.cost_bps), 0.5)
    st.caption(
        "Compra as ações com probabilidade prevista acima do limiar, com peso proporcional à probabilidade; "
        "se nenhuma passar, fica em caixa rendendo a T-bill (^IRX). Escolher o limiar olhando este resultado "
        f"é otimizar dentro do próprio backtest (data snooping). O default ({CONFIG.threshold}) foi fixado a priori."
    )

bt = backtest(str(RESULTS_DIR), stamp, threshold, cost_bps)
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
    cost_cols = st.columns(3)
    cost_cols[0].metric("Giro diário médio (compras + vendas)", f"{daily['turnover'].mean():.1%}",
                        help="Soma de |Δpeso| em cada abertura; metade disso é a fração da carteira trocada no dia.")
    cost_cols[1].metric("Custo total (soma)", f"{daily['cost'].sum():.2%}")
    cost_cols[2].metric("Custo médio por ano", f"{daily['cost'].mean() * metrics.TRADING_DAYS:.2%}")
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
    line_chart(curves(daily, lambda r: r.rolling(63, min_periods=20).std() * np.sqrt(metrics.TRADING_DAYS)),
               "Volatilidade", height=220)
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
    cols = st.columns(5)
    cols[0].metric("Retorno bruto", f"{day['gross']:+.2%}")
    cols[1].metric("Retorno líquido", f"{day['net']:+.2%}", help=f"custo do dia {day['cost']:.3%} (giro {day['turnover']:.1%}, compras + vendas)")
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
    summary, monthly_auc, ic, calibration = model_evaluation(str(RESULTS_DIR), stamp)
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
        st.write(f"Membros do índice sem histórico utilizável no Yahoo (preço em menos da metade dos dias em que eram "
                 f"membros — deslistados ou ticker hoje de outra empresa): {len(missing)}")
        if quality.get("missing_detail"):
            st.dataframe(pd.DataFrame(quality["missing_detail"]).rename(columns={
                "ticker": "Ticker", "member_days": "Dias como membro", "days_with_price": "Dias com preço"}), hide_index=True)
        else:
            st.write(", ".join(missing) or "—")
        st.write(f"Retornos diários com |r| > 50% entre membros (erro de dado ou spin-off): {quality.get('n_extreme_returns', 0)}")
        if quality.get("extreme_returns"):
            st.dataframe(pd.DataFrame(quality["extreme_returns"]), hide_index=True)
        st.write(f"Posições sem retorno realizado (contadas como 0%): {int(daily['n_missing'].sum())}")
        st.json(quality.get("date_ranges", {}))
