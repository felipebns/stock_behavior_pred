"""Resultados do backtest. Rode com: streamlit run app/dashboard.py

Escolha na barra lateral o horizonte h, a janela de treino X e o retreino k: se esse run ainda não existe, o botão
treina (em paralelo) e salva; os runs já feitos abrem na hora e são comparados na aba "Runs".
"""
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from config.config import CONFIG
from engine import metrics
from engine.portfolio import Portfolio
from engine.project import Project, RunKey
from engine.walk_forward import MODELS

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
BEAT = {1.0: "bateu", 0.0: "não bateu"}

st.set_page_config(page_title="Backtest ML — S&P 500", layout="wide")


@st.cache_resource(show_spinner="Carregando as features…")
def load_features(data_out: str, stamp: int):
    return PROJECT.features(with_dataset=False)


@st.cache_resource(show_spinner=False)
def load_run(data_out: str, key: RunKey, stamp: int):
    return PROJECT.load_run(key)


@st.cache_data(show_spinner="Recalculando a carteira…", max_entries=64)
def backtest(data_out: str, key: RunKey, run_stamp: int, features_stamp: int, threshold: float, cost_bps: float):
    features = load_features(data_out, features_stamp)
    predictions = load_run(data_out, key, run_stamp).predictions
    return Portfolio(threshold, cost_bps).backtest(predictions, features.market, features.returns, key.horizon)


def known(data_out: str, key: RunKey, run_stamp: int) -> pd.DataFrame:
    """Predictions whose outcome (the h-session return) is already known: the ones the model is evaluated on."""
    return load_run(data_out, key, run_stamp).predictions.dropna(subset=["label"])


@st.cache_data(show_spinner=False)
def evaluate(data_out: str, key: RunKey, run_stamp: int) -> dict:
    evaluated = known(data_out, key, run_stamp)
    return {
        "summary": metrics.model_metrics(evaluated),
        "monthly_auc": metrics.monthly_auc(evaluated),
        "ic": metrics.daily_ic(evaluated),
        "deciles": metrics.deciles(evaluated),
        "spread": metrics.decile_spread(evaluated),
    }


@st.cache_data(show_spinner=False)
def evaluate_since(data_out: str, key: RunKey, run_stamp: int, start: pd.Timestamp) -> dict:
    evaluated = known(data_out, key, run_stamp)
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
    st.header("Run")
    model = st.selectbox("Modelo", list(MODELS)) if len(MODELS) > 1 else next(iter(MODELS))
    horizon = int(st.number_input("Horizonte h: pregões entre decisões e do rótulo (1 = diário, 5 = semanal, 21 = mensal)",
                                  min_value=1, max_value=63, value=CONFIG.horizon_days, step=1))
    window = int(st.number_input("Janela de treino X: pregões de exemplos (5 = 1 semana, 21 = 1 mês, 63 = 3 meses)",
                                 min_value=1, max_value=1000, value=CONFIG.train_window_days, step=1))
    retrain = int(st.number_input("Retreinar a cada k decisões (1 = em toda decisão)",
                                  min_value=1, max_value=100, value=CONFIG.retrain_every, step=1))
    key = RunKey(horizon, window, retrain, model)
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
    threshold = st.slider("Limiar: probabilidade mínima de bater a mediana", 0.40, 0.80, float(CONFIG.threshold), 0.005,
                          format="%.3f")
    cost_bps = st.number_input("Custo de transação (bps por lado, em cada compra ou venda)", 0.0, 50.0, float(CONFIG.cost_bps), 0.5)
    st.caption(
        "A cada decisão, compra as ações cuja probabilidade prevista de render acima da mediana do índice passa do limiar, "
        "com peso proporcional à probabilidade, e segura até a próxima decisão; se nenhuma passar, fica em caixa rendendo "
        "a T-bill (^IRX). Escolher limiar, h, X ou k olhando este resultado é otimizar dentro do próprio backtest "
        "(data snooping)."
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
st.caption(f"{key.label} · {int(daily['rebalance'].sum())} decisões · {len(daily)} dias de carteira · "
           f"{daily.index.min().date()} → {daily.index.max().date()} · gerado em {run.info.get('generated_at', '?')}")
tab_overview, tab_signal, tab_decision, tab_periods, tab_runs, tab_model = st.tabs(
    ["Visão geral", "Previsto vs realizado", "Carteira por decisão", "Recortes de período", "Runs", "Modelo"])

with tab_overview:
    for col, (label, text) in zip(st.columns(3), labeled(evaluation["summary"], MODEL_FORMATS).items()):
        col.metric(label, text)
    st.caption(
        f"{model} retreinado a cada {retrain} decisão(ões) com os exemplos dos últimos {window} pregões; a cada {horizon} "
        f"pregão(ões), estima a probabilidade de cada ação do índice render acima da mediana do índice nos {horizon} "
        f"pregões seguintes · em média {window_coverage['n_eligible'].mean():.0f} ações elegíveis por dia, de "
        f"{window_coverage['n_members'].mean():.0f} membros do índice."
    )
    st.subheader("Carteira vs S&P 500 Total Return")
    st.dataframe(metrics_table(compare(daily)))
    line_chart(curves(daily, growth), "Capital (1,0 no início)")
    st.caption("Decisão após o fechamento de t, compra na abertura de t+1 e troca só na abertura seguinte à próxima "
               "decisão; o custo incide só nas trocas (retornos abertura → abertura, com dividendos). Arraste ou use a "
               "roda do mouse para aproximar.")
    st.markdown("**Drawdown**")
    line_chart(curves(daily, metrics.drawdown), "Drawdown", height=220, area=True)
    st.markdown("**Nº de ações na carteira**")
    line_chart(daily[["n_positions"]].rename(columns={"n_positions": "Ações"}), "Ações", height=220, y_format=".0f")

with tab_signal:
    st.caption(
        "Em cada decisão, as ações com resultado já conhecido são divididas em 10 grupos pela probabilidade prevista "
        "(decil 1 = menor, 10 = maior). Se o modelo tem sinal, o retorno acima da mediana cresce do decil 1 ao 10 e a "
        "fração que bateu a mediana acompanha a probabilidade prevista (pontos perto da diagonal). AUC: chance de uma "
        "ação que bateu a mediana ter recebido probabilidade maior que uma que não bateu (0,5 = sorteio). IC: correlação "
        "de postos entre probabilidade e retorno em cada decisão."
    )
    table = evaluation["deciles"].rename_axis("decil").reset_index()
    if table.empty:
        st.info("Ainda não há decisão com resultado conhecido neste run.")
    else:
        bars = alt.Chart(table).mark_bar().encode(
            x=alt.X("decil:O", title="Decil de probabilidade prevista"),
            y=alt.Y("relative:Q", title="Retorno médio acima da mediana do dia", axis=alt.Axis(format="%")),
            color=alt.condition(alt.datum.relative > 0, alt.value("#26a65b"), alt.value("#d64541")),
            tooltip=[alt.Tooltip("decil:O"), alt.Tooltip("relative:Q", format="+.3%")],
        )
        low = float(min(table["prob"].min(), table["label"].min())) - 0.02
        high = float(max(table["prob"].max(), table["label"].max())) + 0.02
        scale = alt.Scale(domain=[low, high])
        diagonal = alt.Chart(pd.DataFrame({"x": [low, high]})).mark_line(color="gray", strokeDash=[4, 4]).encode(
            x=alt.X("x:Q", scale=scale), y=alt.Y("x:Q", scale=scale))
        calibration = alt.Chart(table).mark_line(point=True).encode(
            x=alt.X("prob:Q", title="Probabilidade média prevista", scale=scale),
            y=alt.Y("label:Q", title="Fração que bateu a mediana", scale=scale),
            tooltip=[alt.Tooltip("decil:O"), alt.Tooltip("prob:Q", format=".3f"), alt.Tooltip("label:Q", format=".1%")],
        )
        left, right = st.columns(2)
        left.markdown("**Retorno por decil**")
        left.altair_chart(bars.properties(width=CHART_WIDTH // 2 - 20, height=260), width="content")
        right.markdown("**Calibração**")
        right.altair_chart((diagonal + calibration).properties(width=CHART_WIDTH // 2 - 20, height=260), width="content")
        st.dataframe(
            table.set_index("decil").rename(columns={
                "prob": "Prob. média", "label": "Bateu a mediana", "relative": "Retorno acima da mediana"})
            .style.format({"Prob. média": "{:.3f}", "Bateu a mediana": "{:.1%}", "Retorno acima da mediana": "{:+.3%}"}),
        )
        st.markdown("**Spread acumulado: 10% de maior probabilidade − 10% de menor**")
        line_chart(evaluation["spread"].cumsum().to_frame("Spread acumulado"), "Soma dos spreads", height=240)
        st.caption("Em cada decisão, retorno médio das ações com as 10% maiores probabilidades menos o das 10% menores, "
                   "somado ao longo do tempo. Sobe quando a ordenação acerta; não depende do limiar e não inclui custos.")

with tab_decision:
    period_return = daily.groupby("decision_date")[["gross", "net", "bench"]].apply(lambda part: (1.0 + part).prod() - 1.0)
    decisions = list(period_return.index)
    choice = st.selectbox(
        "Decisão",
        options=range(len(decisions)),
        index=len(decisions) - 1,
        format_func=lambda i: (f"{decisions[i].date()}  ·  carteira {period_return['gross'].iloc[i]:+.2%}  ·  "
                               f"S&P {period_return['bench'].iloc[i]:+.2%}"),
    )
    decision = decisions[choice]
    decision_return = period_return.loc[decision]
    days = daily[daily["decision_date"] == decision]
    trade = days.iloc[0]
    cols = st.columns(6)
    cols[0].metric("Retorno bruto", f"{decision_return['gross']:+.2%}")
    cols[1].metric("Retorno líquido", f"{decision_return['net']:+.2%}")
    cols[2].metric(BENCH, f"{decision_return['bench']:+.2%}")
    cols[3].metric("Ações", int(trade["n_positions"]))
    cols[4].metric("Giro na troca", f"{trade['turnover']:.0%}")
    cols[5].metric("Custo", f"{trade['cost']:.3%}")
    st.caption(f"Sinal calculado após o fechamento de {decision.date()}; compra na abertura de {days.index[0].date()} e "
               f"segura por {len(days)} pregão(ões), até a abertura do pregão seguinte a {days.index[-1].date()}.")
    day_predictions = run.predictions[run.predictions["date"] == decision]
    median = day_predictions["fwd_ret"].median()
    held = positions[positions["decision_date"] == decision].merge(day_predictions[["ticker", "label"]], on="ticker")
    if held.empty:
        st.info("Nenhuma ação passou do limiar: carteira em caixa rendendo a T-bill (^IRX) no período.")
    else:
        table = pd.DataFrame({
            "Ticker": held["ticker"], "Prob. prevista": held["prob"], "Peso": held["weight"],
            "Retorno no período": held["realized"], "Acima da mediana": held["realized"] - median,
            "Bateu a mediana?": held["label"].map(BEAT), "Contribuição": held["contribution"],
        }).sort_values("Peso", ascending=False)
        st.dataframe(
            table.style.format({"Prob. prevista": "{:.3f}", "Peso": "{:.2%}", "Retorno no período": "{:+.2%}",
                                "Acima da mediana": "{:+.2%}", "Contribuição": "{:+.3%}"}, na_rep="—")
            .map(shade, subset=["Acima da mediana", "Contribuição"]),
            hide_index=True, height=min(38 * (len(table) + 1), 520),
        )
        st.info(f"Soma das contribuições = {held['contribution'].sum():+.4%} = retorno bruto do período "
                f"({decision_return['gross']:+.4%}).")
        if days["n_missing"].any():
            st.warning("Há ação sem retorno em algum dia do período (deslistagem/halt): o valor dela ficou congelado nesses dias.")
    outcome = day_predictions.assign(relative=day_predictions["fwd_ret"] - median,
                                     beat=day_predictions["label"].map(BEAT)).dropna(subset=["relative"])
    st.markdown(f"**Previsto × realizado em {decision.date()}** ({len(day_predictions)} ações do índice)")
    if outcome.empty:
        st.info("O resultado desta decisão ainda não é conhecido: o horizonte passa do último dado.")
    else:
        dots = alt.Chart(outcome).mark_circle(size=45, opacity=0.7).encode(
            x=alt.X("prob:Q", title="Probabilidade prevista de bater a mediana", scale=alt.Scale(zero=False)),
            y=alt.Y("relative:Q", title="Retorno realizado menos a mediana do dia", axis=alt.Axis(format="%")),
            color=alt.Color("beat:N", title=None, scale=alt.Scale(domain=list(BEAT.values()), range=["#26a65b", "#d64541"])),
            tooltip=["ticker", alt.Tooltip("prob:Q", format=".3f"), alt.Tooltip("relative:Q", format="+.2%")],
        )
        rule = alt.Chart(pd.DataFrame({"limiar": [threshold]})).mark_rule(color="black", strokeDash=[4, 4]).encode(x="limiar:Q")
        st.altair_chart((dots + rule).properties(width=CHART_WIDTH, height=320), width="content")
        st.caption("Cada ponto é uma ação do índice nesta decisão: à direita da linha tracejada, as que passaram do limiar "
                   "e foram compradas; acima de zero, as que renderam mais que a mediana no período.")
    near = day_predictions[day_predictions["prob"] <= threshold].nlargest(10, "prob")
    if len(near):
        st.markdown("**Quase entraram** (maiores probabilidades abaixo do limiar)")
        st.dataframe(
            pd.DataFrame({"Ticker": near["ticker"], "Prob. prevista": near["prob"], "Retorno no período": near["fwd_ret"],
                          "Acima da mediana": near["fwd_ret"] - median})
            .style.format({"Prob. prevista": "{:.3f}", "Retorno no período": "{:+.2%}", "Acima da mediana": "{:+.2%}"},
                          na_rep="—"),
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
            "Giro médio por troca": f"{other_daily.loc[other_daily['rebalance'], 'turnover'].mean():.0%}",
            "Tempo de treino (min)": fmt(load_run(OUT, other, other_stamp).info.get("runtime_minutes", float("nan")), "{:.1f}"),
        }
    st.dataframe(pd.DataFrame(rows).T)

with tab_model:
    st.markdown("**AUC por mês**")
    line_chart(evaluation["monthly_auc"].to_frame("AUC"), "AUC", height=220)
    span = max(5, 63 // horizon)
    st.markdown(f"**IC por decisão (média móvel de {span} decisões)**")
    line_chart(evaluation["ic"].rolling(span, min_periods=max(2, span // 3)).mean().to_frame("IC"), "IC", height=220)
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
        st.write(f"Dias de posição sem retorno (contados como 0%): {int(daily['n_missing'].sum())}")
        st.json(quality["date_ranges"])
