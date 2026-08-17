from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from trading_agent.engine import run_multi_horizon_experiment


st.set_page_config(page_title="Copiloto de Trading", page_icon="📈", layout="wide")
st.title("Copiloto experimental de trading")
st.caption("Swing trading diario a 5–20 ruedas. Solo investigación y simulación.")

with st.sidebar:
    st.header("Configuración")
    symbol_options = {
        "Bitcoin (BTC-USD)": "BTC-USD",
        "Ethereum (ETH-USD)": "ETH-USD",
        "Solana (SOL-USD)": "SOL-USD",
        "S&P 500 ETF (SPY)": "SPY",
        "Nasdaq 100 ETF (QQQ)": "QQQ",
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "Nvidia (NVDA)": "NVDA",
        "Mercado Libre (MELI)": "MELI",
        "Galicia Argentina (GGAL.BA)": "GGAL.BA",
        "Otro símbolo...": None,
    }
    selected_symbol = st.selectbox("Activo", list(symbol_options))
    if symbol_options[selected_symbol] is None:
        symbol = st.text_input("Símbolo de Yahoo Finance", "AMZN").strip().upper()
    else:
        symbol = symbol_options[selected_symbol]
    position_label = st.radio(
        "Mi situación actual",
        ["No tengo posición", "Ya estoy comprado"],
        help="Esto determina si la señal puede ser comprar/mantener/salir.",
    )
    holding_days = 0
    if position_label == "Ya estoy comprado":
        holding_days = st.number_input("Ruedas desde la compra", 0, 60, 0, 1)
    st.text_input("Temporalidad", "1 día (fija)", disabled=True)
    st.caption("Se analizan automáticamente horizontes de 5, 10 y 20 ruedas.")
    fee_pct = st.number_input("Costo por operación (%)", 0.0, 2.0, 0.10, 0.01)
    capital = st.number_input("Capital simulado", 100.0, 10_000_000.0, 10_000.0, 100.0)
    allocation_pct = st.slider("Capital máximo por operación (%)", 5, 100, 25, 5)
    with st.expander("Configuración avanzada"):
        enter = st.slider("Puntaje mínimo para entrar", 0.50, 0.80, 0.58, 0.01)
        exit_ = st.slider("Salir por debajo de", 0.20, 0.49, 0.45, 0.01)
        st.caption("No cambies estos valores mirando qué configuración gana el backtest.")
    run = st.button("Analizar ahora", type="primary", use_container_width=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(ticker: str) -> pd.DataFrame:
    if not ticker:
        raise ValueError("Ingresá un símbolo.")
    frame = yf.download(
        ticker,
        period="10y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        timeout=20,
    )
    if frame.empty:
        raise ValueError("Yahoo Finance no devolvió datos para ese símbolo.")
    # Política conservadora: la vela fechada hoy se considera incompleta, incluso tras el cierre.
    today_utc = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    dates = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    closed = frame.loc[dates < today_utc]
    if len(closed) < 500:
        raise ValueError(f"Solo se recibieron {len(closed)} velas diarias completas; hacen falta al menos 500.")
    return closed


if not run:
    st.info("Elegí el activo y presioná **Analizar ahora**. El sistema usa una única configuración temporal para evitar elegir el resultado más favorable.")
    st.stop()

try:
    with st.spinner("Descargando datos, entrenando y simulando..."):
        prices = load_prices(symbol)
        result = run_multi_horizon_experiment(
            prices,
            interval="1d",
            fee_rate=fee_pct / 100,
            enter_threshold=enter,
            exit_threshold=exit_,
            initial_capital=capital,
            currently_in_position=position_label == "Ya estoy comprado",
            current_holding_bars=int(holding_days),
            max_holding_bars=20,
            allocation=allocation_pct / 100,
        )
except Exception as exc:
    st.error(f"No se pudo completar el análisis: {exc}")
    st.stop()

signal_color = {
    "COMPRAR": "green",
    "MANTENER POSICIÓN": "green",
    "SALIR": "red",
    "ESPERAR": "orange",
    "SEÑAL INCIERTA": "orange",
    "MODELO NO VALIDADO": "red",
}[result.latest_signal]
st.markdown(f"## Señal actual: :{signal_color}[{result.latest_signal}]")
st.write(
    f"Puntajes alcistas entre **{result.score_range[0]:.1%} y {result.score_range[1]:.1%}**. "
    f"{result.agreement}."
)
st.write(f"Horizontes validados: **{result.validated_horizons}/{len(result.results)}**.")
st.caption(f"Última vela completa utilizada: {prices.index[-1]:%Y-%m-%d}.")
if result.latest_signal in {"MODELO NO VALIDADO", "SEÑAL INCIERTA"}:
    st.error("No abrir una operación nueva con esta lectura. El sistema no demostró una ventaja estable.")
else:
    st.warning("Es una señal experimental, no una certeza ni una autorización para operar dinero real.")

rows = []
for horizon, analysis in result.results.items():
    m = analysis.metrics
    rows.append(
        {
            "Horizonte": f"{horizon} ruedas",
            "Puntaje": analysis.latest_probability,
            "Banda": m["direction"],
            "Validado": "Sí" if analysis.validation_passed else "No",
            "AUC": m["roc_auc"],
            "Brier": m["brier"],
            "Brier base": m["baseline_brier"],
            "Retorno 1 año": m["total_return"],
            "Referencia": m["buy_hold_return"],
            "Drawdown": m["max_drawdown"],
            "Operaciones": m["trades"],
        }
    )
st.subheader("Estabilidad y validación")
st.dataframe(
    pd.DataFrame(rows).set_index("Horizonte").style.format(
        {
            "Puntaje": "{:.1%}", "AUC": "{:.3f}", "Brier": "{:.3f}",
            "Brier base": "{:.3f}", "Retorno 1 año": "{:.1%}",
            "Referencia": "{:.1%}", "Drawdown": "{:.1%}",
        }
    ),
    use_container_width=True,
)

primary = result.results[10]
m = primary.metrics
columns = st.columns(6)
columns[0].metric("Retorno estrategia", f"{m['total_return']:.1%}")
columns[1].metric("Referencia misma exposición", f"{m['buy_hold_return']:.1%}")
columns[2].metric("Drawdown máximo", f"{m['max_drawdown']:.1%}")
columns[3].metric("Sharpe", f"{m['sharpe']:.2f}")
columns[4].metric("Operaciones cerradas", f"{m['trades']}")
columns[5].metric("Operaciones ganadoras", f"{m['win_rate']:.1%}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=primary.frame.index, y=primary.frame["equity"], name="Estrategia"))
fig.add_trace(go.Scatter(x=primary.frame.index, y=primary.frame["buy_hold"], name="Referencia"))
fig.update_layout(title="Últimas 252 ruedas fuera de muestra — horizonte 10", yaxis_title="Capital simulado", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

with st.expander("Ver últimas decisiones"):
    table = primary.frame[["Close", "probability", "position", "holding_bars", "strategy_return"]].tail(30).copy()
    table.columns = ["Cierre", "Puntaje", "Posición", "Ruedas en posición", "Retorno estrategia"]
    st.dataframe(table, use_container_width=True)

st.write("Contexto técnico: " + "; ".join(primary.reasons) + ".")
st.caption("Estas observaciones describen indicadores; no son una explicación causal del modelo.")

st.caption(
    "Limitaciones: datos diarios de Yahoo para investigación, sin noticias ni libro de órdenes. "
    "La señal se calcula al cierre y el backtest ejecuta en la apertura siguiente; el desempeño pasado "
    "no garantiza resultados futuros."
)
