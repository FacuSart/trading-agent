from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


FEATURES = [
    "return_1",
    "return_3",
    "return_7",
    "ema_gap_10",
    "ema_gap_30",
    "rsi_14",
    "atr_pct_14",
    "volatility_20",
    "volume_ratio_20",
]


@dataclass(frozen=True)
class BacktestResult:
    frame: pd.DataFrame
    latest_signal: str
    latest_probability: float
    metrics: dict[str, float]
    reasons: list[str]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / window, adjust=False).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + relative_strength))


def build_dataset(prices: pd.DataFrame, fee_rate: float = 0.001) -> pd.DataFrame:
    """Crea variables usando solo información disponible al cierre de cada vela."""
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLCV: {', '.join(sorted(missing))}")

    data = prices.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)

    data["return_1"] = close.pct_change()
    data["return_3"] = close.pct_change(3)
    data["return_7"] = close.pct_change(7)
    data["ema_gap_10"] = close / close.ewm(span=10, adjust=False).mean() - 1
    data["ema_gap_30"] = close / close.ewm(span=30, adjust=False).mean() - 1
    data["rsi_14"] = _rsi(close)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    data["atr_pct_14"] = true_range.rolling(14).mean() / close
    data["volatility_20"] = data["return_1"].rolling(20).std()
    volume_mean = data["Volume"].rolling(20).mean()
    data["volume_ratio_20"] = data["Volume"] / volume_mean.replace(0, np.nan)

    data["forward_return"] = close.shift(-1) / close - 1
    data["target"] = np.where(
        data["forward_return"].notna(),
        (data["forward_return"] > fee_rate).astype(float),
        np.nan,
    )
    return data.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)


def _new_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )


def walk_forward_probabilities(
    dataset: pd.DataFrame,
    train_fraction: float = 0.65,
    refit_every: int = 20,
) -> pd.Series:
    """Predice hacia adelante; cada ajuste ve exclusivamente el pasado."""
    labeled = dataset.dropna(subset=["target"])
    if len(labeled) < 220:
        raise ValueError("Se necesitan al menos 220 velas útiles para entrenar y evaluar.")

    first_test = max(150, int(len(labeled) * train_fraction))
    probabilities = pd.Series(np.nan, index=dataset.index, dtype=float)
    model = None

    for position in range(first_test, len(labeled)):
        if model is None or (position - first_test) % refit_every == 0:
            train = labeled.iloc[:position]
            model = _new_model().fit(train[FEATURES], train["target"].astype(int))
        row = labeled.iloc[[position]]
        probabilities.loc[row.index[0]] = model.predict_proba(row[FEATURES])[:, 1][0]
    return probabilities


def _positions(probabilities: pd.Series, enter: float, exit_: float) -> pd.Series:
    state = 0.0
    values: list[float] = []
    for probability in probabilities:
        if pd.isna(probability):
            values.append(np.nan)
            continue
        if state == 0 and probability >= enter:
            state = 1.0
        elif state == 1 and probability <= exit_:
            state = 0.0
        values.append(state)
    return pd.Series(values, index=probabilities.index, dtype=float)


def _annualization(interval: str) -> int:
    return {
        "1d": 365,
        "1h": 24 * 365,
        "60m": 24 * 365,
        "30m": 48 * 365,
        "15m": 96 * 365,
    }.get(interval, 365)


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def _reasons(row: pd.Series) -> list[str]:
    reasons = []
    reasons.append("precio sobre la EMA de 30" if row["ema_gap_30"] > 0 else "precio bajo la EMA de 30")
    if row["rsi_14"] >= 70:
        reasons.append("RSI en zona de sobrecompra")
    elif row["rsi_14"] <= 30:
        reasons.append("RSI en zona de sobreventa")
    else:
        reasons.append(f"RSI neutral ({row['rsi_14']:.0f})")
    reasons.append(
        "momentum semanal positivo" if row["return_7"] > 0 else "momentum semanal negativo"
    )
    return reasons


def run_experiment(
    prices: pd.DataFrame,
    interval: str = "1d",
    fee_rate: float = 0.001,
    enter_threshold: float = 0.58,
    exit_threshold: float = 0.45,
    initial_capital: float = 10_000,
    currently_in_position: bool = False,
) -> BacktestResult:
    if not 0 <= exit_threshold < enter_threshold <= 1:
        raise ValueError("Los umbrales deben cumplir 0 <= salida < entrada <= 1.")

    data = build_dataset(prices, fee_rate)
    data["probability"] = walk_forward_probabilities(data)
    tested = data.dropna(subset=["probability", "forward_return"]).copy()
    tested["position"] = _positions(tested["probability"], enter_threshold, exit_threshold)
    tested["turnover"] = tested["position"].diff().abs().fillna(tested["position"])
    tested["strategy_return"] = (
        tested["position"] * tested["forward_return"] - tested["turnover"] * fee_rate
    )
    tested["equity"] = initial_capital * (1 + tested["strategy_return"]).cumprod()
    tested["buy_hold"] = initial_capital * (1 + tested["forward_return"]).cumprod()

    periods = _annualization(interval)
    returns = tested["strategy_return"]
    volatility = returns.std(ddof=0)
    sharpe = float(returns.mean() / volatility * sqrt(periods)) if volatility > 0 else 0.0
    metrics = {
        "total_return": float(tested["equity"].iloc[-1] / initial_capital - 1),
        "buy_hold_return": float(tested["buy_hold"].iloc[-1] / initial_capital - 1),
        "max_drawdown": _max_drawdown(tested["equity"]),
        "sharpe": sharpe,
        "trades": float(tested["turnover"].sum()),
        "win_rate": float((returns[returns != 0] > 0).mean()) if (returns != 0).any() else 0.0,
    }

    labeled = data.dropna(subset=["target"])
    final_model = _new_model().fit(labeled[FEATURES], labeled["target"].astype(int))
    latest = data.iloc[-1]
    probability = float(final_model.predict_proba(latest[FEATURES].to_frame().T)[:, 1][0])
    if not currently_in_position and probability >= enter_threshold:
        signal = "COMPRAR"
    elif currently_in_position and probability <= exit_threshold:
        signal = "SALIR"
    elif currently_in_position:
        signal = "MANTENER POSICIÓN"
    else:
        signal = "ESPERAR"

    return BacktestResult(tested, signal, probability, metrics, _reasons(latest))


# Compatibilidad con la primera versión. El motor activo y auditado vive en engine.py.
from .engine import (  # noqa: E402,F401
    FEATURES,
    BacktestResult,
    MultiHorizonResult,
    build_dataset,
    run_experiment,
    run_multi_horizon_experiment,
    walk_forward_predictions,
)
